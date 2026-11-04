"""Frozen validation MPJPE for the absolute-yaw representation."""
from collections import defaultdict
import csv,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .common import G1Kinematics,write_json
from .rot6d69 import decode_native,encode_native
@torch.inference_mode()
def evaluate_fixed(model,native_root,rot_root,rows,output,epoch,step,rank,world,device):
    t=time.time();model.eval();groups=defaultdict(list)
    for row in rows:groups[row['frames']].append(row)
    fk=G1Kinematics(Path(native_root)/'kinematic.xml');local=[];index=0
    for length,group in sorted(groups.items()):
        for start in range(0,len(group),32):
            batch=group[start:start+32];assigned=index%world==rank;index+=1
            if not assigned:continue
            gt=np.stack([np.load(Path(native_root)/'motions'/f"{r['id']}.npy") for r in batch])
            feature=np.stack([np.load(Path(rot_root)/'motions'/f"{r['id']}.npy") for r in batch])
            anchors=torch.as_tensor(np.stack([encode_native(x)[1][0] for x in gt]),device=device,dtype=torch.float32)
            pred=decode_native(model.reconstruct(torch.as_tensor(feature,device=device),sample=False)[0],anchors).cpu().numpy()
            for row,truth,guess in zip(batch,gt,pred):
                gp,pp=fk.positions(truth),fk.positions(guess)
                ge=np.linalg.norm(pp-gp,axis=-1)*1000
                re=np.linalg.norm((pp-guess[:,None,:3])-(gp-truth[:,None,:3]),axis=-1)*1000
                local.append(dict(sample_id=row['id'],frames=len(truth),global_mpjpe_mm=float(ge.mean()),root_relative_mpjpe_mm=float(re.mean())))
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,local);values=[v for part in gathered for v in part]
    else:values=local
    values.sort(key=lambda r:r['sample_id']);assert {r['sample_id'] for r in values}=={r['id'] for r in rows} and len(values)==len(rows)
    summary=dict(complete=True,epoch=epoch,step=step,samples=len(values),frames=sum(r['frames'] for r in values),seconds=time.time()-t,metrics={})
    for key in ('global_mpjpe_mm','root_relative_mpjpe_mm'):
        x=np.array([r[key] for r in values]);summary['metrics'][key]=dict(action_mean=float(x.mean()),action_p95=float(np.percentile(x,95)),pooled=float(np.average(x,weights=[r['frames'] for r in values])))
    if rank==0:
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        with (output/'per_sample.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
        write_json(output/'summary.json',summary)
        print(f"Fixed VAL epoch {epoch}: global MPJPE={summary['metrics']['global_mpjpe_mm']['action_mean']:.4f} mm, root-relative={summary['metrics']['root_relative_mpjpe_mm']['action_mean']:.4f} mm | {len(rows)} clips",flush=True)
    return summary
