"""Update-based schedule and fixed validation MPJPE for rot6d68 continuation."""
from collections import defaultdict
import csv,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .common import G1Kinematics,sha256,write_json
from .rot6d68 import decode_native,encode_native
from .distributed import KEYS,denominators,ddp_objective

def learning_rate(update,total,warmup,hold_until,peak=3e-5,floor=1e-6):
    if not 0<=update<total or not 2<=warmup<=hold_until<total-1:raise ValueError('Invalid continuation schedule')
    if update<warmup:return floor+(peak-floor)*update/(warmup-1)
    if update<hold_until:return peak
    fraction=(update-hold_until)/(total-1-hold_until)
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*fraction))

def fixed_rows(native_root,fixed_root):
    native_root,fixed_root=Path(native_root),Path(fixed_root)
    m=json.loads((fixed_root/'manifest.json').read_text())
    if not m['complete'] or m['split']!='val' or m['native_manifest_sha256']!=sha256(native_root/'manifest.json'):raise ValueError('Wrong frozen validation cohort')
    for filename,key in [('rows.json','rows_sha256'),('samples.txt','samples_sha256')]:
        if sha256(fixed_root/filename)!=m[key]:raise ValueError('Frozen cohort changed')
    rows=json.loads((fixed_root/'rows.json').read_text())
    assert len(rows)==m['samples'] and len({r['id'] for r in rows})==len(rows)
    assert all(r['split']=='val' for r in rows)
    return rows,m

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

def train_epoch(wrapped,ddp,loader,optimizer,device,rank,world,weights,start_step,parent_step,schedule):
    wrapped.train();acc=torch.zeros(len(KEYS)+2,device=device,dtype=torch.float64);began=time.time()
    for i,batch in enumerate(loader):
        lr=learning_rate(start_step+i-parent_step,**schedule)
        for group in optimizer.param_groups:group['lr']=lr
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        terms=ddp(batch,sample=True);values=torch.stack([terms[k].detach() for k in KEYS])
        counts=denominators(batch);global_counts=counts.clone();dist.all_reduce(global_counts)
        finite=torch.isfinite(values).all().int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise FloatingPointError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True);ddp_objective(terms,counts,global_counts,world,weights).backward()
        torch.nn.utils.clip_grad_norm_(wrapped.vae.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        numer=values*counts;dist.all_reduce(numer);values=numer/global_counts.clamp_min(1)
        total=sum(weights[k]*values[j] for j,k in enumerate(KEYS));frames=global_counts[0]
        acc[:-2]+=values.double()*frames;acc[-2]+=total.double()*frames;acc[-1]+=frames
        if rank==0 and (i==0 or i+1==len(loader)):
            print(f'Train batch {i+1}/{len(loader)} | step={start_step+i+1} lr={lr:.7g} loss={float(acc[-2]/acc[-1]):.6f} elapsed={time.time()-began:.1f}s',flush=True)
    return dict(zip((*KEYS,'total'),(acc[:-1]/acc[-1]).cpu().tolist())),len(loader),lr
