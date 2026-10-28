"""Full-test orientation oracle diagnostics; export previously selected visual clips."""
import sys,json,csv,shutil,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.common import G1Kinematics,rotation_6d_to_matrix,extract_root_yaw,yaw_to_matrix,sha256
from g1.model import G1CausalVAE
from g1.rot6d68 import encode_native,decode_native
from g1.data import read_index
OUT=ROOT/'outputs/rot68_epoch1000_analysis';DATA=ROOT/'outputs/bones_g1_native38';ROT=ROOT/'outputs/bones_g1_rot6d68'
def main():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 selections=json.loads((ROOT/'outputs/rot68_reconstruction_comparison/selection.json').read_text())
 selected={r['id']:r for r in selections}
 for r in selections:
  d=OUT/'clips'/r['slug'];d.mkdir(parents=True)
  for key in ('gt','vae_best','rot68'):shutil.copy2(ROOT/'outputs/rot68_reconstruction_comparison/clips'/r['slug']/f'{key}.npy',d/f'{key}.npy')
 rows=read_index(DATA,'test');groups=defaultdict(list)
 for r in rows:groups[r['frames']].append(r)
 fk=G1Kinematics(DATA/'kinematic.xml');summary={}
 for name,run,root in [('native38','g1_causal_textop_500ep',DATA),('rot68_final','g1_rot68_continue_ep0500_1000_bs2048_mpjpe',ROT)]:
  path=ROOT/'outputs'/run/'best.pt';c=torch.load(path,map_location='cpu',weights_only=False)
  model=G1CausalVAE(np.load(root/'mean.npy'),np.load(root/'std.npy'),**c['identity']['model_params']).cuda().eval();model.load_state_dict(c['model'])
  refs={r['sample_id']:r for r in csv.DictReader((path.parent/'eval_best/per_sample.csv').open())}
  records=[];t=time.time()
  with torch.inference_mode():
   for length,group in sorted(groups.items()):
    for start in range(0,len(group),32):
     batch=group[start:start+32];gt=np.stack([np.load(DATA/'motions'/f"{r['id']}.npy") for r in batch]);x=np.stack([np.load(root/'motions'/f"{r['id']}.npy") for r in batch])
     pred=model.reconstruct(torch.tensor(x,device='cuda'),sample=False)[0]
     if name=='rot68_final':pred=decode_native(pred,torch.tensor(np.stack([encode_native(g)[1][0] for g in gt]),device='cuda',dtype=torch.float32))
     for r,g,p in zip(batch,gt,pred.cpu().numpy()):
      gp=fk.positions(g);pp=fk.positions(p);gr=gp-g[:,None,:3];pr=pp-p[:,None,:3]
      rg=rotation_6d_to_matrix(g[:,3:9]);rp=rotation_6d_to_matrix(p[:,3:9])
      local=np.einsum('tji,tkj->tki',rp,pr)
      corrected=np.einsum('tij,tkj->tki',rg,local)
      yawfix=yaw_to_matrix(extract_root_yaw(rg)-extract_root_yaw(rp));yc=np.einsum('tij,tkj->tki',yawfix,pr)
      rec=dict(sample_id=r['id'],frames=length,global_mpjpe_mm=float(np.linalg.norm(pp-gp,axis=-1).mean()*1000),root_relative_mpjpe_mm=float(np.linalg.norm(pr-gr,axis=-1).mean()*1000),gt_yaw_oracle_mm=float(np.linalg.norm(yc-gr,axis=-1).mean()*1000),gt_root_rotation_oracle_mm=float(np.linalg.norm(corrected-gr,axis=-1).mean()*1000))
      for k in ('global_mpjpe_mm','root_relative_mpjpe_mm'):np.testing.assert_allclose(rec[k],float(refs[r['id']][k]),atol=.02,rtol=1e-4)
      records.append(rec)
      if r['id'] in selected and name=='rot68_final':
       chosen=selected[r['id']];np.save(OUT/'clips'/chosen['slug']/'rot68_final.npy',p)
       chosen['metrics']['rot68_final']={k:rec[k] for k in ('global_mpjpe_mm','root_relative_mpjpe_mm')}
    if len(records)%1000<32:print(name,len(records),round(time.time()-t,1),flush=True)
  assert len(records)==8333
  with (OUT/f'{name}_diagnostics.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
  summary[name]=dict(samples=len(records),epoch=c['epoch'],step=c['step'],checkpoint_sha256=sha256(path),metrics={k:float(np.mean([r[k] for r in records])) for k in records[0] if k not in ('sample_id','frames')},seconds=time.time()-t)
  print(json.dumps(summary[name]),flush=True);del model,c
 (OUT/'diagnostics.json').write_text(json.dumps(summary,indent=2));(OUT/'selection.json').write_text(json.dumps(selections,indent=2))
if __name__=='__main__':main()
