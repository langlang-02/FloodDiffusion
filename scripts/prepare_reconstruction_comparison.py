#!/usr/bin/env python3
"""Select representative clips and export VAE predictions with evaluation parity."""
import csv,json,re,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.model import G1CausalVAE
from g1.common import G1Kinematics,metric_values,reduce_values,sha256

def main():
 out=ROOT/'outputs/vae_rvq_comparison';data=ROOT/'outputs/bones_g1_native38';run=ROOT/'outputs/g1_causal_textop_500ep'
 rows=[json.loads(l) for l in (data/'index.jsonl').read_text().splitlines() if json.loads(l)['split']=='test']
 scores={r['sample_id']:r for r in csv.DictReader((run/'eval_best/per_sample.csv').open())}
 rvq={r['sample_id']:r for r in csv.DictReader((out/'rvq_same_test/per_sample.csv').open())}
 selection=[]
 categories=[('walk','Walking',r'^walk_forward'),('turn','Turning',r'^turn_walk'),('dance','Dance',r'^dance'),('jump','Jump',r'^(?:high_jump|jump|Jump_)'),('gesture','Hand gesture',r'^wave'),('stairs','Stair / terrain action',r'^stair_slide_down')]
 for slug,label,pattern in categories:
  candidates=[r for r in rows if re.search(pattern,r['id'],re.I) and '_M_' not in r['id'] and r['frames']>=100]
  if slug=='jump':candidates=[r for r in candidates if r['foot_contact_eligible']]
  assert candidates,slug
  candidates.sort(key=lambda r:float(scores[r['id']]['global_mpjpe_mm']))
  chosen=dict(candidates[len(candidates)//2],slug=slug,label=label,selection='median VAE-best global MPJPE within named category, non-mirrored, >=100 frames',candidate_count=len(candidates))
  selection.append(chosen)
 candidates=[r for r in rows if r['frames']>=100 and '_M_' not in r['id'] and 'walk' in r['id'].lower()]
 candidates.sort(key=lambda r:float(scores[r['id']]['global_mpjpe_mm']))
 selection.append(dict(candidates[int(.95*(len(candidates)-1))],slug='drift',label='Large translation error',selection='95th percentile VAE-best global MPJPE among non-mirrored walking clips >=100 frames',candidate_count=len(candidates)))
 # Audit original RVQ export split by source basename (dates are absent there).
 tracking=Path('/data/langlang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1')
 def source(n):return re.sub(r'_M$','',Path(n).stem.split('_seg')[0])
 old={s:{source(n) for n in (tracking/f'{s}.txt').read_text().splitlines()} for s in ('train','val','test')}
 counts={s:sum(source(r['id'].split('_win')[0]) in names for r in rows) for s,names in old.items()}
 (out/'split_overlap.json').write_text(json.dumps(dict(method='source basename and mirror normalization; old export has no source date, so indicative not definitive',matched_clips=counts,total=len(rows)),indent=2))
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 fk=G1Kinematics(data/'kinematic.xml')
 for r in selection:
  d=out/'clips'/r['slug'];d.mkdir(parents=True,exist_ok=True)
  gt=np.load(data/'motions'/f"{r['id']}.npy");np.save(d/'gt.npy',gt)
  np.save(d/'rvq.npy',np.load(out/'rvq_same_test/predictions'/f"{r['id']}.npy"))
  r['metrics']={'rvq':{k:float(rvq[r['id']][k]) for k in ('global_mpjpe_mm','root_relative_mpjpe_mm')}}
  r['original_rvq_source_split_matches']=[s for s,names in old.items() if source(r['id'].split('_win')[0]) in names]
 for name in ('best','latest'):
  path=run/f'{name}.pt';c=torch.load(path,map_location='cpu',weights_only=False)
  model=G1CausalVAE(np.load(data/'mean.npy'),np.load(data/'std.npy'),**c['identity']['model_params']).cuda().eval();model.load_state_dict(c['model'])
  ref={r['sample_id']:r for r in csv.DictReader((run/f'eval_{name}/per_sample.csv').open())}
  for r in selection:
   d=out/'clips'/r['slug'];gt=np.load(d/'gt.npy')
   with torch.inference_mode():pred=model.reconstruct(torch.tensor(gt[None],device='cuda'),sample=False)[0][0].cpu().numpy()
   np.save(d/f'vae_{name}.npy',pred)
   values=metric_values(gt,pred,fk.positions(gt),fk.positions(pred),20,fk.endpoints)
   metrics={k:reduce_values(k,values[k]) for k in ('global_mpjpe_mm','root_relative_mpjpe_mm')}
   for k,v in metrics.items():np.testing.assert_allclose(v,float(ref[r['id']][k]),rtol=1e-4,atol=.01)
   r['metrics'][f'vae_{name}']=metrics
   r[f'vae_{name}_checkpoint']=dict(epoch=c['epoch'],sha256=sha256(path))
  del model
 (out/'selection.json').write_text(json.dumps(selection,indent=2))
 print(json.dumps(dict(selection=[(r['slug'],r['id'],r['metrics']) for r in selection],overlap=counts)),flush=True)
if __name__=='__main__':main()
