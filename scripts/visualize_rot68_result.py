#!/usr/bin/env python3
import json,csv,sys,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.common import sha256,G1Kinematics,metric_values,reduce_values
from g1.model import G1CausalVAE
from g1.rot6d68 import encode_native,decode_native

def main():
 old=ROOT/'outputs/vae_rvq_comparison';out=ROOT/'outputs/rot68_reconstruction_comparison';run=ROOT/'outputs/g1_rot6d68_textop_500ep_bs2048';rot=ROOT/'outputs/bones_g1_rot6d68'
 out.mkdir(exist_ok=False)
 selections=json.loads((old/'selection.json').read_text());reference={r['sample_id']:r for r in csv.DictReader((run/'eval_best/per_sample.csv').open())}
 path=run/'best.pt';c=torch.load(path,map_location='cpu',weights_only=False)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model=G1CausalVAE(np.load(rot/'mean.npy'),np.load(rot/'std.npy'),**c['identity']['model_params']).cuda().eval();model.load_state_dict(c['model'])
 fk=G1Kinematics(ROOT/'outputs/bones_g1_native38/kinematic.xml')
 for r in selections:
  d=out/'clips'/r['slug'];d.mkdir(parents=True)
  for name in ('gt','rvq','vae_best'):shutil.copy2(old/'clips'/r['slug']/f'{name}.npy',d/f'{name}.npy')
  gt=np.load(d/'gt.npy');feature,anchor=encode_native(gt)
  stored=np.load(rot/'motions'/f"{r['id']}.npy");np.testing.assert_array_equal(feature,stored)
  with torch.inference_mode():
   pred=decode_native(model.reconstruct(torch.tensor(feature[None],device='cuda'),sample=False)[0],torch.tensor(anchor[:1],device='cuda',dtype=torch.float32))[0].cpu().numpy()
  np.save(d/'rot68.npy',pred)
  values=metric_values(gt,pred,fk.positions(gt),fk.positions(pred),20,fk.endpoints)
  r['metrics']['rot68']={k:reduce_values(k,values[k]) for k in ('global_mpjpe_mm','root_relative_mpjpe_mm')}
  for k,v in r['metrics']['rot68'].items():np.testing.assert_allclose(v,float(reference[r['id']][k]),atol=.01,rtol=1e-4)
  r['rot68_checkpoint']=dict(epoch=c['epoch'],sha256=sha256(path));r['metrics'].pop('vae_latest',None)
 (out/'selection.json').write_text(json.dumps(selections,indent=2));print(json.dumps([dict(slug=r['slug'],metrics=r['metrics']['rot68']) for r in selections]),flush=True)
if __name__=='__main__':main()
