#!/usr/bin/env python3
"""Append continuous absolute yaw to canonical rot68; preserve all split/statistics rules."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.common import sha256,write_json,MOMASK,rotation_6d_to_matrix,rotation_matrix_to_6d,yaw_to_matrix
from g1.rot6d69 import encode_native,SPEC
from g1.rot6d68 import validate_rot_data
from train_g1_vae import load_data_identity

def main():
 p=argparse.ArgumentParser();p.add_argument('--native-root',type=Path,required=True);p.add_argument('--rot68-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 native=load_data_identity(a.native_root);validate_rot_data(a.native_root,a.rot68_root)
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'motions').mkdir()
 m=dict(complete=False,representation=SPEC,native_manifest_sha256=sha256(a.native_root/'manifest.json'),rot68_manifest_sha256=sha256(a.rot68_root/'manifest.json'),conversion_sha256=sha256(MOMASK/'dataset/bones-seed_prepare/g1_io.py'),adapter_sha256=sha256(ROOT/'g1/rot6d69.py'),preparation_sha256=sha256(Path(__file__)),smoke=native['smoke'],boundary='Absolute yaw unwrapped on full native clip before random crops; first68 channels exactly preserved',normalization='Training frames only; std floor1e-4')
 write_json(a.output/'manifest.json',m);sums=np.zeros(69);squares=np.zeros(69);count=0;counts={};maximum=0.;t=time.time()
 for i,line in enumerate((a.native_root/'index.jsonl').read_text().splitlines()):
  r=json.loads(line);g=np.load(a.native_root/'motions'/f"{r['id']}.npy");x,anchors=encode_native(g)
  np.testing.assert_array_equal(x[:,:68],np.load(a.rot68_root/'motions'/f"{r['id']}.npy"))
  assert x.shape==(r['frames'],69) and np.isfinite(x).all()
  yaw=x[:,68];c,s=np.cos(yaw),np.sin(yaw);dxy=np.stack((c*x[:,65]-s*x[:,66],s*x[:,65]+c*x[:,66]),axis=-1)
  xy=anchors[0,:2]+np.concatenate((np.zeros((1,2)),np.cumsum(dxy[1:],axis=0)),axis=0)
  rot=yaw_to_matrix(yaw)@rotation_6d_to_matrix(x[:,58:64]);recovered=np.concatenate((xy,x[:,67:68],rotation_matrix_to_6d(rot),x[:,:29]),axis=-1)
  error=float(np.max(np.abs(recovered-g)));maximum=max(maximum,error)
  if error>1e-3:raise ValueError((r['id'],error))
  np.save(a.output/'motions'/f"{r['id']}.npy",x)
  if r['split']=='train':
   y=x.astype(np.float64);sums+=y.sum(0);squares+=(y*y).sum(0);count+=len(y)
  counts[r['split']]=counts.get(r['split'],0)+1
  if (i+1)%10000==0:print(json.dumps(dict(clips=i+1,seconds=time.time()-t)),flush=True)
 mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean*mean,0)).clip(1e-4)
 np.save(a.output/'mean.npy',mean.astype(np.float32));np.save(a.output/'std.npy',std.astype(np.float32))
 for n in ('mean','std'):np.testing.assert_array_equal(np.load(a.output/f'{n}.npy')[:68],np.load(a.rot68_root/f'{n}.npy'))
 m.update(complete=True,counts=counts,train_frames=count,roundtrip_max_abs=maximum,mean_sha256=sha256(a.output/'mean.npy'),std_sha256=sha256(a.output/'std.npy'))
 write_json(a.output/'manifest.json',m);print(json.dumps(m),flush=True)
if __name__=='__main__':main()
