#!/usr/bin/env python3
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.common import sha256,write_json,MOMASK
from g1.rot6d68 import encode_native,SPEC
from train_g1_vae import load_data_identity
from g1_io import local_features_to_model_features

def main():
 p=argparse.ArgumentParser();p.add_argument('--native-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 original=load_data_identity(a.native_root)
 if a.output.exists():raise ValueError('Use a new output directory')
 a.output.mkdir(parents=True);(a.output/'motions').mkdir()
 m=dict(complete=False,representation=SPEC,native_manifest_sha256=sha256(a.native_root/'manifest.json'),conversion_sha256=sha256(MOMASK/'dataset/bones-seed_prepare/g1_io.py'),smoke=original['smoke'],boundary='Convert full prepared clips before random crops; first clip delta zero',normalization='All valid training-split frames only; std floor1e-4')
 write_json(a.output/'manifest.json',m)
 sums=np.zeros(68);squares=np.zeros(68);count=0;maximum=0.;t=time.time();counts={}
 for i,line in enumerate((a.native_root/'index.jsonl').read_text().splitlines()):
  r=json.loads(line);native=np.load(a.native_root/'motions'/f"{r['id']}.npy")
  feature,anchor=encode_native(native)
  assert feature.shape==(r['frames'],68) and np.isfinite(feature).all()
  recovered=local_features_to_model_features(feature,SPEC['fields'],anchor[0]);error=float(np.max(np.abs(recovered-native)))
  maximum=max(maximum,error)
  if error>1e-3:raise ValueError((r['id'],error))
  np.save(a.output/'motions'/f"{r['id']}.npy",feature)
  if r['split']=='train':
   x=feature.astype(np.float64);sums+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
  counts[r['split']]=counts.get(r['split'],0)+1
  if (i+1)%10000==0:print(json.dumps(dict(clips=i+1,seconds=time.time()-t)),flush=True)
 mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean*mean,0)).clip(1e-4)
 np.save(a.output/'mean.npy',mean.astype(np.float32));np.save(a.output/'std.npy',std.astype(np.float32))
 m.update(complete=True,counts=counts,train_frames=count,roundtrip_max_abs=maximum,mean_sha256=sha256(a.output/'mean.npy'),std_sha256=sha256(a.output/'std.npy'))
 write_json(a.output/'manifest.json',m);print(json.dumps(m),flush=True)
if __name__=='__main__':main()
