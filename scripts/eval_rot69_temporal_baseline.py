#!/usr/bin/env python3
"""Add root temporal diagnostics to an existing rot69 checkpoint's frozen validation."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.model import G1CausalVAE
from g1.rot6d69 import validate_rot_data,SPEC
from g1.common import sha256,write_json
from g1.continuation import fixed_rows
from g1.temporal_eval import evaluate_fixed
p=argparse.ArgumentParser()
for key in ('data-root','rot-root','fixed-val','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--device',default='cuda:0');a=p.parse_args()
validate_rot_data(a.data_root,a.rot_root);rows,cohort=fixed_rows(a.data_root,a.fixed_val)
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
assert c['identity']['representation']==SPEC and c['identity']['rot_manifest_sha256']==sha256(a.rot_root/'manifest.json')
assert c['identity']['fixed_val_manifest_sha256']==sha256(a.fixed_val/'manifest.json')
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
m=G1CausalVAE(np.load(a.rot_root/'mean.npy'),np.load(a.rot_root/'std.npy'),**c['identity']['model_params']).to(a.device).eval();m.load_state_dict(c['model'])
s=evaluate_fixed(m,a.data_root,a.rot_root,rows,a.output,c['epoch'],c['step'],0,1,torch.device(a.device))
write_json(a.output/'identity.json',dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),fixed_val_manifest_sha256=sha256(a.fixed_val/'manifest.json'),temporal_metric_sha256=sha256(ROOT/'g1/temporal_metrics.py')))
print(json.dumps(s),flush=True)
