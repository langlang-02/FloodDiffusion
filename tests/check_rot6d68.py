"""Canonical decode/gradient and requested per-GPU batch memory checks."""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.rot6d68 import encode_native,decode_native,SPEC,Rot68Dataset,collate_rot68,Rot68Forward
from g1_io import local_features_to_model_features
from g1.model import G1CausalVAE
from g1.kinematics import TorchG1FK

torch.set_num_threads(2);torch.manual_seed(3407)
root=ROOT/'outputs/bones_g1_native38';rot=ROOT/'outputs/bones_g1_rot6d68'
rows=[json.loads(l) for l in (root/'index.jsonl').read_text().splitlines()][:32]
maximum=0.
for r in rows:
 native=np.load(root/'motions'/f"{r['id']}.npy");feature,anchors=encode_native(native)
 start=min(7,len(native)-3);feature=feature[start:];anchor=anchors[start]
 x=torch.tensor(feature[None],requires_grad=True);a=torch.tensor(anchor[None],dtype=torch.float32)
 decoded=decode_native(x,a)
 ref=local_features_to_model_features(feature,SPEC['fields'],anchor)
 maximum=max(maximum,float(np.abs(decoded.detach().numpy()[0]-ref).max()))
 np.testing.assert_allclose(decoded.detach().numpy()[0],native[start:],atol=3e-5,rtol=3e-5)
 decoded.square().mean().backward();assert torch.isfinite(x.grad).all()
# Compare integration gradient to central differences away from rotation singularities.
x=torch.tensor(feature[None],dtype=torch.float64,requires_grad=True);a=torch.tensor(anchor[None],dtype=torch.float64)
y=decode_native(x,a)[0,-1,:3].square().sum();y.backward()
for index in [(0,1,64),(0,1,65),(0,1,66),(0,len(feature)-1,67)]:
 plus=x.detach().clone();minus=plus.clone();plus[index]+=1e-5;minus[index]-=1e-5
 numeric=(decode_native(plus,a)[0,-1,:3].square().sum()-decode_native(minus,a)[0,-1,:3].square().sum())/2e-5
 torch.testing.assert_close(x.grad[index],numeric,atol=1e-6,rtol=1e-4)
print(json.dumps(dict(canonical_decode_passed=True,max_abs=maximum,gradient_finite_difference_passed=True)),flush=True)
if '--memory' in sys.argv:
 ds=Rot68Dataset(root,rot,'train',65)
 # Production-size model and physical FK losses at the exact requested per-GPU batch.
 item=next(ds[i] for i in range(50) if ds[i]['length']==65)
 batch={k:v.cuda() for k,v in collate_rot68([item]*2048).items()}
 mean,std=(np.load(rot/f'{n}.npy') for n in ('mean','std'))
 model=G1CausalVAE(mean,std,z_dim=32,dim=160,dec_dim=512,num_res_blocks=1,dropout=0.,dim_mult=[1,1,1],temperal_downsample=[True,True]).cuda()
 wrapped=Rot68Forward(model,TorchG1FK(root/'kinematic.xml').cuda())
 optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
 torch.cuda.reset_peak_memory_stats()
 terms=wrapped(batch);terms['total'].backward()
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
 assert all(torch.isfinite(t) for t in terms.values())
 print(json.dumps(dict(batch_per_gpu=2048,frames=65,all_losses_finite=True,gradient_norm=float(norm),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,losses={k:float(v.detach()) for k,v in terms.items()})),flush=True)
