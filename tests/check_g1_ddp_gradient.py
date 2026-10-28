"""Run with torchrun: compare eight-rank masked gradients to a single global batch."""
from pathlib import Path
import sys
import copy
import json
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from g1.model import G1CausalVAE,TextOpLoss,DEFAULT_WEIGHTS
from g1.kinematics import TorchG1FK
from g1.distributed import LossForward,denominators,ddp_objective

rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local)
dist.init_process_group('nccl')
torch.set_num_threads(2)
torch.manual_seed(3407)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
vae=G1CausalVAE(np.zeros(38,np.float32),np.ones(38,np.float32),dim=8,dec_dim=8,z_dim=4,
               dim_mult=[1,1,1],temperal_downsample=[True,True],num_res_blocks=1).cuda()
wrapped=LossForward(vae,TextOpLoss(TorchG1FK(ROOT/'outputs/g1_smoke_data_v2/kinematic.xml').cuda()))
ddp=DDP(wrapped,device_ids=[local],broadcast_buffers=False)
reference=copy.deepcopy(wrapped) if rank==0 else None
# All ranks create the same global input, then take unequal-sized shards.
generator=torch.Generator().manual_seed(123)
count=world*2+1
feature=torch.randn(count,13,38,generator=generator)*.05
feature[:,:,2]+=.8
feature[:,:,3:9]+=torch.tensor([1.,0,0,0,1,0])
length=5+torch.arange(count)%9
eligible=torch.arange(count)%3==0
contact=torch.ones(count,13,2,dtype=torch.bool)
for i,n in enumerate(length):feature[i,n:]=feature[i,n-1]
full={k:v.cuda() for k,v in dict(feature=feature,length=length,eligible=eligible,contact=contact).items()}
batch={k:v[rank::world] for k,v in full.items()}
terms=ddp(batch,sample=False)
counts=denominators(batch);global_counts=counts.clone();dist.all_reduce(global_counts)
loss=ddp_objective(terms,counts,global_counts,world,DEFAULT_WEIGHTS)
loss.backward()
if rank==0:
    target=reference(full,sample=False)['total']
    target.backward()
    largest=0.
    for (name,p),(other,q) in zip(wrapped.vae.named_parameters(),reference.vae.named_parameters()):
        assert name==other
        largest=max(largest,float((p.grad-q.grad).abs().max()))
        torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=5e-4,msg=name)
    print(json.dumps(dict(gradient_parity_passed=True,world_size=world,global_examples=count,
                         max_abs_gradient_error=largest)),flush=True)
dist.barrier();dist.destroy_process_group()
