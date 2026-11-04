"""Four-rank unequal-shard gradient equivalence for all nine loss terms."""
import sys,os,copy,json
from pathlib import Path
import numpy as np
import torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.model import G1CausalVAE
from g1.kinematics import TorchG1FK
from g1.rot6d69 import encode_native,collate_rot69
from g1.root_temporal_loss import TemporalForward,TEMPORAL_WEIGHTS
from g1.root_temporal_ddp import denominators,ddp_objective
rank,local,world=(int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
torch.cuda.set_device(local);dist.init_process_group('nccl');torch.set_num_threads(2);torch.manual_seed(3407)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
mean=np.zeros(69,np.float32);mean[58]=mean[62]=1;mean[67]=.8
vae=G1CausalVAE(mean,np.ones(69,np.float32)*.1,dim=8,dec_dim=8,z_dim=4,dim_mult=[1,1,1],temperal_downsample=[True,True],num_res_blocks=1).cuda()
wrapped=TemporalForward(vae,TorchG1FK(ROOT/'outputs/g1_smoke_data_v2/kinematic.xml').cuda());ddp=DDP(wrapped,device_ids=[local],broadcast_buffers=False);ref=copy.deepcopy(wrapped) if rank==0 else None
rng=np.random.default_rng(123);items=[]
for i in range(world*2+1):
 n=5+i%9;g=rng.normal(0,.02,(n,38)).astype(np.float32);g[:,2]+=.8;g[:,3:9]=np.array([1,0,0,0,1,0]);x,a=encode_native(g)
 items.append(dict(feature=torch.tensor(x),native=torch.tensor(g),anchor=torch.tensor(a[0],dtype=torch.float32),contact=torch.ones(n,2,dtype=torch.bool),eligible=i%3==0,length=n))
full={k:v.cuda() for k,v in collate_rot69(items).items()};batch={k:v[rank::world] for k,v in full.items()}
terms=ddp(batch,sample=False);counts=denominators(batch);total=counts.clone();dist.all_reduce(total)
ddp_objective(terms,counts,total,world,TEMPORAL_WEIGHTS).backward()
if rank==0:
 ref(full,sample=False)['total'].backward();largest=0
 for (name,p),(_,q) in zip(wrapped.vae.named_parameters(),ref.vae.named_parameters()):
  largest=max(largest,float((p.grad-q.grad).abs().max()));torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=1e-3,msg=name)
 print(json.dumps(dict(world=world,examples=len(items),losses=len(TEMPORAL_WEIGHTS),max_gradient_error=largest)),flush=True)
dist.barrier();dist.destroy_process_group()
