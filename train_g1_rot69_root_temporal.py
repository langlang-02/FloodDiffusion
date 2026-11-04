#!/usr/bin/env python3
"""Fresh/resumable absolute-yaw rot6d69 causal VAE with TextOp losses and GT-matched temporal derivatives."""
import argparse
from datetime import datetime,timedelta
import json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from g1.common import ROOT,MOMASK,sha256,write_json
from g1.rot6d69 import Rot69Dataset,collate_rot69,validate_rot_data,SPEC
from g1.continuation import fixed_rows
from g1.temporal_eval import evaluate_fixed
from g1.root_temporal_loss import TemporalForward as Rot69Forward,TEMPORAL_WEIGHTS as DEFAULT_WEIGHTS
from g1.root_temporal_ddp import epoch_pass
from g1.model import G1CausalVAE
from g1.kinematics import TorchG1FK
from g1.distributed import GlobalBatchShard
from train_g1_vae import load_data_identity,seed_worker,checkpoint_save
from train_g1_vae_ddp import rng_state,restore_rng

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--fixed-val',type=Path,required=True)
 p.add_argument('--data-root',type=Path,required=True);p.add_argument('--rot-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--batch-per-gpu',type=int,default=2048);p.add_argument('--epochs',type=int,default=500);p.add_argument('--checkpoint-every',type=int,default=25)
 p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--window',type=int,default=65);p.add_argument('--workers',type=int,default=2)
 p.add_argument('--resume',type=Path);p.add_argument('--allow-smoke',action='store_true');a=p.parse_args()
 rank,local,world=(int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
 torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl',timeout=timedelta(minutes=10))
 torch.set_num_threads(2);torch.manual_seed(3407);np.random.seed(3407+rank);random.seed(3407+rank)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 manifest=load_data_identity(a.data_root);rot_manifest=validate_rot_data(a.data_root,a.rot_root)
 rows,cohort=fixed_rows(a.data_root,a.fixed_val)
 if (manifest['smoke'] or rot_manifest['smoke'] or cohort['smoke']) and not a.allow_smoke:raise ValueError('Smoke data refused')
 if min(a.batch_per_gpu,a.epochs,a.checkpoint_every)<1 or a.window<3:raise ValueError('Invalid configuration')
 params=dict(z_dim=32,dim=160,dec_dim=512,num_res_blocks=1,dropout=0.,dim_mult=[1,1,1],temperal_downsample=[True,True])
 files=['g1/rot6d69.py','g1/common.py','g1/temporal_eval.py','g1/root_temporal_loss.py','g1/root_temporal_ddp.py','g1/temporal_metrics.py','g1/continuation.py','models/vae_wan_1d.py','g1/model.py','g1/kinematics.py','g1/data.py','g1/distributed.py','train_g1_vae.py','train_g1_vae_ddp.py','train_g1_rot69_root_temporal.py','eval_g1_rot69_temporal.py','models/tools/wan_vae_1d.py']
 identity=dict(data_manifest_sha256=sha256(a.data_root/'manifest.json'),rot_manifest_sha256=sha256(a.rot_root/'manifest.json'),
  representation=SPEC,model_params=params,weights=DEFAULT_WEIGHTS,epochs=a.epochs,window=a.window,learning_rate=a.lr,seed=3407,
  batch_size=a.batch_per_gpu*world,batch_per_gpu=a.batch_per_gpu,world_size=world,checkpoint_every=a.checkpoint_every,
  code_sha256={n:sha256(ROOT/n) for n in files},metric_sha256=sha256(MOMASK/'evaluation/g1_metrics.py'),
  smoke=bool(a.allow_smoke or manifest['smoke']),precision='float32 TF32off',posterior_eval='mean',
  root_initialization='GT first XY only; absolute yaw and height predicted',workers_per_rank=a.workers,
  fixed_val_manifest_sha256=sha256(a.fixed_val/'manifest.json'),fixed_val_samples=len(rows),best_selection='minimum fixed filtered-validation global MPJPE action mean; test never selects',lr_schedule='epoch cosine 1e-4 to 1e-6; same formula as original fresh rot68 experiment')
 mean,std=(np.load(a.rot_root/f'{n}.npy') for n in ('mean','std'))
 assert mean.shape==std.shape==(69,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
 model=G1CausalVAE(mean,std,**params).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
 start,best,step=0,math.inf,0;checkpoint=None;best_mpjpe=math.inf;best_epoch=None
 if a.resume:
  checkpoint=torch.load(a.resume,map_location='cpu',weights_only=False)
  if checkpoint['identity']!=identity:raise ValueError('Resume config/data/source identity mismatch')
  model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
  start,best,step=checkpoint['epoch'],checkpoint['best_val'],checkpoint['step']
  best_mpjpe,best_epoch=checkpoint['best_mpjpe'],checkpoint['best_mpjpe_epoch']
 wrapped=Rot69Forward(model,TorchG1FK(a.data_root/'kinematic.xml').to(device));ddp=DDP(wrapped,device_ids=[local],broadcast_buffers=False)
 torch.manual_seed(3407+rank);generator=torch.Generator().manual_seed(3407+rank)
 if checkpoint:restore_rng(checkpoint['rng_per_rank'][rank],generator)
 train=Rot69Dataset(a.data_root,a.rot_root,'train',a.window);val=Rot69Dataset(a.data_root,a.rot_root,'val',a.window)
 sampler=GlobalBatchShard(len(train),a.batch_per_gpu*world,rank,world,3407);val_sampler=GlobalBatchShard(len(val),32,rank,world,training=False)
 def loader(ds,sm):return DataLoader(ds,batch_sampler=sm,num_workers=a.workers,collate_fn=collate_rot69,pin_memory=True,generator=generator,worker_init_fn=seed_worker,multiprocessing_context='spawn' if a.workers else None)
 train_loader,val_loader=loader(train,sampler),loader(val,val_sampler)
 pids=[None]*world;dist.all_gather_object(pids,os.getpid())
 if rank==0:
  if a.resume:
   if math.isfinite(best_mpjpe):
    saved=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
    if saved['best_mpjpe']!=best_mpjpe or saved['epoch']>start:raise ValueError('Resume best mismatch')
    del saved
  else:a.output.mkdir(parents=True,exist_ok=False)
  write_json(a.output/'identity.json',identity);write_json(a.output/'arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
 dist.barrier()
 def status(state,epoch,**extra):
  if rank==0:write_json(a.output/'status.json',dict(state=state,completed_epochs=epoch,target_epochs=a.epochs,pid=os.getpid(),rank_pids=pids,world_size=world,batch_per_gpu=a.batch_per_gpu,global_batch_size=a.batch_per_gpu*world,smoke=identity['smoke'],best_mpjpe_mm=best_mpjpe if math.isfinite(best_mpjpe) else None,best_mpjpe_epoch=best_epoch,**extra))
 status('training',start);began=time.time()
 if rank==0:print(f'ROT6D69 ROOT-TEMPORAL TRAINING | {world} GPUs | batch/GPU={a.batch_per_gpu} | global={a.batch_per_gpu*world} | resume epoch={start} | {len(sampler)} steps/epoch',flush=True)
 for epoch in range(start,a.epochs):
  epoch_start=time.time();sampler.epoch=epoch;lr=a.lr*(.01+.99*.5*(1+math.cos(math.pi*epoch/a.epochs)))
  for group in optimizer.param_groups:group['lr']=lr
  if rank==0:print(f'Epoch {epoch+1}/{a.epochs} | lr={lr:.7g} | training ...',flush=True)
  train_metrics,steps=epoch_pass(wrapped,ddp,train_loader,optimizer,device,rank,world,DEFAULT_WEIGHTS)
  val_metrics,_=epoch_pass(wrapped,ddp,val_loader,None,device,rank,world,DEFAULT_WEIGHTS)
  best=min(best,val_metrics['total']);step+=steps;improved=False;evaluation=None
  if (epoch+1)%a.checkpoint_every==0 or epoch+1==a.epochs:
   evaluation=evaluate_fixed(model,a.data_root,a.rot_root,rows,a.output/'eval_fixed_val'/f'epoch_{epoch+1:04d}',epoch+1,step,rank,world,device)
   score=evaluation['metrics']['global_mpjpe_mm']['action_mean']
   if not math.isfinite(score):raise FloatingPointError('Nonfinite fixed-validation MPJPE')
   improved=score<best_mpjpe
   if improved:best_mpjpe=score;best_epoch=epoch+1
  states=[None]*world;dist.all_gather_object(states,rng_state(generator))
  if rank==0:
   record=dict(epoch=epoch+1,step=step,lr=lr,train=train_metrics,val=val_metrics,seconds=time.time()-began,epoch_seconds=time.time()-epoch_start,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
   record['selection']=dict(best_global_mpjpe_mm=best_mpjpe if math.isfinite(best_mpjpe) else None,best_epoch=best_epoch)
   if evaluation:record['fixed_val']=evaluation['metrics']
   payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),identity=identity,epoch=epoch+1,best_val=best,step=step,rng_per_rank=states,best_mpjpe=best_mpjpe,best_mpjpe_epoch=best_epoch)
   checkpoint_save(a.output/'latest.pt',payload)
   if improved:checkpoint_save(a.output/'best.pt',payload)
   if (epoch+1)%a.checkpoint_every==0 or epoch+1==a.epochs:checkpoint_save(a.output/f'epoch_{epoch+1:04d}.pt',payload)
   if evaluation:write_json(a.output/'eval_fixed_val'/f'epoch_{epoch+1:04d}'/'identity.json',dict(checkpoint_sha256=sha256(a.output/f'epoch_{epoch+1:04d}.pt'),fixed_val_manifest_sha256=identity['fixed_val_manifest_sha256']))
   with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   remaining=(time.time()-began)/(epoch+1-start)*(a.epochs-epoch-1)
   print(f"Epoch {epoch+1}/{a.epochs} complete | train={train_metrics['total']:.6f} val={val_metrics['total']:.6f} best MPJPE={best_mpjpe:.3f}mm @ep{best_epoch} | {record['epoch_seconds']:.1f}s | ETA {timedelta(seconds=int(remaining))} ({datetime.now()+timedelta(seconds=remaining):%m-%d %H:%M:%S}) | peak GPU0 {record['peak_memory_gib']:.2f} GiB",flush=True)
   del payload
  dist.barrier();status('training',epoch+1)
 dist.barrier();del ddp,wrapped,model,optimizer,train_loader,val_loader;dist.destroy_process_group();torch.cuda.empty_cache()
 if rank==0:
  status('evaluating',a.epochs)
  from eval_g1_rot69_temporal import evaluate
  for name in ('best','latest'):evaluate(a.data_root,a.rot_root,a.output/f'{name}.pt',a.output/f'eval_{name}',str(device),32,allow_smoke=a.allow_smoke)
  status('complete',a.epochs,evaluation_complete=True)
if __name__=='__main__':main()
