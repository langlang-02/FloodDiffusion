#!/usr/bin/env python3
"""Continue a rot68 checkpoint with update-based LR and frozen validation MPJPE selection."""
import argparse
from datetime import datetime,timedelta
import json,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from g1.common import ROOT,MOMASK,sha256,write_json
from g1.rot6d68 import Rot68Dataset,collate_rot68,Rot68Forward,validate_rot_data,SPEC
from g1.model import G1CausalVAE,DEFAULT_WEIGHTS
from g1.kinematics import TorchG1FK
from g1.distributed import GlobalBatchShard
from g1.continuation import fixed_rows,evaluate_fixed,train_epoch
from train_g1_vae import load_data_identity,seed_worker,checkpoint_save
from train_g1_vae_ddp import epoch_pass,rng_state,restore_rng

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('data-root','rot-root','fixed-val','output','parent'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--resume',type=Path);p.add_argument('--additional-epochs',type=int,default=500);p.add_argument('--batch-per-gpu',type=int,default=2048)
 p.add_argument('--eval-every',type=int,default=25);p.add_argument('--workers',type=int,default=2)
 p.add_argument('--warmup-steps',type=int,default=300);p.add_argument('--hold-until',type=int,default=2000)
 p.add_argument('--peak-lr',type=float,default=3e-5);p.add_argument('--floor-lr',type=float,default=1e-6);p.add_argument('--allow-smoke',action='store_true');a=p.parse_args()
 rank,local,world=(int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
 torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl',timeout=timedelta(minutes=15))
 torch.set_num_threads(2);torch.manual_seed(3407+rank);np.random.seed(3407+rank);random.seed(3407+rank)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 native=load_data_identity(a.data_root);rot=validate_rot_data(a.data_root,a.rot_root);rows,cohort=fixed_rows(a.data_root,a.fixed_val)
 parent=torch.load(a.parent,map_location='cpu',weights_only=False);pi=parent['identity'];parent_hash=sha256(a.parent)
 if pi['data_manifest_sha256']!=sha256(a.data_root/'manifest.json') or pi['rot_manifest_sha256']!=sha256(a.rot_root/'manifest.json'):raise ValueError('Parent data mismatch')
 if pi['representation']!=SPEC or pi['weights']!=DEFAULT_WEIGHTS:raise ValueError('Parent model/loss mismatch')
 for name,h in pi['code_sha256'].items():
  if sha256(ROOT/name)!=h:raise ValueError(f'Parent component changed: {name}')
 if pi['metric_sha256']!=sha256(MOMASK/'evaluation/g1_metrics.py'):raise ValueError('Metric source changed')
 if (native['smoke'] or rot['smoke'] or pi['smoke'] or cohort['smoke']) and not a.allow_smoke:raise ValueError('Smoke input refused')
 if world!=pi['world_size'] or a.batch_per_gpu!=pi['batch_per_gpu']:raise ValueError('Continuation preserves world/batch config')
 train=Rot68Dataset(a.data_root,a.rot_root,'train',pi['window']);val=Rot68Dataset(a.data_root,a.rot_root,'val',pi['window'])
 sampler=GlobalBatchShard(len(train),a.batch_per_gpu*world,rank,world,pi['seed']);vs=GlobalBatchShard(len(val),32,rank,world,training=False)
 schedule=dict(total=a.additional_epochs*len(sampler),warmup=a.warmup_steps,hold_until=a.hold_until,peak=a.peak_lr,floor=a.floor_lr)
 from g1.continuation import learning_rate
 learning_rate(0,**schedule);learning_rate(schedule['total']-1,**schedule)
 identity=dict(pi);identity.update(epochs=parent['epoch']+a.additional_epochs,learning_rate=a.peak_lr,checkpoint_every=a.eval_every,
  parent_checkpoint=str(a.parent.resolve()),parent_checkpoint_sha256=parent_hash,parent_epoch=parent['epoch'],parent_step=parent['step'],
  continuation_schedule=schedule,fixed_val_manifest_sha256=sha256(a.fixed_val/'manifest.json'),fixed_val_samples=cohort['samples'],
  best_selection='minimum frozen filtered-validation global_mpjpe_mm/action_mean; baseline included; test never selects',
  workers_per_rank=a.workers,smoke=bool(a.allow_smoke or pi['smoke']))
 identity['code_sha256']=dict(pi['code_sha256'])
 for f in ('g1/continuation.py','train_g1_rot68_continue.py'):identity['code_sha256'][f]=sha256(ROOT/f)
 checkpoint=parent
 if a.resume:
  checkpoint=torch.load(a.resume,map_location='cpu',weights_only=False)
  if checkpoint['identity']!=identity:raise ValueError('Continuation resume identity mismatch')
 start,step=checkpoint['epoch'],checkpoint['step']
 mean,std=(np.load(a.rot_root/f'{n}.npy') for n in ('mean','std'))
 model=G1CausalVAE(mean,std,**pi['model_params']).to(device);model.load_state_dict(checkpoint['model'])
 optimizer=torch.optim.Adam(model.parameters(),lr=a.peak_lr);optimizer.load_state_dict(checkpoint['optimizer'])
 wrapped=Rot68Forward(model,TorchG1FK(a.data_root/'kinematic.xml').to(device));ddp=DDP(wrapped,device_ids=[local],broadcast_buffers=False)
 generator=torch.Generator().manual_seed(pi['seed']+rank);restore_rng(checkpoint['rng_per_rank'][rank],generator)
 def loader(ds,sm):return DataLoader(ds,batch_sampler=sm,num_workers=a.workers,collate_fn=collate_rot68,pin_memory=True,generator=generator,worker_init_fn=seed_worker,multiprocessing_context='spawn' if a.workers else None)
 tl,vl=loader(train,sampler),loader(val,vs)
 pids=[None]*world;dist.all_gather_object(pids,os.getpid())
 if rank==0:
  if a.resume:
   best_saved=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
   if best_saved['best_mpjpe']!=checkpoint['best_mpjpe'] or best_saved['epoch']>start:raise ValueError('Inconsistent best checkpoint')
   del best_saved
  else:a.output.mkdir(parents=True,exist_ok=False)
  write_json(a.output/'identity.json',identity);write_json(a.output/'arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
 dist.barrier()
 best=checkpoint.get('best_mpjpe',float('inf'));best_epoch=checkpoint.get('best_mpjpe_epoch',start);best_loss=checkpoint['best_val']
 def status(state,epoch,**extra):
  if rank==0:write_json(a.output/'status.json',dict(state=state,completed_epochs=epoch,target_epochs=identity['epochs'],parent_epoch=identity['parent_epoch'],pid=os.getpid(),rank_pids=pids,world_size=world,batch_per_gpu=a.batch_per_gpu,global_batch_size=a.batch_per_gpu*world,smoke=identity['smoke'],best_mpjpe_mm=best,best_mpjpe_epoch=best_epoch,**extra))
 def payload(epoch,states):return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),identity=identity,epoch=epoch,step=step,best_val=best_loss,best_mpjpe=best,best_mpjpe_epoch=best_epoch,rng_per_rank=states)
 def states():
  values=[None]*world;dist.all_gather_object(values,rng_state(generator));return values
 def log(record):
  if rank==0:
   with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 if not a.resume:
  baseline=evaluate_fixed(model,a.data_root,a.rot_root,rows,a.output/'eval_fixed_val'/f'epoch_{start:04d}',start,step,rank,world,device)
  best=baseline['metrics']['global_mpjpe_mm']['action_mean'];best_epoch=start
  initial=payload(start,states())
  if rank==0:
   checkpoint_save(a.output/'best.pt',initial);checkpoint_save(a.output/'latest.pt',initial)
   write_json(a.output/'eval_fixed_val'/f'epoch_{start:04d}'/'identity.json',dict(checkpoint_sha256=sha256(a.output/'latest.pt'),fixed_val_manifest_sha256=identity['fixed_val_manifest_sha256']))
  log(dict(epoch=start,step=step,baseline=True,fixed_val=baseline['metrics'],selection=dict(best_global_mpjpe_mm=best,best_epoch=best_epoch)))
  del initial
 dist.barrier();status('training',start);began=time.time();parent_epoch=parent['epoch'];parent_step=parent['step']
 del checkpoint,parent
 if rank==0:print(f'CONTINUATION epochs {start+1}..{identity["epochs"]} | {world}-GPU batch/GPU={a.batch_per_gpu} | fixed VAL={len(rows)} | best criterion=global MPJPE | LR schedule={schedule}',flush=True)
 for epoch in range(start,identity['epochs']):
  tick=time.time();sampler.epoch=epoch
  train_values,steps,lr=train_epoch(wrapped,ddp,tl,optimizer,device,rank,world,DEFAULT_WEIGHTS,step,parent_step,schedule);step+=steps
  val_values,_=epoch_pass(wrapped,ddp,vl,None,device,rank,world,DEFAULT_WEIGHTS);best_loss=min(best_loss,val_values['total'])
  evaluation=None;improved=False
  if (epoch+1-parent_epoch)%a.eval_every==0 or epoch+1==identity['epochs']:
   evaluation=evaluate_fixed(model,a.data_root,a.rot_root,rows,a.output/'eval_fixed_val'/f'epoch_{epoch+1:04d}',epoch+1,step,rank,world,device)
   score=evaluation['metrics']['global_mpjpe_mm']['action_mean'];improved=score<best
   if improved:best=score;best_epoch=epoch+1
  record=dict(epoch=epoch+1,step=step,continuation_step=step-parent_step,lr=lr,train=train_values,val=val_values,seconds=time.time()-began,epoch_seconds=time.time()-tick,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,selection=dict(best_global_mpjpe_mm=best,best_epoch=best_epoch))
  if evaluation:record['fixed_val']=evaluation['metrics']
  state=payload(epoch+1,states())
  if rank==0:
   checkpoint_save(a.output/'latest.pt',state)
   if improved:checkpoint_save(a.output/'best.pt',state)
   if evaluation:
    path=a.output/f'epoch_{epoch+1:04d}.pt';checkpoint_save(path,state)
    write_json(a.output/'eval_fixed_val'/f'epoch_{epoch+1:04d}'/'identity.json',dict(checkpoint_sha256=sha256(path),fixed_val_manifest_sha256=identity['fixed_val_manifest_sha256']))
  log(record);del state
  dist.barrier();status('training',epoch+1)
  if rank==0:
   eta=(time.time()-began)/(epoch+1-start)*(identity['epochs']-epoch-1)
   print(f"Epoch {epoch+1}/{identity['epochs']} | updates {step} (+{step-parent_step}) | train={train_values['total']:.6f} val={val_values['total']:.6f} | best MPJPE={best:.3f} mm @ep{best_epoch} | ETA {timedelta(seconds=int(eta))} ({datetime.now()+timedelta(seconds=eta):%m-%d %H:%M:%S})",flush=True)
 dist.barrier();del ddp,wrapped,model,optimizer,tl,vl;dist.destroy_process_group();torch.cuda.empty_cache()
 if rank==0:
  status('evaluating',identity['epochs'])
  from eval_g1_rot6d68 import evaluate
  for name in ('best','latest'):evaluate(a.data_root,a.rot_root,a.output/f'{name}.pt',a.output/f'eval_{name}',str(device),32,allow_smoke=a.allow_smoke)
  status('complete',identity['epochs'],evaluation_complete=True)
if __name__=='__main__':main()
