#!/usr/bin/env python3
"""Continue a G1 causal VAE checkpoint using torchrun DDP, then auto-evaluate."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from g1.common import ROOT,MOMASK,sha256,write_json
from g1.data import MotionDataset,collate_motions
from g1.distributed import GlobalBatchShard,LossForward,KEYS,denominators,ddp_objective
from g1.kinematics import TorchG1FK
from g1.model import G1CausalVAE,TextOpLoss,DEFAULT_WEIGHTS
from train_g1_vae import load_data_identity,checkpoint_save,seed_worker


def emit(rank, value):
    if rank==0:
        print(json.dumps(value),flush=True)


def rng_state(generator):
    return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_rng=generator.get_state())


def restore_rng(state,generator):
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    generator.set_state(state['loader_rng'])


def epoch_pass(wrapped,ddp,loader,optimizer,device,rank,world,weights):
    training=optimizer is not None
    wrapped.train(training)
    accumulator=torch.zeros(len(KEYS)+2,device=device,dtype=torch.float64)
    started=time.time()
    steps=0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            terms=ddp(batch,sample=True) if training else wrapped(batch,sample=False)
            values=torch.stack([terms[k].detach() for k in KEYS])
            if training:
                counts=denominators(batch)
                global_counts=counts.clone()
                dist.all_reduce(global_counts)
                finite=torch.isfinite(values).all().to(torch.int32)
                dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError('Nonfinite loss on at least one DDP rank')
                objective=ddp_objective(terms,counts,global_counts,world,weights)
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(wrapped.vae.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                numerator=values*counts
                dist.all_reduce(numerator)
                values=numerator/global_counts.clamp_min(1)
                frames=global_counts[0]
            else:
                # No collectives inside unequal-length validation shards.
                if not torch.isfinite(values).all():
                    raise FloatingPointError('Nonfinite validation loss')
                frames=batch['length'].sum()
            total=sum(weights[k]*values[i] for i,k in enumerate(KEYS))
            accumulator[:-2] += values.double()*frames
            accumulator[-2] += total.double()*frames
            accumulator[-1] += frames
            steps+=1
            if rank==0 and (steps==1 or steps%50==0 or steps==len(loader)):
                emit(rank,dict(event='batch_progress',phase='train' if training else 'val',
                    batch=steps,batches=len(loader),mean_total=float(accumulator[-2]/accumulator[-1]),
                    elapsed_s=time.time()-started,world_size=world))
    if not training:
        dist.all_reduce(accumulator)
    if accumulator[-1]<=0:
        raise ValueError('Empty global epoch')
    means=(accumulator[:-1]/accumulator[-1]).cpu().tolist()
    return dict(zip((*KEYS,'total'),means)),steps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2,help='Data workers per rank')
    p.add_argument('--val-batch-size',type=int,default=32,help='Preserve original validation batch grouping')
    p.add_argument('--allow-smoke',action='store_true')
    args=p.parse_args()
    rank,local,world=(int(os.environ[n]) for n in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    manifest=load_data_identity(args.data_root)
    parent_sha=sha256(args.resume)
    ckpt=torch.load(args.resume,map_location='cpu',weights_only=False)
    identity=dict(ckpt['identity'])
    if identity['data_manifest_sha256']!=sha256(args.data_root/'manifest.json'):
        raise ValueError('Checkpoint/data identity mismatch')
    if (identity['smoke'] or manifest['smoke']) and not args.allow_smoke:
        raise ValueError('Smoke input refused for production')
    if identity['weights']!=DEFAULT_WEIGHTS:
        raise ValueError('Unexpected TextOp weights')
    for key,path in [('model_source_sha256',ROOT/'models/tools/wan_vae_1d.py'),
        ('adapter_sha256',ROOT/'g1/model.py'),('fk_sha256',ROOT/'g1/kinematics.py'),
        ('dataset_source_sha256',ROOT/'g1/data.py'),('metric_sha256',MOMASK/'evaluation/g1_metrics.py')]:
        if identity[key]!=sha256(path):
            raise ValueError(f'Checkpoint component changed: {key}')
    migration='distributed' not in identity
    distributed=dict(world_size=world,global_batch_size=identity['batch_size'],
        nominal_batch_per_rank=identity['batch_size']//world,training_sampler='global_batches_exact_no_duplicates',
        reduction='per-term global valid-count weighting; DDP gradient mean corrected',
        validation='original contiguous batches partitioned across ranks without padding')
    if not migration:
        if identity['distributed']!=distributed or identity['trainer_sha256']!=sha256(Path(__file__)) or identity['ddp_helper_sha256']!=sha256(ROOT/'g1/distributed.py'):
            raise ValueError('DDP resume configuration/source differs')
    else:
        identity['single_trainer_sha256']=identity['trainer_sha256']
        identity['trainer_sha256']=sha256(Path(__file__))
        identity['ddp_helper_sha256']=sha256(ROOT/'g1/distributed.py')
        identity['distributed']=distributed
    if identity['batch_size']%world:
        raise ValueError('Global batch must be divisible by world size')
    seed=identity['seed']+rank+10000*ckpt['epoch']
    torch.manual_seed(seed);np.random.seed(seed%(2**32));random.seed(seed)
    generator=torch.Generator().manual_seed(seed)
    mean,std=(np.load(args.data_root/f'{n}.npy') for n in ('mean','std'))
    model=G1CausalVAE(mean,std,**identity['model_params']).to(device)
    model.load_state_dict(ckpt['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=identity['learning_rate'])
    optimizer.load_state_dict(ckpt['optimizer'])
    wrapped=LossForward(model,TextOpLoss(TorchG1FK(args.data_root/'kinematic.xml').to(device),fps=manifest['fps']))
    ddp=DDP(wrapped,device_ids=[local],broadcast_buffers=False)
    if 'rng_per_rank' in ckpt:
        if len(ckpt['rng_per_rank'])!=world:
            raise ValueError('RNG rank count mismatch')
        restore_rng(ckpt['rng_per_rank'][rank],generator)
    elif rank==0:
        # Other ranks get independent deterministic streams on the one-time migration.
        state={k:ckpt[k] for k in ('torch_rng','numpy_rng','python_rng','loader_rng')}
        state['cuda_rng']=ckpt['cuda_rng'][0]
        restore_rng(state,generator)
    train_ds=MotionDataset(args.data_root,'train',identity['window'])
    val_ds=MotionDataset(args.data_root,'val',identity['window'])
    train_sampler=GlobalBatchShard(len(train_ds),identity['batch_size'],rank,world,identity['seed'])
    val_sampler=GlobalBatchShard(len(val_ds),args.val_batch_size,rank,world,training=False)
    def loader(dataset,sampler):
        return DataLoader(dataset,batch_sampler=sampler,num_workers=args.workers,
            collate_fn=collate_motions,generator=generator,worker_init_fn=seed_worker,
            pin_memory=True,multiprocessing_context='spawn' if args.workers else None)
    train_loader,val_loader=loader(train_ds,train_sampler),loader(val_ds,val_sampler)
    start_epoch,best,step=ckpt['epoch'],ckpt['best_val'],ckpt['step']
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=True)
        best_path=args.resume.parent/'best.pt'
        if not best_path.exists():
            raise ValueError('Best checkpoint missing')
        if best_path.resolve()!=(args.output/'best.pt').resolve():
            shutil.copy2(best_path,args.output/'best.pt')
        old_best=torch.load(args.output/'best.pt',map_location='cpu',weights_only=False)
        if old_best['epoch']>start_epoch or old_best['identity']['data_manifest_sha256']!=identity['data_manifest_sha256']:
            raise ValueError('Best checkpoint does not belong to this continuation')
        if old_best['best_val']!=best:
            raise ValueError('Best checkpoint score differs from resume state')
        del old_best
        write_json(args.output/'identity.json',identity)
        write_json(args.output/'ddp_arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
        if migration:
            write_json(args.output/'ddp_migration.json',dict(parent_checkpoint=str(args.resume.resolve()),
                parent_sha256=parent_sha,parent_epoch=start_epoch,parent_step=step,
                previous_identity=ckpt['identity'],new_identity=identity,
                stochastic_trajectory='new independent rank streams; not bitwise equivalent to single GPU'))
    dist.barrier()
    pids=[None]*world
    dist.all_gather_object(pids,os.getpid())
    def status(state,epoch,**extra):
        if rank==0:
            write_json(args.output/'status.json',dict(state=state,completed_epochs=epoch,
                target_epochs=identity['epochs'],pid=os.getpid(),rank_pids=pids,world_size=world,
                global_batch_size=identity['batch_size'],smoke=identity['smoke'],**extra))
    status('training',start_epoch)
    emit(rank,dict(event='ddp_started',resume_epoch=start_epoch,world_size=world,rank_pids=pids,
        global_batch_size=identity['batch_size'],target_epochs=identity['epochs'],migration=migration))
    del ckpt
    started=time.time()
    for epoch in range(start_epoch,identity['epochs']):
        train_sampler.epoch=epoch
        lr=identity['learning_rate']*(.01+.99*.5*(1+math.cos(math.pi*epoch/identity['epochs'])))
        for group in optimizer.param_groups:group['lr']=lr
        emit(rank,dict(event='epoch_start',epoch=epoch+1,target_epochs=identity['epochs'],lr=lr,world_size=world))
        train,steps=epoch_pass(wrapped,ddp,train_loader,optimizer,device,rank,world,identity['weights'])
        val,_=epoch_pass(wrapped,ddp,val_loader,None,device,rank,world,identity['weights'])
        step+=steps
        improved=val['total']<best
        best=min(best,val['total'])
        states=[None]*world
        dist.all_gather_object(states,rng_state(generator))
        if rank==0:
            record=dict(epoch=epoch+1,step=step,lr=lr,train=train,val=val,seconds=time.time()-started,world_size=world)
            with (args.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            emit(rank,record)
            checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,
                best_val=best,step=step,identity=identity,rng_per_rank=states)
            checkpoint_save(args.output/'latest.pt',checkpoint)
            if improved:checkpoint_save(args.output/'best.pt',checkpoint)
            if (epoch+1)%identity['checkpoint_every']==0 or epoch+1==identity['epochs']:
                checkpoint_save(args.output/f'epoch_{epoch+1:04d}.pt',checkpoint)
            del checkpoint
        dist.barrier()
        status('training',epoch+1)
    dist.barrier()
    del ddp,wrapped,optimizer,model,train_loader,val_loader
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    if rank==0:
        status('evaluating',identity['epochs'])
        from eval_g1_vae import evaluate
        for name in ('best','latest'):
            evaluate(args.data_root,args.output/f'{name}.pt',args.output/f'eval_{name}',
                     str(device),args.val_batch_size,allow_smoke=args.allow_smoke)
        status('complete',identity['epochs'],evaluation_complete=True)


if __name__=='__main__':main()
