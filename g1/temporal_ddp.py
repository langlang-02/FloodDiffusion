"""Denominator-correct DDP reduction including pair/triple-frame temporal losses."""
import json,time
import torch
import torch.distributed as dist
from .temporal_loss import TEMPORAL_WEIGHTS
KEYS=tuple(TEMPORAL_WEIGHTS)

def emit(rank,value):
    if rank==0:print(json.dumps(value),flush=True)

def denominators(batch):
    length=batch['length'];frames=length.sum().float();pairs=(length-1).clamp_min(0).sum().float();triples=(length-2).clamp_min(0).sum().float()
    values=dict(rec=frames,body_trans=frames,body_rot=frames,dof_pos=frames,
                dof_vel=pairs,kl=(1+(length-1+3)//4).sum().float(),foot_contact=(length*batch['eligible']).sum().float(),
                dof_acc=triples,root_rot_vel=pairs,root_rot_acc=triples)
    return torch.stack([values[k] for k in KEYS])

def ddp_objective(terms,local_counts,global_counts,world,weights):
    return sum(weights[k]*terms[k]*world*local_counts[i]/global_counts[i].clamp_min(1) for i,k in enumerate(KEYS))

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

