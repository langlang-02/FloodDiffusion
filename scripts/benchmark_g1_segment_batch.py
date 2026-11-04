#!/usr/bin/env python3
"""Isolated single-GPU end-to-end batch benchmark; never updates production state."""
import argparse,copy,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train_g1_ldf_segment import collate
from g1.common import sha256
from g1.data import read_index
from g1.segment_data import SegmentMotions,conditions,load_labels
from g1.segment_generation import SegmentDiffusion
from g1.model import G1CausalVAE
from g1.generation import encode_motion

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,required=True)
 p.add_argument('--warmup-samples',type=int,default=1024);p.add_argument('--measure-samples',type=int,default=4096);a=p.parse_args()
 assert a.warmup_samples%a.batch==0 and a.measure_samples%a.batch==0
 torch.set_num_threads(2);torch.manual_seed(3407);np.random.seed(3407);random.seed(3407)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device('cuda');c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=c['identity']['config']
 labels,eligible,manifest=load_labels(cfg['segment_data'],cfg['data_root'])
 rows=[r for r in read_index(cfg['data_root'],'train') if r['id'] in eligible]
 choice=np.random.default_rng(20261103).permutation(len(rows))[:a.warmup_samples+a.measure_samples];rows=[rows[i] for i in choice]
 model=SegmentDiffusion(cfg['clip'],device,cfg['hidden_dim'],cfg['layers'],False).to(device);model.load_state_dict(c['model']);model.train()
 ema=copy.deepcopy(model.model).eval().requires_grad_(False);ema.load_state_dict(c['ema'])
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=(.9,.99),weight_decay=0.);optimizer.load_state_dict(c['optimizer'])
 epoch,step=c['epoch'],c['step'];del c
 parent=torch.load(ROOT/cfg['vae'],map_location='cpu',weights_only=False)
 vae=G1CausalVAE(np.zeros(69,np.float32),np.ones(69,np.float32),**parent['identity']['model_params']).to(device)
 vae.load_state_dict(parent['model']);vae.eval().requires_grad_(False);del parent
 loader=DataLoader(SegmentMotions(rows,cfg['rot_root'],labels),batch_size=a.batch,num_workers=4,collate_fn=collate,pin_memory=True)
 iterator=iter(loader);timings=[];warm_steps=a.warmup_samples//a.batch;measure_steps=a.measure_samples//a.batch
 def update():
  features,lengths,texts,_=next(iterator);features=features.to(device,non_blocking=True);lengths=lengths.to(device,non_blocking=True)
  latent=encode_motion(vae,features);optimizer.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(dict(feature=latent,feature_length=1+(lengths-1+3)//4,**conditions(texts)))['total']
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
  with torch.no_grad():
   for e,m in zip(ema.parameters(),model.model.parameters()):e.lerp_(m,.01)
  return float(loss.detach())
 for i in range(warm_steps):update()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
 for i in range(measure_steps):
  start=time.perf_counter();loss=update();torch.cuda.synchronize();timings.append(time.perf_counter()-start)
  if (i+1)%4==0 or i+1==measure_steps:print(json.dumps(dict(batch=a.batch,measured_step=i+1,total_steps=measure_steps,seconds=time.perf_counter()-began,loss=loss)),flush=True)
 elapsed=time.perf_counter()-began
 result=dict(batch_per_gpu=a.batch,activation_checkpointing=False,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(),checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),epoch=epoch,step=step,warmup_samples=a.warmup_samples,measured_samples=a.measure_samples,measured_steps=measure_steps,elapsed_seconds=elapsed,samples_per_second=a.measure_samples/elapsed,mean_step_seconds=float(np.mean(timings)),median_step_seconds=float(np.median(timings)),step_seconds=timings,first_half_samples_per_second=(a.measure_samples/2)/sum(timings[:measure_steps//2]),second_half_samples_per_second=(a.measure_samples/2)/sum(timings[measure_steps//2:]),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,measured_ids=[r['id'] for r in rows[a.warmup_samples:]],scope='single GPU: data loading, frozen VAE, frozen CLIP/cache, diffusion forward/backward, AdamW, EMA; excludes DDP allreduce, validation and checkpoint saving; production keeps running')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('measured_ids','step_seconds')}),flush=True)
if __name__=='__main__':main()
