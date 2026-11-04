"""Train fresh segment-conditioned G1 diffusion forcing with frozen rot69 VAE."""
import argparse,copy,json,math,os,random,sys,time
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'momask-codes/dataset/bones-seed_prepare'))
from g1.common import sha256,write_json,G1Kinematics
from g1.data import read_index,DESCRIPTION_COLUMNS
from g1.model import G1CausalVAE
from g1.rot6d69 import decode_native,validate_rot_data
from g1.continuation import fixed_rows
from g1.generation import encode_motion,decode_motion
from g1.segment_generation import SegmentDiffusion
from g1.segment_data import SegmentMotions,conditions,load_labels

def collate(items):
    n=max(len(x[0]) for x in items);features=[]
    for x,_,_ in items:features.append(torch.cat((x,x[-1:].expand(n-len(x),-1))))
    return torch.stack(features),torch.tensor([len(x[0]) for x in items]),[x[1] for x in items],[x[2] for x in items]

def rng():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
def restore(s):
    torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda']);np.random.set_state(s['numpy']);random.setstate(s['python'])
def save(path,payload):
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)

@torch.inference_mode()
def evaluate(model,vae,loader,a,rank,world,device,epoch):
    state=rng();torch.manual_seed(9917+rank);np.random.seed(9917+rank);random.seed(9917+rank)
    model.eval();fk=G1Kinematics(Path(a.data_root)/'kinematic.xml');records=[]
    for features,lengths,texts,rows in loader:
        lens=1+(lengths.to(device)-1+3)//4
        with torch.autocast('cuda',dtype=torch.bfloat16):
            generated=model.generate(dict(feature_length=lens,**conditions(texts)))['generated']
        # Decode equal padded latent lengths together; causal VAE preserves valid prefix.
        longest=max(len(z) for z in generated)
        z=torch.stack([torch.cat((v,v[-1:].expand(longest-len(v),-1))) for v in generated])
        decoded=decode_motion(vae,z)
        anchors=torch.tensor(np.stack([np.load(Path(a.data_root)/'motions'/f"{r['id']}.npy")[0,:3] for r in rows]),device=device)
        predictions=decode_native(decoded,anchors).cpu().numpy()
        for row,pred in zip(rows,predictions):
            gt=np.load(Path(a.data_root)/'motions'/f"{row['id']}.npy");pred=pred[:len(gt)]
            gp,pp=fk.positions(gt),fk.positions(pred)
            ge=float(np.linalg.norm(pp-gp,axis=-1).mean()*1000)
            re=float(np.linalg.norm((pp-pred[:,None,:3])-(gp-gt[:,None,:3]),axis=-1).mean()*1000)
            if not math.isfinite(ge+re):raise FloatingPointError('Nonfinite generated metrics')
            records.append(dict(id=row['id'],global_mpjpe_mm=ge,root_relative_mpjpe_mm=re))
        if rank==0:print(f'VAL epoch={epoch} local clips={len(records)}/{len(loader.dataset)}',flush=True)
    all_records=[None]*world;dist.all_gather_object(all_records,records)
    records=sorted([r for part in all_records for r in part],key=lambda r:r['id'])
    assert len(records)==len({r['id'] for r in records})
    summary={k:float(np.mean([r[k] for r in records])) for k in ('global_mpjpe_mm','root_relative_mpjpe_mm')}
    summary['samples']=len(records)
    if rank==0:write_json(Path(a.output)/f'val_epoch_{epoch:04d}.json',dict(metrics=summary,per_sample=records,seed=9917,protocol='time-aligned segment text generation, GT duration and initial XY; equal-clip means; paired MPJPE is not text alignment or diversity'))
    restore(state);return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vae',default='outputs/g1_rot69_root_temporal_500ep_bs2048_mpjpe/best.pt')
    p.add_argument('--data-root',default='outputs/bones_g1_native38');p.add_argument('--rot-root',default='outputs/bones_g1_rot6d69')
    p.add_argument('--fixed-val',default='outputs/rot68_fixed_val_tracker_v1')
    p.add_argument('--clip',default='/data/langlang/cache/clip/ViT-B-32.pt');p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=500);p.add_argument('--batch-per-gpu',type=int,default=512)
    p.add_argument('--segment-data',default='outputs/g1_segment_labels_v1')
    p.add_argument('--activation-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--smoke-batches',type=int,default=2)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--eval-every',type=int,default=0,help='Periodic generation validation interval; 0 disables periodic validation (final evaluation retained)');p.add_argument('--hidden-dim',type=int,default=1024)
    p.add_argument('--layers',type=int,default=8);p.add_argument('--smoke',action='store_true');p.add_argument('--resume')
    a=p.parse_args();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    torch.set_num_threads(2);torch.manual_seed(3407);np.random.seed(3407+rank);random.seed(3407+rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output)
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        if (out/'identity.json').exists() and not a.resume:raise ValueError('Existing experiment; use --resume')
    dist.barrier()
    manifest=validate_rot_data(a.data_root,a.rot_root);valrows,_=fixed_rows(a.data_root,a.fixed_val)
    trainrows=read_index(a.data_root,'train')
    assert not {r['source_group'] for r in trainrows}&{r['source_group'] for r in valrows}
    native_manifest=json.loads((Path(a.data_root)/'manifest.json').read_text())
    metadata=Path(native_manifest['source_root'])/'metadata/seed_metadata_v004.parquet'
    assert sha256(metadata)==native_manifest['metadata_sha256']
    labels,eligible_train,segment_manifest=load_labels(a.segment_data,a.data_root)
    trainrows=[r for r in trainrows if r['id'] in eligible_train]
    if len(trainrows)!=segment_manifest['training_samples']:raise ValueError('Segment training list mismatch')
    production_samples=len(trainrows)
    if a.smoke:trainrows=trainrows[:world*a.batch_per_gpu*a.smoke_batches];valrows=valrows[:world*2]
    files=['train_g1_ldf_segment.py','g1/segment_generation.py','g1/segment_data.py','g1/generation.py','models/diffusion_forcing_wan.py','models/tools/wan_model.py','models/tools/wan_vae_1d.py','g1/model.py','g1/rot6d69.py']
    identity=dict(config={k:v for k,v in vars(a).items() if k not in ('resume','output')},world_size=world,
        vae_sha256=sha256(a.vae),clip_sha256=sha256(a.clip),data_sha256=sha256(Path(a.data_root)/'manifest.json'),
        rot_sha256=sha256(Path(a.rot_root)/'manifest.json'),cohort_sha256=sha256(Path(a.fixed_val)/'manifest.json'),
        metadata_sha256=sha256(metadata),segment_manifest_sha256=sha256(Path(a.segment_data)/'manifest.json'),segment_audit=segment_manifest['stats'],initialization='fresh generator and optimizer; frozen specified VAE and CLIP',code_sha256={f:sha256(ROOT/f) for f in files},
        train_samples=len(trainrows),val_samples=len(valrows),latent='frozen posterior mean, unscaled',
        selection='minimum fixed-val generated global MPJPE equal-clip mean',text='frozen CLIP token states; source-time-aligned per-latent segment conditions; empty gap conditions',
        precision='frozen VAE fp32, diffusion bf16 autocast',ema=.99)
    parent=torch.load(a.vae,map_location='cpu',weights_only=False)
    vae=G1CausalVAE(np.zeros(69,np.float32),np.ones(69,np.float32),**parent['identity']['model_params']).to(device)
    vae.load_state_dict(parent['model'],strict=True);vae.eval().requires_grad_(False);del parent
    model=SegmentDiffusion(a.clip,device,a.hidden_dim,a.layers,a.activation_checkpointing).to(device)
    ema=copy.deepcopy(model.model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.99),weight_decay=0.)
    start=step=0;best=float('inf');schedule_total=None
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False)
        if ck['identity']!=identity:raise ValueError('Resume identity mismatch')
        model.load_state_dict(ck['model']);ema.load_state_dict(ck['ema']);optimizer.load_state_dict(ck['optimizer'])
        start,step,best=ck['epoch'],ck['step'],ck['best'];schedule_total=ck.get('schedule_total');restore(ck['rng'][rank]);del ck
    else:torch.manual_seed(3407+rank)
    run=None
    if rank==0:
        write_json(out/'identity.json',identity)
        if not a.smoke:
            import wandb
            track=out/'wandb_run.json';old=json.loads(track.read_text()) if track.exists() else {}
            run=wandb.init(project='flood-g1-generation',entity='langlang-02-ustc',name=out.name,
                id=old.get('id'),resume='must' if old else None,config=identity,dir=str(out),mode='online',allow_val_change=True)
            write_json(track,dict(id=run.id,url=run.url,entity=run.entity,project=run.project))
        print(f'Training {len(trainrows)} clips, frozen VAE, world={world}, global batch={a.batch_per_gpu*world}',flush=True)
    dist.barrier();ddp=DDP(model,device_ids=[local])
    dataset=SegmentMotions(trainrows,a.rot_root,labels,True)
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=3407,drop_last=True)
    loader=DataLoader(dataset,batch_size=a.batch_per_gpu,sampler=sampler,num_workers=a.workers,collate_fn=collate,pin_memory=True)
    valloader=DataLoader(SegmentMotions(valrows[rank::world],a.rot_root,labels,False),batch_size=8,num_workers=0,collate_fn=collate)
    total=schedule_total or (step+(a.epochs-start)*len(loader));began=time.time();initial_step=step
    for epoch in range(start,a.epochs):
        sampler.set_epoch(epoch);model.train();acc=0.;count=0;epstart=time.time()
        for features,lengths,texts,rows in loader:
            features=features.to(device);lengths=lengths.to(device)
            latent=encode_motion(vae,features)
            lr=a.lr*min((step+1)/1000,1.) if step<1000 else 1e-6+.5*(a.lr-1e-6)*(1+math.cos(math.pi*(step-1000)/max(total-1001,1)))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=ddp(dict(feature=latent,feature_length=1+(lengths-1+3)//4,**conditions(texts)))['total']
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():
                for e,m in zip(ema.parameters(),model.model.parameters()):e.lerp_(m,.01)
            step+=1;acc+=float(loss.detach())*len(rows);count+=len(rows)
            if step%10==0 or step==initial_step+1:
                metric=loss.detach().clone();dist.all_reduce(metric);metric/=world
                counts=torch.tensor(list(model.last_condition_stats.values()),device=device,dtype=torch.float64);dist.all_reduce(counts)
                peak=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device=device);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
                if rank==0:
                    elapsed=time.time()-began;eta=elapsed/max(step-initial_step,1)*(total-step)
                    log={'train/loss':float(metric),'train/grad_norm':float(grad),'lr':lr,'epoch':epoch+1,'step':step,'eta_hours':eta/3600,'train/global_batch':int(counts[0]),'condition/cfg_dropout_fraction':float(counts[1]/counts[0]),'condition/gap_fraction':float(counts[2]/counts[3]),'condition/segments_per_sample':float(counts[4]/counts[0]),'memory/peak_allocated_gib':float(peak),'train/samples_per_second':(step-initial_step)*a.batch_per_gpu*world/elapsed}
                    print(json.dumps(log),flush=True)
                    if run:run.log(log,step=step)
        values=torch.tensor([acc,count],device=device,dtype=torch.float64);dist.all_reduce(values)
        validation=None;improved=False
        if (a.eval_every>0 and (epoch+1)%a.eval_every==0) or epoch+1==a.epochs:
            trained=model.model;model.model=ema
            validation=evaluate(model,vae,valloader,a,rank,world,device,epoch+1);model.model=trained
            assert validation['samples']==len(valrows)
            improved=validation['global_mpjpe_mm']<best;best=min(best,validation['global_mpjpe_mm'])
        states=[None]*world;dist.all_gather_object(states,rng())
        if rank==0:
            payload=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),identity=identity,
                         epoch=epoch+1,step=step,best=best,rng=states,schedule_total=total)
            save(out/'latest.pt',payload)
            if improved:save(out/'best.pt',payload)
            if a.smoke or (epoch+1)%25==0 or epoch+1==a.epochs:save(out/f'epoch_{epoch+1:04d}.pt',payload)
            record=dict(epoch=epoch+1,step=step,train_loss=float(values[0]/values[1]),epoch_seconds=time.time()-epstart,validation=validation)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            write_json(out/'status.json',record);print(json.dumps(record),flush=True)
            if run:
                logs={'epoch/train_loss':record['train_loss'],'epoch/seconds':record['epoch_seconds']}
                if validation:logs.update({f'fixed_val/{k}':v for k,v in validation.items()});logs['selection/best_global_mpjpe_mm']=best
                run.log(logs,step=step)
        dist.barrier()
    if run:run.finish()
    dist.destroy_process_group()
if __name__=='__main__':main()
