#!/usr/bin/env python3
"""Evaluate frozen RVQ on exactly the native clips used for VAE evaluation."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
MOMASK=ROOT.parent/'momask-codes'
sys.path.insert(0,str(MOMASK))
from evaluation.eval_g1_rvq import reconstruct_batch
from evaluation.g1_metrics import G1Kinematics,metric_values,reduce_values
from g1_visualize.reconstruct_g1_rvq import build_model
from g1_io import model_features_to_feature_bank,local_features_to_model_features
from utils.get_opt import get_opt

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    data=ROOT/'outputs/bones_g1_native38'
    out=ROOT/'outputs/vae_rvq_comparison/rvq_same_test'
    out.mkdir(parents=True,exist_ok=False)
    directory=Path('/data/langlang/checkpoints/momask/bones_g1_rvq_rot6d68_v1_nq6')
    path=directory/'model/frozen_latest_ep0499_it8998000.tar'
    spec=json.loads((directory/'meta/representation.json').read_text())
    torch.set_num_threads(4);torch.manual_seed(3407)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    assert ckpt['representation']==spec
    opt=get_opt(str(directory/'opt.txt'),torch.device('cuda:0'))
    model=build_model(opt,spec['feature_dim'],torch.device('cuda:0'))
    model.load_state_dict(ckpt.get('vq_model',ckpt.get('net')))
    mean,std=(np.load(directory/'meta'/f'{n}.npy') for n in ('mean','std'))
    rows=[json.loads(l) for l in (data/'index.jsonl').read_text().splitlines() if json.loads(l)['split']=='test']
    identity=dict(checkpoint=str(path),checkpoint_sha256=sha(path),epoch=int(ckpt['ep']),samples=len(rows),
        split_sha256=sha(data/'test.txt'),mjcf_sha256=sha(data/'kinematic.xml'),metric_sha256=sha(MOMASK/'evaluation/g1_metrics.py'),
        representation=spec,protocol='g1_rvq_reconstruction_v1',fps=20,
        input='VAE native38 test clips, converted with canonical MoMask feature bank; first deltas zero for independent clip',
        caveat='VAE held-out split; RVQ training-source overlap audited separately; not historical RVQ benchmark')
    (out/'identity.json').write_text(json.dumps(identity,indent=2))
    (out/'samples.txt').write_text('\n'.join(r['id'] for r in rows)+'\n')
    groups=defaultdict(list)
    for r in rows:groups[r['frames']].append(r)
    fk=G1Kinematics(data/'kinematic.xml');totals={};results=[];roundtrip=0.;delta=0.;checked=False;t0=time.time()
    predictions=out/'predictions';predictions.mkdir()
    with (out/'per_sample.csv').open('w') as stream:
        writer=None
        for length,group in sorted(groups.items()):
            for start in range(0,len(group),32):
                batch=group[start:start+32]
                gt=[np.load(data/'motions'/f"{r['id']}.npy") for r in batch]
                banks,anchors=zip(*(model_features_to_feature_bank(x) for x in gt))
                features=[b[:,spec['indices']] for b in banks]
                output=reconstruct_batch(model,features,mean,std,opt.stride_t**opt.down_t,'cuda:0')
                if not checked and len(batch)>1:
                    single=reconstruct_batch(model,features[:1],mean,std,opt.stride_t**opt.down_t,'cuda:0')[0]
                    delta=float(np.max(np.abs(single-output[0])))
                    np.testing.assert_allclose(single,output[0],atol=1e-4,rtol=1e-4);checked=True
                for r,truth,feature,anchor,pred in zip(batch,gt,features,anchors,output):
                    recovered=local_features_to_model_features(feature,spec['fields'],anchor[0])
                    err=float(np.max(np.abs(recovered-truth)));roundtrip=max(roundtrip,err)
                    assert err<1e-3,(r['id'],err)
                    native=local_features_to_model_features(pred,spec['fields'],anchor[0])
                    np.save(predictions/f"{r['id']}.npy",native)
                    values=metric_values(truth,native,fk.positions(truth),fk.positions(native),20,fk.endpoints)
                    result=dict(sample_id=r['id'],frames=length,foot_contact_eligible=r['foot_contact_eligible'])
                    for k,v in values.items():
                        result[k]=reduce_values(k,v)
                        s,n=totals.get(k,(0.,0));totals[k]=(s+float((v*v).sum() if 'rmse' in k else v.sum()),n+v.size)
                    if writer is None:writer=csv.DictWriter(stream,fieldnames=list(result));writer.writeheader()
                    writer.writerow(result);results.append(result)
                if len(results)%500<len(batch):print(f'RVQ {len(results)}/{len(rows)} {time.time()-t0:.1f}s',flush=True)
    summary=dict(complete=True,epoch=int(ckpt['ep']),samples=len(results),frames=sum(r['frames'] for r in results),seconds=time.time()-t0,
        batch_single_checked=checked,batch_single_max_abs=delta,roundtrip_max_abs=roundtrip,metrics={})
    for k,(s,n) in totals.items():summary['metrics'][k]=dict(action_mean=float(np.mean([r[k] for r in results])),action_p95=float(np.percentile([r[k] for r in results],95)),pooled=float(np.sqrt(s/n) if 'rmse' in k else s/n),observations=n)
    assert len(results)==8333 and sha(path)==identity['checkpoint_sha256']
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
