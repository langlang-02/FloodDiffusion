#!/usr/bin/env python3
"""Evaluate native G1 reconstructions with the unmodified MoMask G1 metrics."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from g1.common import G1Kinematics, JOINT_NAMES, ENDPOINT_NAMES, MOMASK, sha256, write_json, metric_values, reduce_values
from g1.temporal_metrics import temporal_values
from g1.data import read_index
from g1.model import G1CausalVAE
from g1.rot6d69 import validate_rot_data,encode_native,decode_native,SPEC
from train_g1_vae import load_data_identity


def evaluate(data_root, rot_root, checkpoint, output, device='cuda:0', batch_size=32, allow_smoke=False):
    data_root, checkpoint, output = map(Path, (data_root, checkpoint, output))
    rot_root=Path(rot_root)
    rot_manifest=validate_rot_data(data_root,rot_root)
    manifest = load_data_identity(data_root)
    checksum = sha256(checkpoint)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ckpt['identity']['representation']!=SPEC:raise ValueError('Representation mismatch')
    if ckpt['identity']['rot_manifest_sha256']!=sha256(rot_root/'manifest.json'):
        raise ValueError('Rot68 dataset identity mismatch')
    if ckpt['identity']['data_manifest_sha256'] != sha256(data_root/'manifest.json'):
        raise ValueError('Checkpoint and evaluation dataset identity differ')
    if (manifest['smoke'] or ckpt['identity']['smoke']) and not allow_smoke:
        raise ValueError('Refusing benchmark publication from a smoke checkpoint/data')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows = read_index(data_root, 'test')
    model = G1CausalVAE(np.load(rot_root/'mean.npy'), np.load(rot_root/'std.npy'),
                       **ckpt['identity']['model_params']).to(device).eval()
    model.load_state_dict(ckpt['model'])
    identity = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checksum, epoch=ckpt['epoch'],
        data_manifest_sha256=sha256(data_root/'manifest.json'), split_sha256=sha256(data_root/'test.txt'),
        mjcf_sha256=sha256(data_root/'kinematic.xml'), metric_source=str(MOMASK/'evaluation/g1_metrics.py'),
        metric_sha256=sha256(MOMASK/'evaluation/g1_metrics.py'), metrics_protocol='g1_rvq_reconstruction_v1_plus_root_temporal',temporal_metric_sha256=sha256(Path(__file__).parent/'g1/temporal_metrics.py'),
        dataset_protocol='flood_g1_source_group_rot6d69_v1', rot_manifest_sha256=sha256(rot_root/'manifest.json'), sample_count=len(rows), fps=manifest['fps'],
        joint_names=JOINT_NAMES, endpoint_names=ENDPOINT_NAMES, posterior='mean',
        root_reference='GT first-frame XY only; predicted absolute yaw; integrated heading-local XY deltas; predicted height',
        padding='repeat tail to 1+4k, trim all padding before metrics',
        smoke=bool(manifest['smoke'] or ckpt['identity']['smoke']))
    if output.exists():
        if not (output/'identity.json').exists() or json.loads((output/'identity.json').read_text()) != json.loads(json.dumps(identity)):
            raise ValueError('Existing evaluation output belongs to another run')
        if (output/'summary.json').exists():
            summary = json.loads((output/'summary.json').read_text())
            if summary.get('complete'):
                print(f'Already complete: {output}', flush=True)
                return summary
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'identity.json', identity)
    write_json(output/'manifest.json', dict(**identity, complete=False))
    (output/'samples.txt').write_text('\n'.join(r['id'] for r in rows)+'\n')
    fk = G1Kinematics(data_root/'kinematic.xml')
    groups = defaultdict(list)
    for row in rows:
        groups[row['frames']].append(row)
    totals, per_sample, joint_abs, joint_sq = {}, [], np.zeros(29), np.zeros(29)
    frame_count, batch_delta, batch_checked = 0, 0., False
    started = time.time()
    with (output/'per_sample.csv').open('w') as stream, torch.inference_mode():
        writer = None
        for length, group in sorted(groups.items()):
            for start in range(0, len(group), batch_size):
                batch = group[start:start+batch_size]
                gt = np.stack([np.load(data_root/'motions'/(row['id']+'.npy')) for row in batch])
                if gt.shape != (len(batch),length,38) or not np.isfinite(gt).all():
                    raise ValueError('Invalid feature arrays')
                local=np.stack([np.load(rot_root/'motions'/(row['id']+'.npy')) for row in batch])
                anchors=torch.as_tensor(np.stack([encode_native(x)[1][0] for x in gt]),dtype=torch.float32,device=device)
                features=torch.as_tensor(local,dtype=torch.float32,device=device)
                prediction=decode_native(model.reconstruct(features,sample=False)[0],anchors).cpu().numpy()
                if len(batch)>1 and not batch_checked:
                    single=decode_native(model.reconstruct(features[:1],sample=False)[0],anchors[:1]).cpu().numpy()
                    batch_delta = float(np.max(np.abs(single[0]-prediction[0])))
                    np.testing.assert_allclose(single[0], prediction[0], atol=1e-4, rtol=1e-4)
                    batch_checked = True
                for row, truth, pred in zip(batch, gt, prediction):
                    values = metric_values(truth, pred, fk.positions(truth), fk.positions(pred), manifest['fps'], fk.endpoints)
                    values.update(temporal_values(truth,pred,manifest['fps']))
                    result = dict(sample_id=row['id'], frames=length, foot_contact_eligible=row['foot_contact_eligible'])
                    for key, value in values.items():
                        result[key] = reduce_values(key, value)
                        total, count = totals.get(key, (0.,0))
                        total += float(np.square(value).sum() if 'rmse' in key else value.sum())
                        totals[key] = (total, count+value.size)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(result))
                        writer.writeheader()
                    writer.writerow(result)
                    per_sample.append(result)
                    angles = np.rad2deg(pred[:,9:]-truth[:,9:])
                    joint_abs += np.abs(angles).sum(0)
                    joint_sq += np.square(angles).sum(0)
                    frame_count += length
                if len(per_sample)%500 < len(batch):
                    stream.flush()
                    print(f'eval {checkpoint.name} {len(per_sample)}/{len(rows)}', flush=True)
    if len(per_sample) != len(rows) or sha256(checkpoint) != checksum:
        raise RuntimeError('Incomplete evaluation or checkpoint changed')
    summary = dict(complete=True, smoke=identity['smoke'], samples=len(rows), frames=frame_count,
                   epoch=ckpt['epoch'], seconds=time.time()-started, metrics={},
                   batch_single_checked=batch_checked, batch_single_max_abs=batch_delta)
    for key,(total,count) in totals.items():
        pooled = np.sqrt(total/count) if 'rmse' in key else total/count
        summary['metrics'][key] = dict(action_mean=float(np.mean([r[key] for r in per_sample])),
            action_p95=float(np.percentile([r[key] for r in per_sample],95)), pooled=float(pooled), observations=count)
    # Terrain subset is also reported, never removed from the main test score.
    summary['contact_subsets'] = {}
    for eligible in (True,False):
        subset = [r for r in per_sample if r['foot_contact_eligible']==eligible]
        if subset:
            summary['contact_subsets'][str(eligible)] = dict(samples=len(subset), **{
                key:float(np.mean([r[key] for r in subset])) for key in ('global_mpjpe_mm','root_relative_mpjpe_mm')})
    with (output/'per_joint.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['joint','mae_deg','rmse_deg'])
        writer.writerows(zip(JOINT_NAMES, joint_abs/frame_count, np.sqrt(joint_sq/frame_count)))
    write_json(output/'summary.json', summary)
    write_json(output/'manifest.json', dict(**identity, complete=True))
    print(json.dumps(dict(output=str(output), **summary)), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--rot-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--allow-smoke', action='store_true')
    args = p.parse_args()
    evaluate(args.data_root,args.rot_root,args.checkpoint,args.output,args.device,args.batch_size,args.allow_smoke)


if __name__=='__main__':
    main()
