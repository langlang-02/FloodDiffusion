"""Evaluate all held-out G1 windows in native physical space after training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.vae_wan_1d import VAEWanModel
from scripts.g1.common import MotionDataset, atomic_json, collate, resolve_view
from vendor.momask_g1.g1_io import local_features_to_model_features
from vendor.momask_g1.g1_metrics import G1Kinematics, JOINT_NAMES, ground_truth_native, metric_values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--data', required=True)
    p.add_argument('--mjcf', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--limit', type=int, default=0, help='Smoke only; reported as partial')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    indices, signature = resolve_view(args.data, cfg, allow_partial=checkpoint.get('smoke', False))
    if signature != checkpoint['dataset_signature']:
        raise ValueError('Evaluation dataset does not match the training checkpoint')
    model = VAEWanModel(**cfg['model']).to(args.device)
    model.load_state_dict(checkpoint['ema'])
    model.eval()
    dataset = MotionDataset(args.data, 'test', indices)
    if args.limit:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.limit, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)
    kinematics = G1Kinematics(args.mjcf)
    totals, counts = {}, {}
    samples, frames = 0, 0
    started = time.monotonic()
    with (args.output / 'per_sequence.csv').open('w', newline='') as out:
        writer = None
        with torch.inference_mode():
            for batch in loader:
                # Evaluation uses float32 and posterior mean, without sampling/clipping.
                x = batch['feature'].to(args.device)
                decoded = model.decode(model.encode(x)).float().cpu().numpy()
                for item, reconstruction in zip(batch['items'], decoded):
                    length = len(item['feature'])
                    if len(reconstruction) < length:
                        raise ValueError('Decoder truncated valid frames')
                    reconstruction = reconstruction[:length]
                    pred = local_features_to_model_features(reconstruction, cfg['fields'], item['anchors'][0])
                    gt = ground_truth_native(item['bank'], item['anchors'])
                    values = metric_values(gt, pred, kinematics.positions(gt), kinematics.positions(pred),
                                           cfg['fps'], kinematics.endpoints)
                    row = {'sample': item['name'], 'frames': length}
                    for key, value in values.items():
                        array = np.square(value) if 'rmse' in key else value
                        totals[key] = totals.get(key, 0.0) + float(np.sum(array, dtype=np.float64))
                        counts[key] = counts.get(key, 0) + int(array.size)
                        mean = float(np.mean(array))
                        row[key] = float(np.sqrt(mean)) if 'rmse' in key else mean
                    if writer is None:
                        writer = csv.DictWriter(out, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    if samples < 8:
                        np.savez_compressed(args.output / f'reconstruction_{samples:02d}.npz',
                            sample=item['name'], reference_features=item['feature'], reconstruction_features=reconstruction,
                            reference_native=gt, reconstruction_native=pred, initial_anchor=item['anchors'][0])
                    samples += 1
                    frames += length
                if samples % 256 < args.batch_size:
                    print(f'Evaluated {samples}/{len(dataset)} windows', flush=True)
    metrics = {k: float(np.sqrt(totals[k] / counts[k])) if 'rmse' in k else totals[k] / counts[k] for k in totals}
    report = {'complete': True, 'partial': bool(args.limit), 'smoke_checkpoint': checkpoint.get('smoke', False),
              'representation': cfg['id'], 'checkpoint': str(args.checkpoint.resolve()), 'step': checkpoint['step'],
              'weights': 'EMA', 'latent_decode': 'posterior_mean_float32', 'dataset_signature': signature,
              'split': 'test', 'samples': samples, 'frames': frames, 'metrics': metrics,
              'joint_names': JOINT_NAMES, 'mjcf': str(args.mjcf.resolve()),
              'mjcf_sha256': hashlib.sha256(args.mjcf.read_bytes()).hexdigest(),
              'initial_state': 'ground-truth first-frame XYZ/yaw only; integrate predicted increments thereafter',
              'ground_truth': 'full ground-truth anchors and bank joint angles/tilt',
              'mpjpe_definition': '29 named hinge joint anchors; frame/joint weighted Euclidean error in mm',
              'root_relative_definition': 'subtract each trajectory root translation; no yaw/Procrustes alignment',
              'clamp_joint_limits': False, 'elapsed_seconds': time.monotonic() - started}
    atomic_json(args.output / 'metrics.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
