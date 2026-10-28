#!/usr/bin/env python3
"""Train FloodDiffusion causal VAE on G1; auto-evaluate best and final checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from g1.common import ROOT, MOMASK, sha256, write_json
from g1.data import MotionDataset, collate_motions
from g1.kinematics import TorchG1FK
from g1.model import G1CausalVAE, TextOpLoss, DEFAULT_WEIGHTS


def load_data_identity(root):
    root = Path(root)
    m = json.loads((root/'manifest.json').read_text())
    if not m.get('complete') or m['format'] != 'flood_g1_native38_v1' or m['fps'] != 20:
        raise ValueError('Require complete native38 20Hz prepared data')
    for name, key in [('mean.npy','mean_sha256'), ('std.npy','std_sha256'),
                      ('index.jsonl','index_sha256'), ('kinematic.xml','mjcf_sha256')]:
        if sha256(root/name) != m[key]:
            raise ValueError(f'Data checksum mismatch: {name}')
    for split, checksum in m['split_sha256'].items():
        if sha256(root/f'{split}.txt') != checksum:
            raise ValueError(f'Split checksum mismatch: {split}')
    groups = {}
    with (root/'index.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            previous = groups.setdefault(row['source_group'], row['split'])
            if previous != row['split']:
                raise ValueError(f'Source leakage: {row["source_group"]}')
    return m


def checkpoint_save(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def epoch_pass(model, loss_fn, loader, optimizer, device, limit=0):
    training = optimizer is not None
    model.train(training)
    totals, frames, steps = {}, 0, 0
    started = time.time()
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {key:value.to(device, non_blocking=True) for key,value in batch.items()}
            pred, mu, logvar = model.reconstruct(batch['feature'], sample=training)
            losses = loss_fn(model, pred, batch['feature'], mu, logvar, batch['length'],
                             batch['contact'], batch['eligible'])
            if not all(torch.isfinite(x) for x in losses.values()):
                raise FloatingPointError('Nonfinite loss; no checkpoint advanced')
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            n = int(batch['length'].sum())
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.) + float(value.detach()) * n
            frames += n
            steps += 1
            if steps == 1 or steps % 50 == 0 or steps == len(loader):
                print(json.dumps(dict(event='batch_progress', phase='train' if training else 'val',
                    batch=steps, batches=len(loader), mean_total=totals['total']/frames,
                    elapsed_s=time.time()-started)), flush=True)
            if limit and steps >= limit:
                break
    if not steps:
        raise ValueError('Empty data loader')
    return {k:v/frames for k,v in totals.items()}, steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--checkpoint-every', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--val-batch-size', type=int, default=32)
    p.add_argument('--window', type=int, default=65)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--z-dim', type=int, default=32)
    p.add_argument('--dim', type=int, default=160)
    p.add_argument('--dec-dim', type=int, default=512)
    p.add_argument('--resume', type=Path)
    p.add_argument('--allow-smoke', action='store_true')
    p.add_argument('--max-batches', type=int, default=0, help='Smoke validation only')
    args = p.parse_args()
    if min(args.epochs,args.checkpoint_every,args.batch_size,args.val_batch_size) < 1 or args.window < 3:
        p.error('Positive epochs/checkpoint/batch sizes and window>=3 required')
    if args.max_batches and not args.allow_smoke:
        p.error('max-batches is allowed only for smoke tests')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this session. Production training was not started.')
    manifest = load_data_identity(args.data_root)
    if manifest.get('smoke') and not args.allow_smoke:
        raise ValueError('Refusing production training on smoke data')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    mean, std = (np.load(args.data_root/f'{n}.npy') for n in ('mean','std'))
    if mean.shape != (38,) or std.shape != (38,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError('Invalid normalization')
    params = dict(z_dim=args.z_dim, dim=args.dim, dec_dim=args.dec_dim,
                  num_res_blocks=1, dropout=0., dim_mult=[1,1,1], temperal_downsample=[True,True])
    model = G1CausalVAE(mean, std, **params).to(args.device)
    loss_fn = TextOpLoss(TorchG1FK(args.data_root/'kinematic.xml').to(args.device), fps=manifest['fps'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(MotionDataset(args.data_root,'train',args.window),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_motions,
        generator=generator, worker_init_fn=seed_worker, pin_memory=args.device.startswith('cuda'))
    val_loader = DataLoader(MotionDataset(args.data_root,'val',args.window),
        batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_motions,
        worker_init_fn=seed_worker, pin_memory=args.device.startswith('cuda'))
    identity = dict(data_manifest_sha256=sha256(args.data_root/'manifest.json'), model_params=params,
        weights=DEFAULT_WEIGHTS, epochs=args.epochs, window=args.window, learning_rate=args.lr,
        batch_size=args.batch_size, seed=args.seed, checkpoint_every=args.checkpoint_every,
        model_source_sha256=sha256(ROOT/'models/tools/wan_vae_1d.py'),
        adapter_sha256=sha256(ROOT/'g1/model.py'), fk_sha256=sha256(ROOT/'g1/kinematics.py'),
        trainer_sha256=sha256(Path(__file__)), dataset_source_sha256=sha256(ROOT/'g1/data.py'),
        metric_sha256=sha256(MOMASK/'evaluation/g1_metrics.py'),
        smoke=bool(args.allow_smoke or args.max_batches or manifest.get('smoke')),
        precision='float32, TF32 off', posterior_eval='mean',
        validation='all valid frames; deterministic posterior mean; frame-weighted loss')
    start_epoch, best, global_step = 0, math.inf, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ckpt['identity'] != identity:
            raise ValueError('Resume identity differs from data/model/training configuration')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, best, global_step = ckpt['epoch'], ckpt['best_val'], ckpt['step']
        torch.set_rng_state(ckpt['torch_rng'])
        if torch.cuda.is_available() and ckpt['cuda_rng']:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        np.random.set_state(ckpt['numpy_rng'])
        random.setstate(ckpt['python_rng'])
        generator.set_state(ckpt['loader_rng'])
        if not args.output.exists() or not (args.output/'best.pt').exists():
            raise ValueError('Resume in original run directory with its best.pt')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'identity.json', identity)
    write_json(args.output/'arguments.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    started = time.time()
    write_json(args.output/'status.json', dict(state='training', completed_epochs=start_epoch, target_epochs=args.epochs,
                                             pid=os.getpid(), smoke=identity['smoke']))
    for epoch in range(start_epoch, args.epochs):
        # Epoch-defined schedule is reproducible across restarts.
        lr = args.lr * (.01 + .99 * .5 * (1+math.cos(math.pi*epoch/args.epochs)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        print(json.dumps(dict(event='epoch_start', epoch=epoch+1, target_epochs=args.epochs, lr=lr)), flush=True)
        train, steps = epoch_pass(model, loss_fn, train_loader, optimizer, args.device, args.max_batches)
        val, _ = epoch_pass(model, loss_fn, val_loader, None, args.device, args.max_batches)
        global_step += steps
        improved = val['total'] < best
        best = min(best, val['total'])
        record = dict(epoch=epoch+1, step=global_step, lr=lr, train=train, val=val, seconds=time.time()-started)
        with (args.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1,
            best_val=best, step=global_step, identity=identity, torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(), loader_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        checkpoint_save(args.output/'latest.pt', checkpoint)
        if improved:
            checkpoint_save(args.output/'best.pt', checkpoint)
        if (epoch+1) % args.checkpoint_every == 0 or epoch+1 == args.epochs:
            checkpoint_save(args.output/f'epoch_{epoch+1:04d}.pt', checkpoint)
        write_json(args.output/'status.json', dict(state='training', completed_epochs=epoch+1,
            target_epochs=args.epochs, pid=os.getpid(), smoke=identity['smoke']))
    del optimizer, train_loader, val_loader, loss_fn, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_json(args.output/'status.json', dict(state='evaluating', completed_epochs=args.epochs,
                                             target_epochs=args.epochs, pid=os.getpid(), smoke=identity['smoke']))
    from eval_g1_vae import evaluate
    for name in ('best','latest'):
        evaluate(args.data_root, args.output/f'{name}.pt', args.output/f'eval_{name}',
                 args.device, args.val_batch_size, allow_smoke=args.allow_smoke)
    write_json(args.output/'status.json', dict(state='complete', completed_epochs=args.epochs,
        target_epochs=args.epochs, smoke=identity['smoke'], evaluation_complete=True))


if __name__ == '__main__':
    main()
