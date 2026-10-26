"""DDP G1 training using FloodDiffusion's unmodified Causal-VAE network/loss."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from models.vae_wan_1d import VAEWanModel
from scripts.g1.common import MotionDataset, atomic_json, collate, read_config, resolve_view, seed_all


def save_checkpoint(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


@torch.no_grad()
def validate(model, loader, device, enabled):
    model.eval()
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    for batch in loader:
        x = batch['feature'].to(device)
        lengths = batch['feature_length'].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled):
            y = model.decode(model.encode(x))
        n = min(x.shape[1], y.shape[1])
        mask = torch.arange(n, device=device)[None, :] < lengths[:, None]
        error = torch.nn.functional.smooth_l1_loss(
            (y[:, :n].float() - model.mean) / model.std,
            (x[:, :n] - model.mean) / model.std, reduction='none')
        sums[0] += (error * mask[..., None]).sum()
        sums[1] += mask.sum() * x.shape[-1]
        sums[2] += (error[..., 29:58] * mask[..., None]).sum()
        sums[3] += mask.sum() * 29
    if dist.is_initialized():
        dist.all_reduce(sums)
    recon, velocity = (sums[0] / sums[1]).item(), (sums[2] / sums[3]).item()
    return {'recons': recon, 'velocity': velocity, 'selection_loss': recon + 0.5 * velocity}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--smoke', action='store_true', help='Tiny synthetic-independent run; never a production result')
    p.add_argument('--smoke-full-model', action='store_true')
    args = p.parse_args()
    args.smoke = args.smoke or args.smoke_full_model
    cfg = read_config(args.config)
    indices, signature = resolve_view(args.data, cfg, allow_partial=args.smoke)
    rank, world, local = int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1)), int(os.getenv('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local}' if args.device == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    seed_all(cfg['seed'])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_cfg = cfg['training']
    if args.smoke:
        cfg = copy.deepcopy(cfg)
        if not args.smoke_full_model:
            cfg['model'].update(dim=16, dec_dim=16, z_dim=4)
        train_cfg = dict(train_cfg, max_steps=2, batch_size_per_gpu=2, validate_every=2,
                         checkpoint_every=2, log_every=1, workers=0, eval_batch_size=2)
        if args.smoke_full_model:
            train_cfg['batch_size_per_gpu'] = cfg['training']['batch_size_per_gpu']
    if world != 4 and not args.smoke:
        raise ValueError('Production training requires exactly four DDP ranks')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.output / 'last.pt').exists():
        raise ValueError('Existing run: explicitly resume or use a new output directory')
    mean = np.load(Path(args.data) / 'Mean.npy')[indices]
    std = np.load(Path(args.data) / 'Std.npy')[indices]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid training statistics')
    model = VAEWanModel(**cfg['model']).to(device)
    model.mean.copy_(torch.from_numpy(mean).to(device))
    model.std.copy_(torch.from_numpy(std).to(device))
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'],
                                 betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
    step, epoch, offset, best = 0, 0, 0, float('inf')
    restored_rng = None
    if args.resume:
        ckpt = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        if ckpt['config'] != cfg or ckpt['dataset_signature'] != signature or ckpt['world_size'] != world:
            raise ValueError('Resume configuration, data, or world size mismatch')
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step, epoch, offset, best = ckpt['step'], ckpt['epoch'], ckpt['offset'], ckpt['best']
        restored_rng = ckpt['rng'][rank]
    wrapped = DDP(model, device_ids=[local] if device.type == 'cuda' else None) if world > 1 else model
    # Initialize identical model/EMA weights, then use independent sampling RNGs.
    seed_all(cfg['seed'] + rank)
    dataset = MotionDataset(args.data, 'train', indices, window=train_cfg['window'])
    val_dataset = MotionDataset(args.data, 'val', indices)
    if args.smoke:
        count = world * train_cfg['batch_size_per_gpu'] * 2
        dataset = Subset(dataset, [i % min(len(dataset), 16) for i in range(count)])
        val_dataset = Subset(val_dataset, list(range(min(len(val_dataset), 8))))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=cfg['seed'], drop_last=True)
    loader = DataLoader(dataset, batch_size=train_cfg['batch_size_per_gpu'], sampler=sampler,
                        collate_fn=collate, num_workers=train_cfg['workers'], pin_memory=device.type == 'cuda',
                        drop_last=True, persistent_workers=train_cfg['workers'] > 0)
    if not len(loader):
        raise ValueError('Training split is too small for this batch/world size')
    val_loader = DataLoader(Subset(val_dataset, list(range(rank, len(val_dataset), world))),
                           batch_size=train_cfg['eval_batch_size'], collate_fn=collate,
                           num_workers=train_cfg['workers'], pin_memory=device.type == 'cuda')
    enabled = device.type == 'cuda' and train_cfg['precision'] == 'bf16'
    if restored_rng:
        random.setstate(restored_rng['python']); np.random.set_state(restored_rng['numpy'])
        torch.set_rng_state(restored_rng['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(restored_rng['cuda'], device)
    if rank == 0:
        atomic_json(args.output / 'run.json', {'config': cfg, 'dataset_signature': signature,
            'world_size': world, 'global_batch_size': train_cfg['batch_size_per_gpu'] * world,
            'smoke': args.smoke, 'torch': torch.__version__, 'data': str(Path(args.data).resolve()),
            'loss': 'Unmodified VAEWanModel.forward; normalized SmoothL1 + 0.5 joint-delta SmoothL1 + 1e-5 KL',
            'resume_note': 'Optimizer/model/RNG restored; worker crop streams may differ after resume.'})
    started = time.monotonic()
    while step < train_cfg['max_steps']:
        sampler.set_epoch(epoch)
        wrapped.train()
        for batch_index, batch in enumerate(loader):
            if batch_index < offset:
                continue
            lr = train_cfg['learning_rate'] * min(1.0, (step + 1) / train_cfg['warmup_steps'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            payload = {key: batch[key].to(device, non_blocking=True) for key in ('feature', 'feature_length')}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled):
                losses = wrapped(payload)
                loss = losses['total']
            finite = torch.isfinite(loss).int()
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - train_cfg['ema_decay'])
            step += 1
            offset = batch_index + 1
            if step % train_cfg['log_every'] == 0:
                values = torch.stack([losses[k].detach().float() for k in ('total', 'recons', 'velocity', 'kl')])
                if world > 1:
                    dist.all_reduce(values)
                if rank == 0:
                    record = dict(zip(('total', 'recons', 'velocity', 'kl'), (values / world).tolist()))
                    record.update(step=step, epoch=epoch, lr=lr, elapsed_seconds=time.monotonic() - started)
                    with (args.output / 'train.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
                    print(json.dumps(record), flush=True)
            final = step == train_cfg['max_steps']
            improved = False
            if step % train_cfg['validate_every'] == 0 or final:
                result = validate(ema, val_loader, device, enabled)
                improved = result['selection_loss'] < best
                best = min(best, result['selection_loss'])
                if rank == 0:
                    result.update(step=step, best=best)
                    with (args.output / 'validation.jsonl').open('a') as f:
                        f.write(json.dumps(result) + '\n')
                    print('validation', json.dumps(result), flush=True)
                wrapped.train()
            if step % train_cfg['checkpoint_every'] == 0 or final or improved:
                rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                       'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, rng)
                else:
                    states[0] = rng
                if rank == 0:
                    state = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                                 config=cfg, dataset_signature=signature, step=step, epoch=epoch,
                                 offset=offset, best=best, world_size=world, rng=states, smoke=args.smoke)
                    save_checkpoint(args.output / 'last.pt', state)
                    if improved:
                        save_checkpoint(args.output / 'best.pt', state)
            if final:
                break
        epoch += 1
        offset = 0
    if world > 1:
        dist.barrier()
    if rank == 0:
        atomic_json(args.output / 'training_complete.json', {'step': step, 'best_validation_loss': best,
                    'checkpoint': str(args.output / 'best.pt'), 'smoke': args.smoke,
                    'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0})
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
