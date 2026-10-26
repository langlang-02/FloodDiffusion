"""Shared, explicit data contracts for G1 Causal-VAE experiments."""
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_view(root, config, allow_partial=False):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if not manifest['complete'] or (manifest['partial'] and not allow_partial):
        raise ValueError('Training requires a complete, nonpartial packed bank')
    info = manifest['source_info']
    for key, field in [('bank_format', 'format'), ('fps', 'target_fps'),
                       ('velocity_unit', 'velocity_unit'), ('velocity_frame', 'velocity_frame')]:
        if config[key] != info[field]:
            raise ValueError(f'Dataset/representation mismatch: {key}')
    indices = []
    for field in config['fields']:
        start, end = info['feature_layout'][field]
        indices.extend(range(start, end))
    if len(indices) != config['model']['input_dim']:
        raise ValueError('Representation dimension mismatch')
    signature = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
    return indices, signature


class MotionDataset(Dataset):
    def __init__(self, root, split, indices, window=None):
        self.root, self.split, self.indices = Path(root), split, indices
        self.window = window
        self.names = (self.root / f'{split}.txt').read_text().splitlines()
        self.offsets = np.load(self.root / f'{split}_offsets.npy')
        self.features = np.load(self.root / f'{split}_features.npy', mmap_mode='r')
        self.anchors = np.load(self.root / f'{split}_anchors.npy', mmap_mode='r')
        if len(self.offsets) != len(self.names) + 1 or self.offsets[-1] != len(self.features):
            raise ValueError('Packed offset mismatch')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        start, end = self.offsets[index:index + 2]
        if self.window and end - start > self.window:
            start += random.randint(0, int(end - start - self.window))
            end = start + self.window
        bank = np.array(self.features[start:end], copy=True)
        return {'feature': bank[:, self.indices].copy(), 'bank': bank,
                'anchors': np.array(self.anchors[start:end], copy=True), 'name': self.names[index]}


def collate(items):
    lengths = [len(x['feature']) for x in items]
    # Wan's causal codec maps 4k+1 frames to k+1 latents and back.
    size = ((max(lengths) - 1 + 3) // 4) * 4 + 1
    features = np.stack([np.pad(x['feature'], ((0, size - n), (0, 0)), mode='edge')
                         for x, n in zip(items, lengths)])
    return {'feature': torch.from_numpy(features), 'feature_length': torch.tensor(lengths), 'items': items}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
