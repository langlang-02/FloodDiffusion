"""Verify transferred packed-bank hashes and source/mirror split disjointness."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['complete'] and not manifest['partial'], 'Incomplete/partial data'
    for name, expected in manifest['sha256'].items():
        h = hashlib.sha256()
        with (root / name).open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        assert h.hexdigest() == expected, f'Checksum mismatch: {name}'
    groups = {}
    for split in ('train', 'val', 'test'):
        names = (root / f'{split}.txt').read_text().splitlines()
        groups[split] = {n.rsplit('_seg', 1)[0].removesuffix('_M') for n in names}
        expected = manifest['source_info']['split_sha256'][split]
        assert hashlib.sha256((root / f'{split}.txt').read_bytes()).hexdigest() == expected
        offsets = np.load(root / f'{split}_offsets.npy')
        assert len(offsets) == len(names) + 1 and (np.diff(offsets) >= 3).all()
        features = np.load(root / f'{split}_features.npy', mmap_mode='r')
        assert features.shape == (offsets[-1], 72)
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    print(json.dumps({'verified': True, 'splits': manifest['splits'], 'source_mirror_leakage': 0}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root')
    verify(p.parse_args().root)
