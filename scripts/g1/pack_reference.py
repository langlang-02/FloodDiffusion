"""Pack an existing audited MoMask bank without changing splits or statistics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    info = json.loads((args.source / 'dataset_info.json').read_text())
    assert info['complete'] and not info.get('partial')
    manifest = {'complete': False, 'partial': bool(args.limit), 'source_info': info, 'splits': {}}
    lengths_manifest = {}
    with (args.source / 'metadata/windows.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            lengths_manifest[record['sample_id']] = record['frames']
    for split in ('train', 'val', 'test'):
        names = (args.source / f'{split}.txt').read_text().splitlines()
        if args.limit:
            names = names[:args.limit]
        # Inspect headers only before allocating contiguous arrays.
        source_dir = args.source / 'new_joint_vecs'
        if not (source_dir / f'{names[0]}.npy').exists():
            source_dir = args.source / 'merged_joint_vecs'
        lengths = [lengths_manifest[name] for name in names]
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        features = np.lib.format.open_memmap(args.output / f'{split}_features.npy', mode='w+', dtype='float32', shape=(int(offsets[-1]), 72))
        anchors = np.lib.format.open_memmap(args.output / f'{split}_anchors.npy', mode='w+', dtype='float64', shape=(int(offsets[-1]), 4))
        def load_sample(name):
            return (np.load(source_dir / f'{name}.npy')[:, :72],
                    np.load(args.source / 'anchors' / f'{name}.npy'))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for block in range(0, len(names), 512):
                for i, (x, a) in enumerate(pool.map(load_sample, names[block:block + 512]), start=block):
                    if x.shape != (lengths[i], 72) or a.shape != (lengths[i], 4) or not np.isfinite(x).all() or not np.isfinite(a).all():
                        raise ValueError(names[i])
                    features[offsets[i]:offsets[i + 1]] = x
                    anchors[offsets[i]:offsets[i + 1]] = a
                if block % 10240 == 0:
                    print(split, min(block + 512, len(names)), '/', len(names), flush=True)
        features.flush(); anchors.flush()
        np.save(args.output / f'{split}_offsets.npy', offsets)
        (args.output / f'{split}.txt').write_text('\n'.join(names) + '\n')
        manifest['splits'][split] = {'samples': len(names), 'frames': int(offsets[-1])}
        del features, anchors
    for name in ('Mean.npy', 'Std.npy', 'dataset_info.json'):
        shutil.copy2(args.source / name, args.output / name)
    manifest['sha256'] = {}
    for path in sorted(args.output.iterdir()):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        manifest['sha256'][path.name] = h.hexdigest()
    manifest['complete'] = True
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    main()
