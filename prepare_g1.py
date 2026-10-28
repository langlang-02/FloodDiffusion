#!/usr/bin/env python3
"""Prepare raw BONES-SEED G1 CSVs, preserving original source-group splits."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from g1.common import (G1Kinematics, canonicalize, csv_values_to_model_features, read_g1_csv,
                       resample_features, sha256, write_json)
from g1.data import contact_policy, source_group, split_for_group, TERRAIN_RULES


def export_kinematic_xml(source, target):
    """Preserve body/joint/inertial transforms, omit unavailable visual mesh assets."""
    root = ET.parse(source).getroot()
    if root.findall('.//include'):
        raise ValueError('Use an expanded MJCF without includes')
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ('asset', 'geom', 'site', 'camera', 'light', 'actuator', 'sensor',
                             'contact', 'tendon', 'equality', 'keyframe'):
                parent.remove(child)
    compiler = root.find('compiler')
    if compiler is not None:
        compiler.attrib.pop('meshdir', None)
        compiler.attrib.pop('texturedir', None)
    ET.indent(root)
    ET.ElementTree(root).write(target, encoding='unicode')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=Path('/data/langlang/dataset/supermimic/raw/bones_seed'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mjcf', type=Path, required=True)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--max-frames', type=int, default=196)
    p.add_argument('--min-frames', type=int, default=40)
    p.add_argument('--limit-per-split', type=int, default=0, help='Smoke only; zero means full dataset')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--source-fps', type=float, default=120.)
    p.add_argument('--fps', type=float, default=20.)
    args = p.parse_args()
    if args.fps != 20 or args.min_frames < 3 or args.max_frames < args.min_frames or args.limit_per_split < 0:
        p.error('Require 20Hz, max >= min >= 3, limit >= 0')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = args.raw_root / 'metadata/seed_metadata_v004.parquet'
    temporal_path = args.raw_root / 'metadata/seed_metadata_v002_temporal_labels.jsonl'
    rows = pd.read_parquet(metadata).fillna('').to_dict('records')
    if len({r['filename'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source names')
    temporal = {}
    if temporal_path.exists():
        with temporal_path.open() as stream:
            for line in stream:
                item = json.loads(line)
                temporal[item['filename']] = ' '.join(str(e.get('description', '')) for e in item.get('events', []))
    manifest = dict(format='flood_g1_native38_v1', fps=args.fps, feature_dim=38,
                    metadata_sha256=sha256(metadata), source_root=str(args.raw_root.resolve()),
                    temporal_sha256=sha256(temporal_path) if temporal_path.exists() else None,
                    source_mjcf=str(args.mjcf.resolve()), source_mjcf_sha256=sha256(args.mjcf),
                    split_seed=args.seed, split_method='sha256(date|actor|original_take); 90/5/5; mirrors grouped',
                    max_frames=args.max_frames, min_frames=args.min_frames,
                    contact_policy=TERRAIN_RULES, contact_height_m=.08, contact_displacement_squared_m2=.002,
                    normalization='training split only; first-frame XY/yaw canonicalization; per-frame mean/std',
                    smoke=bool(args.limit_per_split), complete=False)
    write_json(args.output / 'manifest.json', manifest)
    selected, split_sources, excluded, reasons = [], Counter(), Counter(), Counter()
    with (args.output / 'contact_audit.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['filename','source_group','split','foot_contact_eligible','reasons'])
        writer.writeheader()
        for row in rows:
            group = source_group(row)
            split = split_for_group(group, args.seed)
            eligible, why = contact_policy(row, temporal.get(row['filename'], ''))
            writer.writerow(dict(filename=row['filename'], source_group=group, split=split,
                                 foot_contact_eligible=int(eligible), reasons=';'.join(why)))
            excluded[split] += not eligible
            reasons.update(why)
            split_sources[split] += 1
            row.update(group=group, split=split, eligible=eligible, reasons=why)
            selected.append(row)
    manifest.update(source_count=len(rows), source_split_counts=dict(split_sources),
                    excluded_contact_sources=dict(excluded), exclusion_reason_counts=dict(reasons))
    write_json(args.output / 'manifest.json', manifest)
    if args.audit_only:
        write_json(args.output / 'audit_summary.json', dict(source_count=len(rows), split_counts=dict(split_sources),
                   excluded_contact_sources=dict(excluded), reason_counts=dict(reasons)))
        print(json.dumps(manifest), flush=True)
        return
    export_kinematic_xml(args.mjcf, args.output / 'kinematic.xml')
    manifest['mjcf_sha256'] = sha256(args.output / 'kinematic.xml')
    fk = G1Kinematics(args.output / 'kinematic.xml')
    foot_indices = fk.endpoints[2:]
    for directory in ('motions', 'contacts'):
        (args.output / directory).mkdir()
    ids = {s: [] for s in ('train','val','test')}
    used = Counter()
    total = np.zeros(38, np.float64)
    total_sq = np.zeros(38, np.float64)
    frame_count = 0
    short = 0
    started = time.time()
    with (args.output / 'index.jsonl').open('w') as stream:
        for row in selected:
            split = row['split']
            if args.limit_per_split and used[split] >= args.limit_per_split:
                continue
            path = (args.raw_root / row['move_g1_path']).resolve()
            if not path.is_relative_to(args.raw_root.resolve()):
                raise ValueError('Source path escapes raw root')
            native = resample_features(csv_values_to_model_features(read_g1_csv(path)), args.source_fps, args.fps)
            if not np.isfinite(native).all():
                raise ValueError(f'Nonfinite source {path}')
            if len(native) < args.min_frames:
                short += 1
                continue
            native = native.astype(np.float32)
            points = fk.positions(native)[:, foot_indices]
            contact = np.zeros((len(native), 2), dtype=bool)
            contact[1:] = ((np.diff(points, axis=0)**2).sum(-1)<.002) & (points[1:, :, 2]<.08)
            source_hash = sha256(path)
            for start in range(0, len(native)-args.min_frames+1, args.max_frames):
                stop = min(start+args.max_frames, len(native))
                sample = f"{row['filename']}_win{start:06d}"
                if Path(sample).name != sample:
                    raise ValueError('Unsafe sample ID')
                motion = canonicalize(native[start:stop])
                np.save(args.output / 'motions' / (sample+'.npy'), motion)
                np.save(args.output / 'contacts' / (sample+'.npy'), contact[start:stop])
                item = dict(id=sample, split=split, source_group=row['group'], source=row['move_g1_path'],
                            source_sha256=source_hash, start_frame=start, frames=len(motion),
                            foot_contact_eligible=row['eligible'], exclusion_reasons=row['reasons'])
                stream.write(json.dumps(item)+'\n')
                ids[split].append(sample)
                if split == 'train':
                    total += motion.sum(0, dtype=np.float64)
                    total_sq += np.square(motion.astype(np.float64)).sum(0)
                    frame_count += len(motion)
            used[split] += 1
            if sum(used.values()) % 500 == 0:
                stream.flush()
                print(f'converted sources={sum(used.values())} seconds={time.time()-started:.1f}', flush=True)
    if not all(ids.values()) or not frame_count:
        raise ValueError('Preparation produced an empty split')
    mean = total/frame_count
    std = np.sqrt(np.maximum(total_sq/frame_count-mean**2, 0)).clip(.01)
    np.save(args.output / 'mean.npy', mean.astype(np.float32))
    np.save(args.output / 'std.npy', std.astype(np.float32))
    for split, names in ids.items():
        (args.output / f'{split}.txt').write_text('\n'.join(names)+'\n')
    manifest.update(complete=True, source_used=dict(used), windows={k:len(v) for k,v in ids.items()},
                    train_stat_frames=frame_count, too_short_sources=short,
                    mean_sha256=sha256(args.output/'mean.npy'), std_sha256=sha256(args.output/'std.npy'),
                    index_sha256=sha256(args.output/'index.jsonl'),
                    split_sha256={s:sha256(args.output/f'{s}.txt') for s in ids})
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
