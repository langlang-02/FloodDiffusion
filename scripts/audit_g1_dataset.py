#!/usr/bin/env python3
"""Audit every prepared window and recompute training-only normalization."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from g1.common import (G1Kinematics, canonicalize, csv_values_to_model_features,
                       read_g1_csv, resample_features, sha256, write_json)
from train_g1_vae import load_data_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--raw-checks',type=int,default=32)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    root=args.data_root
    manifest=load_data_identity(root)
    with (root/'index.jsonl').open() as stream:
        rows=[json.loads(line) for line in stream]
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate window IDs')
    expected=set(ids)
    if {p.stem for p in (root/'motions').glob('*.npy')}!=expected:
        raise ValueError('Motion file inventory differs from index')
    if {p.stem for p in (root/'contacts').glob('*.npy')}!=expected:
        raise ValueError('Contact file inventory differs from index')
    split_ids={s:set((root/f'{s}.txt').read_text().splitlines()) for s in ('train','val','test')}
    if set.union(*split_ids.values())!=expected or sum(map(len,split_ids.values()))!=len(expected):
        raise ValueError('Splits must partition the full index')
    with (root/'contact_audit.csv').open() as stream:
        policy={r['filename']:r for r in csv.DictReader(stream)}
    count, frames, eligible_frames=Counter(),Counter(),Counter()
    contacts, eligible_contacts=Counter(),Counter()
    total=np.zeros(38,np.float64)
    total_sq=np.zeros(38,np.float64)
    source_groups={}
    for i,row in enumerate(rows):
        motion=np.load(root/'motions'/(row['id']+'.npy'))
        contact=np.load(root/'contacts'/(row['id']+'.npy'))
        if motion.shape!=(row['frames'],38) or motion.dtype!=np.float32 or not np.isfinite(motion).all():
            raise ValueError(f'Invalid motion {row["id"]}')
        if contact.shape!=(len(motion),2) or contact.dtype!=np.bool_:
            raise ValueError(f'Invalid contact mask {row["id"]}')
        if not manifest['min_frames']<=len(motion)<=manifest['max_frames']:
            raise ValueError('Clip length out of bounds')
        if np.max(np.abs(motion[0,:2]))>1e-6:
            raise ValueError('Canonical XY origin mismatch')
        source=Path(row['source']).stem
        decision=policy[source]
        if row['foot_contact_eligible']!=(decision['foot_contact_eligible']=='1') or row['split']!=decision['split']:
            raise ValueError('Index/contact-audit decision mismatch')
        split=row['split']
        if row['id'] not in split_ids[split]:
            raise ValueError('Wrong split membership')
        source_groups.setdefault(row['source_group'],set()).add(split)
        count[split]+=1
        frames[split]+=len(motion)
        contacts[split]+=int(contact.sum())
        if row['foot_contact_eligible']:
            eligible_frames[split]+=len(motion)
            eligible_contacts[split]+=int(contact.sum())
        if split=='train':
            total+=motion.sum(0,dtype=np.float64)
            total_sq+=np.square(motion.astype(np.float64)).sum(0)
        if (i+1)%10000==0:
            print(f'audited {i+1}/{len(rows)} windows',flush=True)
    if any(len(s)!=1 for s in source_groups.values()):
        raise ValueError('Source group crosses splits')
    mean=total/frames['train']
    std=np.sqrt(np.maximum(total_sq/frames['train']-mean**2,0)).clip(.01)
    stored_mean,stored_std=(np.load(root/f'{n}.npy') for n in ('mean','std'))
    np.testing.assert_allclose(mean.astype(np.float32),stored_mean,atol=0,rtol=0)
    np.testing.assert_allclose(std.astype(np.float32),stored_std,atol=0,rtol=0)
    if frames['train']!=manifest['train_stat_frames'] or dict(count)!=manifest['windows']:
        raise ValueError('Manifest counts mismatch')
    # Deterministic source/representation checks include all splits.
    fk=G1Kinematics(root/'kinematic.xml')
    check_indices=np.random.default_rng(3407).choice(len(rows),min(args.raw_checks,len(rows)),replace=False)
    checks=[]
    for i in sorted(check_indices):
        row=rows[int(i)]
        raw=Path(manifest['source_root'])/row['source']
        if sha256(raw)!=row['source_sha256']:
            raise ValueError(f'Original CSV changed: {raw}')
        native=resample_features(csv_values_to_model_features(read_g1_csv(raw)),120.,manifest['fps'])
        start=row['start_frame'];stop=start+row['frames']
        expected_motion=canonicalize(native[start:stop].astype(np.float32))
        actual=np.load(root/'motions'/(row['id']+'.npy'))
        np.testing.assert_array_equal(actual,expected_motion)
        feet=fk.positions(native.astype(np.float32))[:,fk.endpoints[2:]]
        expected_contact=np.zeros((len(feet),2),dtype=bool)
        expected_contact[1:]=((np.diff(feet,axis=0)**2).sum(-1)<.002)&(feet[1:,:,2]<.08)
        np.testing.assert_array_equal(np.load(root/'contacts'/(row['id']+'.npy')),expected_contact[start:stop])
        checks.append(row['id'])
    result=dict(complete=True,data_manifest_sha256=sha256(root/'manifest.json'),
        windows=dict(count),frames=dict(frames),contact_eligible_frames=dict(eligible_frames),
        contact_labels=dict(contacts),eligible_contact_labels=dict(eligible_contacts),
        source_groups=len(source_groups),source_group_leakage=0,train_stats_exact_match=True,
        original_csv_checks=checks,smoke=manifest['smoke'])
    output=args.output or root/'audit.json'
    write_json(output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
