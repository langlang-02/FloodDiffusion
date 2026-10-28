"""BONES data loading, source-group splits, and auditable contact exclusions."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from .common import canonicalize

# Deliberately avoid bare 'step': ordinary flat-ground stepping stays eligible.
TERRAIN_RULES = {
    'stairs': r'\b(stairs?|staircases?|stairways?|upstairs|downstairs|stairwells?|escalators?)\b',
    'incline': r'\b(ramps?|slopes?|inclines?|uphill|downhill|hills?|hillsides?|mountains?)\b',
    'uneven': r'\b(uneven|rocky|rough terrain|stepping stones?|curbs?|kerbs?|ledges?|platforms?)\b',
    'climbing': r'\b(ladders?|climb(?:s|ed|ing)?|obstacles?|hurdles?)\b',
    'elevated_transition': r'\b(?:step(?:s|ped|ping)?|jump(?:s|ed|ing)?|hop(?:s|ped|ping)?|com(?:e|es|ing)|g(?:et|ets|etting))\s+(?:up|down|onto|off|from|form)\b[^.!?\n]{0,60}\b(?:box|bench|chair|table|log)\b',
    'elevated_support': r'\b(?:stand(?:s|ing)?|stood|walk(?:s|ed|ing)?|step(?:s|ped|ping)?)\s+on\s+(?:(?:a|the)\s+)?(?:chair|table|bench|box|log)\b',
    'terrain_cn': r'楼梯|台阶|斜坡|坡道|爬梯|崎岖|高台',
}
DESCRIPTION_COLUMNS = ('content_natural_desc_1', 'content_natural_desc_2', 'content_natural_desc_3',
                       'content_natural_desc_4', 'content_technical_description',
                       'content_short_description', 'content_short_description_2')


def contact_policy(row, temporal_text=''):
    parts = [str(row.get(k, '') or '') for k in DESCRIPTION_COLUMNS]
    known = any(p.strip().lower() not in ('', 'nan', 'none') for p in parts)
    parts += [str(row.get(k, '') or '') for k in ('filename', 'category', 'content_props', 'content_name')]
    text = ' '.join(parts + [temporal_text]).replace('_', ' ').lower()
    reasons = [name for name, pattern in TERRAIN_RULES.items() if re.search(pattern, text)]
    if not known and not temporal_text.strip():
        reasons.append('missing_description')
    return not reasons, reasons


def source_group(row):
    # Original take/performer/date keeps mirrored and styled variants together.
    name = str(row.get('take_org_name') or row.get('take_name') or row['filename'])
    name = re.sub(r'_M$', '', name, flags=re.IGNORECASE)
    return '|'.join((str(row.get('take_date', '')), str(row.get('take_actor', '')), name))


def split_for_group(group, seed=3407):
    value = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16], 16) / 2**64
    return 'test' if value < .05 else ('val' if value < .10 else 'train')


def read_index(root, split):
    root = Path(root)
    ids = (root / f'{split}.txt').read_text().splitlines()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'Empty or duplicate split {split}')
    rows = {}
    with (root / 'index.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['id'] in rows:
                raise ValueError('Duplicate index ID')
            rows[row['id']] = row
    result = [rows[n] for n in ids]
    if any(r['split'] != split for r in result):
        raise ValueError('Split/index mismatch')
    return result


class MotionDataset(Dataset):
    def __init__(self, root, split, window=65):
        self.root = Path(root)
        self.rows = read_index(root, split)
        self.window = window
        self.training = split == 'train'

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        motion = np.load(self.root / 'motions' / (row['id'] + '.npy'))
        contact = np.load(self.root / 'contacts' / (row['id'] + '.npy'))
        start = np.random.randint(len(motion)-self.window+1) if self.training and len(motion)>self.window else 0
        stop = min(start+self.window, len(motion)) if self.training else len(motion)
        motion = canonicalize(motion[start:stop])
        contact = contact[start:stop]
        length = len(motion)
        if length < self.window:
            motion = np.concatenate((motion, np.repeat(motion[-1:], self.window-length, axis=0)))
            contact = np.pad(contact, ((0, self.window-length), (0, 0)))
        return dict(feature=torch.from_numpy(motion.copy()), length=length,
                    contact=torch.from_numpy(contact.copy()), eligible=row['foot_contact_eligible'])


def collate_motions(items):
    max_length = max(int(x['length']) for x in items)
    features, contacts = [], []
    for item in items:
        length = int(item['length'])
        feature, contact = item['feature'][:length], item['contact'][:length]
        padding = max_length-length
        features.append(torch.cat((feature, feature[-1:].expand(padding, -1))))
        contacts.append(torch.cat((contact, torch.zeros(padding, 2, dtype=contact.dtype))))
    return dict(feature=torch.stack(features), contact=torch.stack(contacts),
                length=torch.tensor([x['length'] for x in items]),
                eligible=torch.tensor([x['eligible'] for x in items], dtype=torch.bool))
