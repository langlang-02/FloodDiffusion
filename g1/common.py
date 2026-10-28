"""Shared identities and canonical MoMask G1 conversion/metric imports."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
MOMASK = Path(os.environ.get('MOMASK_ROOT', ROOT.parent / 'momask-codes')).resolve()
if not (MOMASK / 'evaluation/g1_metrics.py').is_file():
    raise FileNotFoundError(f'MoMask G1 evaluator missing: {MOMASK}')
sys.path.append(str(MOMASK))
sys.path.append(str(MOMASK / 'dataset/bones-seed_prepare'))
from g1_io import (G1_FEATURE_DIM, G1_FEATURE_COLUMNS, csv_values_to_model_features,
                   read_g1_csv, resample_features, rotation_6d_to_matrix,
                   rotation_matrix_to_6d, extract_root_yaw, yaw_to_matrix)
from evaluation.g1_metrics import G1Kinematics, JOINT_NAMES, ENDPOINT_NAMES, metric_values, reduce_values


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def canonicalize(native):
    """Use only first-frame XY/yaw; keep absolute height and all subsequent motion."""
    out = native.copy()
    rotations = rotation_6d_to_matrix(native[:, 3:9])
    inverse = yaw_to_matrix(-extract_root_yaw(rotations[:1])[0])
    origin = native[0, :3].copy()
    origin[2] = 0
    out[:, :3] = (native[:, :3] - origin) @ inverse.T
    out[:, 3:9] = rotation_matrix_to_6d(inverse @ rotations)
    return out
