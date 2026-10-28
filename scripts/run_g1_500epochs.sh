#!/usr/bin/env bash
# Foreground pipeline; launch in Zellij on a host exposing CUDA.
set -euo pipefail
cd "$(dirname "$0")/.."
G1_PYTHON=${G1_PYTHON:-/data/langlang/envs/gentrack-momask/bin/python}
G1_RAW_ROOT=${G1_RAW_ROOT:-/data/langlang/dataset/supermimic/raw/bones_seed}
G1_DATA_ROOT=${G1_DATA_ROOT:-$PWD/outputs/bones_g1_native38}
G1_RUN_ROOT=${G1_RUN_ROOT:-$PWD/outputs/g1_causal_textop_500ep}
G1_MJCF=${G1_MJCF:-$PWD/../MimicLite/active-adaptation/active_adaptation/assets/G1/g1_29dof_nohand/g1_29dof_nohand.xml}
G1_DEVICE=${G1_DEVICE:-cuda:0}
G1_BATCH_SIZE=${G1_BATCH_SIZE:-256}
G1_WORKERS=${G1_WORKERS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# Fail before an expensive preparation if this session cannot train.
"$G1_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: production training was not started"'
mkdir -p outputs
"$G1_PYTHON" scripts/verify_g1_sources.py --raw-root "$G1_RAW_ROOT" --output outputs/source_verification.json
if [[ ! -f "$G1_DATA_ROOT/manifest.json" ]]; then
    "$G1_PYTHON" prepare_g1.py --raw-root "$G1_RAW_ROOT" --output "$G1_DATA_ROOT" --mjcf "$G1_MJCF"
fi
"$G1_PYTHON" scripts/audit_g1_dataset.py --data-root "$G1_DATA_ROOT"
G1_RESUME_ARGS=()
if [[ -f "$G1_RUN_ROOT/latest.pt" ]]; then
    G1_RESUME_ARGS=(--resume "$G1_RUN_ROOT/latest.pt")
fi
exec "$G1_PYTHON" scripts/train_with_wandb.py --output "$G1_RUN_ROOT" -- \
    "$G1_PYTHON" train_g1_vae.py \
    --data-root "$G1_DATA_ROOT" --output "$G1_RUN_ROOT" \
    --epochs 500 --checkpoint-every 25 --window 65 \
    --batch-size "$G1_BATCH_SIZE" --workers "$G1_WORKERS" --device "$G1_DEVICE" \
    "${G1_RESUME_ARGS[@]}"
