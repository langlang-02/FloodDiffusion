#!/usr/bin/env bash
set -euo pipefail
cd /data/langlang/Project/FloodDiffusion
[[ $(id -un) == langlang ]] || { echo 'Run as langlang'; exit 1; }
unset WANDB_API_KEY
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_ENTITY=langlang-02-ustc WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
/data/langlang/envs/gentrack-momask/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
 train_g1_ldf.py --output outputs/g1_ldf_rot69_root_temporal_clip_500ep --epochs 500 --batch-per-gpu 256 --eval-every 0 "$@" \
 2>&1 | tee -a outputs/g1_ldf_rot69_root_temporal_clip_500ep.log
