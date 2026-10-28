#!/usr/bin/env bash
set -euo pipefail
cd /data/langlang/Project/FloodDiffusion
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2
/data/langlang/envs/gentrack-momask/bin/python scripts/train_with_wandb.py \
 --output outputs/g1_rot6d68_textop_500ep_bs2048 -- \
 /data/langlang/envs/gentrack-momask/bin/python -m torch.distributed.run \
 --standalone --nproc_per_node=8 train_g1_rot6d68.py \
 --data-root outputs/bones_g1_native38 --rot-root outputs/bones_g1_rot6d68 \
 --output outputs/g1_rot6d68_textop_500ep_bs2048 \
 --batch-per-gpu 2048 --epochs 500 --checkpoint-every 25 --workers 2 "$@" \
 2>&1 | tee -a outputs/g1_rot6d68_textop_500ep_bs2048.log
