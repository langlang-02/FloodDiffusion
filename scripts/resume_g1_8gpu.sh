#!/usr/bin/env bash
set -euo pipefail
cd /data/langlang/Project/FloodDiffusion
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2
exec /data/langlang/envs/gentrack-momask/bin/python scripts/train_with_wandb.py \
  --output outputs/g1_causal_textop_500ep -- \
  /data/langlang/envs/gentrack-momask/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 train_g1_vae_ddp.py \
  --data-root outputs/bones_g1_native38 \
  --output outputs/g1_causal_textop_500ep \
  --resume outputs/g1_causal_textop_500ep/latest.pt \
  --workers 2 --val-batch-size 32
