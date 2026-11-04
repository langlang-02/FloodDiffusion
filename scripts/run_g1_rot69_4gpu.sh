#!/usr/bin/env bash
set -euo pipefail
cd /data/langlang/Project/FloodDiffusion
if [[ $(id -un) != langlang ]]; then
  echo 'Run as langlang to use the intended W&B account.' >&2
  exit 1
fi
unset WANDB_API_KEY
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_ENTITY=langlang-02-ustc
export WANDB_PROJECT=flood-g1-vae
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=2
/data/langlang/envs/gentrack-momask/bin/python scripts/train_with_wandb.py \
 --output outputs/g1_rot69_absyaw_500ep_bs2048_mpjpe -- \
 /data/langlang/envs/gentrack-momask/bin/python -m torch.distributed.run \
 --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29670 \
 train_g1_rot6d69.py \
 --data-root outputs/bones_g1_native38 --rot-root outputs/bones_g1_rot6d69 \
 --fixed-val outputs/rot68_fixed_val_tracker_v1 \
 --output outputs/g1_rot69_absyaw_500ep_bs2048_mpjpe \
 --epochs 500 --batch-per-gpu 2048 --checkpoint-every 25 --lr 1e-4 --workers 2 "$@" \
 2>&1 | tee -a outputs/g1_rot69_absyaw_500ep_bs2048_mpjpe.log
