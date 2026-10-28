#!/usr/bin/env bash
set -euo pipefail
cd /data/langlang/Project/FloodDiffusion
if [[ $(id -un) != langlang ]]; then
  echo 'Run this launcher as langlang so W&B uses /home/langlang credentials.' >&2
  exit 1
fi
unset WANDB_API_KEY
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_ENTITY=langlang-02-ustc
export WANDB_PROJECT=flood-g1-vae
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2
/data/langlang/envs/gentrack-momask/bin/python scripts/train_with_wandb.py \
 --output outputs/g1_rot68_continue_ep0500_1000_bs2048_mpjpe -- \
 /data/langlang/envs/gentrack-momask/bin/python -m torch.distributed.run \
 --standalone --nproc_per_node=8 train_g1_rot68_continue.py \
 --data-root outputs/bones_g1_native38 --rot-root outputs/bones_g1_rot6d68 \
 --fixed-val outputs/rot68_fixed_val_tracker_v1 \
 --parent outputs/g1_rot6d68_textop_500ep_bs2048/latest.pt \
 --output outputs/g1_rot68_continue_ep0500_1000_bs2048_mpjpe \
 --additional-epochs 500 --batch-per-gpu 2048 --eval-every 25 \
 --warmup-steps 300 --hold-until 2000 --peak-lr 3e-5 --floor-lr 1e-6 --workers 2 "$@" \
 2>&1 | tee -a outputs/g1_rot68_continue_ep0500_1000_bs2048_mpjpe.log
