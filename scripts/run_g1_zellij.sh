#!/usr/bin/env bash
set -euo pipefail
cd /data/langlang/Project/FloodDiffusion
bash scripts/resume_g1_8gpu.sh 2>&1 | tee -a outputs/g1_500epochs.ddp8_bs1024.log
