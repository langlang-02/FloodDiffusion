#!/usr/bin/env python3
from pathlib import Path
import render_reconstruction_comparison as renderer
renderer.OUT=Path(__file__).resolve().parents[1]/'outputs/rot68_reconstruction_comparison'
renderer.KEYS=['gt','rvq','vae_best','rot68']
renderer.LABELS=['Ground truth','RVQ rot6d68 | epoch 499','VAE native38 | best epoch 149','VAE rot6d68 | best epoch 500']
if __name__=='__main__':renderer.main()
