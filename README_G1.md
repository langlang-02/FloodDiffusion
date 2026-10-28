# BONES-SEED G1 causal VAE reconstruction

This entry point trains the repository's **Wan 1D causal VAE** on native 29-DoF
Unitree G1 motion. It does not train the diffusion model or the HumanML3D
text-motion evaluator. Raw data is read from the existing, checksum-verified
BOS download at `/data/langlang/dataset/supermimic/raw/bones_seed`.

## Historical native38 run: 500 epochs

From this repository, on a CUDA-enabled host:

```bash
bash scripts/run_g1_500epochs.sh
```

The script verifies the original six downloaded files, prepares the data if
needed, audits every prepared window and its training-only statistics, trains **500 epochs**, and automatically evaluates `best.pt` and
`latest.pt` on the full held-out test split. It resumes an existing `latest.pt`
in the same run directory. CUDA preflight failure means training did not start.
Install the standalone dependencies with `pip install -r requirements-g1.txt`
(Python 3.11 tested), and keep the sibling `momask-codes` checkout available. No text encoder or physics rollout is required. Current launchers enable
online W&B; the original native38 run predates that integration.

Defaults can be changed through environment variables:

| Variable | Default |
|---|---|
| `G1_PYTHON` | `/data/langlang/envs/gentrack-momask/bin/python` |
| `G1_RAW_ROOT` | `/data/langlang/dataset/supermimic/raw/bones_seed` |
| `G1_DATA_ROOT` | `outputs/bones_g1_native38` |
| `G1_RUN_ROOT` | `outputs/g1_causal_textop_500ep` |
| `G1_DEVICE` | `cuda:0` |
| `G1_BATCH_SIZE` | `256` |
| `G1_WORKERS` | `8` |
| `G1_MJCF` | sibling MimicLite `g1_29dof_nohand.xml` |
| `MOMASK_ROOT` | sibling `momask-codes` |

`latest.pt` is atomically replaced after each epoch. `best.pt` is replaced on
validation improvement. Numbered checkpoints are kept every **25 epochs** and
at the final epoch: 20 scheduled snapshots for a 500-epoch run. Checkpoints
include the optimizer, epoch, RNG states, loader generator, normalization,
model and data identities. Resume requires the same configuration, source
code identities and data. Validation is deterministic, uses posterior means
and all valid frames, and selects the minimum frame-weighted total loss.

Default model dimensions are encoder 160, decoder 512, latent 32; temporal
compression is 4. Adam starts at 1e-4, with an epoch cosine schedule to 1% of
the starting rate, and gradient clipping at 1. Training uses posterior
samples and 65-frame random windows; validation uses entire prepared clips.

## Data and contact supervision

The 38D representation is imported from MoMask's canonical G1 conversion:
`[root_xyz_m(3), root_rotation_6d(6), hinge_angles_rad(29)]`. Root 6D uses the
first two **columns** of a rotation matrix. CSV translations in cm and joint
angles in degrees are converted by the existing MoMask helpers. Original
120Hz motion is resampled to 20Hz, including SLERP for root orientation.
Only the first frame's XY/yaw establishes a shared local coordinate system;
absolute height is retained. No later ground-truth anchors are injected into
reconstruction. All 29 hinges are trained, including wrists.

Original takes are grouped by date, performer and original-take name,
including mirrored variants. A fixed SHA256 split (seed 3407) assigns groups
to 90% train / 5% validation / 5% test **before** making non-overlapping clips
of at most 196 frames, discarding tails shorter than 40 frames. Statistics
use only training clips. This is a new source-group split, **not a claim of
matching an existing MoMask test list**. A numerical cross-model comparison
requires evaluating both models on the same IDs and robot XML.

`contact_audit.csv` records every source's eligibility and exclusion reason.
The policy reads all global descriptions, source names and temporal labels.
Explicit stairs, inclines, uneven ground, climbing, obstacles and elevated
support transitions disable **only** foot-contact supervision for the entire
source clip. Unknown descriptions are conservatively excluded. Flat-ground
stepping, ordinary gait initiation and jumping in place are not excluded just
because the text contains “step” or “jump”. This is a reproducible semantic
rule set, not a manually certified terrain annotation; inspect the audit
before changing rules. Terrain clips stay in all other losses and in the test
set. The evaluator reports contact-eligible/excluded subsets in addition to
its full-test score.

GT contact is computed from ankle-roll joint anchors at 20Hz, following
TextOp's squared per-frame displacement `<0.002 m²` and height `<0.08 m`
thresholds. The first source frame has no measured velocity and is left
unlabelled. Ground height is zero; the terrain exclusion prevents applying
this flat-ground heuristic to identified elevated/uneven-surface actions.

## Losses

All reconstruction terms use Huber loss, delta 1.0, with padding excluded.

| Term | Weight | Domain |
|---|---:|---|
| `rec` | 1 | normalized native38 features |
| `kl` | 1e-4 | mean KL to unit Gaussian, valid latents only |
| `body_trans` | 0.05 | FK root + 29 hinge-body origins, metres |
| `body_rot` | 0.01 | FK global unit quaternions |
| `dof_pos` | 0.03 | 29 hinge angles, radians |
| `dof_vel` | 1e-5 | finite-difference hinge velocities, rad/s |
| `foot_contact` | 0.01 | GT-contact-masked foot position reconstruction |

These are the seven active losses in the checked TextOp FeatureVersion=3
configuration. Inactive optional delta, smoothness, quantization and drift
terms are not silently enabled. Adaptations to native G1 are explicit:
TextOp's 23-DoF extended-body skeleton is replaced by the 29-DoF skeleton;
quaternion signs are aligned before Huber so equivalent rotations do not
create an error; terrain-ineligible clips are removed from the contact loss
denominator. Non-contact feet have masked zero position errors, as in
TextOp. An all-excluded batch gives exactly zero contact loss and gradient.

FK is differentiable Torch code using MuJoCo-compiled transforms, hinge axes,
pivots and reference angles. `kinematic.xml` is derived from the supplied
MJCF by removing rendering assets/geometries while preserving body, joint,
inertial and hierarchy definitions. It needs no missing STL files. Its SHA256
is recorded, and the same file is used for training FK and evaluation.

## Causality and lengths

The upstream encoder exposes a latent at frame 0, followed by latents at
frames 4, 8, 12, ... . Later input frames do not alter earlier latents.
Encoding/decoding supports streaming cache parity. This is block-causal:
reconstructing the frames of a four-frame block uses that block's latent;
it is not a zero-lookahead per-frame reconstruction claim. Input is padded
by repeating the last valid frame to length `1+4k`; output is trimmed back
to its original length. This avoids the upstream wrapper's dropped tail
frames. Padding is excluded from feature, geometry, velocity, KL and metric
reductions.

## Evaluation

```bash
python eval_g1_vae.py \
  --data-root outputs/bones_g1_native38 \
  --checkpoint outputs/g1_causal_textop_500ep/best.pt \
  --output outputs/g1_causal_textop_500ep/eval_best
```

The evaluator directly imports `evaluation/g1_metrics.py` from MoMask:
29 named hinge **joint anchors**, world-space MPJPE and root-translation-
relative MPJPE in **mm**, equal-clip `action_mean`, `action_p95`, and frame/
joint-weighted `pooled` aggregation. Root rotation is not removed for the
root-relative metric. No Procrustes alignment, motion clipping, contact
correction or dynamics is applied. Equal-length batches use deterministic
posterior means, float32, TF32 off. Batch/single agreement is checked.

Each evaluation writes `identity.json`, `manifest.json`, `samples.txt`,
`per_sample.csv`, `per_joint.csv` and `summary.json`. Only `complete: true`
with `smoke: false` is a completed production evaluation. Source/model/
checkpoint/split/XML hashes and the exact imported metric checksum are saved.

## Validation and smoke runs

```bash
python prepare_g1.py --output outputs/g1_smoke_data \
  --mjcf /path/to/g1_29dof_nohand.xml --limit-per-split 2
python -m unittest discover -s tests -p test_g1_vae.py -v
python train_g1_vae.py --data-root outputs/g1_smoke_data \
  --output outputs/g1_smoke_run --epochs 2 --checkpoint-every 1 \
  --batch-size 2 --val-batch-size 2 --workers 0 --device cpu \
  --window 17 --z-dim 4 --dim 8 --dec-dim 8 \
  --allow-smoke --max-batches 2
```

The reduced smoke model verifies the end-to-end path, not reconstruction
quality. Production training refuses smoke-prepared data unless explicitly
marked as a smoke run. An incomplete preparation is also rejected.

## Historical native38 eight-GPU continuation

`scripts/resume_g1_8gpu.sh` resumes `outputs/g1_causal_textop_500ep/latest.pt`
with eight DDP ranks. Stop the previous trainer before using the same output
folder. The final global training batch was 8192 (1024 per GPU), changed at
epoch 137 by explicit user request; the optimizer,
learning-rate schedule, 500-epoch target and 25-epoch snapshots are inherited
from the checkpoint. Each epoch also updates `latest.pt` and, when improved,
`best.pt`. Completed training automatically evaluates both with MoMask metrics.

Training batches contain each example exactly once, including the uneven last
batch. Loss gradients use global valid-frame/contact denominators. Validation
retains the original contiguous batches of 32 and assigns whole batches to
ranks, without duplicated samples. Checkpoints retain plain VAE state dicts
and each rank's RNG state. Migration records the original checkpoint hash;
new per-rank random streams mean the trajectory is not bitwise identical to
continued single-GPU training. The DDP source and helper hashes are checked
on subsequent DDP resumes.

The eight-rank gradient check can be run with:

```bash
torchrun --standalone --nproc_per_node=8 tests/check_g1_ddp_gradient.py
```

Use Zellij for future training sessions (user preference). Historical session:

```bash
zellij attach flood-g1-ddp8
```

The historical training pane ran `scripts/run_g1_zellij.sh` and uses `tee` to
display output and save `outputs/g1_500epochs.ddp8_bs1024.log`. Here BS1024
means per GPU; the global batch is 8192. The previous batch256 checkpoint
is preserved under `bs256_archive/epoch_0137/`, and the exact migration is
recorded in `batch_size_migration.json`. Optimizer state and epoch-based
learning rates are preserved; optimizer updates per epoch fall from 582
to 19. Future sessions should also run in Zellij with visible logs.

## rot6d68 + TextOp experiment

The independent `g1_rot6d68_textop_500ep_bs2048` run starts from random weights
with 8 GPUs, 2048 examples per GPU (global16384), 65-frame training crops,
500 epochs, Adam1e-4 with the same cosine epoch schedule, and snapshots every25
epochs plus best/latest each epoch. `zellij attach flood-rot68-bs2048` displays
live training losses and epoch ETA. This completed run did not use W&B;
future launches now enable W&B through the wrapper described below. Launch entry point:
`scripts/run_g1_rot68_8gpu.sh`; resume in the same directory by appending
`--resume outputs/g1_rot6d68_textop_500ep_bs2048/latest.pt` after stopping the
existing trainer.

Inputs use the canonical MoMask rot6d68 field order: joint_angles29,
joint_vel29, root_roll_pitch_rot6d6, root_yaw_vel1, root_xy_vel_body2,
root_height1. Deltas are per frame in the current-heading frame. The native
source-group splits and contact eligibility are unchanged. Conversion runs
on complete prepared clips before training crops; each prepared clip's
first deltas are zero. Statistics use training frames only, std floor1e-4.

The rec loss operates on normalized68D features. The other six terms retain
the prior TextOp weights. Differentiable integration uses only the initial
GT XY/yaw anchor; predicted height and all subsequent root motion come from
model outputs. Physical losses operate on decoded native38 trajectories.
DOF velocity is differentiated from reconstructed joint angles (rad/s),
not read from the predicted velocity channels. rec overlaps with dof_pos
as in the original seven-loss setup. Terrain actions are excluded only
from the contact term. No clipping or postprocessing is used for metrics.

`g1/rot6d68.py` contains the representation adapter and physical-loss wrapper.
`train_g1_rot6d68.py` trains and automatically invokes `eval_g1_rot6d68.py`
for both best and latest on all8333 test clips, using the original MoMask
metric implementation. Checkpoint/source/data hashes protect resumed runs.


## W&B for subsequent training

User preference: use Zellij and enable online W&B recording for subsequent
training. The native38 single/DDP and rot6d68 launch scripts now invoke
`scripts/train_with_wandb.py` around the trainer. Default project is
`flood-g1-vae`; WANDB_PROJECT/WANDB_ENTITY may override the destination.
The existing completed experiments have not been uploaded retroactively.

The wrapper records configuration, epoch/update count, learning rate, all
train/validation loss terms (raw and weighted), available epoch time/peak
memory, and final best/latest evaluation metrics. Only one wrapper runs per
DDP job. The stable run ID and URL are saved under
`outputs/.wandb_tracking/<experiment>/tracking.json`; online restarts reuse
that ID. Training stdout remains visible through tee in Zellij. The wrapper
uses the existing logs so trained-source hashes and old checkpoints remain
unchanged. Directly invoking a trainer without the wrapper does not log W&B.

For a new experiment use the wrapper with its matching output directory:

```bash
python scripts/train_with_wandb.py --output outputs/NEW_RUN --   python -m torch.distributed.run --standalone --nproc_per_node=8   train_g1_rot6d68.py --output outputs/NEW_RUN [other training arguments]
```

Online initialization must succeed before launching the trainer. Offline
mode is explicit (`--mode offline` or WANDB_MODE=offline), not a silent
fallback. Authentication was verified through the API for the configured
local account; the logger integration was smoke-tested offline, without
starting training or uploading synthetic metrics.

## MPJPE-selected continuation: epochs 501–1000

`scripts/run_g1_rot68_continue_8gpu.sh` runs as Linux user `langlang`, using
that user's verified W&B account langlang-02 / langlang-02-ustc at
api.forge.coreweave.com. Project: flood-g1-vae. Session:
`zellij attach flood-rot68-continue` from the langlang login.

The parent is the completed rot68 epoch500 checkpoint. Its Adam moments,
optimizer update count and eight RNG states are restored. Per-GPU batch2048,
global16384, all seven losses and training dataset are unchanged. Additional
500epochs contain5000updates: the first300updates warm from1e-6 to3e-5,
then hold3e-5 until update2000, then cosine-decay to1e-6 at update5000.
The previous run and checkpoint remain untouched in a separate directory.

The fixed cohort `outputs/rot68_fixed_val_tracker_v1` contains8107 validation
clips after the historical MimicLite tracker filter (490 removed). Its
samples, source-audit and hashes are frozen; no train/test source overlaps.
The epoch500 baseline and epochs525,550,...,1000 are evaluated with posterior
means and the unchanged MoMask FK/MPJPE definition. `best.pt` is chosen by
**global MPJPE action mean**, not validation total loss. Root-relative MPJPE
is also logged. All epochs still log full-validation losses and save
`latest.pt`; evaluation epochs save numbered snapshots and per-clip metrics.
W&B records fixed_val metrics and selection/best_global_mpjpe_mm alongside
training losses and LR. Final best/latest full-test evaluation runs automatically.

The continuation runner supports strict resume with the same arguments plus
`--resume outputs/g1_rot68_continue_ep0500_1000_bs2048_mpjpe/latest.pt`.
Resume retains the original parent, fixed cohort and schedule budget; do not
reset the LR schedule or parent checkpoint on a restart.

### 2026-10-27: continuation resized to seven GPUs

Run `scripts/run_g1_rot68_continue_7gpu.sh` as langlang in Zellij. Resumed epoch 613 / update 6130 with model, Adam and rank 0–6 RNG states retained. GPUs 0–6, per-GPU batch 2048, global batch 14336, 11 updates/epoch. Target remains epoch 1000. LR stays 3e-5 through epoch 700 then cosine-decays to 1e-6 at epoch 1000; no new warmup. Actual optimizer steps remain cumulative; virtual schedule origin 4887 preserves epoch progress. Fixed validation and MPJPE best selection every 25 epochs remain unchanged. Same online W&B run 3ygxoqxi. Original eight-GPU state is archived under the run directory `migration_8_to_7_ep0613`. World-size changes alter sample-to-rank assignments and are not bitwise equivalent to continuing on eight GPUs.

## Completed continuation and reconstruction diagnostics (2026-10-28)

Training completed at epoch 1000 / optimizer update 10387. The best checkpoint
is epoch 1000, selected on the frozen, tracker-filtered validation cohort:
global MPJPE 109.55 mm and root-relative MPJPE 25.42 mm. Full-test results
(8333 clips, including terrain actions) are 115.57 mm and 26.89 mm respectively.
The eight-GPU continuation launcher describes the original phase; after the
recorded migration, only the seven-GPU launcher matches the latest checkpoint.
It expects the already migrated checkpoint and is not a general world-size
migration utility. The completed run will not train additional epochs on resume.

Small numerical results and verification metadata are committed in
`docs/results/rot68_epoch1000.json`. Raw data, checkpoints, predictions,
W&B files and rendered media stay under ignored directories. Historical run
paths and source hashes are intentional: do not reformat hashed trainers or
adapters when restoring an existing checkpoint.

The two representations differ in more than root translation. Native38 stores
XYZ, full root rotation6D and 29 joint angles. Rot68 stores 29 joint angles,
29 angle deltas, tilt rotation6D, integrated yaw delta, integrated heading-local
XY delta and height. Root-relative MPJPE removes XYZ but retains orientation
errors. On the same test set, replacing only predicted yaw with GT reduces
rot68 root-relative MPJPE from 26.89 to 13.44 mm; replacing full root rotation
reduces it to 10.87 mm. Native38 gives 7.60 and 6.22 mm before/after full rotation
replacement. These GT-assisted diagnostics are not model benchmark scores,
and error components are not additive. The native38 best checkpoint also had
79962 optimizer updates versus 10387 here; this is not a controlled comparison
of representations or training budgets.

### Reproduce the saved visual comparison

The analysis scripts use existing prepared data, evaluated checkpoints and the
previous `outputs/rot68_reconstruction_comparison/selection.json` and clip exports.
That selection originates in `scripts/prepare_reconstruction_comparison.py`,
with the epoch500 exports from `scripts/visualize_rot68_result.py`. Keep those
local artifacts to reproduce the same seven clips. Rendered robot meshes come
from the sibling MimicLite checkout. EGL, an installed DejaVu Sans font and
FFmpeg support are required for rendering.

```bash
python scripts/analyze_rot68_final.py
MUJOCO_GL=egl python scripts/render_rot68_final.py --follow
MUJOCO_GL=egl python scripts/render_rot68_final.py --follow --root-align
python scripts/report_rot68_final.py
```

Choose a free GPU using `CUDA_VISIBLE_DEVICES` for inference and
`MUJOCO_EGL_DEVICE_ID` for rendering; leave GPU 7 reserved. The analyzer creates
`outputs/rot68_epoch1000_analysis` exclusively and refuses to overwrite an
existing result. It checks every clip against saved MoMask evaluation metrics.
The HTML viewer shows GT/native38/rot68 epoch500/rot68 epoch1000 side by side,
with separate original-world and GT-root-translation-aligned diagnostic videos.
The stairs clip is explicitly marked as terrain-dependent with no original
stairs geometry. All 14 exported videos were verified at 1920x576, 20 FPS,
with exact source frame counts.
