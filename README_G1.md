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

## Absolute-yaw experiment: rot6d69 (2026-10-29)

`scripts/run_g1_rot69_4gpu.sh` starts a fresh VAE on GPUs 4–7, 2048 examples
per GPU (global 8192), for 500 epochs. This experiment explicitly uses GPU 7,
superseding the previous reservation for the completed seven-GPU run. Run as
Linux user langlang in Zellij with online W&B. Output:
`outputs/g1_rot69_absyaw_500ep_bs2048_mpjpe`.

The original 68 channels are preserved exactly; channel 68 (zero-based) is
continuous, unwrapped absolute yaw in radians in the native clip's canonical
coordinate frame. Unwrapping precedes training crops. The decoder uses this
predicted absolute yaw directly for full root orientation and rotating local
XY increments. It uses only GT initial XY, not GT initial yaw. The original
yaw-delta and joint-delta outputs remain supervised feature channels; they
are not integrated to recover root orientation or joint angles. No sine/cosine
expansion is added. All seven losses and their weights remain unchanged,
with rec averaged over 69 normalized channels. The original 68 training
means/stds are verified unchanged; the new yaw statistics use training only.

Data preparation: `scripts/prepare_g1_rot6d69.py --native-root
outputs/bones_g1_native38 --rot68-root outputs/bones_g1_rot6d68 --output
outputs/bones_g1_rot6d69`. It checks every clip's unchanged first 68 channels
and full-trajectory roundtrip. Original source splits and foot-contact
eligibility are retained. Terrain actions remain in training except for
contact supervision. The fixed 8107-clip filtered validation set is unchanged.

Model dimensions, Adam, 65-frame crops, FP32/TF32-off and the original fresh
rot68 epoch-cosine LR formula (1e-4 to approximately 1e-6) are retained.
There are 19 optimizer updates per epoch, 9500 total. Every epoch saves latest;
every 25 epochs evaluates fixed-validation global MPJPE and saves a numbered
checkpoint, selecting best only by that metric. The first best.pt is produced
at epoch 25; earlier latest checkpoints can still resume. Final best/latest
are evaluated on the full test split using unchanged MoMask metrics.

Resume by appending `--resume
outputs/g1_rot69_absyaw_500ep_bs2048_mpjpe/latest.pt` to the launcher. Source,
data, cohort and full configuration identities must match; Adam and all four
RNG streams are restored. Previous native38/rot68 hashed sources are unchanged.

Production session: `zellij attach flood-rot69-absyaw` as langlang.
W&B run: https://app.forge.coreweave.com/langlang-02-ustc/flood-g1-vae/runs/cyk9zyux .
Four-rank smoke/resume and unchanged-MoMask validation are checked by
`tests/check_rot69_pipeline.py`; representation invariants by
`python -m unittest discover -s tests -p test_rot69.py -v`.

## GT-matched temporal-loss prototype (2026-10-30; smoke only)

The original rot69 checkpoint reduced spatial error but the seven historical
visual clips had higher root angular velocity/acceleration errors. Full-test
hinge acceleration error slightly improved, so root-driven whole-body jitter
must be separated from joint-coordinate errors. This experiment starts fresh
with the same seed, model, 69D inputs, splits, batch and LR schedule as rot69;
only the training objective changes. The proposed production configuration was
500 epochs on GPUs4–7, batch2048 per GPU (global8192), with online W&B as
langlang. It was never production-launched; its reserved output directory is
`outputs/g1_rot69_temporal_500ep_bs2048_mpjpe`.

| Loss | Prior weight | Temporal experiment |
|---|---:|---:|
| body_rot | 0.01 | 0.1 |
| dof_vel | 1e-5 | 0.01 |
| dof_acc | absent | 1e-4 |
| root_rot_vel | absent | 0.01 |
| root_rot_acc | absent | 1e-4 |

The other five original losses are unchanged. There are ten active terms,
not seven. Derivatives match GT derivatives with Huber delta1; no zero-motion
penalty is applied. Joint acceleration is in rad/s². Root angular velocity
is the world-frame SO(3) logarithm of `R[t+1] R[t]^T`, scaled by20Hz, in rad/s;
its first difference times20 defines angular acceleration. Full decoded root
rotation (absolute yaw and predicted tilt) is supervised, not just yaw. Valid
pair/triple masks and DDP denominators exclude padded boundaries.

`g1/temporal_loss.py`, `g1/temporal_ddp.py`, `g1/temporal_metrics.py`,
`g1/temporal_eval.py`, `train_g1_rot69_temporal.py` and
`eval_g1_rot69_temporal.py` are independent of old hashed training sources.
Launch `scripts/run_g1_rot69_temporal_4gpu.sh` as langlang. Every25 epochs,
fixed validation records unchanged MPJPE plus root angular velocity and
acceleration RMSE and joint velocity/acceleration RMSE. Best remains selected
by global MPJPE; inspect temporal metrics on the saved periodic checkpoints
as well. Final full-test evaluation includes both root temporal metrics.
An improvement in jitter is a hypothesis to test, not a guaranteed result.

Validation: `tests/test_temporal_loss.py` covers SciPy parity, identity gradients,
GT-motion preservation, padding/short clips, oscillation gradients and counts.
`tests/check_temporal_ddp.py` checks all10 terms on unequal4-rank shards against
a single global batch. The old rot69 checkpoint's frozen validation baseline
is evaluated by `scripts/eval_rot69_temporal_baseline.py` and stored in
`outputs/rot69_temporal_diagnostics/baseline_fixed_val`.

## Approved root-temporal configuration (2026-10-31)

The earlier ten-term temporal prototype was only smoke-tested, never launched
as production. The approved experiment uses **nine** terms: original seven
with `body_rot=0.05`, `dof_vel=0.01`, plus `root_rot_vel=0.01` and
`root_rot_acc=1e-4`. There is **no dof_acc loss**. Joint acceleration remains
an evaluation metric. Other weights, data/representation, initialization seed,
model dimensions, LR schedule and frozen validation cohort are unchanged.

New code is `g1/root_temporal_loss.py`, `g1/root_temporal_ddp.py` and
`train_g1_rot69_root_temporal.py`; it reuses the temporal evaluator. Start via
`scripts/run_g1_rot69_root_temporal_4gpu.sh` as langlang, or resume by adding
`--resume outputs/g1_rot69_root_temporal_500ep_bs2048_mpjpe/latest.pt`.
User approved physical GPUs **3,4,5,7** because GPU6 is occupied. Train fresh
for500epochs,2048 examples per GPU (global8192),19 updates/epoch,9500 total.
Output is `outputs/g1_rot69_root_temporal_500ep_bs2048_mpjpe`.

View the session from langlang with
`env -u XDG_RUNTIME_DIR zellij attach flood-rot69-root-temporal`.
From root use
`runuser -u langlang -- env -u XDG_RUNTIME_DIR zellij attach flood-rot69-root-temporal`.
Every25epochs records fixed-val MPJPE and root/joint derivative errors. Best
continues to minimize global MPJPE. Prior rot69 frozen-val temporal baseline:
root angular velocity RMSE0.50590rad/s, acceleration15.22557rad/s²;
global MPJPE86.80392mm, root-relative18.12506mm (8107clips).

Production W&B run for the nine-term configuration:
https://app.forge.coreweave.com/langlang-02-ustc/flood-g1-vae/runs/9a7vrwyz .
The approved run completed all 500 epochs / 9500 updates; best is epoch500.
The older ten-term prototype remains unlaunched. Tests `test_root_temporal_loss.py`,
`check_root_temporal_ddp.py` and `check_root_temporal_pipeline.py` verify the
final nine-term objective, uneven-shard gradients, checkpoint selection,
metric parity and resume behavior.

## Completed rot69 reconstruction results and videos

All full-test numbers below use the same 8333 clips and equal-clip action means,
with unchanged MoMask G1 joint metrics. Terrain-dependent test clips remain
included. Both rot69 runs trained fresh for 500 epochs / 9500 updates.

| Model | Global MPJPE (mm) | Root-relative MPJPE (mm) | Root yaw error (deg) |
|---|---:|---:|---:|
| rot68 epoch1000 | 115.57 | 26.89 | 6.46 |
| rot69 original epoch500 | 91.96 | 18.73 | 3.76 |
| rot69 root-temporal epoch500 | 89.54 | 17.49 | 2.97 |

On the separate fixed 8107-clip terrain-filtered validation cohort, root angular
velocity RMSE decreased from 0.50590 to 0.19747 rad/s and angular acceleration
RMSE from 15.22557 to 4.31970 rad/s². Best selection used global validation
MPJPE (84.62153 mm), not test results or temporal metrics.
Best checkpoint: `outputs/g1_rot69_root_temporal_500ep_bs2048_mpjpe/best.pt`.
Machine-readable summaries are in `docs/results/rot69_reconstruction.json`;
weights, data and videos remain local under ignored `outputs/`.

Generate fixed historical comparisons with `scripts/visualize_rot69_comparison.py`
(original rot69 versus rot68) or `scripts/visualize_rot69_root_temporal.py`
(root-temporal versus original rot69). Each writes a new output directory and
verifies individual clip metrics against the saved official evaluation CSVs.
Render with the corresponding `scripts/render_rot69_comparison.py` or
`scripts/render_rot69_root_temporal.py`, once with `--follow` and once with
`--follow --root-align`. The renderers require MuJoCo EGL and the local MimicLite
G1 mesh assets. Run from the configured environment with an available GPU.

The latest page is `outputs/rot69_root_temporal_comparison/index.html`: seven
unchanged clips, 14 synchronized three-column videos at 1440x576, 20 FPS.
Columns are GT, original rot69, root-temporal rot69. `root_aligned` videos replace
predicted root XYZ with GT XYZ for visualization only, preserving predicted
rotation and joint angles. Overlay metrics always describe original unaligned
predictions. No smoothing, joint clipping or physics correction is applied.

## G1 text-conditioned generation (2026-11-01)

Launch `bash scripts/run_g1_ldf_4gpu.sh` as langlang. Session:
`env -u XDG_RUNTIME_DIR zellij attach flood-g1-ldf-rot69`.
The independent `train_g1_ldf.py` adapts upstream Wan diffusion forcing to the
frozen epoch500 `g1_rot69_root_temporal_500ep_bs2048_mpjpe/best.pt` VAE.
32D posterior means, no latent rescaling; complete prepared clips (up to196
frames) retain their original source-group splits. Each training visit randomly
chooses a source-level BONES description. Text uses frozen local CLIP ViT-B/32
**token hidden states**, not the upstream UMT5 encoder. Original Wan structure:
1024 hidden, 8 layers, 8 heads, velocity target, chunk5, CFG5, dropout0.1, EMA0.99.
A local SDPA adapter honors actual key lengths without requiring FlashAttention;
old hashed model sources are unchanged.

GPUs4–7, batch64 per GPU/global256, 500 epochs, AdamW2e-4, 1000-update warmup
then cosine to1e-6. Frozen VAE FP32; diffusion BF16 autocast. DistributedSampler
drops at most3 training examples per epoch, rotating with the shuffle. Online
W&B uses langlang's credentials, entity langlang-02-ustc, project
flood-g1-generation. Output: `outputs/g1_ldf_rot69_root_temporal_clip_500ep`.
Every epoch saves latest; every25 epochs generates the entire frozen8107-clip
validation cohort with fixed noise seeds, GT duration and only initial GT XY.
Best selection uses equal-clip global MPJPE; root-relative MPJPE is also logged.
Paired generative MPJPE is a reproducible checkpoint criterion, not a measure
of language alignment, diversity, or reconstruction quality. Test is untouched.

Append `--resume outputs/g1_ldf_rot69_root_temporal_clip_500ep/latest.pt` to the
launcher for recovery. Model, EMA, optimizer, step, best metric, and per-rank RNG
states are retained; source/data/cohort/parent/text encoder hashes must match.
W&B run identity and URL are in `wandb_run.json`; logs include steps and ETA.
`tests/check_g1_generation.py` verifies attention padding and exact frozen-VAE
encode/decode parity. Tiny four-rank train/resume and production-size four-rank
batch64 train/generate/checkpoint smoke tests passed before production launch.

### Batch migration (2026-11-01)

User requested batch256 per GPU/global1024. At epoch2/update1164, the batch64
checkpoint, identity and original trainer/launcher were archived in
`batch64_archive_epoch0002/` inside the run directory. Resume starts from
`batch256_resume.pt` with unchanged model, EMA, optimizer and rank RNG state.
There are now146 updates/epoch; the persisted schedule horizon is73872 total
cumulative updates, ending at epoch500. Warmup is not restarted. W&B run
zbhcjme4 and Zellij session flood-g1-ldf-rot69 are reused. The launch script now
defaults to batch256. `batch_size_migration.json` records the old/new identities
and original checkpoint hash. After the next epoch save, normal recovery uses
`--resume outputs/g1_ldf_rot69_root_temporal_clip_500ep/latest.pt`.

### Disable periodic generation validation (2026-11-01)

At user request, `--eval-every 0` now disables the every25-epoch generation
validation. Final epoch500 validation is retained. Every epoch still writes
latest.pt, and numbered checkpoints remain every25 epochs. There is no periodic
MPJPE-based best selection while validation is disabled. Migration occurred at
epoch4/update1456; `periodic_val_archive_epoch0004/` preserves the prior state,
and `validation_policy_migration.json` records the exact identities. Model,
EMA, optimizer, RNG and LR schedule are retained. The launcher defaults to
batch256 and eval-every0. Initial migrated checkpoint is
`no_periodic_val_resume.pt`; use latest.pt after the next saved epoch. Same
langlang Zellij session and W&B run continue.

## Fresh segment-conditioned G1 generation (2026-11-03)

The source-caption experiment was stopped at epoch435/update64382 at the user's
request; its final saved state and launch sources are preserved under
`stopped_for_segment_epoch0435/` in that experiment directory. It is not the
initialization for the new model.

`train_g1_ldf_segment.py` starts a fresh1024-wide/8-layer Wan generator on
physical GPUs4–7 with actual batch512/GPU/global2048,500epochs, AdamW2e-4,
1000-update warmup, cosine decay to1e-6, EMA0.99. The original rot69 root-temporal
VAE best.pt and CLIP ViT-B/32 remain frozen. Original69D inputs and32D posterior
means are unchanged. Launch as langlang with
`bash scripts/run_g1_ldf_segment_4gpu.sh`; Zellij session is
`env -u XDG_RUNTIME_DIR zellij attach flood-g1-ldf-segment`.
Output: `outputs/g1_ldf_rot69_segment_clip_500ep_bs512`. New online W&B run uses
langlang-02 / langlang-02-ustc, project flood-g1-generation.

Temporal labels are frozen under `outputs/g1_segment_labels_v1`. Rebuild with
`g1.segment_data.prepare(native_root, temporal_jsonl, new_output_directory)`.
Events use left-closed/right-open source seconds; latent j maps to
`(start_frame + min(4*j, frames-1))/20`. Unlabelled gaps receive empty text,
not an invented motion label;20 entirely unlabelled train windows are excluded,
leaving148885 samples. There are38 event/window intersections with no latent
anchor; these are reported rather than artificially extended. Source group
train/val/test splits remain unchanged. Conflicting overlaps, invalid times and
duplicate temporal records fail preparation. The manifest retains counts,
exclusions and source/code hashes. Existing windows remain40–196frames.

Text conditions switch inside a continuous motion window. `g1/segment_generation.py`
groups queries by sample/text context; it projects text per group without
materializing a separate full text embedding for every motion token. Numerical
output and gradient parity against upstream frame-wise conditioning is checked.
Random-prefix diffusion forcing and last-five-token velocity supervision are
unchanged. CFG drops the entire sample's text with probability0.1. Gap and CFG
dropout ratios are logged separately. Activation checkpointing is enabled:
full-size smoke peak allocation was11.35GiB versus39.29GiB without checkpointing;
recomputation provides memory headroom for varying segment counts. No gradient
accumulation substitutes for actual batch512. The final incomplete batch is
smaller; DistributedSampler drops one example per epoch with shuffled ordering.
There are73 updates/epoch and36500 updates over500epochs.

The batch accepts ordered `text` lists and `feature_text_end` latent endpoints;
inference supports both explicit multi-stage plans and live prompt switching
while retaining stream state. `scripts/preview_g1_segments.py --checkpoint PATH
--output NEW_DIRECTORY` generates a fixed wave/walk/squat plan and renders G1
MuJoCo meshes. The smoke preview is only a pipeline check, not generation quality.

Periodic generation validation remains disabled. Latest is saved every epoch,
numbered snapshots every25epochs, and the frozen8107-clip cohort is generated
only at the final epoch using aligned segment conditions. Paired global/root-
relative MPJPE is not a language alignment or diversity metric. Test is unused.
Resume by appending `--resume outputs/g1_ldf_rot69_segment_clip_500ep_bs512/latest.pt`
to the launcher; optimizer, EMA, schedule and rank RNG states are restored with
strict source/data/config identity checks.

Validation: `tests/check_g1_segments.py` covers source-time offsets, event
boundaries, gaps, short events, padding, single/multi-condition Wan output and
gradient parity, sample/interval isolation, original FD loss parity and staged
inference. Four-rank actual batch512 train/final-eval/checkpoint/resume passed;
resumed loss matches within1e-4. Live prompt switches retain stream state, and
the137-frame/20FPS G1 pipeline video was verified.

2026-11-03 production update: resumed epoch2/step146 with batch128/GPU
(global512), activation checkpointing disabled, GPUs4–7. Model, EMA, optimizer
and rank RNG states were preserved and verified exactly. Remaining498 epochs
use291 updates/epoch; cumulative schedule horizon145064; warmup continues.
Output retains its historical bs512 suffix. Migration details: output file
`batch_size_migration_128.json`; previous state: `batch512_archive_epoch0002`.
Same W&B b6k4p4u3 and langlang Zellij flood-g1-ldf-segment. At step250,
throughput570 samples/s, estimated remaining training36.1h (early estimate,
excluding final evaluation). Launcher now selects128/no checkpointing.
