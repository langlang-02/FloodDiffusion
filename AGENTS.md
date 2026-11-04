# Training preferences for this project

- Run long training jobs in Zellij, with visible logs and ETA.
- Enable online W&B for subsequent training. Use the credentials of Linux user
  `langlang` (/home/langlang), never substitute root's W&B account. The verified
  account is langlang-02 / langlang-02-ustc, with the configured service at
  https://api.forge.coreweave.com. Never print or copy credential values.
- Evaluate a fixed, frozen validation cohort every 25 epochs and select best
  by lowest global MPJPE (equal-clip action mean, mm). Also log root-relative
  MPJPE. Keep the test split for final evaluation, not checkpoint selection.
- Current cohort is `outputs/rot68_fixed_val_tracker_v1`: original validation
  sources filtered with the exact historical MimicLite flat_ground_v1 rules.
- Include the starting checkpoint's validation baseline among best candidates
  during continuation. Preserve source/checkpoint/cohort hashes and resume state.
- The older native38 and rot68 trainers are historical loss-selected runs;
  do not change their hashed source in place to add new continuation policies.
  The current continuation entry is `scripts/run_g1_rot68_continue_7gpu.sh` (GPUs 0–6; reserve GPU 7).

- Current fresh experiment: `scripts/run_g1_rot69_4gpu.sh`, rot68 plus one
  continuous absolute-yaw channel (69D). User explicitly requested GPUs 4–7,
  per-GPU batch 2048, 500 epochs; GPU 7 reservation applies only to the older run.

- Latest approved run (2026-10-31): `scripts/run_g1_rot69_root_temporal_4gpu.sh`.
  User approved GPUs 3,4,5,7 (GPU6 occupied), per-GPU2048, fresh500epochs.
  Use body_rot=.05, dof_vel=.01, root_rot_vel=.01, root_rot_acc=1e-4;
  do not enable dof_acc. Previous ten-term prototype was never production-launched.
