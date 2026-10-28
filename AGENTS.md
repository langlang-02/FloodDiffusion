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
