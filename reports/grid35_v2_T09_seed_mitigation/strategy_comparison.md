# Grid35 V2 T01 - first-action seed-mitigation strategy comparison (offline)

Offline comparison of 4 candidate first-action stabilization strategies, computed entirely from the existing 20-seed sweep (`reports/grid35_v2_T01_seed_sweep/seed_sweep.csv`). **No new inference was run.**

- Source sweep checkpoint: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model`
- Reference Shadow observation: `T09` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T09/shadow_synthetic_T09.json`)
- Task: `Pick up the cube and place it in the target area.`
- Nearest-training-demo match: episode 30, frame 7
- Deterministic RNG seed used for resampling-permutation simulation: `20260808` (n_permutations=20000)
- 3-sample / 5-sample medians: full exhaustive combination enumeration (no sampling approximation)

## Safety thresholds used (read-only, not modified)

| joint | WOULD_CLAMP threshold (deg) |
|---|---:|
| shoulder_pan | 4.58 |
| shoulder_lift | 5.16 |
| elbow_flex | 5.73 |
| wrist_flex | 4.01 |
| wrist_roll | 1.15 |
| gripper | 9.17 |

## GT immediate delta (reference, evaluation-only - never used in selection logic)

| joint | GT immediate delta (deg) |
|---|---:|
| shoulder_pan | -0.6154 |
| shoulder_lift | +0.1758 |
| elbow_flex | -2.0659 |
| wrist_flex | +0.1319 |
| wrist_roll | +0.3516 |
| gripper | +0.1635 |

## Strategy comparison table

| strategy | clamp-free rate | avg clamp joints | GT L2 mean | GT L2 median | GT L2 p95 | GT L2 max | shoulder_lift mean±std (range) | elbow_flex mean±std (range) | avg # inference | max # inference | failure rate |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| single_random_sample | 30.0% | 0.900 | 7.242 | 6.403 | 11.837 | 14.305 | +6.472±3.006 (14.857) | -4.225±2.235 (10.478) | 1.00 | 1 | 70.0% |
| 3_sample_median | 20.2% | 0.886 | 6.709 | 6.009 | 10.194 | 11.427 | +6.296±1.558 (6.576) | -4.288±1.166 (6.280) | 3.00 | 3 | 79.8% |
| 5_sample_median | 13.1% | 0.901 | 6.523 | 5.977 | 9.002 | 10.779 | +6.145±1.161 (6.068) | -4.338±0.795 (4.428) | 5.00 | 5 | 86.9% |
| safety_pass_resampling_max3 | 68.7% | 0.403 | 4.966 | 5.264 | 6.270 | 6.897 | +4.054±1.864 (6.084) | -3.818±1.494 (6.832) | 2.17 | 3 | 31.4% |
| safety_pass_resampling_max5 | 87.3% | 0.163 | 4.710 | 5.072 | 6.270 | 6.897 | +3.753±1.998 (6.084) | -3.767±1.301 (6.832) | 2.69 | 5 | 12.7% |

## Baseline-only: fix one good seed (NOT evaluated as a deployment candidate)

Selection rule: min GT L2 among clamp-free seeds in the 0..19 sweep (retrospective, GT-informed - NOT reproducible at deploy time). Chosen seed: **12**.

| metric | value |
|---|---:|
| clamp-free | yes |
| avg clamp joint count | 0 |
| GT L2 (deg) | 1.737 |
| shoulder_lift (deg) | -0.924 |
| elbow_flex (deg) | -2.788 |

> Degenerate stats (std=0, range=0) are expected and diagnostic, not a virtue: this reflects a single observation (one seed x one fixed Shadow reference state), not a distribution. It says nothing about what this seed would produce against a different observation.

## Additional analysis

### Does 3-median / 5-median actually reduce clamp rate vs single sample?

- single-sample clamp-free rate: 30.0%
- 3-median clamp-free rate: 20.2% (not higher than single-sample)
- 5-median clamp-free rate: 13.1% (not higher than single-sample)

### Does median also improve GT L2?

- single-sample mean GT L2: 7.242 deg
- 3-median mean GT L2: 6.709 deg (improves)
- 5-median mean GT L2: 6.523 deg (improves)

### Resampling: max-3 vs max-5 success-rate difference

- max-3 clamp-free (success) rate: 68.7% (avg 2.17 inference calls, failure rate 31.4%)
- max-5 clamp-free (success) rate: 87.3% (avg 2.69 inference calls, failure rate 12.7%)
- delta (max5 - max3): +18.6 percentage points

### Clamp-free does not imply low GT error

> seed 3 is clamp-free (0 joints clamp) but has GT L2=6.270 deg, worse than 4/14 clamped seeds. Clamp-free selection (the only signal available at real deploy time, since GT is unknown) does not guarantee low GT error - it only guarantees the action was not truncated by the Safety Gate.

- clamp-free seeds [12, 2, 6, 11, 18, 3], GT L2 values [1.737, 4.043, 4.44, 5.072, 5.715, 6.27] (mean 4.546)
- clamped seeds mean GT L2: 8.397
- **Implication for deployment:** since clamp status is the only signal available at inference time (GT is unknown), a strategy that optimizes purely for clamp-free status (e.g. safety-pass resampling) reduces Safety-Gate intervention but offers **no guarantee** on trajectory accuracy. It is a safety/stability mitigation, not an accuracy mitigation.

## Recommendation

**Primary recommendation: safety-pass resampling, cap=5** (clamp-free rate 87.3%, failure rate 12.7%, avg 2.69 inference calls, GT L2 mean 4.710 deg). It directly targets the deployment failure mode this sweep was run to investigate (Safety Gate truncating first-action joints), it is the only strategy of the four whose stopping rule is deploy-time-realistic (uses only clamp status, never GT), and its inference cost is adaptive (only pays for extra calls when the first draw actually clamps) rather than fixed.

**Secondary / fallback recommendation: safety-pass resampling, cap=3** (clamp-free rate 68.7%, avg 2.17 calls) if the extra latency budget for a 5th inference call is not available on the deployment hardware - it captures most of the benefit of cap=5 at lower worst-case latency.

3-sample and 5-sample median are **not** recommended as the primary mitigation: median clamp-free rate (3-median 20.2%, 5-median 13.1%) does not clearly dominate single-sample (30.0%) or resampling, while always paying the fixed 3x/5x inference cost regardless of whether the first draw was already clamp-free - resampling only pays that cost when needed.

The 'fix one good seed' baseline (seed 12, clamp-free, GT L2 1.737 deg here) is **not recommended for deployment**: it was selected using GT that will not be available at deploy time, it reflects a single observation against one fixed Shadow reference state, and this whole sweep's own premise - that SmolVLA's flow-matching noise draw materially shifts the first action - means nothing guarantees that seed's favorable behavior transfers to a different observation, task, or checkpoint. It remains useful only as a debug/regression reference, not as a runtime policy.

## Next step: T01-T10 full Shadow validation

Carry safety-pass resampling (cap=5, fallback cap=3) into the T01-T10 full Shadow verification pass as the candidate mitigation, but re-run this same offline analysis *per task* once each task has its own seed sweep - this T01-only result (clamp-free 87.3% at cap=5) should not be assumed to hold for T02-T10 without their own 20-seed sweeps, since clamp behavior is a function of each task's own first-action delta distribution relative to the (task-independent) Safety Gate thresholds. Concretely: (1) run `sweep_grid35_first_action_seed.py` (or equivalent) for each of T02..T10 against its own Shadow reference observation; (2) re-run this script per task; (3) only adopt a single fixed cap/strategy repo-wide once its clamp-free rate and GT L2 are confirmed acceptable across all 10 tasks, not just T01.

