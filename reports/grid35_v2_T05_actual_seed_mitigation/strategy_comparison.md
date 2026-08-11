# Grid35 V2 T01 - first-action seed-mitigation strategy comparison (offline)

Offline comparison of 4 candidate first-action stabilization strategies, computed entirely from the existing 20-seed sweep (`reports/grid35_v2_T01_seed_sweep/seed_sweep.csv`). **No new inference was run.**

- Source sweep checkpoint: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model`
- Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T05_actual/shadow_patched.json`)
- Task: `Pick up the cube and place it in the target area.`
- Nearest-training-demo match: episode 33, frame 25
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
| shoulder_pan | +0.1758 |
| shoulder_lift | +3.8681 |
| elbow_flex | -4.6154 |
| wrist_flex | +0.2198 |
| wrist_roll | -0.0879 |
| gripper | -0.3216 |

## Strategy comparison table

| strategy | clamp-free rate | avg clamp joints | GT L2 mean | GT L2 median | GT L2 p95 | GT L2 max | shoulder_lift mean±std (range) | elbow_flex mean±std (range) | avg # inference | max # inference | failure rate |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| single_random_sample | 10.0% | 1.400 | 4.773 | 4.123 | 9.031 | 11.115 | +6.269±3.064 (15.281) | -6.733±2.270 (10.549) | 1.00 | 1 | 90.0% |
| 3_sample_median | 4.2% | 1.589 | 3.776 | 3.374 | 7.008 | 8.450 | +6.076±1.547 (6.868) | -6.773±1.195 (6.384) | 3.00 | 3 | 95.8% |
| 5_sample_median | 1.6% | 1.721 | 3.555 | 3.320 | 5.455 | 7.581 | +5.935±1.139 (5.832) | -6.819±0.810 (4.744) | 5.00 | 5 | 98.4% |
| safety_pass_resampling_max3 | 29.0% | 1.106 | 2.882 | 2.421 | 5.425 | 5.425 | +3.924±2.152 (6.299) | -5.121±0.982 (4.227) | 2.70 | 3 | 71.0% |
| safety_pass_resampling_max5 | 45.1% | 0.860 | 3.018 | 2.421 | 5.425 | 5.425 | +3.362±2.490 (6.299) | -5.033±0.895 (4.227) | 4.03 | 5 | 54.9% |

## Baseline-only: fix one good seed (NOT evaluated as a deployment candidate)

Selection rule: min GT L2 among clamp-free seeds in the 0..19 sweep (retrospective, GT-informed - NOT reproducible at deploy time). Chosen seed: **2**.

| metric | value |
|---|---:|
| clamp-free | yes |
| avg clamp joint count | 0 |
| GT L2 (deg) | 1.485 |
| shoulder_lift (deg) | +3.992 |
| elbow_flex (deg) | -4.354 |

> Degenerate stats (std=0, range=0) are expected and diagnostic, not a virtue: this reflects a single observation (one seed x one fixed Shadow reference state), not a distribution. It says nothing about what this seed would produce against a different observation.

## Additional analysis

### Does 3-median / 5-median actually reduce clamp rate vs single sample?

- single-sample clamp-free rate: 10.0%
- 3-median clamp-free rate: 4.2% (not higher than single-sample)
- 5-median clamp-free rate: 1.6% (not higher than single-sample)

### Does median also improve GT L2?

- single-sample mean GT L2: 4.773 deg
- 3-median mean GT L2: 3.776 deg (improves)
- 5-median mean GT L2: 3.555 deg (improves)

### Resampling: max-3 vs max-5 success-rate difference

- max-3 clamp-free (success) rate: 29.0% (avg 2.70 inference calls, failure rate 71.0%)
- max-5 clamp-free (success) rate: 45.1% (avg 4.03 inference calls, failure rate 54.9%)
- delta (max5 - max3): +16.1 percentage points

> Note: max-5 mean GT L2 (3.018 deg) is *slightly worse* than max-3 (2.882 deg), even though max-5 succeeds (clamp-free) more often. This is not a bug: the stopping rule never looks at GT, so the *extra* successes cap=5 finds beyond what cap=3 already found are exactly the additional clamp-free seeds only reachable with more attempts (e.g. seed 12 in this sweep - clamp-free but a comparatively large GT L2) - a direct instance of the 'clamp-free does not imply low GT error' finding below, not evidence that more resampling attempts make actions worse.

### Clamp-free does not imply low GT error

> seed 12 is clamp-free (0 joints clamp) but has GT L2=5.425 deg, worse than 13/18 clamped seeds. Clamp-free selection (the only signal available at real deploy time, since GT is unknown) does not guarantee low GT error - it only guarantees the action was not truncated by the Safety Gate.

- clamp-free seeds [2, 12], GT L2 values [1.485, 5.425] (mean 3.455)
- clamped seeds mean GT L2: 4.919
- **Implication for deployment:** since clamp status is the only signal available at inference time (GT is unknown), a strategy that optimizes purely for clamp-free status (e.g. safety-pass resampling) reduces Safety-Gate intervention but offers **no guarantee** on trajectory accuracy. It is a safety/stability mitigation, not an accuracy mitigation.

## Recommendation

**Primary recommendation: safety-pass resampling, cap=5** (clamp-free rate 45.1%, failure rate 54.9%, avg 4.03 inference calls, GT L2 mean 3.018 deg). It directly targets the deployment failure mode this sweep was run to investigate (Safety Gate truncating first-action joints), it is the only strategy of the four whose stopping rule is deploy-time-realistic (uses only clamp status, never GT), and its inference cost is adaptive (only pays for extra calls when the first draw actually clamps) rather than fixed.

**Secondary / fallback recommendation: safety-pass resampling, cap=3** (clamp-free rate 29.0%, avg 2.70 calls) if the extra latency budget for a 5th inference call is not available on the deployment hardware - it captures most of the benefit of cap=5 at lower worst-case latency.

3-sample and 5-sample median are **not** recommended as the primary mitigation: median clamp-free rate (3-median 4.2%, 5-median 1.6%) does not clearly dominate single-sample (10.0%) or resampling, while always paying the fixed 3x/5x inference cost regardless of whether the first draw was already clamp-free - resampling only pays that cost when needed.

The 'fix one good seed' baseline (seed 2, clamp-free, GT L2 1.485 deg here) is **not recommended for deployment**: it was selected using GT that will not be available at deploy time, it reflects a single observation against one fixed Shadow reference state, and this whole sweep's own premise - that SmolVLA's flow-matching noise draw materially shifts the first action - means nothing guarantees that seed's favorable behavior transfers to a different observation, task, or checkpoint. It remains useful only as a debug/regression reference, not as a runtime policy.

## Next step: T01-T10 full Shadow validation

Carry safety-pass resampling (cap=5, fallback cap=3) into the T01-T10 full Shadow verification pass as the candidate mitigation, but re-run this same offline analysis *per task* once each task has its own seed sweep - this T01-only result (clamp-free 45.1% at cap=5) should not be assumed to hold for T02-T10 without their own 20-seed sweeps, since clamp behavior is a function of each task's own first-action delta distribution relative to the (task-independent) Safety Gate thresholds. Concretely: (1) run `sweep_grid35_first_action_seed.py` (or equivalent) for each of T02..T10 against its own Shadow reference observation; (2) re-run this script per task; (3) only adopt a single fixed cap/strategy repo-wide once its clamp-free rate and GT L2 are confirmed acceptable across all 10 tasks, not just T01.

