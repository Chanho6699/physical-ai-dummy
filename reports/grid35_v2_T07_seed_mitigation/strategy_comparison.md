# Grid35 V2 T01 - first-action seed-mitigation strategy comparison (offline)

Offline comparison of 4 candidate first-action stabilization strategies, computed entirely from the existing 20-seed sweep (`reports/grid35_v2_T01_seed_sweep/seed_sweep.csv`). **No new inference was run.**

- Source sweep checkpoint: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model`
- Reference Shadow observation: `T07` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T07/shadow_synthetic_T07.json`)
- Task: `Pick up the cube and place it in the target area.`
- Nearest-training-demo match: episode 32, frame 0
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
| shoulder_pan | -1.0549 |
| shoulder_lift | +0.2637 |
| elbow_flex | -1.8901 |
| wrist_flex | -0.8352 |
| wrist_roll | +0.0879 |
| gripper | -0.1106 |

## Strategy comparison table

| strategy | clamp-free rate | avg clamp joints | GT L2 mean | GT L2 median | GT L2 p95 | GT L2 max | shoulder_lift mean±std (range) | elbow_flex mean±std (range) | avg # inference | max # inference | failure rate |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| single_random_sample | 20.0% | 1.050 | 7.987 | 7.098 | 12.476 | 15.299 | +6.988±3.110 (15.391) | -4.533±2.280 (10.711) | 1.00 | 1 | 80.0% |
| 3_sample_median | 12.3% | 1.061 | 7.464 | 7.041 | 11.170 | 11.979 | +6.809±1.584 (6.972) | -4.611±1.186 (6.387) | 3.00 | 3 | 87.7% |
| 5_sample_median | 6.9% | 1.059 | 7.302 | 7.035 | 9.344 | 11.449 | +6.678±1.152 (6.181) | -4.672±0.807 (4.517) | 5.00 | 5 | 93.1% |
| safety_pass_resampling_max3 | 51.7% | 0.634 | 5.380 | 5.939 | 6.961 | 7.050 | +4.163±1.858 (5.705) | -3.893±1.540 (6.643) | 2.42 | 3 | 48.3% |
| safety_pass_resampling_max5 | 72.5% | 0.360 | 5.053 | 5.137 | 6.561 | 7.050 | +3.759±2.073 (5.705) | -3.720±1.330 (6.643) | 3.27 | 5 | 27.5% |

## Baseline-only: fix one good seed (NOT evaluated as a deployment candidate)

Selection rule: min GT L2 among clamp-free seeds in the 0..19 sweep (retrospective, GT-informed - NOT reproducible at deploy time). Chosen seed: **12**.

| metric | value |
|---|---:|
| clamp-free | yes |
| avg clamp joint count | 0 |
| GT L2 (deg) | 2.694 |
| shoulder_lift (deg) | -0.545 |
| elbow_flex (deg) | -3.095 |

> Degenerate stats (std=0, range=0) are expected and diagnostic, not a virtue: this reflects a single observation (one seed x one fixed Shadow reference state), not a distribution. It says nothing about what this seed would produce against a different observation.

## Additional analysis

### Does 3-median / 5-median actually reduce clamp rate vs single sample?

- single-sample clamp-free rate: 20.0%
- 3-median clamp-free rate: 12.3% (not higher than single-sample)
- 5-median clamp-free rate: 6.9% (not higher than single-sample)

### Does median also improve GT L2?

- single-sample mean GT L2: 7.987 deg
- 3-median mean GT L2: 7.464 deg (improves)
- 5-median mean GT L2: 7.302 deg (improves)

### Resampling: max-3 vs max-5 success-rate difference

- max-3 clamp-free (success) rate: 51.7% (avg 2.42 inference calls, failure rate 48.3%)
- max-5 clamp-free (success) rate: 72.5% (avg 3.27 inference calls, failure rate 27.5%)
- delta (max5 - max3): +20.8 percentage points

### Clamp-free does not imply low GT error

> seed 11 is clamp-free (0 joints clamp) but has GT L2=5.942 deg, worse than 0/16 clamped seeds. Clamp-free selection (the only signal available at real deploy time, since GT is unknown) does not guarantee low GT error - it only guarantees the action was not truncated by the Safety Gate.

- clamp-free seeds [12, 2, 6, 11], GT L2 values [2.694, 4.759, 5.137, 5.942] (mean 4.633)
- clamped seeds mean GT L2: 8.826
- **Implication for deployment:** since clamp status is the only signal available at inference time (GT is unknown), a strategy that optimizes purely for clamp-free status (e.g. safety-pass resampling) reduces Safety-Gate intervention but offers **no guarantee** on trajectory accuracy. It is a safety/stability mitigation, not an accuracy mitigation.

## Recommendation

**Primary recommendation: safety-pass resampling, cap=5** (clamp-free rate 72.5%, failure rate 27.5%, avg 3.27 inference calls, GT L2 mean 5.053 deg). It directly targets the deployment failure mode this sweep was run to investigate (Safety Gate truncating first-action joints), it is the only strategy of the four whose stopping rule is deploy-time-realistic (uses only clamp status, never GT), and its inference cost is adaptive (only pays for extra calls when the first draw actually clamps) rather than fixed.

**Secondary / fallback recommendation: safety-pass resampling, cap=3** (clamp-free rate 51.7%, avg 2.42 calls) if the extra latency budget for a 5th inference call is not available on the deployment hardware - it captures most of the benefit of cap=5 at lower worst-case latency.

3-sample and 5-sample median are **not** recommended as the primary mitigation: median clamp-free rate (3-median 12.3%, 5-median 6.9%) does not clearly dominate single-sample (20.0%) or resampling, while always paying the fixed 3x/5x inference cost regardless of whether the first draw was already clamp-free - resampling only pays that cost when needed.

The 'fix one good seed' baseline (seed 12, clamp-free, GT L2 2.694 deg here) is **not recommended for deployment**: it was selected using GT that will not be available at deploy time, it reflects a single observation against one fixed Shadow reference state, and this whole sweep's own premise - that SmolVLA's flow-matching noise draw materially shifts the first action - means nothing guarantees that seed's favorable behavior transfers to a different observation, task, or checkpoint. It remains useful only as a debug/regression reference, not as a runtime policy.

## Next step: T01-T10 full Shadow validation

Carry safety-pass resampling (cap=5, fallback cap=3) into the T01-T10 full Shadow verification pass as the candidate mitigation, but re-run this same offline analysis *per task* once each task has its own seed sweep - this T01-only result (clamp-free 72.5% at cap=5) should not be assumed to hold for T02-T10 without their own 20-seed sweeps, since clamp behavior is a function of each task's own first-action delta distribution relative to the (task-independent) Safety Gate thresholds. Concretely: (1) run `sweep_grid35_first_action_seed.py` (or equivalent) for each of T02..T10 against its own Shadow reference observation; (2) re-run this script per task; (3) only adopt a single fixed cap/strategy repo-wide once its clamp-free rate and GT L2 are confirmed acceptable across all 10 tasks, not just T01.

