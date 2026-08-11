# Grid35 V2 T01 - first-action seed-mitigation strategy comparison (offline)

Offline comparison of 4 candidate first-action stabilization strategies, computed entirely from the existing 20-seed sweep (`reports/grid35_v2_T01_seed_sweep/seed_sweep.csv`). **No new inference was run.**

- Source sweep checkpoint: `outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model`
- Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
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
| single_random_sample | 20.0% | 1.200 | 4.617 | 3.976 | 8.671 | 10.099 | +5.127±3.013 (14.965) | -7.080±2.303 (10.638) | 1.00 | 1 | 80.0% |
| 3_sample_median | 11.2% | 1.281 | 3.599 | 3.350 | 6.790 | 8.094 | +4.942±1.523 (6.822) | -7.106±1.220 (6.494) | 3.00 | 3 | 88.8% |
| 5_sample_median | 6.2% | 1.323 | 3.387 | 3.284 | 5.112 | 7.173 | +4.808±1.109 (5.838) | -7.149±0.826 (4.950) | 5.00 | 5 | 93.8% |
| safety_pass_resampling_max3 | 51.5% | 0.732 | 3.104 | 2.494 | 6.505 | 6.505 | +3.386±2.300 (7.343) | -4.800±1.286 (3.862) | 2.43 | 3 | 48.5% |
| safety_pass_resampling_max5 | 71.9% | 0.425 | 3.308 | 2.494 | 6.505 | 6.505 | +2.960±2.538 (7.343) | -4.468±1.367 (3.862) | 3.28 | 5 | 28.1% |

## Baseline-only: fix one good seed (NOT evaluated as a deployment candidate)

Selection rule: min GT L2 among clamp-free seeds in the 0..19 sweep (retrospective, GT-informed - NOT reproducible at deploy time). Chosen seed: **2**.

| metric | value |
|---|---:|
| clamp-free | yes |
| avg clamp joint count | 0 |
| GT L2 (deg) | 1.766 |
| shoulder_lift (deg) | +2.918 |
| elbow_flex (deg) | -4.718 |

> Degenerate stats (std=0, range=0) are expected and diagnostic, not a virtue: this reflects a single observation (one seed x one fixed Shadow reference state), not a distribution. It says nothing about what this seed would produce against a different observation.

## Additional analysis

### Does 3-median / 5-median actually reduce clamp rate vs single sample?

- single-sample clamp-free rate: 20.0%
- 3-median clamp-free rate: 11.2% (not higher than single-sample)
- 5-median clamp-free rate: 6.2% (not higher than single-sample)

### Does median also improve GT L2?

- single-sample mean GT L2: 4.617 deg
- 3-median mean GT L2: 3.599 deg (improves)
- 5-median mean GT L2: 3.387 deg (improves)

### Resampling: max-3 vs max-5 success-rate difference

- max-3 clamp-free (success) rate: 51.5% (avg 2.43 inference calls, failure rate 48.5%)
- max-5 clamp-free (success) rate: 71.9% (avg 3.28 inference calls, failure rate 28.1%)
- delta (max5 - max3): +20.4 percentage points

> Note: max-5 mean GT L2 (3.308 deg) is *slightly worse* than max-3 (3.104 deg), even though max-5 succeeds (clamp-free) more often. This is not a bug: the stopping rule never looks at GT, so the *extra* successes cap=5 finds beyond what cap=3 already found are exactly the additional clamp-free seeds only reachable with more attempts (e.g. seed 12 in this sweep - clamp-free but a comparatively large GT L2) - a direct instance of the 'clamp-free does not imply low GT error' finding below, not evidence that more resampling attempts make actions worse.

### Clamp-free does not imply low GT error

> seed 12 is clamp-free (0 joints clamp) but has GT L2=6.505 deg, worse than 12/16 clamped seeds. Clamp-free selection (the only signal available at real deploy time, since GT is unknown) does not guarantee low GT error - it only guarantees the action was not truncated by the Safety Gate.

- clamp-free seeds [2, 8, 19, 12], GT L2 values [1.766, 2.06, 3.986, 6.505] (mean 3.579)
- clamped seeds mean GT L2: 4.876
- **Implication for deployment:** since clamp status is the only signal available at inference time (GT is unknown), a strategy that optimizes purely for clamp-free status (e.g. safety-pass resampling) reduces Safety-Gate intervention but offers **no guarantee** on trajectory accuracy. It is a safety/stability mitigation, not an accuracy mitigation.

## Recommendation

**Primary recommendation: safety-pass resampling, cap=5** (clamp-free rate 71.9%, failure rate 28.1%, avg 3.28 inference calls, GT L2 mean 3.308 deg). It directly targets the deployment failure mode this sweep was run to investigate (Safety Gate truncating first-action joints), it is the only strategy of the four whose stopping rule is deploy-time-realistic (uses only clamp status, never GT), and its inference cost is adaptive (only pays for extra calls when the first draw actually clamps) rather than fixed.

**Secondary / fallback recommendation: safety-pass resampling, cap=3** (clamp-free rate 51.5%, avg 2.43 calls) if the extra latency budget for a 5th inference call is not available on the deployment hardware - it captures most of the benefit of cap=5 at lower worst-case latency.

3-sample and 5-sample median are **not** recommended as the primary mitigation: median clamp-free rate (3-median 11.2%, 5-median 6.2%) does not clearly dominate single-sample (20.0%) or resampling, while always paying the fixed 3x/5x inference cost regardless of whether the first draw was already clamp-free - resampling only pays that cost when needed.

The 'fix one good seed' baseline (seed 2, clamp-free, GT L2 1.766 deg here) is **not recommended for deployment**: it was selected using GT that will not be available at deploy time, it reflects a single observation against one fixed Shadow reference state, and this whole sweep's own premise - that SmolVLA's flow-matching noise draw materially shifts the first action - means nothing guarantees that seed's favorable behavior transfers to a different observation, task, or checkpoint. It remains useful only as a debug/regression reference, not as a runtime policy.

## Next step: T01-T10 full Shadow validation

Carry safety-pass resampling (cap=5, fallback cap=3) into the T01-T10 full Shadow verification pass as the candidate mitigation, but re-run this same offline analysis *per task* once each task has its own seed sweep - this T01-only result (clamp-free 71.9% at cap=5) should not be assumed to hold for T02-T10 without their own 20-seed sweeps, since clamp behavior is a function of each task's own first-action delta distribution relative to the (task-independent) Safety Gate thresholds. Concretely: (1) run `sweep_grid35_first_action_seed.py` (or equivalent) for each of T02..T10 against its own Shadow reference observation; (2) re-run this script per task; (3) only adopt a single fixed cap/strategy repo-wide once its clamp-free rate and GT L2 are confirmed acceptable across all 10 tasks, not just T01.

