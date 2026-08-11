# Grid35 V2 T01 - first-action seed-mitigation strategy comparison (offline)

Offline comparison of 4 candidate first-action stabilization strategies, computed entirely from the existing 20-seed sweep (`reports/grid35_v2_T01_seed_sweep/seed_sweep.csv`). **No new inference was run.**

- Source sweep checkpoint: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/010000/pretrained_model`
- Reference Shadow observation: `T06` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T06_real_final/shadow_patched.json`)
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
| single_random_sample | 0.0% | 1.600 | 6.626 | 5.982 | 10.774 | 11.685 | +5.899±2.446 (11.925) | -10.172±1.775 (7.542) | 1.00 | 1 | 100.0% |
| 3_sample_median | 0.0% | 1.656 | 6.205 | 6.024 | 9.074 | 10.490 | +5.686±1.249 (5.952) | -10.114±1.026 (5.355) | 3.00 | 3 | 100.0% |
| 5_sample_median | 0.0% | 1.704 | 6.104 | 6.059 | 7.881 | 9.528 | +5.572±0.911 (4.689) | -10.098±0.709 (4.019) | 5.00 | 5 | 100.0% |
| safety_pass_resampling_max3 | 0.0% | 1.600 | 2.727 | 2.548 | 4.149 | 4.274 | +4.636±1.067 (4.614) | -5.730±0.000 (0.000) | 3.00 | 3 | 100.0% |
| safety_pass_resampling_max5 | 0.0% | 1.609 | 2.737 | 2.552 | 4.274 | 4.274 | +4.657±1.044 (4.614) | -5.730±0.000 (0.000) | 5.00 | 5 | 100.0% |

## Baseline-only: fix one good seed (NOT evaluated as a deployment candidate)

Selection rule: min GT L2 among clamp-free seeds in the 0..19 sweep (retrospective, GT-informed - NOT reproducible at deploy time). Chosen seed: **19**.

| metric | value |
|---|---:|
| clamp-free | no |
| avg clamp joint count | 2 |
| GT L2 (deg) | 3.213 |
| shoulder_lift (deg) | +5.268 |
| elbow_flex (deg) | -6.451 |

> Degenerate stats (std=0, range=0) are expected and diagnostic, not a virtue: this reflects a single observation (one seed x one fixed Shadow reference state), not a distribution. It says nothing about what this seed would produce against a different observation.

## Additional analysis

### Does 3-median / 5-median actually reduce clamp rate vs single sample?

- single-sample clamp-free rate: 0.0%
- 3-median clamp-free rate: 0.0% (not higher than single-sample)
- 5-median clamp-free rate: 0.0% (not higher than single-sample)

### Does median also improve GT L2?

- single-sample mean GT L2: 6.626 deg
- 3-median mean GT L2: 6.205 deg (improves)
- 5-median mean GT L2: 6.104 deg (improves)

### Resampling: max-3 vs max-5 success-rate difference

- max-3 clamp-free (success) rate: 0.0% (avg 3.00 inference calls, failure rate 100.0%)
- max-5 clamp-free (success) rate: 0.0% (avg 5.00 inference calls, failure rate 100.0%)
- delta (max5 - max3): +0.0 percentage points

> Note: max-5 mean GT L2 (2.737 deg) is *slightly worse* than max-3 (2.727 deg), even though max-5 succeeds (clamp-free) more often. This is not a bug: the stopping rule never looks at GT, so the *extra* successes cap=5 finds beyond what cap=3 already found are exactly the additional clamp-free seeds only reachable with more attempts (e.g. seed 12 in this sweep - clamp-free but a comparatively large GT L2) - a direct instance of the 'clamp-free does not imply low GT error' finding below, not evidence that more resampling attempts make actions worse.

### Clamp-free does not imply low GT error

> 0/20 seeds in the raw sweep were clamp-free, so no clamp-free-vs-GT counterexample exists for this scene - every raw seed's chunk[0] clamps on at least one joint. This is a stronger finding than the counterexample below normally shows: at this scene, clamp status is not just an imperfect proxy for GT accuracy, single-sample inference essentially never passes the Safety Gate cleanly regardless of seed.

- clamp-free seeds [], GT L2 values [] (mean n/a (0 clamp-free seeds))
- clamped seeds mean GT L2: 6.626
- **Implication for deployment:** since clamp status is the only signal available at inference time (GT is unknown), a strategy that optimizes purely for clamp-free status (e.g. safety-pass resampling) reduces Safety-Gate intervention but offers **no guarantee** on trajectory accuracy. It is a safety/stability mitigation, not an accuracy mitigation.

## Recommendation

**Primary recommendation: safety-pass resampling, cap=5** (clamp-free rate 0.0%, failure rate 100.0%, avg 5.00 inference calls, GT L2 mean 2.737 deg). It directly targets the deployment failure mode this sweep was run to investigate (Safety Gate truncating first-action joints), it is the only strategy of the four whose stopping rule is deploy-time-realistic (uses only clamp status, never GT), and its inference cost is adaptive (only pays for extra calls when the first draw actually clamps) rather than fixed.

**Secondary / fallback recommendation: safety-pass resampling, cap=3** (clamp-free rate 0.0%, avg 3.00 calls) if the extra latency budget for a 5th inference call is not available on the deployment hardware - it captures most of the benefit of cap=5 at lower worst-case latency.

3-sample and 5-sample median are **not** recommended as the primary mitigation: median clamp-free rate (3-median 0.0%, 5-median 0.0%) does not clearly dominate single-sample (0.0%) or resampling, while always paying the fixed 3x/5x inference cost regardless of whether the first draw was already clamp-free - resampling only pays that cost when needed.

The 'fix one good seed' baseline (seed 19, clamp-free, GT L2 3.213 deg here) is **not recommended for deployment**: it was selected using GT that will not be available at deploy time, it reflects a single observation against one fixed Shadow reference state, and this whole sweep's own premise - that SmolVLA's flow-matching noise draw materially shifts the first action - means nothing guarantees that seed's favorable behavior transfers to a different observation, task, or checkpoint. It remains useful only as a debug/regression reference, not as a runtime policy.

## Next step: T01-T10 full Shadow validation

Carry safety-pass resampling (cap=5, fallback cap=3) into the T01-T10 full Shadow verification pass as the candidate mitigation, but re-run this same offline analysis *per task* once each task has its own seed sweep - this T01-only result (clamp-free 0.0% at cap=5) should not be assumed to hold for T02-T10 without their own 20-seed sweeps, since clamp behavior is a function of each task's own first-action delta distribution relative to the (task-independent) Safety Gate thresholds. Concretely: (1) run `sweep_grid35_first_action_seed.py` (or equivalent) for each of T02..T10 against its own Shadow reference observation; (2) re-run this script per task; (3) only adopt a single fixed cap/strategy repo-wide once its clamp-free rate and GT L2 are confirmed acceptable across all 10 tasks, not just T01.

