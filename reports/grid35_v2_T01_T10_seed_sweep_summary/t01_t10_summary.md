# Grid35 V2 clean SmolVLA 7.5k - T01-T10 seed-sweep + mitigation generalization summary (synthetic offline-proxy)

Aggregates the per-scene 20-seed (0..19) sweeps and offline mitigation-strategy comparisons for all 10 Shadow scenes against one frozen checkpoint (`outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model`). **No training, no Safety Gate threshold changes, no robot writes.**

## Scene data provenance

| scene | source |
|---|---|
| T01 | **real SO-101 hardware Shadow capture** |
| T02 | synthetic offline proxy (held-out `midpoint_test10` dataset, frame 0) |
| T03 | synthetic offline proxy (held-out `midpoint_test10` dataset, frame 0) |
| T04 | synthetic offline proxy (held-out `midpoint_test10` dataset, frame 0) |
| T05 | synthetic offline proxy (held-out `midpoint_test10` dataset, frame 0) |
| T06 | synthetic offline proxy (held-out `midpoint_test10` dataset, frame 0) |
| T07 | synthetic offline proxy (held-out `midpoint_test10` dataset, frame 0) |
| T08 | synthetic offline proxy (held-out `midpoint_test10` dataset, frame 0) |
| T09 | synthetic offline proxy (held-out `midpoint_test10` dataset, frame 0) |
| T10 | synthetic offline proxy (held-out `midpoint_test10` dataset, frame 0) |

> T02-T10 could not be captured live at the time this synthetic summary was built: that analysis session had no SO-101 hardware attached (no `/dev/serial/by-id/*`, no `/dev/video*`). Per explicit user decision (2026-08-08), T02-T10 instead use frame_index=0 (state + workspace/wrist image) of episodes 1..9 of the held-out `data/so101_cube_xy_midpoint_test10_v2_clean` dataset (never used in training), sequentially mapped episode N -> T0(N+1) - 9 genuinely distinct held-out grid positions. See `scripts/build_synthetic_midpoint_shadow_reports.py` for full detail. **Real hardware captures for T02-T10 now exist** (see `reports/grid35_v2_T01_T10_actual_seed_sweep_summary/`, built with `--variant actual` of this same script) - that is the actual-data policy-decision reference; this synthetic summary is retained for its original purpose (probing generalization across spatially distinct positions, which the actual T02-T10 captures turned out not to provide - see the synthetic-vs-actual comparison report for the full discussion).

## Main comparison table

| scene | single clamp-free | resample<=3 success | resample<=5 success | shoulder_lift clamp rate | elbow_flex clamp rate | single GT L2 (deg) | resample5 GT L2 (deg) | resample5 avg #infer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T01 | 20.0% | 51.5% | 71.9% | 45% | 75% | 4.62 | 3.31 | 3.28 |
| T02 | 55.0% | 92.7% | 99.2% | 45% | 5% | 5.94 | 4.08 | 1.74 |
| T03 | 55.0% | 92.7% | 99.2% | 45% | 15% | 6.49 | 4.50 | 1.74 |
| T04 | 45.0% | 85.7% | 97.0% | 50% | 30% | 6.85 | 4.62 | 2.04 |
| T05 | 55.0% | 92.7% | 99.2% | 45% | 10% | 6.68 | 4.76 | 1.74 |
| T06 | 45.0% | 85.8% | 97.1% | 55% | 10% | 6.60 | 4.49 | 2.04 |
| T07 | 20.0% | 51.7% | 72.5% | 75% | 30% | 7.99 | 5.05 | 3.27 |
| T08 | 15.0% | 41.2% | 60.7% | 85% | 30% | 8.50 | 5.11 | 3.63 |
| T09 | 30.0% | 68.7% | 87.3% | 70% | 20% | 7.24 | 4.71 | 2.69 |
| T10 | 55.0% | 92.7% | 99.2% | 45% | 10% | 6.27 | 4.33 | 1.74 |
| **AVG (T01-T10)** | **39.5%** | **75.5%** | **88.3%** | **56%** | **24%** | **6.72** | **4.50** | **2.39** |

(median3/median5 strategies are reported in the per-scene CSV/JSON for completeness only - not shown here since T01 already found median unreliable, and here median clamp-free rate falls below single-sample's in T01/T06/T07/T08/T09.)

## Per-joint clamp-rate trend (single-sample, all 6 joints, averaged across T01-T10)

| joint | avg clamp rate | worst scene | worst scene clamp rate |
|---|---:|---|---:|
| shoulder_pan | 0% | T10 | 0% |
| shoulder_lift | 56% | T08 | 85% |
| elbow_flex | 24% | T01 | 75% |
| wrist_flex | 0% | T10 | 0% |
| wrist_roll | 0% | T10 | 0% |
| gripper | 0% | T10 | 0% |

## Best / worst scenes

- single-sample clamp-free: worst = **T08** (15.0%), best = **T02** (55.0%)
- resample<=3: worst = **T08** (41.2%), best = **T02** (92.7%)
- resample<=5: worst = **T08** (60.7%), best = **T02** (99.2%)

## Conclusions

### Q1: Does T01's sampling-noise problem repeat in T02-T10?

**Yes, it repeats across all 10 scenes.** Every scene (T01-T10) shows the same pattern T01 first surfaced: re-running the identical frozen checkpoint against the identical fixed observation, varying only the flow-matching RNG seed (0..19), swings chunk-index-0 between WOULD_CLAMP and clean. Single-sample clamp-free rate ranges from 15% (T08) to 55% (T02) - never close to 0% or 100%, i.e. never a scene where the seed choice stops mattering. 10/10 scenes have single-sample clamp-free rate below 60%. This is a property of the checkpoint's flow-matching noise sensitivity at this training step (007500), not an artifact specific to T01's particular reference state.

### Q2: Does safety-pass resampling's benefit generalize across scenes?

**Yes, directionally, on every scene** - safety-pass resampling raises the clamp-free rate over the single-sample baseline in all 10/10 scenes at both cap=3 and cap=5, and resample5 clamp-free rate is always >= resample3's (monotonic improvement with cap, as expected since cap=5 draws are a strict superset of cap=3's same permutation prefix). Average clamp-free rate: single 39.5% -> resample3 75.5% -> resample5 88.3%. However the *magnitude* of the benefit is scene-dependent: 6/10 scenes reach >=95% clamp-free at cap=5, but 3/10 stay below 80% (worst: T08 60.7%, T01 71.9%). GT L2 also improves alongside clamp-free rate in every scene (resample5 mean L2 < single-sample mean L2 in all 10 scenes), consistent with T01. Per-joint, shoulder_lift (avg clamp rate 56%) and elbow_flex (avg clamp rate 24%) remain the two joints that clamp by far the most often across every scene, matching T01's finding - the mitigation generalizes to the same failure mode, not a T01-specific one. The 3/5-sample median strategy does **not** generalize as a fix: in T06/T07/T08/T09 it actually *lowers* clamp-free rate below the single-sample baseline (same qualitative failure T01 already flagged) - retained here only as a reference point, not a candidate.

### Q3: Is resample<=5 justified as the real Shadow-runtime mitigation candidate?

**Conditionally, not unconditionally.** Resample<=5 is a clear, consistent improvement over both single-sample and median on every one of the 10 scenes tested, its stopping rule is deploy-time-realistic (WOULD_CLAMP status only, no GT), and its cost is adaptive (avg 2.39 inference calls across scenes, not a fixed 5x tax). That supports adopting it as the Shadow-runtime mitigation candidate. But the residual failure rate at cap=5 is not uniformly small: T08 39.3%, T01 28.1% still fail to find a clamp-free draw within 5 attempts, meaning on those scenes the Safety Gate would still end up clipping roughly 1 in 3-to-2.5 real runs even with resampling active. Before treating resample<=5 as sufficient on its own, either (a) raise the cap for scenes with low observed clamp-free base rates, or (b) treat resample<=5's residual failure rate as an accepted, monitored Safety-Gate-clip rate rather than a solved problem. It should not be adopted repo-wide as a single fixed cap without per-scene validation - the cap=3 vs cap=5 gap itself varies by scene (see the comparison table): a cap tuned to T02/T03/T04/T05/T06/T10's easy behavior (all >=95% clamp-free at cap=5) would under-serve T08/T01/T07 (all <80% clamp-free at cap=5).

### Q4: Do any scenes need policy retraining / checkpoint comparison instead?

**Yes - T08/T07 in particular** (plus T01, which already triggered this whole investigation) have a single-sample clamp-free rate at or below 20% - a materially thinner safe-seed pool than the other scenes (T08 15%, T01 20%, T07 20%, vs the T01-T10 average of 39.5%), and resample5 still leaves T08 39.3%, T01 28.1%, T07 27.5% residual failure rate respectively. The signal is not uniform across this group, though: T08/T07 also have above-average checkpoint GT-L2 error (T08 8.50 deg, T07 7.99 deg vs the T01-T10 average of 6.72 deg) - i.e. for T08/T07, resampling more is fighting a checkpoint that is *also* less accurate at that scene geometry, not just noisier, which is the stronger checkpoint/training-data-review signal (does the training grid have thin coverage there?). T01 is the counterexample: its GT-L2 (4.62 deg) is *below* the T01-T10 average despite the thin safe-seed pool, so a thin pool alone does not always imply a systematically-off policy - only when it co-occurs with above-average GT-L2 (as for T08/T07) is checkpoint/training-data review clearly warranted over pure sampling-based mitigation.

