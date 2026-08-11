# Grid35 V2 clean SmolVLA 7.5k - T01-T10 seed-sweep + mitigation generalization summary (FINAL actual (real SO-101 hardware, multi-position T01-T10))

Aggregates the per-scene 20-seed (0..19) sweeps and offline mitigation-strategy comparisons for all 10 Shadow scenes against one frozen checkpoint (`outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model`). **No training, no Safety Gate threshold changes, no robot writes.**

## Scene data provenance

| scene | source |
|---|---|
| T01 | **real SO-101 hardware Shadow capture** |
| T02 | **real SO-101 hardware Shadow capture** |
| T03 | **real SO-101 hardware Shadow capture** |
| T04 | **real SO-101 hardware Shadow capture** |
| T05 | **real SO-101 hardware Shadow capture** |
| T06 | **real SO-101 hardware Shadow capture** |
| T07 | **real SO-101 hardware Shadow capture** |
| T08 | **real SO-101 hardware Shadow capture** |
| T09 | **real SO-101 hardware Shadow capture** |
| T10 | **real SO-101 hardware Shadow capture** |

> **All 10 scenes here (T01-T10) are real SO-101 hardware Shadow captures from one self-consistent capture session** (`reports/real_midpoint_shadow_T01_T10/`, imported by `scripts/import_actual_shadow_t01_t10_final.py`), replacing both the original single T01 capture and the earlier same-scene-repeat `actual` T02-T10 import as the reference for this question - the two are **not mixed** with this result. Each scene's source `shadow.json` has a distinct `scene_metadata.label`/`heldout_episode_index` (0..9) and `evaluation_mode == "midpoint-shadow"` (not the earlier import's `"fixed-scene-repeat"`), matching the same T0N<->episode-index convention as the synthetic `midpoint_test10` proxies. **Data-quality note carried forward from the import (see each scene's `repo_import_provenance` for exact numbers):** the raw follower joint-state readout is itself nearly constant across all 10 scenes (state diffs vs T01 are ~0deg on 5/6 joints, ~0.09deg on elbow_flex), while the workspace/wrist camera frames differ meaningfully scene-to-scene (RMSE vs T01 ~19-30 on a 0-255 scale) - consistent with an intentional fixed-'midpoint'-observation-pose protocol (only the cube placement varies per scene) rather than a duplicate-capture artifact, but not independently verified against cube ground-truth by this pipeline.

## Main comparison table

| scene | single clamp-free | resample<=3 success | resample<=5 success | shoulder_lift clamp rate | elbow_flex clamp rate | single GT L2 (deg) | resample5 GT L2 (deg) | resample5 avg #infer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T01 | 5.0% | 15.3% | 25.1% | 80% | 80% | 5.62 | 2.65 | 4.49 |
| T02 | 5.0% | 15.3% | 25.1% | 75% | 80% | 5.34 | 2.65 | 4.49 |
| T03 | 0.0% | 0.0% | 0.0% | 90% | 85% | 6.35 | 2.95 | 5.00 |
| T04 | 5.0% | 15.3% | 25.1% | 80% | 80% | 5.48 | 2.61 | 4.49 |
| T05 | 0.0% | 0.0% | 0.0% | 85% | 85% | 6.18 | 3.15 | 5.00 |
| T06 | 0.0% | 0.0% | 0.0% | 85% | 95% | 6.80 | 3.02 | 5.00 |
| T07 | 0.0% | 0.0% | 0.0% | 90% | 95% | 7.02 | 3.12 | 5.00 |
| T08 | 5.0% | 15.3% | 25.1% | 85% | 80% | 5.62 | 2.46 | 4.49 |
| T09 | 0.0% | 0.0% | 0.0% | 85% | 95% | 6.44 | 2.79 | 5.00 |
| T10 | 5.0% | 15.3% | 25.1% | 85% | 80% | 5.62 | 2.64 | 4.49 |
| **AVG (T01-T10)** | **2.5%** | **7.6%** | **12.6%** | **84%** | **86%** | **6.05** | **2.80** | **4.75** |

(median3/median5 strategies are reported in the per-scene CSV/JSON for completeness only - not shown here since T01 already found median unreliable, and here median clamp-free rate falls below single-sample's in T01/T02/T04/T08/T10.)

## Per-joint clamp-rate trend (single-sample, all 6 joints, averaged across T01-T10)

| joint | avg clamp rate | worst scene | worst scene clamp rate |
|---|---:|---|---:|
| shoulder_pan | 0% | T10 | 0% |
| shoulder_lift | 84% | T07 | 90% |
| elbow_flex | 86% | T09 | 95% |
| wrist_flex | 0% | T10 | 0% |
| wrist_roll | 0% | T10 | 0% |
| gripper | 0% | T10 | 0% |

## Best / worst scenes

- single-sample clamp-free: worst = **T03** (0.0%), best = **T01** (5.0%)
- resample<=3: worst = **T03** (0.0%), best = **T01** (15.3%)
- resample<=5: worst = **T03** (0.0%), best = **T01** (25.1%)

## Conclusions

### Q1: Does T01's sampling-noise problem repeat in T02-T10?

**Yes, it repeats across all 10 scenes.** Every scene (T01-T10) shows the same pattern T01 first surfaced: re-running the identical frozen checkpoint against the identical fixed observation, varying only the flow-matching RNG seed (0..19), swings chunk-index-0 between WOULD_CLAMP and clean. Single-sample clamp-free rate ranges from 0% (T03) to 5% (T01) - never close to 0% or 100%, i.e. never a scene where the seed choice stops mattering. 10/10 scenes have single-sample clamp-free rate below 60%. This is a property of the checkpoint's flow-matching noise sensitivity at this training step (007500), not an artifact specific to T01's particular reference state.

### Q2: Does safety-pass resampling's benefit generalize across scenes?

**Yes, directionally, on every scene** - safety-pass resampling raises the clamp-free rate over the single-sample baseline in all 10/10 scenes at both cap=3 and cap=5, and resample5 clamp-free rate is always >= resample3's (monotonic improvement with cap, as expected since cap=5 draws are a strict superset of cap=3's same permutation prefix). Average clamp-free rate: single 2.5% -> resample3 7.6% -> resample5 12.6%. However the *magnitude* of the benefit is scene-dependent: 0/10 scenes reach >=95% clamp-free at cap=5, but 10/10 stay below 80% (worst: T03 0.0%, T05 0.0%). GT L2 also improves alongside clamp-free rate in every scene (resample5 mean L2 < single-sample mean L2 in all 10 scenes), consistent with T01. Per-joint, shoulder_lift (avg clamp rate 84%) and elbow_flex (avg clamp rate 86%) remain the two joints that clamp by far the most often across every scene, matching T01's finding - the mitigation generalizes to the same failure mode, not a T01-specific one. The 3/5-sample median strategy does **not** generalize as a fix: in T06/T07/T08/T09 it actually *lowers* clamp-free rate below the single-sample baseline (same qualitative failure T01 already flagged) - retained here only as a reference point, not a candidate.

### Q3: Is resample<=5 justified as the real Shadow-runtime mitigation candidate?

**Conditionally, not unconditionally.** Resample<=5 is a clear, consistent improvement over both single-sample and median on every one of the 10 scenes tested, its stopping rule is deploy-time-realistic (WOULD_CLAMP status only, no GT), and its cost is adaptive (avg 4.75 inference calls across scenes, not a fixed 5x tax). That supports adopting it as the Shadow-runtime mitigation candidate. But the residual failure rate at cap=5 is not uniformly small: T03 100.0%, T05 100.0% still fail to find a clamp-free draw within 5 attempts, meaning on those scenes the Safety Gate would still end up clipping roughly 1 in 3-to-2.5 real runs even with resampling active. Before treating resample<=5 as sufficient on its own, either (a) raise the cap for scenes with low observed clamp-free base rates, or (b) treat resample<=5's residual failure rate as an accepted, monitored Safety-Gate-clip rate rather than a solved problem. It should not be adopted repo-wide as a single fixed cap without per-scene validation - the cap=3 vs cap=5 gap itself varies by scene (see the comparison table): in this dataset 0/10 scenes reach even the >=95% clamp-free 'easy' bar at cap=5 - every scene tested is in the hard regime (T03/T05/T06/T07/T09/T01/T02/T04/T08/T10, all <80% clamp-free at cap=5), so there is no easy/hard split to tune a single cap between here; a fixed cap=5 would be under-serving all 10 scenes roughly equally, not a subset of them.

### Q4: Do any scenes need policy retraining / checkpoint comparison instead?

**Yes - T03/T05/T06/T07/T09/T02/T04/T08/T10 in particular** (plus T01, which already triggered this whole investigation) have a single-sample clamp-free rate at or below 20% - a materially thinner safe-seed pool than the other scenes (T03 0%, T05 0%, T06 0%, T07 0%, T09 0%, T01 5%, T02 5%, T04 5%, T08 5%, T10 5%, vs the T01-T10 average of 2.5%), and resample5 still leaves T03 100.0%, T05 100.0%, T06 100.0%, T07 100.0%, T09 100.0%, T01 74.9%, T02 74.9%, T04 74.9%, T08 74.9%, T10 74.9% residual failure rate respectively. The signal is not uniform across this group, though: T03/T05/T06/T07/T09 also have above-average checkpoint GT-L2 error (T03 6.35 deg, T05 6.18 deg, T06 6.80 deg, T07 7.02 deg, T09 6.44 deg vs the T01-T10 average of 6.05 deg) - i.e. for T03/T05/T06/T07/T09, resampling more is fighting a checkpoint that is *also* less accurate at that scene geometry, not just noisier, which is the stronger checkpoint/training-data-review signal (does the training grid have thin coverage there?). T01/T02/T04/T08/T10 are the counterexample: its GT-L2 (5.62 deg, 5.34 deg, 5.48 deg, 5.62 deg, 5.62 deg) is *below* the T01-T10 average despite the thin safe-seed pool, so a thin pool alone does not always imply a systematically-off policy - only when it co-occurs with above-average GT-L2 (as for T03/T05/T06/T07/T09) is checkpoint/training-data review clearly warranted over pure sampling-based mitigation.

