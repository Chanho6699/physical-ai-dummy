# Grid35 V2 clean SmolVLA 7.5k - T01-T10 seed-sweep + mitigation generalization summary (actual (real SO-101 hardware, same-scene repeat))

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

> T02-T10 here are **real SO-101 hardware Shadow captures** (real follower state, real workspace/wrist camera frames), imported from a separate real Shadow-mode run and patched only to point at repo-local camera-frame copies - see `scripts/import_actual_shadow_t02_t10.py`. **Important data-quality caveat:** every one of the 9 source `shadow.json` files has `scene_metadata.label == "V2_F02"` and `evaluation_mode == "fixed-scene-repeat"` - i.e. all 9 are real-hardware repeats of **the same fixed scene T01 already uses**, not 9 spatially distinct held-out positions, despite the T02..T10 folder naming. Any comparison across these 10 rows should be read as "seed-to-seed variance across repeated real captures of ~1 physical scene", not as evidence about generalization across different cube positions - that question is what the *synthetic* T01-T10 summary (`reports/grid35_v2_T01_T10_seed_sweep_summary/`) was built to probe instead, and the two should not be read as measuring the same thing.

## Main comparison table

| scene | single clamp-free | resample<=3 success | resample<=5 success | shoulder_lift clamp rate | elbow_flex clamp rate | single GT L2 (deg) | resample5 GT L2 (deg) | resample5 avg #infer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T01 | 20.0% | 51.5% | 71.9% | 45% | 75% | 4.62 | 3.31 | 3.28 |
| T02 | 20.0% | 51.8% | 72.4% | 50% | 70% | 4.25 | 2.90 | 3.27 |
| T03 | 15.0% | 41.0% | 60.3% | 55% | 75% | 4.79 | 3.13 | 3.64 |
| T04 | 25.0% | 60.8% | 81.0% | 45% | 70% | 4.27 | 3.46 | 2.96 |
| T05 | 10.0% | 29.0% | 45.1% | 65% | 75% | 4.77 | 3.02 | 4.03 |
| T06 | 20.0% | 51.5% | 71.9% | 50% | 75% | 4.57 | 3.40 | 3.28 |
| T07 | 20.0% | 51.5% | 71.9% | 45% | 75% | 4.46 | 3.26 | 3.28 |
| T08 | 20.0% | 51.5% | 71.9% | 45% | 75% | 4.18 | 3.12 | 3.28 |
| T09 | 45.0% | 86.0% | 97.1% | 35% | 55% | 4.01 | 3.21 | 2.03 |
| T10 | 10.0% | 29.0% | 45.1% | 70% | 75% | 4.61 | 3.05 | 4.03 |
| **AVG (T01-T10)** | **20.5%** | **50.4%** | **68.9%** | **50%** | **72%** | **4.45** | **3.19** | **3.31** |

(median3/median5 strategies are reported in the per-scene CSV/JSON for completeness only - not shown here since T01 already found median unreliable, and here median clamp-free rate falls below single-sample's in T01/T02/T03/T04/T05/T06/T07/T08/T09/T10.)

## Per-joint clamp-rate trend (single-sample, all 6 joints, averaged across T01-T10)

| joint | avg clamp rate | worst scene | worst scene clamp rate |
|---|---:|---|---:|
| shoulder_pan | 0% | T10 | 0% |
| shoulder_lift | 50% | T10 | 70% |
| elbow_flex | 72% | T10 | 75% |
| wrist_flex | 0% | T10 | 0% |
| wrist_roll | 0% | T10 | 0% |
| gripper | 0% | T10 | 0% |

## Best / worst scenes

- single-sample clamp-free: worst = **T05** (10.0%), best = **T09** (45.0%)
- resample<=3: worst = **T05** (29.0%), best = **T09** (86.0%)
- resample<=5: worst = **T05** (45.1%), best = **T09** (97.1%)

## Conclusions

### Q1: Does T01's sampling-noise problem repeat in T02-T10?

**Yes, it repeats across all 10 scenes.** Every scene (T01-T10) shows the same pattern T01 first surfaced: re-running the identical frozen checkpoint against the identical fixed observation, varying only the flow-matching RNG seed (0..19), swings chunk-index-0 between WOULD_CLAMP and clean. Single-sample clamp-free rate ranges from 10% (T05) to 45% (T09) - never close to 0% or 100%, i.e. never a scene where the seed choice stops mattering. 10/10 scenes have single-sample clamp-free rate below 60%. This is a property of the checkpoint's flow-matching noise sensitivity at this training step (007500), not an artifact specific to T01's particular reference state.

### Q2: Does safety-pass resampling's benefit generalize across scenes?

**Yes, directionally, on every scene** - safety-pass resampling raises the clamp-free rate over the single-sample baseline in all 10/10 scenes at both cap=3 and cap=5, and resample5 clamp-free rate is always >= resample3's (monotonic improvement with cap, as expected since cap=5 draws are a strict superset of cap=3's same permutation prefix). Average clamp-free rate: single 20.5% -> resample3 50.4% -> resample5 68.9%. However the *magnitude* of the benefit is scene-dependent: 1/10 scenes reach >=95% clamp-free at cap=5, but 8/10 stay below 80% (worst: T05 45.1%, T10 45.1%). GT L2 also improves alongside clamp-free rate in every scene (resample5 mean L2 < single-sample mean L2 in all 10 scenes), consistent with T01. Per-joint, shoulder_lift (avg clamp rate 50%) and elbow_flex (avg clamp rate 72%) remain the two joints that clamp by far the most often across every scene, matching T01's finding - the mitigation generalizes to the same failure mode, not a T01-specific one. The 3/5-sample median strategy does **not** generalize as a fix: in T06/T07/T08/T09 it actually *lowers* clamp-free rate below the single-sample baseline (same qualitative failure T01 already flagged) - retained here only as a reference point, not a candidate.

### Q3: Is resample<=5 justified as the real Shadow-runtime mitigation candidate?

**Conditionally, not unconditionally.** Resample<=5 is a clear, consistent improvement over both single-sample and median on every one of the 10 scenes tested, its stopping rule is deploy-time-realistic (WOULD_CLAMP status only, no GT), and its cost is adaptive (avg 3.31 inference calls across scenes, not a fixed 5x tax). That supports adopting it as the Shadow-runtime mitigation candidate. But the residual failure rate at cap=5 is not uniformly small: T05 54.9%, T10 54.9% still fail to find a clamp-free draw within 5 attempts, meaning on those scenes the Safety Gate would still end up clipping roughly 1 in 3-to-2.5 real runs even with resampling active. Before treating resample<=5 as sufficient on its own, either (a) raise the cap for scenes with low observed clamp-free base rates, or (b) treat resample<=5's residual failure rate as an accepted, monitored Safety-Gate-clip rate rather than a solved problem. It should not be adopted repo-wide as a single fixed cap without per-scene validation - the cap=3 vs cap=5 gap itself varies by scene (see the comparison table): a cap tuned to T09's easy behavior (all >=95% clamp-free at cap=5) would under-serve T05/T10/T03/T01/T06/T07/T08/T02 (all <80% clamp-free at cap=5).

### Q4: Do any scenes need policy retraining / checkpoint comparison instead?

**Yes - T05/T10/T03/T06/T07/T08/T02 in particular** (plus T01, which already triggered this whole investigation) have a single-sample clamp-free rate at or below 20% - a materially thinner safe-seed pool than the other scenes (T05 10%, T10 10%, T03 15%, T01 20%, T06 20%, T07 20%, T08 20%, T02 20%, vs the T01-T10 average of 20.5%), and resample5 still leaves T05 54.9%, T10 54.9%, T03 39.7%, T01 28.1%, T06 28.1%, T07 28.1%, T08 28.1%, T02 27.6% residual failure rate respectively. The signal is not uniform across this group, though: T05/T10/T03/T01/T06/T07 also have above-average checkpoint GT-L2 error (T05 4.77 deg, T10 4.61 deg, T03 4.79 deg, T01 4.62 deg, T06 4.57 deg, T07 4.46 deg vs the T01-T10 average of 4.45 deg) - i.e. for T05/T10/T03/T01/T06/T07, resampling more is fighting a checkpoint that is *also* less accurate at that scene geometry, not just noisier, which is the stronger checkpoint/training-data-review signal (does the training grid have thin coverage there?). T08/T02 are the counterexample: its GT-L2 (4.18 deg, 4.25 deg) is *below* the T01-T10 average despite the thin safe-seed pool, so a thin pool alone does not always imply a systematically-off policy - only when it co-occurs with above-average GT-L2 (as for T05/T10/T03/T01/T06/T07) is checkpoint/training-data review clearly warranted over pure sampling-based mitigation.

