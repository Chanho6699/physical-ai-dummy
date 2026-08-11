# Synthetic proxy vs actual (real hardware) T02-T10 - comparison, NOT averaged together

Per explicit user instruction, synthetic and actual T02-T10 results are never averaged together anywhere in this repo. **The actual T01-T10 results (`reports/grid35_v2_T01_T10_actual_seed_sweep_summary/`) are the reference for any real policy decision** - this report only quantifies how the two datasets differ and why.

## Data-quality caveat (read this before the table)

The *synthetic* T02-T10 scenes are 9 genuinely distinct held-out grid positions (`data/so101_cube_xy_midpoint_test10_v2_clean`, episodes 1-9). The *actual* T02-T10 captures turned out to be 9 real-hardware repeat captures of **the same fixed scene T01 already uses** (every source `shadow.json` has `scene_metadata.label == "V2_F02"`, `evaluation_mode == "fixed-scene-repeat"` - see `scripts/import_actual_shadow_t02_t10.py`). The two datasets were never sampling the same thing, so differences below reflect that mismatch in what was actually captured, not proxy modeling error.

## Per-scene diff table (actual minus synthetic)

| scene | synthetic single CF | actual single CF | diff | synthetic r5 | actual r5 | diff | synthetic shoulder_lift clamp | actual shoulder_lift clamp | diff | synthetic elbow_flex clamp | actual elbow_flex clamp | diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T02 | 55.0% | 20.0% | -35.0pp | 99.2% | 72.4% | -26.8pp | 45% | 50% | +5pp | 5% | 70% | +65pp |
| T03 | 55.0% | 15.0% | -40.0pp | 99.2% | 60.3% | -38.9pp | 45% | 55% | +10pp | 15% | 75% | +60pp |
| T04 | 45.0% | 25.0% | -20.0pp | 97.0% | 81.0% | -16.1pp | 50% | 45% | -5pp | 30% | 70% | +40pp |
| T05 | 55.0% | 10.0% | -45.0pp | 99.2% | 45.1% | -54.2pp | 45% | 65% | +20pp | 10% | 75% | +65pp |
| T06 | 45.0% | 20.0% | -25.0pp | 97.1% | 71.9% | -25.2pp | 55% | 50% | -5pp | 10% | 75% | +65pp |
| T07 | 20.0% | 20.0% | +0.0pp | 72.5% | 71.9% | -0.5pp | 75% | 45% | -30pp | 30% | 75% | +45pp |
| T08 | 15.0% | 20.0% | +5.0pp | 60.7% | 71.9% | +11.2pp | 85% | 45% | -40pp | 30% | 75% | +45pp |
| T09 | 30.0% | 45.0% | +15.0pp | 87.3% | 97.1% | +9.8pp | 70% | 35% | -35pp | 20% | 55% | +35pp |
| T10 | 55.0% | 10.0% | -45.0pp | 99.2% | 45.1% | -54.2pp | 45% | 70% | +25pp | 10% | 75% | +65pp |

(T01 omitted - identical real-hardware result in both summaries by construction, diff is trivially zero: single 20.0%, resample5 71.9%.)

## Judgment

### How well did the synthetic proxy reproduce the actual observation trend?

**Not in absolute magnitude - by design, not by failure.** Averaged over T02-T10: synthetic single-sample clamp-free 41.7% vs actual 20.6%; synthetic resample5 90.2% vs actual 68.5%. The actual numbers sit much closer to T01's own real-hardware numbers (single 20.0%, resample5 71.9%) than to the synthetic T02-T10 average - which is exactly what should happen once you know the 'actual T02-T10' captures are real-hardware repeats of T01's own scene, not 9 different positions: the synthetic proxy was modeling a *different, harder* experiment (spatial generalization across genuinely distinct held-out grid positions) than what the actual capture ended up providing (repeat-capture robustness on one already-characterized scene). It would be wrong to read the gap as 'the synthetic proxy method is unreliable' - no scene-matched ground truth exists yet to make that call, since the actual data doesn't cover the positions the synthetic data covered.

### Does the sampling-noise problem repeat across the actual captures?

**Yes.** All 9/9 actual T02-T10 captures show the same seed-to-seed swing between WOULD_CLAMP and clean chunk-index-0 actions that T01 first surfaced, now confirmed across 9 independent real-hardware capture sessions (not just T01's own single capture) - single-sample clamp-free rate ranges 10%-45% across the group, never near 0% or 100%. This is a meaningfully stronger form of evidence than the synthetic run gave for T01 alone: it shows the effect survives real camera/robot-repositioning noise across repeated real captures of the same scene, not just repeated *inference* on one fixed captured frame.

### Does resample<=5 improve consistently in the actual data?

**Yes, in direction, on all 9/9 actual scenes** - resample5 clamp-free rate exceeds single-sample on every actual T02-T10 capture (9/9 scenes improve), averaging 20.6% -> 68.5%. Magnitude varies by capture instance though: T09 reaches 97.1% while T05/T10 only reach 45.1% - since these are repeat captures of one scene, that spread is real-hardware capture-to-capture variance (camera framing/robot pose micro-differences interacting with the same flow-matching noise sensitivity), not evidence about how the mitigation performs on different scene geometries.

### Is resample<=5 justified as the Shadow-runtime candidate, on actual-data evidence?

**Real hardware data now supports the resampling *direction* robustly, but still cannot confirm resampling generalizes across different scene geometries** - and that gap matters for a runtime-adoption decision. The actual T02-T10 captures strengthen confidence that resample<=5's benefit is not an artifact of T01's one particular capture (it reproduces across 9 independently-captured real sessions of that scene), but because all 9 are the *same* physical scene, they provide no additional evidence about scenes with different cube positions - that evidence still only exists in the synthetic proxy data, which is not real hardware. Before adopting resample<=5 repo-wide as the Shadow runtime candidate, the genuinely missing piece is **real Shadow captures at spatially distinct cube positions** (i.e. actually moving the cube, not re-capturing the same spot) - neither dataset built so far provides that combination (real hardware AND spatially distinct).

### Which scene is still most fragile, and is it a policy/data problem?

**T05/T10** (tied) show the lowest resample5 success in the actual data (45.1%, single-sample 10.0%). But because every actual T02-T10 capture is the same physical scene as T01, this does **not** indicate T05/T10 are a weak *position* - it indicates that particular real capture session (camera framing/lighting/robot pose at that moment) landed in a harder region of the same underlying seed-noise distribution T01 already has. Treating it as a distinct 'weak scene' requiring its own checkpoint/data investigation would misattribute ordinary repeat-capture variance to a scene-geometry problem that these 9 captures cannot actually test.

