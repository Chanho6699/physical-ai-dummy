# Grid35 V2-clean: episode start-segment (0-0.5s) temporal-alignment audit

Dataset: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_xy_grid35_v2_clean` - 35 training episodes, 30fps, start segment = frames 0-14 (0.0s-0.467s).

Movement-vs-noise threshold used: cumulative displacement from frame-0 state >= **1.0 deg**, sustained for **3 consecutive frames** (see script docstring for the data-based justification; noise-floor evidence in section 2 below).

## 1. Frame 0-14 GT action-state delta (shoulder_lift / elbow_flex)

Aggregated across all 35 episodes, per frame index (0=episode start).

| frame | time(s) | joint | mean\|delta\| | median\|delta\| | p95\|delta\| | max\|delta\| |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 0.000 | shoulder_lift | 0.337 | 0.352 | 0.527 | 0.527 |
| 0 | 0.000 | elbow_flex | 1.873 | 1.890 | 2.242 | 2.330 |
| 1 | 0.033 | shoulder_lift | 0.337 | 0.352 | 0.527 | 0.527 |
| 1 | 0.033 | elbow_flex | 1.875 | 1.890 | 2.242 | 2.330 |
| 2 | 0.067 | shoulder_lift | 0.337 | 0.352 | 0.527 | 0.527 |
| 2 | 0.067 | elbow_flex | 1.875 | 1.890 | 2.242 | 2.330 |
| 3 | 0.100 | shoulder_lift | 0.337 | 0.352 | 0.527 | 0.527 |
| 3 | 0.100 | elbow_flex | 1.875 | 1.890 | 2.242 | 2.330 |
| 4 | 0.133 | shoulder_lift | 0.342 | 0.352 | 0.527 | 0.527 |
| 4 | 0.133 | elbow_flex | 1.883 | 1.890 | 2.268 | 2.418 |
| 5 | 0.167 | shoulder_lift | 0.342 | 0.352 | 0.527 | 0.527 |
| 5 | 0.167 | elbow_flex | 1.883 | 1.890 | 2.268 | 2.418 |
| 6 | 0.200 | shoulder_lift | 0.352 | 0.352 | 0.527 | 0.615 |
| 6 | 0.200 | elbow_flex | 1.888 | 1.890 | 2.268 | 2.418 |
| 7 | 0.233 | shoulder_lift | 0.357 | 0.352 | 0.527 | 0.703 |
| 7 | 0.233 | elbow_flex | 1.893 | 1.890 | 2.268 | 2.418 |
| 8 | 0.267 | shoulder_lift | 0.359 | 0.352 | 0.554 | 0.703 |
| 8 | 0.267 | elbow_flex | 1.908 | 1.978 | 2.295 | 2.418 |
| 9 | 0.300 | shoulder_lift | 0.377 | 0.352 | 0.642 | 0.791 |
| 9 | 0.300 | elbow_flex | 1.915 | 1.978 | 2.356 | 2.505 |
| 10 | 0.333 | shoulder_lift | 0.394 | 0.352 | 0.703 | 0.791 |
| 10 | 0.333 | elbow_flex | 1.930 | 1.978 | 2.356 | 2.505 |
| 11 | 0.367 | shoulder_lift | 0.422 | 0.440 | 0.791 | 1.319 |
| 11 | 0.367 | elbow_flex | 1.958 | 1.978 | 2.418 | 2.593 |
| 12 | 0.400 | shoulder_lift | 0.462 | 0.440 | 0.818 | 2.374 |
| 12 | 0.400 | elbow_flex | 2.018 | 1.978 | 2.470 | 3.209 |
| 13 | 0.433 | shoulder_lift | 0.517 | 0.440 | 0.818 | 3.604 |
| 13 | 0.433 | elbow_flex | 2.084 | 1.978 | 2.699 | 3.824 |
| 14 | 0.467 | shoulder_lift | 0.555 | 0.440 | 0.905 | 4.220 |
| 14 | 0.467 | elbow_flex | 2.144 | 1.978 | 3.033 | 4.264 |

For reference, Safety Gate WOULD_CLAMP thresholds (read-only, `configs/safety_gate.yaml`): shoulder_lift=5.16deg, elbow_flex=5.73deg.

## 2. Noise-floor evidence (frame-to-frame |state[t]-state[t-1]|, frames 1-14, all episodes)

| joint | n samples | frac exactly 0 | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | 490 | 0.996 | 0.0000 | 0.0000 | 0.0000 | 0.3516 |
| shoulder_lift | 490 | 0.980 | 0.0000 | 0.0000 | 0.1758 | 0.9670 |
| elbow_flex | 490 | 0.998 | 0.0000 | 0.0000 | 0.0000 | 0.1758 |
| wrist_flex | 490 | 1.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wrist_roll | 490 | 1.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gripper | 490 | 0.992 | 0.0000 | 0.0000 | 0.0000 | 0.0690 |

## 3. 35-episode aggregate table: first movement frames + start-segment delta

| episode | length | \|delta\|@f0 SL | \|delta\|@f0 EF | \|delta\|@f14 SL | \|delta\|@f14 EF | state 1st-move (any) | state 1st-move SL | state 1st-move EF | action 1st-move (any) | action 1st-move SL | action 1st-move EF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 329 | 0.264 | 1.802 | 0.264 | 1.802 | 23 | 23 | 25 | 20 | 20 | 21 |
| 1 | 329 | 0.264 | 1.802 | 0.352 | 1.802 | 34 | 34 | 36 | 31 | 31 | 31 |
| 2 | 328 | 0.088 | 1.802 | 0.088 | 1.802 | 29 | 30 | 29 | 24 | 27 | 24 |
| 3 | 329 | 0.176 | 1.802 | 0.264 | 1.978 | 25 | 25 | 26 | 21 | 22 | 21 |
| 4 | 329 | 0.527 | 1.978 | 0.527 | 1.978 | 28 | 28 | 31 | 26 | 26 | 27 |
| 5 | 328 | 0.440 | 1.714 | 0.967 | 1.978 | 23 | 23 | 25 | 21 | 22 | 21 |
| 6 | 329 | 0.176 | 2.066 | 0.176 | 2.066 | 24 | 24 | 25 | 21 | 21 | 21 |
| 7 | 329 | 0.352 | 1.802 | 0.352 | 1.802 | 31 | 31 | 32 | 28 | 28 | 28 |
| 8 | 329 | 0.440 | 1.978 | 0.440 | 1.978 | 25 | 25 | 28 | 22 | 22 | 24 |
| 9 | 329 | 0.527 | 2.066 | 0.440 | 2.066 | 25 | 25 | 26 | 21 | 21 | 22 |
| 10 | 329 | 0.176 | 1.802 | 0.264 | 1.978 | 32 | 32 | 33 | 29 | 29 | 29 |
| 11 | 329 | 0.088 | 1.978 | 0.264 | 1.978 | 29 | 29 | 31 | 23 | 25 | 26 |
| 12 | 328 | 0.264 | 1.626 | 0.615 | 1.714 | 30 | 30 | 32 | 28 | 28 | 28 |
| 13 | 329 | 0.440 | 1.890 | 0.440 | 1.890 | 34 | 34 | 35 | 31 | 32 | 31 |
| 14 | 329 | 0.352 | 1.802 | 0.527 | 1.890 | 25 | 30 | 25 | 20 | 27 | 20 |
| 15 | 329 | 0.440 | 2.242 | 0.440 | 2.242 | 29 | 29 | 30 | 25 | 25 | 25 |
| 16 | 329 | 0.527 | 2.066 | 0.352 | 2.242 | 29 | 29 | 32 | 25 | 25 | 26 |
| 17 | 328 | 0.440 | 2.066 | 0.440 | 2.066 | 24 | 24 | 25 | 20 | 20 | 21 |
| 18 | 328 | 0.088 | 0.308 | 0.703 | 0.923 | 24 | 24 | 27 | 20 | 20 | 21 |
| 19 | 328 | 0.352 | 1.802 | 0.527 | 1.978 | 26 | 26 | 28 | 23 | 23 | 23 |
| 20 | 329 | 0.527 | 2.242 | 0.264 | 2.681 | 24 | 25 | 24 | 20 | 21 | 20 |
| 21 | 329 | 0.527 | 2.242 | 0.264 | 2.505 | 20 | 20 | 20 | 16 | 16 | 16 |
| 22 | 329 | 0.527 | 2.242 | 0.527 | 2.242 | 24 | 24 | 24 | 20 | 20 | 20 |
| 23 | 329 | 0.176 | 1.890 | 0.264 | 2.066 | 26 | 26 | 27 | 23 | 23 | 23 |
| 24 | 329 | 0.352 | 2.330 | 4.220 | 4.264 | 15 | 15 | 17 | 12 | 12 | 13 |
| 25 | 329 | 0.352 | 1.011 | 0.527 | 1.363 | 20 | 20 | 21 | 16 | 17 | 16 |
| 26 | 328 | 0.527 | 2.242 | 0.088 | 2.418 | 23 | 24 | 23 | 18 | 20 | 18 |
| 27 | 329 | 0.176 | 1.978 | 0.703 | 2.506 | 19 | 19 | 21 | 15 | 15 | 17 |
| 28 | 329 | 0.264 | 1.802 | 0.879 | 3.648 | 17 | 17 | 18 | 13 | 14 | 13 |
| 29 | 328 | 0.527 | 1.714 | 0.615 | 2.066 | 26 | 28 | 28 | 22 | 25 | 23 |
| 30 | 329 | 0.176 | 2.066 | 0.791 | 2.769 | 20 | 20 | 22 | 17 | 17 | 17 |
| 31 | 329 | 0.264 | 1.802 | 0.352 | 1.978 | 26 | 34 | 27 | 23 | 30 | 23 |
| 32 | 329 | 0.264 | 1.890 | 0.527 | 2.154 | 20 | 20 | 22 | 17 | 17 | 18 |
| 33 | 328 | 0.352 | 1.802 | 0.615 | 2.330 | 24 | 24 | 26 | 20 | 20 | 21 |
| 34 | 328 | 0.352 | 1.890 | 0.352 | 1.890 | 22 | 22 | 24 | 19 | 19 | 20 |

Median state first-movement frame (any joint): **25.0** (~0.833s). Median action first-movement frame (any joint): **21.0** (~0.700s). Median (state-action) lead = **4.0** frames.

## 4. Dataset-wide horizon trajectory (frames 1-30) vs. actual-Shadow policy first-action

Dataset-wide mean `state[t]-state[0]` and `action[t]-state[0]` displacement (degrees):

| joint | series | t=1(0.03s) | t=2(0.07s) | t=3(0.10s) | t=4(0.13s) | t=5(0.17s) | t=6(0.20s) | t=7(0.23s) | t=8(0.27s) | t=9(0.30s) | t=10(0.33s) | t=11(0.37s) | t=12(0.40s) | t=13(0.43s) | t=14(0.47s) | t=15(0.50s) | t=16(0.53s) | t=17(0.57s) | t=18(0.60s) | t=19(0.63s) | t=20(0.67s) | t=21(0.70s) | t=22(0.73s) | t=23(0.77s) | t=24(0.80s) | t=25(0.83s) | t=26(0.87s) | t=27(0.90s) | t=28(0.93s) | t=29(0.97s) | t=30(1.00s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shoulder_lift | state[t]-state[0] | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.01 | +0.01 | +0.02 | +0.07 | +0.13 | +0.20 | +0.27 | +0.37 | +0.53 | +0.76 | +1.04 | +1.34 | +1.74 | +2.32 | +3.02 | +3.79 | +4.60 | +5.47 | +6.36 | +7.38 |
| shoulder_lift | action[t]-state[0] | +0.11 | +0.11 | +0.11 | +0.12 | +0.12 | +0.13 | +0.14 | +0.15 | +0.18 | +0.21 | +0.24 | +0.30 | +0.37 | +0.46 | +0.59 | +0.78 | +1.01 | +1.29 | +1.63 | +2.07 | +2.63 | +3.31 | +4.10 | +4.96 | +5.88 | +6.82 | +7.82 | +8.83 | +9.88 | +10.95 |
| elbow_flex | state[t]-state[0] | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.01 | -0.01 | -0.04 | -0.07 | -0.13 | -0.20 | -0.31 | -0.46 | -0.64 | -0.87 | -1.18 | -1.57 | -2.04 | -2.56 | -3.13 | -3.75 | -4.43 |
| elbow_flex | action[t]-state[0] | -1.88 | -1.88 | -1.88 | -1.88 | -1.88 | -1.89 | -1.89 | -1.91 | -1.92 | -1.93 | -1.96 | -2.02 | -2.08 | -2.15 | -2.25 | -2.39 | -2.58 | -2.80 | -3.04 | -3.34 | -3.72 | -4.20 | -4.75 | -5.33 | -5.95 | -6.58 | -7.29 | -8.03 | -8.84 | -9.78 |

Actual-Shadow policy first-action mean delta, averaged over T01-T10 (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_T01_T10_actual_seed_sweep_summary/t01_t10_summary.json`):

| joint | T01-T10 avg policy delta (deg) |
|---|---:|
| shoulder_lift | +5.61 |
| elbow_flex | -6.46 |

Estimated dataset horizon (frames/seconds) whose GT displacement best matches that policy delta:

| joint | basis | target delta (deg) | matching horizon (frames) | matching horizon (s) |
|---|---|---:|---:|---:|
| shoulder_lift | state[t]-state[0] | +5.61 | 28.2 | 0.939 |
| shoulder_lift | action[t]-state[0] | +5.61 | 24.7 | 0.824 |
| elbow_flex | state[t]-state[0] | -6.46 | 30.0 | 1.000 |
| elbow_flex | action[t]-state[0] | -6.46 | 25.8 | 0.860 |

Cross-reference: the earlier live-pipeline trace (`reports/grid35_v2_first_action_diagnostic_T01/`) found the *nearest-neighbor* GT state to the actual Shadow T01 observation at episode 33, frame 25 (L2~2.2deg) - i.e. **not** an episode-0/start-of-episode frame, but ~0.83s into a demonstration. Per-scene nearest-demo matches (T01-T10), read from the actual-Shadow summary JSON:

| scene | nearest episode | nearest frame | L2 dist (deg) |
|---|---:|---:|---:|
| T01 | 33 | 25 | 2.22 |
| T02 | 33 | 25 | 2.30 |
| T03 | 33 | 25 | 2.30 |
| T04 | 33 | 25 | 2.30 |
| T05 | 33 | 25 | 2.30 |
| T06 | 33 | 25 | 2.28 |
| T07 | 33 | 25 | 2.28 |
| T08 | 33 | 25 | 2.19 |
| T09 | 33 | 25 | 2.19 |
| T10 | 33 | 25 | 2.19 |

## 5. Safety threshold comparison (read-only)

| joint | WOULD_CLAMP (deg) | REJECT (deg, x5) | GT mean\|delta\|@frame0 | GT mean\|delta\|@frame14 | actual policy mean delta |
|---|---:|---:|---:|---:|---:|
| shoulder_lift | 5.16 | 25.80 | 0.337 | 0.555 | +5.61 |
| elbow_flex | 5.73 | 28.65 | 1.873 | 2.144 | -6.46 |

## 6. Verdict

### A. GT start action itself large?
**False** - frame-0 mean|delta|: shoulder_lift=0.337deg, elbow_flex=1.873deg, vs. WOULD_CLAMP thresholds {'shoulder_lift': 5.16, 'elbow_flex': 5.73}.

### B. GT normal, policy alone large?
**True** - policy delta / GT frame-0 delta ratio: shoulder_lift=16.7x, elbow_flex=3.5x.

### C. Static-hold action-label misalignment?
**False** - mean (state_first_move - action_first_move) lead = 3.57 frames (n=35 episodes with both defined); frame-14 GT mean|delta|: shoulder_lift=0.555deg, elbow_flex=2.144deg.

### D. Policy first-action resembles which future GT frame?
Approximate mean matching frame across shoulder_lift/elbow_flex (state-based): **29.1 frames** (~0.97s) into a typical demonstration.

### E. Root-cause lean
E leaning towards D/training-coverage: GT start-segment delta (frames 0-14) stays well under the WOULD_CLAMP threshold and does not grow enough within the 0.5s window to explain the actual-Shadow policy bias; the policy's delivered first-action delta magnitude matches dataset GT displacement several hundred ms to ~1s further into a typical demonstration than frame 0 (see estimated_matching_horizon). This is consistent with an undertrained/undifferentiated flow-matching policy (D) rather than a training-time state/action mislabeling bug (C) - the dataset's own frame-0 labels are not the source of the oversized first action.
