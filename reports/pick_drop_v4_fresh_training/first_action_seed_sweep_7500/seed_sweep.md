# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v4/smolvla_pick_drop_v4_fresh/checkpoints/007500/pretrained_model`
Task: `Pick up the cube and drop it into the bin.`
Seeds swept: 0..19 (20 seeds, 1 inference call each)

## Safety thresholds used (read-only, not modified)

| joint | WOULD_CLAMP threshold (deg) |
|---|---:|
| shoulder_pan | 4.58 |
| shoulder_lift | 5.16 |
| elbow_flex | 5.73 |
| wrist_flex | 4.01 |
| wrist_roll | 1.15 |
| gripper | 9.17 |

## Nearest-training-demo immediate GT delta (reference)

| joint | GT immediate delta (deg) |
|---|---:|
| shoulder_pan | -0.7033 |
| shoulder_lift | +3.5165 |
| elbow_flex | -4.4396 |
| wrist_flex | -1.0989 |
| wrist_roll | +0.0879 |
| gripper | -0.3945 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -1.47 | +3.08 | -10.37 | -0.16 | +0.25 | -0.15 | 1 | elbow_flex | 6.07 |
| 1 | -1.21 | +10.38 | -9.36 | -0.30 | +0.62 | -0.32 | 2 | shoulder_lift, elbow_flex | 8.51 |
| 2 | +0.54 | -0.47 | -3.82 | -0.22 | +0.41 | -0.16 | 0 | - | 4.33 |
| 3 | +0.67 | +0.80 | -7.11 | +0.46 | +0.24 | +0.14 | 1 | elbow_flex | 4.38 |
| 4 | -1.46 | +1.95 | -6.91 | -0.13 | +0.83 | +0.62 | 1 | elbow_flex | 3.41 |
| 5 | -0.43 | +4.18 | -7.32 | +0.60 | +0.58 | -0.31 | 1 | elbow_flex | 3.45 |
| 6 | +0.25 | -0.42 | -5.55 | +0.31 | -0.07 | -0.08 | 0 | - | 4.45 |
| 7 | -0.23 | +4.26 | -7.45 | +0.56 | +0.21 | -0.47 | 1 | elbow_flex | 3.55 |
| 8 | -0.83 | +0.19 | -3.24 | +0.35 | +0.92 | -0.19 | 0 | - | 3.92 |
| 9 | -0.09 | +0.92 | -6.02 | -0.78 | +0.22 | -0.50 | 1 | elbow_flex | 3.12 |
| 10 | -0.45 | +5.27 | -8.44 | -0.05 | +0.22 | +0.15 | 2 | shoulder_lift, elbow_flex | 4.54 |
| 11 | -0.12 | +1.25 | -6.27 | -0.03 | +0.62 | +0.30 | 1 | elbow_flex | 3.28 |
| 12 | +0.27 | -4.07 | -3.19 | -0.87 | -0.22 | +0.55 | 0 | - | 7.82 |
| 13 | +0.83 | +1.51 | -5.29 | +0.16 | +0.62 | +0.04 | 0 | - | 3.03 |
| 14 | -0.94 | +2.61 | -7.19 | -0.52 | +0.35 | -0.16 | 1 | elbow_flex | 2.98 |
| 15 | -1.89 | +6.50 | -9.29 | -0.36 | +0.54 | -0.41 | 2 | shoulder_lift, elbow_flex | 5.88 |
| 16 | -1.65 | +5.46 | -9.09 | +0.94 | +0.45 | +0.45 | 2 | shoulder_lift, elbow_flex | 5.59 |
| 17 | -1.15 | +1.76 | -3.82 | -1.20 | +0.26 | -0.38 | 0 | - | 1.92 |
| 18 | +1.10 | +1.74 | -6.34 | +0.14 | -0.43 | +0.23 | 1 | elbow_flex | 3.49 |
| 19 | +0.70 | +0.74 | -1.04 | -2.66 | +1.18 | +0.81 | 1 | wrist_roll | 5.13 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.38 | 0.89 | -1.89 | +1.10 | 0/20 | 0% |
| shoulder_lift | +2.38 | 2.99 | -4.07 | +10.38 | 4/20 | 20% |
| elbow_flex | -6.35 | 2.37 | -10.37 | -1.04 | 13/20 | 65% |
| wrist_flex | -0.19 | 0.77 | -2.66 | +0.94 | 0/20 | 0% |
| wrist_roll | +0.39 | 0.37 | -0.43 | +1.18 | 1/20 | 5% |
| gripper | +0.01 | 0.38 | -0.50 | +0.81 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 6/20 ([2, 6, 8, 12, 13, 17])
- clamp-joint-count distribution (count -> #seeds): {'0': 6, '1': 10, '2': 4}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.44, std=1.61, min=1.92, max=8.51
