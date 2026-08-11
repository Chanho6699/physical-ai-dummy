# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65_reweight3/smolvla_pick_drop_combined65_reweight3_fresh/checkpoints/005000/pretrained_model`
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
| shoulder_pan | +0.1758 |
| shoulder_lift | +3.8681 |
| elbow_flex | -4.6154 |
| wrist_flex | +0.2198 |
| wrist_roll | -0.0879 |
| gripper | -0.3216 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -1.78 | +1.25 | -11.07 | -2.13 | +0.03 | -0.56 | 1 | elbow_flex | 7.61 |
| 1 | -1.85 | +10.09 | -5.82 | -4.05 | +0.12 | -0.82 | 3 | shoulder_lift, elbow_flex, wrist_flex | 7.92 |
| 2 | +1.26 | -3.34 | -0.93 | -3.59 | +0.03 | -0.60 | 0 | - | 9.02 |
| 3 | +1.54 | -2.29 | -5.69 | -1.30 | -0.04 | -0.36 | 0 | - | 6.58 |
| 4 | -3.24 | -0.96 | -4.03 | -3.19 | +0.28 | +0.52 | 0 | - | 6.92 |
| 5 | +0.42 | +2.98 | -4.48 | -0.84 | +0.13 | -1.34 | 0 | - | 1.75 |
| 6 | +0.94 | -3.49 | -2.83 | -1.78 | -0.25 | -0.78 | 0 | - | 7.88 |
| 7 | -0.04 | +3.11 | -5.78 | -2.31 | -0.04 | -1.60 | 1 | elbow_flex | 3.16 |
| 8 | -1.58 | -2.32 | -0.07 | -2.47 | +0.23 | -0.85 | 0 | - | 8.34 |
| 9 | +0.19 | -1.86 | -4.51 | -2.36 | -0.02 | -1.07 | 0 | - | 6.33 |
| 10 | -0.20 | +4.03 | -4.22 | -3.63 | -0.03 | -0.26 | 0 | - | 3.90 |
| 11 | -0.34 | -2.48 | -4.04 | -2.76 | +0.15 | -0.51 | 0 | - | 7.06 |
| 12 | +0.19 | -10.46 | +0.82 | -4.45 | -0.36 | +0.02 | 2 | shoulder_lift, wrist_flex | 16.02 |
| 13 | +1.83 | -0.66 | -2.64 | -1.92 | +0.15 | -0.84 | 0 | - | 5.66 |
| 14 | -1.17 | +0.19 | -4.71 | -4.04 | -0.02 | -0.80 | 1 | wrist_flex | 5.80 |
| 15 | -2.00 | +5.52 | -6.04 | -3.99 | +0.08 | -1.03 | 2 | shoulder_lift, elbow_flex | 5.27 |
| 16 | -2.87 | +5.80 | -8.03 | -0.34 | +0.17 | -0.00 | 2 | shoulder_lift, elbow_flex | 5.01 |
| 17 | -1.50 | -0.51 | -0.21 | -5.27 | -0.05 | -1.61 | 1 | wrist_flex | 8.56 |
| 18 | +2.70 | -3.65 | -0.53 | -3.62 | -0.46 | -0.41 | 0 | - | 9.72 |
| 19 | +1.86 | -4.08 | +4.85 | -6.87 | +0.42 | +0.05 | 1 | wrist_flex | 14.36 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.28 | 1.63 | -3.24 | +2.70 | 0/20 | 0% |
| shoulder_lift | -0.16 | 4.40 | -10.46 | +10.09 | 4/20 | 20% |
| elbow_flex | -3.50 | 3.40 | -11.07 | +4.85 | 5/20 | 25% |
| wrist_flex | -3.05 | 1.51 | -6.87 | -0.34 | 5/20 | 25% |
| wrist_roll | +0.03 | 0.20 | -0.46 | +0.42 | 0/20 | 0% |
| gripper | -0.64 | 0.54 | -1.61 | +0.52 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 11/20 ([2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 18])
- clamp-joint-count distribution (count -> #seeds): {'0': 11, '1': 5, '2': 3, '3': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=7.34, std=3.27, min=1.75, max=16.02
