# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65/smolvla_pick_drop_combined65_fresh/checkpoints/010000/pretrained_model`
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
| 0 | -1.76 | +4.63 | -11.83 | -0.20 | +0.15 | -0.25 | 1 | elbow_flex | 7.52 |
| 1 | -1.12 | +11.90 | -9.16 | -1.74 | +0.27 | -0.48 | 2 | shoulder_lift, elbow_flex | 9.54 |
| 2 | +0.12 | +2.14 | -5.56 | -1.06 | +0.22 | -0.23 | 0 | - | 2.37 |
| 3 | +0.43 | +2.32 | -8.25 | +0.74 | +0.14 | -0.15 | 1 | elbow_flex | 4.00 |
| 4 | -1.71 | +3.02 | -6.40 | -1.04 | +0.37 | +1.09 | 1 | elbow_flex | 3.35 |
| 5 | +0.16 | +5.05 | -7.24 | +0.73 | +0.28 | -0.96 | 1 | elbow_flex | 3.01 |
| 6 | +0.19 | +2.04 | -7.30 | +0.57 | +0.06 | -0.48 | 1 | elbow_flex | 3.27 |
| 7 | -0.53 | +6.16 | -8.40 | -0.17 | +0.15 | -1.06 | 2 | shoulder_lift, elbow_flex | 4.57 |
| 8 | -0.46 | +1.33 | -4.14 | -0.74 | +0.33 | -0.31 | 0 | - | 2.86 |
| 9 | -0.60 | +1.74 | -6.71 | -0.62 | +0.13 | -1.01 | 1 | elbow_flex | 3.28 |
| 10 | -0.48 | +6.38 | -7.67 | -1.06 | +0.17 | +0.07 | 2 | shoulder_lift, elbow_flex | 4.23 |
| 11 | -0.50 | +2.31 | -6.96 | -0.36 | +0.33 | +0.28 | 1 | elbow_flex | 3.04 |
| 12 | -0.26 | -1.37 | -4.71 | -1.20 | +0.00 | +0.64 | 0 | - | 5.53 |
| 13 | +0.85 | +3.32 | -6.51 | -0.03 | +0.32 | -0.36 | 1 | elbow_flex | 2.14 |
| 14 | -0.66 | +3.19 | -6.50 | -1.59 | +0.17 | -0.19 | 1 | elbow_flex | 2.85 |
| 15 | -1.13 | +7.39 | -8.23 | -1.67 | +0.24 | -0.31 | 2 | shoulder_lift, elbow_flex | 5.56 |
| 16 | -1.93 | +6.31 | -9.22 | +0.91 | +0.28 | +0.42 | 2 | shoulder_lift, elbow_flex | 5.72 |
| 17 | -1.08 | +4.25 | -5.18 | -2.00 | +0.20 | -0.75 | 0 | - | 2.69 |
| 18 | +0.60 | +2.80 | -6.16 | -0.56 | -0.06 | +0.02 | 1 | elbow_flex | 2.11 |
| 19 | -0.04 | +1.67 | -1.40 | -3.48 | +0.53 | +0.26 | 0 | - | 5.44 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.50 | 0.76 | -1.93 | +0.85 | 0/20 | 0% |
| shoulder_lift | +3.83 | 2.77 | -1.37 | +11.90 | 5/20 | 25% |
| elbow_flex | -6.88 | 2.12 | -11.83 | -1.40 | 15/20 | 75% |
| wrist_flex | -0.73 | 1.06 | -3.48 | +0.91 | 0/20 | 0% |
| wrist_roll | +0.21 | 0.13 | -0.06 | +0.53 | 0/20 | 0% |
| gripper | -0.19 | 0.54 | -1.06 | +1.09 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 5/20 ([2, 8, 12, 17, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 5, '1': 10, '2': 5}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.15, std=1.87, min=2.11, max=9.54
