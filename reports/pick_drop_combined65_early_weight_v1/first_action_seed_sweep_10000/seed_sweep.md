# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65/combined65_early_weight_v1/checkpoints/010000/pretrained_model`
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
| 0 | -1.89 | +3.98 | -12.32 | -0.16 | +0.17 | -0.09 | 1 | elbow_flex | 7.99 |
| 1 | -0.92 | +10.28 | -8.25 | -1.51 | +0.26 | -0.33 | 2 | shoulder_lift, elbow_flex | 7.66 |
| 2 | +0.23 | +1.30 | -4.85 | -1.11 | +0.21 | +0.10 | 0 | - | 2.95 |
| 3 | +0.49 | +1.78 | -8.30 | +0.62 | +0.16 | +0.09 | 1 | elbow_flex | 4.29 |
| 4 | -1.77 | +2.80 | -6.59 | -1.01 | +0.37 | +1.20 | 1 | elbow_flex | 3.59 |
| 5 | +0.10 | +3.83 | -6.27 | +0.79 | +0.28 | -0.69 | 1 | elbow_flex | 1.83 |
| 6 | +0.22 | +1.51 | -6.78 | +0.41 | +0.06 | -0.25 | 1 | elbow_flex | 3.21 |
| 7 | -0.55 | +5.37 | -7.62 | -0.25 | +0.15 | -0.90 | 2 | shoulder_lift, elbow_flex | 3.52 |
| 8 | -0.66 | +1.17 | -3.74 | -0.75 | +0.34 | -0.09 | 0 | - | 3.15 |
| 9 | -0.38 | +1.64 | -6.40 | -0.66 | +0.15 | -0.74 | 1 | elbow_flex | 3.07 |
| 10 | -0.21 | +5.31 | -7.34 | -0.93 | +0.16 | +0.25 | 2 | shoulder_lift, elbow_flex | 3.37 |
| 11 | -0.33 | +1.77 | -6.67 | -0.33 | +0.34 | +0.57 | 1 | elbow_flex | 3.18 |
| 12 | -0.06 | -1.74 | -4.15 | -1.44 | +0.01 | +0.96 | 0 | - | 6.01 |
| 13 | +1.13 | +2.30 | -5.94 | -0.00 | +0.33 | -0.09 | 1 | elbow_flex | 2.33 |
| 14 | -0.40 | +3.05 | -6.43 | -1.51 | +0.18 | +0.01 | 1 | elbow_flex | 2.73 |
| 15 | -0.97 | +6.12 | -7.63 | -1.57 | +0.22 | -0.21 | 2 | shoulder_lift, elbow_flex | 4.33 |
| 16 | -2.04 | +5.34 | -8.33 | +0.72 | +0.29 | +0.53 | 2 | shoulder_lift, elbow_flex | 4.69 |
| 17 | -1.03 | +3.64 | -4.91 | -1.89 | +0.21 | -0.61 | 0 | - | 2.50 |
| 18 | +0.68 | +2.37 | -5.85 | -0.62 | -0.05 | +0.33 | 1 | elbow_flex | 2.27 |
| 19 | +0.32 | +0.94 | -0.85 | -3.35 | +0.54 | +0.34 | 0 | - | 6.03 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.40 | 0.83 | -2.04 | +1.13 | 0/20 | 0% |
| shoulder_lift | +3.14 | 2.47 | -1.74 | +10.28 | 5/20 | 25% |
| elbow_flex | -6.46 | 2.21 | -12.32 | -0.85 | 15/20 | 75% |
| wrist_flex | -0.73 | 0.99 | -3.35 | +0.79 | 0/20 | 0% |
| wrist_roll | +0.22 | 0.13 | -0.05 | +0.54 | 0/20 | 0% |
| gripper | +0.02 | 0.53 | -0.90 | +1.20 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 5/20 ([2, 8, 12, 17, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 5, '1': 10, '2': 5}
- L2 error vs nearest-demo immediate GT delta (deg): mean=3.94, std=1.69, min=1.83, max=7.99
