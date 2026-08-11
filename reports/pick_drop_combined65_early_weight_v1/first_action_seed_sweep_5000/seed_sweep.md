# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65/combined65_early_weight_v1/checkpoints/005000/pretrained_model`
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
| 0 | -4.57 | +6.82 | -17.47 | +0.78 | +0.41 | +0.45 | 2 | shoulder_lift, elbow_flex | 14.06 |
| 1 | -2.79 | +14.51 | -11.33 | -0.53 | +0.54 | -0.07 | 2 | shoulder_lift, elbow_flex | 12.97 |
| 2 | -1.61 | +1.59 | -4.85 | -0.35 | +0.45 | +0.15 | 0 | - | 3.04 |
| 3 | -0.82 | +4.40 | -11.60 | +2.22 | +0.38 | +0.48 | 1 | elbow_flex | 7.41 |
| 4 | -6.04 | +5.17 | -9.26 | +0.02 | +0.74 | +1.94 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 8.23 |
| 5 | -1.23 | +6.67 | -6.73 | +1.96 | +0.54 | -0.90 | 2 | shoulder_lift, elbow_flex | 4.25 |
| 6 | -1.11 | +3.20 | -8.05 | +1.38 | +0.18 | -0.00 | 1 | elbow_flex | 3.93 |
| 7 | -1.89 | +7.92 | -8.45 | +0.77 | +0.31 | -1.26 | 2 | shoulder_lift, elbow_flex | 6.06 |
| 8 | -4.37 | +2.91 | -4.52 | +0.36 | +0.67 | +0.31 | 0 | - | 4.75 |
| 9 | -2.05 | +2.72 | -7.09 | -0.03 | +0.36 | -0.84 | 1 | elbow_flex | 3.60 |
| 10 | -2.26 | +9.62 | -9.88 | -0.27 | +0.38 | +0.58 | 2 | shoulder_lift, elbow_flex | 8.25 |
| 11 | -2.30 | +3.52 | -8.67 | +0.66 | +0.66 | +1.06 | 1 | elbow_flex | 5.03 |
| 12 | -2.51 | -5.01 | -4.48 | -1.29 | +0.11 | +1.73 | 0 | - | 9.63 |
| 13 | -0.60 | +4.55 | -6.95 | +0.84 | +0.63 | +0.33 | 1 | elbow_flex | 2.80 |
| 14 | -3.28 | +5.77 | -9.03 | -1.11 | +0.43 | +0.40 | 2 | shoulder_lift, elbow_flex | 6.13 |
| 15 | -4.02 | +9.99 | -11.36 | -0.57 | +0.49 | +0.13 | 2 | shoulder_lift, elbow_flex | 10.09 |
| 16 | -5.38 | +9.83 | -11.73 | +2.64 | +0.57 | +0.96 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 11.18 |
| 17 | -4.78 | +5.46 | -4.27 | -2.05 | +0.45 | -0.98 | 2 | shoulder_pan, shoulder_lift | 5.75 |
| 18 | -0.83 | +3.14 | -6.22 | -0.33 | +0.03 | +0.41 | 1 | elbow_flex | 2.23 |
| 19 | -2.74 | +1.68 | -0.05 | -3.93 | +0.94 | +1.02 | 0 | - | 7.37 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.76 | 1.58 | -6.04 | -0.60 | 3/20 | 15% |
| shoulder_lift | +5.22 | 3.98 | -5.01 | +14.51 | 10/20 | 50% |
| elbow_flex | -8.10 | 3.63 | -17.47 | -0.05 | 15/20 | 75% |
| wrist_flex | +0.06 | 1.48 | -3.93 | +2.64 | 0/20 | 0% |
| wrist_roll | +0.46 | 0.21 | +0.03 | +0.94 | 0/20 | 0% |
| gripper | +0.30 | 0.82 | -1.26 | +1.94 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([2, 8, 12, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 6, '2': 8, '3': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.84, std=3.31, min=2.23, max=14.06
