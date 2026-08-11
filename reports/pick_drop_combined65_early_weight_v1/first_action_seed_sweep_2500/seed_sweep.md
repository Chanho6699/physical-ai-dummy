# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65/combined65_early_weight_v1/checkpoints/002500/pretrained_model`
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
| 0 | -6.22 | +3.20 | -17.46 | -0.61 | +0.33 | +0.64 | 2 | shoulder_pan, elbow_flex | 14.43 |
| 1 | -5.50 | +14.84 | -3.10 | -4.67 | +0.53 | -0.15 | 3 | shoulder_pan, shoulder_lift, wrist_flex | 13.39 |
| 2 | -2.44 | -8.47 | +7.25 | -3.47 | +0.41 | +1.02 | 2 | shoulder_lift, elbow_flex | 17.76 |
| 3 | -1.66 | -4.74 | -4.35 | +0.92 | +0.24 | +1.11 | 0 | - | 8.96 |
| 4 | -9.43 | -1.06 | +0.78 | -3.08 | +0.75 | +2.61 | 1 | shoulder_pan | 12.88 |
| 5 | -2.81 | +4.11 | +1.26 | +1.10 | +0.47 | -1.26 | 0 | - | 6.75 |
| 6 | -1.01 | -5.38 | +0.22 | -0.32 | +0.04 | +0.83 | 1 | shoulder_lift | 10.58 |
| 7 | -3.71 | +5.43 | -2.61 | -1.07 | +0.20 | -1.46 | 1 | shoulder_lift | 4.96 |
| 8 | -5.90 | -5.55 | +9.63 | -2.63 | +0.70 | +0.10 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 18.37 |
| 9 | -2.62 | -5.17 | +0.32 | -2.09 | +0.27 | -0.62 | 1 | shoulder_lift | 10.92 |
| 10 | -3.95 | +5.90 | -1.31 | -3.75 | +0.29 | +2.02 | 1 | shoulder_lift | 7.31 |
| 11 | -3.37 | -4.83 | +0.30 | -1.71 | +0.65 | +1.72 | 0 | - | 10.99 |
| 12 | -2.98 | -20.77 | +7.31 | -4.45 | -0.16 | +3.54 | 3 | shoulder_lift, elbow_flex, wrist_flex | 28.21 |
| 13 | +1.31 | -2.67 | +3.97 | -1.24 | +0.56 | +0.42 | 0 | - | 10.99 |
| 14 | -5.07 | -0.65 | -1.06 | -5.03 | +0.27 | +1.39 | 2 | shoulder_pan, wrist_flex | 9.55 |
| 15 | -6.15 | +7.32 | -3.29 | -5.06 | +0.47 | -0.10 | 3 | shoulder_pan, shoulder_lift, wrist_flex | 9.05 |
| 16 | -7.23 | +8.80 | -6.02 | +2.34 | +0.51 | +1.19 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 9.39 |
| 17 | -6.22 | -1.19 | +7.99 | -7.28 | +0.30 | -1.00 | 3 | shoulder_pan, elbow_flex, wrist_flex | 16.80 |
| 18 | -0.88 | -6.89 | +3.50 | -2.99 | -0.34 | +2.15 | 1 | shoulder_lift | 14.12 |
| 19 | -2.42 | -11.36 | +18.72 | -11.82 | +1.09 | +2.75 | 3 | shoulder_lift, elbow_flex, wrist_flex | 30.64 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -3.91 | 2.48 | -9.43 | +1.31 | 8/20 | 40% |
| shoulder_lift | -1.46 | 7.82 | -20.77 | +14.84 | 12/20 | 60% |
| elbow_flex | +1.10 | 7.10 | -17.46 | +18.72 | 7/20 | 35% |
| wrist_flex | -2.84 | 3.10 | -11.82 | +2.34 | 6/20 | 30% |
| wrist_roll | +0.38 | 0.31 | -0.34 | +1.09 | 0/20 | 0% |
| gripper | +0.85 | 1.35 | -1.46 | +3.54 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([3, 5, 11, 13])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 6, '2': 3, '3': 7}
- L2 error vs nearest-demo immediate GT delta (deg): mean=13.30, std=6.41, min=4.96, max=30.64
