# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65/smolvla_pick_drop_combined65_fresh/checkpoints/007500/pretrained_model`
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
| 0 | -1.72 | +5.58 | -13.66 | -0.00 | +0.17 | +0.27 | 2 | shoulder_lift, elbow_flex | 9.42 |
| 1 | -0.94 | +13.24 | -10.17 | -1.70 | +0.30 | -0.05 | 2 | shoulder_lift, elbow_flex | 11.13 |
| 2 | +0.53 | +2.50 | -5.97 | -0.93 | +0.25 | +0.10 | 1 | elbow_flex | 2.34 |
| 3 | +0.87 | +2.97 | -9.26 | +0.85 | +0.15 | +0.45 | 1 | elbow_flex | 4.89 |
| 4 | -1.53 | +3.29 | -7.19 | -1.04 | +0.41 | +1.54 | 1 | elbow_flex | 3.90 |
| 5 | +0.43 | +6.26 | -8.50 | +0.79 | +0.29 | -0.64 | 2 | shoulder_lift, elbow_flex | 4.63 |
| 6 | +0.56 | +2.76 | -7.90 | +0.48 | +0.04 | +0.04 | 1 | elbow_flex | 3.52 |
| 7 | -0.13 | +6.79 | -8.57 | -0.31 | +0.14 | -0.84 | 2 | shoulder_lift, elbow_flex | 4.99 |
| 8 | -0.20 | +1.80 | -4.38 | -0.74 | +0.37 | +0.17 | 0 | - | 2.42 |
| 9 | -0.29 | +2.15 | -7.68 | -0.48 | +0.15 | -0.60 | 1 | elbow_flex | 3.63 |
| 10 | -0.30 | +7.44 | -8.49 | -0.95 | +0.17 | +0.41 | 2 | shoulder_lift, elbow_flex | 5.47 |
| 11 | -0.07 | +2.62 | -7.56 | -0.28 | +0.38 | +0.78 | 1 | elbow_flex | 3.46 |
| 12 | +0.02 | -1.42 | -5.60 | -1.04 | +0.03 | +1.18 | 0 | - | 5.72 |
| 13 | +1.27 | +4.10 | -7.31 | +0.13 | +0.33 | +0.19 | 1 | elbow_flex | 2.99 |
| 14 | -0.47 | +4.05 | -7.61 | -1.35 | +0.20 | +0.28 | 1 | elbow_flex | 3.51 |
| 15 | -0.86 | +8.40 | -8.97 | -1.76 | +0.27 | +0.18 | 2 | shoulder_lift, elbow_flex | 6.70 |
| 16 | -1.63 | +6.83 | -9.69 | +0.49 | +0.29 | +0.87 | 2 | shoulder_lift, elbow_flex | 6.27 |
| 17 | -1.01 | +4.84 | -5.73 | -1.89 | +0.22 | -0.30 | 1 | elbow_flex | 2.86 |
| 18 | +0.84 | +3.63 | -6.88 | -0.67 | -0.09 | +0.70 | 1 | elbow_flex | 2.72 |
| 19 | +0.42 | +2.42 | -2.17 | -3.68 | +0.57 | +0.85 | 0 | - | 5.02 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.21 | 0.84 | -1.72 | +1.27 | 0/20 | 0% |
| shoulder_lift | +4.51 | 3.02 | -1.42 | +13.24 | 7/20 | 35% |
| elbow_flex | -7.66 | 2.30 | -13.66 | -2.17 | 17/20 | 85% |
| wrist_flex | -0.70 | 1.05 | -3.68 | +0.85 | 0/20 | 0% |
| wrist_roll | +0.23 | 0.15 | -0.09 | +0.57 | 0/20 | 0% |
| gripper | +0.28 | 0.59 | -0.84 | +1.54 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 3/20 ([8, 12, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 3, '1': 10, '2': 7}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.78, std=2.22, min=2.34, max=11.13
