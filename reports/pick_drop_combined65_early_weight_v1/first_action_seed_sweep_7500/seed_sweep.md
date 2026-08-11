# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65/combined65_early_weight_v1/checkpoints/007500/pretrained_model`
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
| 0 | -1.90 | +4.18 | -13.67 | +0.01 | +0.21 | +0.22 | 1 | elbow_flex | 9.32 |
| 1 | -0.88 | +11.05 | -8.94 | -1.44 | +0.31 | -0.16 | 2 | shoulder_lift, elbow_flex | 8.62 |
| 2 | +0.54 | +1.14 | -4.89 | -1.01 | +0.26 | +0.29 | 0 | - | 3.11 |
| 3 | +0.84 | +1.86 | -8.97 | +0.73 | +0.18 | +0.51 | 1 | elbow_flex | 4.94 |
| 4 | -1.78 | +2.57 | -6.99 | -1.07 | +0.45 | +1.46 | 1 | elbow_flex | 4.03 |
| 5 | +0.30 | +4.54 | -7.19 | +0.88 | +0.32 | -0.45 | 1 | elbow_flex | 2.78 |
| 6 | +0.54 | +1.67 | -7.22 | +0.39 | +0.07 | +0.16 | 1 | elbow_flex | 3.47 |
| 7 | -0.24 | +5.57 | -7.73 | -0.30 | +0.17 | -0.77 | 2 | shoulder_lift, elbow_flex | 3.65 |
| 8 | -0.49 | +1.05 | -3.70 | -0.75 | +0.40 | +0.16 | 0 | - | 3.26 |
| 9 | -0.11 | +1.43 | -6.96 | -0.52 | +0.20 | -0.38 | 1 | elbow_flex | 3.49 |
| 10 | -0.09 | +5.78 | -7.93 | -0.81 | +0.19 | +0.46 | 2 | shoulder_lift, elbow_flex | 4.06 |
| 11 | -0.01 | +1.31 | -6.83 | -0.30 | +0.41 | +0.95 | 1 | elbow_flex | 3.69 |
| 12 | +0.21 | -2.25 | -4.73 | -1.26 | +0.07 | +1.32 | 0 | - | 6.51 |
| 13 | +1.55 | +2.46 | -6.37 | +0.12 | +0.37 | +0.30 | 1 | elbow_flex | 2.75 |
| 14 | -0.29 | +3.22 | -7.11 | -1.33 | +0.23 | +0.30 | 1 | elbow_flex | 3.12 |
| 15 | -0.80 | +6.68 | -8.13 | -1.63 | +0.28 | -0.00 | 2 | shoulder_lift, elbow_flex | 4.99 |
| 16 | -1.87 | +5.53 | -8.82 | +0.48 | +0.33 | +0.83 | 2 | shoulder_lift, elbow_flex | 5.12 |
| 17 | -1.06 | +3.60 | -5.15 | -1.78 | +0.26 | -0.31 | 0 | - | 2.45 |
| 18 | +0.88 | +2.54 | -6.39 | -0.68 | -0.06 | +0.87 | 1 | elbow_flex | 2.76 |
| 19 | +0.77 | +1.12 | -1.18 | -3.51 | +0.61 | +0.85 | 0 | - | 5.96 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.19 | 0.94 | -1.90 | +1.55 | 0/20 | 0% |
| shoulder_lift | +3.25 | 2.73 | -2.25 | +11.05 | 5/20 | 25% |
| elbow_flex | -6.94 | 2.42 | -13.67 | -1.18 | 15/20 | 75% |
| wrist_flex | -0.69 | 1.00 | -3.51 | +0.88 | 0/20 | 0% |
| wrist_roll | +0.26 | 0.15 | -0.06 | +0.61 | 0/20 | 0% |
| gripper | +0.33 | 0.58 | -0.77 | +1.46 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 5/20 ([2, 8, 12, 17, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 5, '1': 10, '2': 5}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.40, std=1.86, min=2.45, max=9.32
