# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65/smolvla_pick_drop_combined65_fresh/checkpoints/002500/pretrained_model`
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
| 0 | -6.46 | +5.17 | -14.97 | -0.08 | +0.36 | +1.67 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 12.54 |
| 1 | -5.14 | +15.57 | -2.61 | -4.03 | +0.61 | +0.59 | 3 | shoulder_pan, shoulder_lift, wrist_flex | 13.74 |
| 2 | -3.68 | -6.97 | +7.65 | -2.60 | +0.43 | +1.78 | 2 | shoulder_lift, elbow_flex | 17.19 |
| 3 | -2.82 | -2.36 | -3.91 | +1.77 | +0.32 | +2.20 | 0 | - | 7.56 |
| 4 | -9.65 | -0.82 | +1.28 | -2.37 | +0.74 | +3.29 | 1 | shoulder_pan | 13.18 |
| 5 | -4.13 | +5.79 | +1.85 | +1.83 | +0.55 | -0.79 | 1 | shoulder_lift | 8.20 |
| 6 | -1.96 | -3.57 | +1.21 | +0.80 | +0.10 | +1.27 | 0 | - | 9.84 |
| 7 | -5.23 | +6.64 | -2.01 | +0.05 | +0.22 | -0.76 | 2 | shoulder_pan, shoulder_lift | 6.63 |
| 8 | -6.45 | -3.93 | +9.46 | -1.67 | +0.75 | +0.82 | 2 | shoulder_pan, elbow_flex | 17.56 |
| 9 | -4.14 | -3.22 | +0.67 | -0.87 | +0.31 | -0.11 | 0 | - | 9.91 |
| 10 | -4.58 | +7.36 | -0.64 | -3.22 | +0.33 | +2.75 | 1 | shoulder_lift | 8.48 |
| 11 | -3.86 | -3.49 | +0.59 | -1.12 | +0.64 | +2.45 | 0 | - | 10.37 |
| 12 | -5.49 | -20.16 | +7.05 | -3.43 | -0.21 | +4.39 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 27.95 |
| 13 | +0.38 | -1.29 | +4.65 | -0.53 | +0.61 | +0.95 | 0 | - | 10.73 |
| 14 | -6.37 | +0.76 | -1.05 | -4.05 | +0.29 | +2.31 | 2 | shoulder_pan, wrist_flex | 9.51 |
| 15 | -6.03 | +8.78 | -2.65 | -4.11 | +0.51 | +0.67 | 3 | shoulder_pan, shoulder_lift, wrist_flex | 9.31 |
| 16 | -7.84 | +9.56 | -5.37 | +2.67 | +0.56 | +1.75 | 2 | shoulder_pan, shoulder_lift | 10.39 |
| 17 | -7.47 | +1.11 | +7.34 | -6.01 | +0.36 | -0.22 | 3 | shoulder_pan, elbow_flex, wrist_flex | 15.75 |
| 18 | -2.60 | -5.68 | +4.15 | -2.64 | -0.37 | +2.62 | 1 | shoulder_lift | 13.88 |
| 19 | -3.75 | -9.67 | +19.87 | -10.94 | +1.18 | +3.62 | 4 | shoulder_lift, elbow_flex, wrist_flex, wrist_roll | 30.66 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -4.86 | 2.21 | -9.65 | +0.38 | 10/20 | 50% |
| shoulder_lift | -0.02 | 7.78 | -20.16 | +15.57 | 11/20 | 55% |
| elbow_flex | +1.63 | 6.81 | -14.97 | +19.87 | 6/20 | 30% |
| wrist_flex | -2.03 | 3.03 | -10.94 | +2.67 | 5/20 | 25% |
| wrist_roll | +0.41 | 0.33 | -0.37 | +1.18 | 1/20 | 5% |
| gripper | +1.56 | 1.40 | -0.79 | +4.39 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 5/20 ([3, 6, 9, 11, 13])
- clamp-joint-count distribution (count -> #seeds): {'0': 5, '1': 4, '2': 5, '3': 5, '4': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=13.17, std=6.16, min=6.63, max=30.66
