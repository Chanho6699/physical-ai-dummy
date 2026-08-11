# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/reinforcement30_only/reinforcement30_only_v1/checkpoints/002500/pretrained_model`
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
| shoulder_pan | +0.2637 |
| shoulder_lift | +3.7802 |
| elbow_flex | -4.3517 |
| wrist_flex | -1.0989 |
| wrist_roll | -0.0879 |
| gripper | -0.3945 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -8.54 | +3.04 | -22.30 | -0.57 | +0.16 | +0.97 | 2 | shoulder_pan, elbow_flex | 20.06 |
| 1 | -6.85 | +18.84 | -14.37 | -2.96 | +0.20 | +0.53 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 19.55 |
| 2 | -4.46 | -1.79 | -5.65 | -2.47 | +0.16 | +1.19 | 0 | - | 7.71 |
| 3 | -3.73 | -1.36 | -14.34 | +0.42 | +0.15 | +0.61 | 1 | elbow_flex | 12.06 |
| 4 | -12.50 | +3.84 | -13.50 | -2.76 | +0.26 | +4.72 | 2 | shoulder_pan, elbow_flex | 16.61 |
| 5 | -3.25 | +6.46 | -7.49 | +1.24 | +0.21 | -1.15 | 2 | shoulder_lift, elbow_flex | 5.96 |
| 6 | -3.35 | -1.90 | -6.05 | -1.17 | +0.04 | -0.28 | 1 | elbow_flex | 6.95 |
| 7 | -5.20 | +7.57 | -11.07 | -0.17 | +0.10 | -1.33 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 9.55 |
| 8 | -9.19 | +4.53 | -7.95 | -1.52 | +0.24 | +1.44 | 2 | shoulder_pan, elbow_flex | 10.32 |
| 9 | -5.04 | -2.69 | -8.52 | -1.58 | +0.15 | -0.63 | 2 | shoulder_pan, elbow_flex | 9.36 |
| 10 | -5.18 | +9.29 | -11.13 | -2.96 | +0.15 | +1.46 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 10.63 |
| 11 | -6.19 | +0.36 | -13.89 | -1.71 | +0.24 | +2.42 | 2 | shoulder_pan, elbow_flex | 12.36 |
| 12 | -7.35 | -15.03 | -1.64 | -4.63 | +0.03 | +2.64 | 3 | shoulder_pan, shoulder_lift, wrist_flex | 21.00 |
| 13 | -0.31 | +1.87 | -7.57 | -0.83 | +0.23 | +0.92 | 1 | elbow_flex | 4.03 |
| 14 | -7.78 | +2.12 | -12.44 | -3.97 | +0.17 | +1.48 | 2 | shoulder_pan, elbow_flex | 12.03 |
| 15 | -9.04 | +12.58 | -14.88 | -3.37 | +0.21 | +0.94 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 16.78 |
| 16 | -10.71 | +12.93 | -16.82 | +2.22 | +0.22 | +2.60 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 19.49 |
| 17 | -8.11 | +4.39 | -3.27 | -5.54 | +0.16 | -0.72 | 2 | shoulder_pan, wrist_flex | 9.57 |
| 18 | -2.57 | -1.56 | -4.46 | -3.11 | -0.06 | -0.25 | 0 | - | 6.37 |
| 19 | -3.33 | +2.36 | +2.21 | -10.14 | +0.27 | +3.20 | 1 | wrist_flex | 12.36 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -6.13 | 2.96 | -12.50 | -0.31 | 13/20 | 65% |
| shoulder_lift | +3.29 | 6.97 | -15.03 | +18.84 | 7/20 | 35% |
| elbow_flex | -9.76 | 5.67 | -22.30 | +2.21 | 15/20 | 75% |
| wrist_flex | -2.28 | 2.61 | -10.14 | +2.22 | 3/20 | 15% |
| wrist_roll | +0.16 | 0.08 | -0.06 | +0.27 | 0/20 | 0% |
| gripper | +1.04 | 1.51 | -1.33 | +4.72 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 2/20 ([2, 18])
- clamp-joint-count distribution (count -> #seeds): {'0': 2, '1': 4, '2': 8, '3': 6}
- L2 error vs nearest-demo immediate GT delta (deg): mean=12.14, std=5.01, min=4.03, max=21.00
