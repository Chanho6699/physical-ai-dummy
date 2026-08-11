# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/reinforcement30_only/reinforcement30_only_v1/checkpoints/005000/pretrained_model`
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
| 0 | -1.95 | +2.45 | -11.97 | -1.50 | +0.15 | +0.88 | 1 | elbow_flex | 8.16 |
| 1 | -1.43 | +11.66 | -10.61 | -2.69 | +0.20 | +1.32 | 2 | shoulder_lift, elbow_flex | 10.47 |
| 2 | -0.05 | -1.70 | -4.78 | -1.90 | +0.17 | +1.25 | 0 | - | 5.81 |
| 3 | +0.63 | -0.41 | -9.28 | -0.61 | +0.14 | +1.31 | 1 | elbow_flex | 6.72 |
| 4 | -3.32 | +0.83 | -7.96 | -2.01 | +0.23 | +2.29 | 1 | elbow_flex | 6.54 |
| 5 | -0.71 | +4.14 | -7.53 | -0.19 | +0.20 | +0.34 | 1 | elbow_flex | 3.56 |
| 6 | +0.49 | -0.12 | -7.26 | -0.61 | -0.01 | +1.24 | 1 | elbow_flex | 5.16 |
| 7 | -0.65 | +4.97 | -9.44 | -0.54 | +0.06 | +0.64 | 1 | elbow_flex | 5.43 |
| 8 | -2.09 | +0.18 | -5.71 | -1.49 | +0.23 | +1.37 | 0 | - | 4.87 |
| 9 | -0.08 | -1.91 | -5.66 | -1.78 | +0.14 | +0.58 | 0 | - | 5.97 |
| 10 | -1.12 | +5.64 | -8.91 | -2.22 | +0.14 | +1.66 | 2 | shoulder_lift, elbow_flex | 5.63 |
| 11 | -0.91 | +0.10 | -7.79 | -1.50 | +0.22 | +1.58 | 1 | elbow_flex | 5.56 |
| 12 | -0.88 | -6.68 | -3.44 | -2.70 | -0.01 | +2.27 | 1 | shoulder_lift | 11.01 |
| 13 | +1.07 | +0.28 | -6.82 | -1.41 | +0.21 | +1.22 | 1 | elbow_flex | 4.67 |
| 14 | -1.79 | +1.67 | -7.53 | -2.57 | +0.16 | +1.12 | 1 | elbow_flex | 4.82 |
| 15 | -2.59 | +7.26 | -10.12 | -2.70 | +0.21 | +0.90 | 2 | shoulder_lift, elbow_flex | 7.60 |
| 16 | -3.18 | +4.86 | -11.40 | +0.44 | +0.20 | +1.80 | 1 | elbow_flex | 8.36 |
| 17 | -1.65 | +1.79 | -5.03 | -3.19 | +0.17 | +0.31 | 0 | - | 3.61 |
| 18 | +1.06 | +0.80 | -7.06 | -1.97 | -0.10 | +1.89 | 1 | elbow_flex | 4.78 |
| 19 | -0.52 | -0.01 | -3.46 | -5.79 | +0.28 | +2.47 | 1 | wrist_flex | 6.79 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.98 | 1.26 | -3.32 | +1.07 | 0/20 | 0% |
| shoulder_lift | +1.79 | 3.79 | -6.68 | +11.66 | 4/20 | 20% |
| elbow_flex | -7.59 | 2.39 | -11.97 | -3.44 | 14/20 | 70% |
| wrist_flex | -1.85 | 1.30 | -5.79 | +0.44 | 1/20 | 5% |
| wrist_roll | +0.15 | 0.09 | -0.10 | +0.28 | 0/20 | 0% |
| gripper | +1.32 | 0.61 | +0.31 | +2.47 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([2, 8, 9, 17])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 13, '2': 3}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.28, std=1.96, min=3.56, max=11.01
