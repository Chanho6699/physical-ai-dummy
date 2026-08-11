# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/reinforcement30_only/reinforcement30_only_v1/checkpoints/010000/pretrained_model`
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
| 0 | -1.82 | +2.42 | -9.76 | -0.35 | +0.19 | -0.14 | 1 | elbow_flex | 6.01 |
| 1 | -1.57 | +8.85 | -9.14 | -1.43 | +0.22 | +0.19 | 2 | shoulder_lift, elbow_flex | 7.25 |
| 2 | -0.55 | -1.19 | -5.11 | -0.90 | +0.20 | +0.26 | 0 | - | 5.14 |
| 3 | -0.55 | +0.13 | -7.68 | -0.10 | +0.18 | +0.16 | 1 | elbow_flex | 5.14 |
| 4 | -2.13 | +1.39 | -7.16 | -0.53 | +0.26 | +0.95 | 1 | elbow_flex | 4.65 |
| 5 | -0.81 | +2.70 | -7.32 | +0.32 | +0.22 | -0.38 | 1 | elbow_flex | 3.64 |
| 6 | -0.68 | -1.38 | -6.11 | -0.34 | +0.03 | +0.24 | 1 | elbow_flex | 5.62 |
| 7 | -1.09 | +2.86 | -7.80 | -0.33 | +0.13 | -0.11 | 1 | elbow_flex | 3.91 |
| 8 | -1.34 | -0.02 | -5.58 | -0.38 | +0.25 | +0.06 | 0 | - | 4.40 |
| 9 | -0.66 | -0.89 | -5.90 | -0.74 | +0.18 | -0.25 | 1 | elbow_flex | 5.03 |
| 10 | -1.55 | +3.53 | -7.95 | -1.20 | +0.19 | +0.33 | 1 | elbow_flex | 4.11 |
| 11 | -0.97 | +1.22 | -7.46 | -0.38 | +0.24 | +0.55 | 1 | elbow_flex | 4.38 |
| 12 | -0.61 | -4.50 | -3.69 | -1.05 | +0.02 | +0.33 | 0 | - | 8.38 |
| 13 | +0.32 | +0.70 | -6.99 | -0.32 | +0.23 | +0.04 | 1 | elbow_flex | 4.17 |
| 14 | -1.50 | +1.72 | -7.24 | -0.99 | +0.21 | -0.07 | 1 | elbow_flex | 3.99 |
| 15 | -1.87 | +5.47 | -8.89 | -1.36 | +0.23 | -0.09 | 2 | shoulder_lift, elbow_flex | 5.32 |
| 16 | -2.47 | +3.57 | -9.32 | +0.68 | +0.23 | +0.50 | 1 | elbow_flex | 6.03 |
| 17 | -1.49 | +1.08 | -5.39 | -1.61 | +0.21 | -0.21 | 0 | - | 3.44 |
| 18 | -0.50 | -0.38 | -6.00 | -1.09 | -0.08 | +0.32 | 1 | elbow_flex | 4.59 |
| 19 | -0.98 | +0.84 | -3.66 | -3.42 | +0.28 | +0.84 | 0 | - | 4.21 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.14 | 0.65 | -2.47 | +0.32 | 0/20 | 0% |
| shoulder_lift | +1.41 | 2.72 | -4.50 | +8.85 | 2/20 | 10% |
| elbow_flex | -6.91 | 1.68 | -9.76 | -3.66 | 15/20 | 75% |
| wrist_flex | -0.77 | 0.83 | -3.42 | +0.68 | 0/20 | 0% |
| wrist_roll | +0.18 | 0.09 | -0.08 | +0.28 | 0/20 | 0% |
| gripper | +0.18 | 0.34 | -0.38 | +0.95 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 5/20 ([2, 8, 12, 17, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 5, '1': 13, '2': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.97, std=1.20, min=3.44, max=8.38
