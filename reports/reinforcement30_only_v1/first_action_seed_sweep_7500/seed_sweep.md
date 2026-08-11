# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/reinforcement30_only/reinforcement30_only_v1/checkpoints/007500/pretrained_model`
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
| 0 | -2.23 | +2.72 | -9.05 | -0.28 | +0.19 | -0.03 | 1 | elbow_flex | 5.50 |
| 1 | -1.93 | +9.06 | -7.71 | -1.59 | +0.22 | +0.27 | 2 | shoulder_lift, elbow_flex | 6.69 |
| 2 | -0.57 | -1.09 | -3.92 | -1.09 | +0.20 | +0.23 | 0 | - | 5.00 |
| 3 | -0.53 | +0.44 | -6.90 | -0.02 | +0.18 | +0.30 | 1 | elbow_flex | 4.47 |
| 4 | -2.45 | +0.81 | -6.22 | -0.57 | +0.26 | +1.14 | 1 | elbow_flex | 4.74 |
| 5 | -0.95 | +2.84 | -6.24 | +0.36 | +0.23 | -0.44 | 1 | elbow_flex | 2.85 |
| 6 | -0.81 | -0.51 | -5.33 | -0.22 | +0.01 | +0.48 | 0 | - | 4.70 |
| 7 | -0.99 | +3.43 | -6.75 | -0.50 | +0.11 | -0.10 | 1 | elbow_flex | 2.81 |
| 8 | -1.58 | +0.19 | -4.32 | -0.35 | +0.25 | +0.31 | 0 | - | 4.18 |
| 9 | -0.81 | -1.00 | -4.77 | -0.65 | +0.18 | -0.31 | 0 | - | 4.94 |
| 10 | -1.58 | +4.25 | -6.93 | -1.22 | +0.19 | +0.38 | 1 | elbow_flex | 3.31 |
| 11 | -1.43 | +0.83 | -6.65 | -0.36 | +0.24 | +0.71 | 1 | elbow_flex | 4.33 |
| 12 | -0.89 | -4.62 | -2.71 | -1.29 | +0.00 | +0.68 | 0 | - | 8.71 |
| 13 | -0.08 | +0.50 | -6.14 | -0.20 | +0.24 | +0.21 | 1 | elbow_flex | 3.92 |
| 14 | -1.74 | +1.88 | -6.22 | -1.04 | +0.20 | +0.08 | 1 | elbow_flex | 3.38 |
| 15 | -2.34 | +5.92 | -7.90 | -1.50 | +0.23 | +0.11 | 2 | shoulder_lift, elbow_flex | 4.95 |
| 16 | -2.75 | +3.49 | -8.52 | +0.71 | +0.23 | +0.64 | 1 | elbow_flex | 5.57 |
| 17 | -1.71 | +1.32 | -4.02 | -1.69 | +0.21 | -0.22 | 0 | - | 3.25 |
| 18 | -0.25 | +0.80 | -5.10 | -1.18 | -0.09 | +0.56 | 0 | - | 3.26 |
| 19 | -0.96 | +0.84 | -2.15 | -3.83 | +0.29 | +1.10 | 0 | - | 4.98 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.33 | 0.74 | -2.75 | -0.08 | 0/20 | 0% |
| shoulder_lift | +1.61 | 2.79 | -4.62 | +9.06 | 2/20 | 10% |
| elbow_flex | -5.88 | 1.79 | -9.05 | -2.15 | 12/20 | 60% |
| wrist_flex | -0.83 | 0.94 | -3.83 | +0.71 | 0/20 | 0% |
| wrist_roll | +0.18 | 0.09 | -0.09 | +0.29 | 0/20 | 0% |
| gripper | +0.30 | 0.42 | -0.44 | +1.14 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 8/20 ([2, 6, 8, 9, 12, 17, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 8, '1': 10, '2': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.58, std=1.37, min=2.81, max=8.71
