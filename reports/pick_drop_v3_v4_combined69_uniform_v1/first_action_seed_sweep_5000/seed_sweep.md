# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v3_v4_combined69/smolvla_pick_drop_v3_v4_combined69_uniform_fresh/checkpoints/005000/pretrained_model`
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
| shoulder_pan | -0.7033 |
| shoulder_lift | +3.5165 |
| elbow_flex | -4.4396 |
| wrist_flex | -1.0989 |
| wrist_roll | +0.0879 |
| gripper | -0.3945 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -1.77 | +1.54 | -10.46 | -0.32 | +0.17 | +0.91 | 1 | elbow_flex | 6.61 |
| 1 | -0.67 | +10.10 | -5.32 | -1.68 | +0.46 | +0.97 | 1 | shoulder_lift | 6.82 |
| 2 | -0.08 | -1.54 | -2.22 | -0.23 | +0.39 | +1.31 | 0 | - | 5.89 |
| 3 | +0.19 | -0.12 | -7.54 | +1.14 | +0.14 | +1.36 | 1 | elbow_flex | 5.63 |
| 4 | -3.25 | -1.09 | -4.26 | -0.66 | +0.68 | +2.81 | 0 | - | 6.21 |
| 5 | -0.21 | +2.40 | -3.96 | +0.79 | +0.40 | +0.73 | 0 | - | 2.57 |
| 6 | -0.07 | -0.98 | -4.17 | +0.43 | -0.05 | +1.49 | 0 | - | 5.16 |
| 7 | -1.13 | +4.37 | -6.48 | +0.62 | +0.11 | +0.35 | 1 | elbow_flex | 2.93 |
| 8 | -1.56 | -1.37 | -0.22 | +0.07 | +0.97 | +1.56 | 0 | - | 6.96 |
| 9 | -0.62 | -0.72 | -4.22 | -0.21 | +0.16 | +0.59 | 0 | - | 4.45 |
| 10 | -0.95 | +4.97 | -5.47 | -0.87 | +0.17 | +1.42 | 0 | - | 2.56 |
| 11 | +0.08 | -2.02 | -2.78 | -0.68 | +0.62 | +1.32 | 0 | - | 6.12 |
| 12 | -1.21 | -7.12 | -0.64 | -0.80 | -0.34 | +2.94 | 1 | shoulder_lift | 11.80 |
| 13 | +1.17 | -0.25 | -2.58 | -0.12 | +0.71 | +1.73 | 0 | - | 5.19 |
| 14 | -1.74 | +0.35 | -3.94 | -1.27 | +0.25 | +1.43 | 0 | - | 3.83 |
| 15 | -1.81 | +4.70 | -5.01 | -1.41 | +0.44 | +0.84 | 0 | - | 2.17 |
| 16 | -3.13 | +4.90 | -8.45 | +1.72 | +0.31 | +2.40 | 1 | elbow_flex | 6.30 |
| 17 | -1.96 | +2.07 | -1.53 | -1.85 | +0.47 | +0.42 | 0 | - | 3.67 |
| 18 | -0.00 | +0.44 | -4.18 | +0.01 | -0.56 | +1.94 | 0 | - | 4.14 |
| 19 | +0.61 | -1.23 | +2.47 | -3.11 | +1.60 | +2.09 | 1 | wrist_roll | 9.19 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.91 | 1.14 | -3.25 | +1.17 | 0/20 | 0% |
| shoulder_lift | +0.97 | 3.53 | -7.12 | +10.10 | 2/20 | 10% |
| elbow_flex | -4.05 | 2.88 | -10.46 | +2.47 | 4/20 | 20% |
| wrist_flex | -0.42 | 1.09 | -3.11 | +1.72 | 0/20 | 0% |
| wrist_roll | +0.36 | 0.45 | -0.56 | +1.60 | 1/20 | 5% |
| gripper | +1.43 | 0.71 | +0.35 | +2.94 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 13/20 ([2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18])
- clamp-joint-count distribution (count -> #seeds): {'0': 13, '1': 7}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.41, std=2.28, min=2.17, max=11.80
