# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v3_v4_combined69/smolvla_pick_drop_v3_v4_combined69_uniform_fresh/checkpoints/010000/pretrained_model`
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
| 0 | -0.34 | +1.20 | -8.40 | -0.38 | -0.08 | +0.44 | 1 | elbow_flex | 4.74 |
| 1 | +0.11 | +8.80 | -6.62 | -1.04 | +0.13 | +0.57 | 2 | shoulder_lift, elbow_flex | 5.86 |
| 2 | +1.23 | -0.53 | -2.97 | -0.57 | +0.03 | +0.70 | 0 | - | 4.87 |
| 3 | +1.08 | -0.40 | -6.07 | +0.12 | -0.04 | +0.88 | 1 | elbow_flex | 4.93 |
| 4 | -1.09 | -0.49 | -4.43 | -0.63 | +0.36 | +1.45 | 0 | - | 4.46 |
| 5 | +0.96 | +1.91 | -4.89 | +0.16 | +0.18 | +0.19 | 0 | - | 2.74 |
| 6 | +0.92 | -1.26 | -4.39 | -0.20 | -0.22 | +0.79 | 0 | - | 5.26 |
| 7 | +0.46 | +3.81 | -6.05 | +0.03 | -0.05 | -0.03 | 1 | elbow_flex | 2.33 |
| 8 | +0.59 | -0.47 | -1.93 | -0.17 | +0.51 | +0.86 | 0 | - | 5.14 |
| 9 | +0.24 | -0.49 | -4.02 | -0.98 | -0.14 | +0.27 | 0 | - | 4.19 |
| 10 | +0.55 | +3.62 | -5.59 | -0.75 | -0.06 | +0.76 | 0 | - | 2.09 |
| 11 | +0.59 | -0.42 | -4.81 | -0.30 | +0.25 | +1.01 | 0 | - | 4.46 |
| 12 | +0.36 | -5.38 | -1.83 | -0.71 | -0.40 | +1.49 | 1 | shoulder_lift | 9.54 |
| 13 | +1.95 | +0.16 | -3.80 | -0.37 | +0.30 | +0.81 | 0 | - | 4.55 |
| 14 | +0.01 | +0.51 | -4.46 | -0.97 | +0.03 | +0.68 | 0 | - | 3.27 |
| 15 | -0.12 | +4.27 | -5.99 | -0.92 | +0.13 | +0.40 | 1 | elbow_flex | 1.99 |
| 16 | -0.86 | +4.26 | -7.49 | +0.93 | +0.17 | +1.29 | 1 | elbow_flex | 4.10 |
| 17 | -0.24 | +1.83 | -2.80 | -1.45 | +0.04 | +0.28 | 0 | - | 2.52 |
| 18 | +0.98 | -0.00 | -3.80 | -0.33 | -0.55 | +0.91 | 0 | - | 4.28 |
| 19 | +1.02 | +0.08 | -0.07 | -2.30 | +0.91 | +1.50 | 0 | - | 6.29 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | +0.42 | 0.72 | -1.09 | +1.95 | 0/20 | 0% |
| shoulder_lift | +1.05 | 2.83 | -5.38 | +8.80 | 2/20 | 10% |
| elbow_flex | -4.52 | 1.96 | -8.40 | -0.07 | 6/20 | 30% |
| wrist_flex | -0.54 | 0.66 | -2.30 | +0.93 | 0/20 | 0% |
| wrist_roll | +0.08 | 0.31 | -0.55 | +0.91 | 0/20 | 0% |
| gripper | +0.76 | 0.43 | -0.03 | +1.50 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 13/20 ([2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 17, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 13, '1': 6, '2': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.38, std=1.69, min=1.99, max=9.54
