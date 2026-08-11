# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v4/smolvla_pick_drop_v4_fresh/checkpoints/010000/pretrained_model`
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
| 0 | -1.58 | +2.65 | -10.95 | +0.17 | +0.10 | -0.07 | 1 | elbow_flex | 6.76 |
| 1 | -1.03 | +9.85 | -9.85 | -0.05 | +0.38 | -0.14 | 2 | shoulder_lift, elbow_flex | 8.41 |
| 2 | +0.43 | -0.74 | -4.57 | +0.12 | +0.18 | +0.12 | 0 | - | 4.60 |
| 3 | +0.33 | +0.68 | -7.65 | +0.77 | +0.04 | +0.26 | 1 | elbow_flex | 4.83 |
| 4 | -1.56 | +1.83 | -7.33 | -0.01 | +0.59 | +0.83 | 1 | elbow_flex | 3.85 |
| 5 | -0.62 | +3.75 | -7.21 | +0.83 | +0.41 | -0.25 | 1 | elbow_flex | 3.41 |
| 6 | -0.14 | -0.23 | -6.22 | +0.69 | -0.22 | +0.19 | 1 | elbow_flex | 4.60 |
| 7 | -0.57 | +3.85 | -7.91 | +0.90 | -0.00 | -0.33 | 1 | elbow_flex | 4.03 |
| 8 | -0.74 | +0.05 | -4.10 | +0.66 | +0.71 | +0.18 | 0 | - | 3.99 |
| 9 | -0.35 | +0.28 | -6.06 | -0.49 | -0.01 | -0.34 | 1 | elbow_flex | 3.69 |
| 10 | -0.51 | +4.81 | -8.70 | +0.17 | +0.02 | +0.38 | 1 | elbow_flex | 4.70 |
| 11 | -0.37 | +1.44 | -7.27 | +0.34 | +0.38 | +0.56 | 1 | elbow_flex | 3.94 |
| 12 | +0.15 | -3.90 | -3.85 | -0.67 | -0.42 | +0.85 | 0 | - | 7.61 |
| 13 | +0.61 | +1.36 | -5.63 | +0.36 | +0.44 | +0.16 | 0 | - | 3.21 |
| 14 | -0.95 | +2.49 | -7.73 | -0.31 | +0.12 | +0.16 | 1 | elbow_flex | 3.58 |
| 15 | -1.63 | +6.53 | -9.97 | -0.07 | +0.33 | +0.13 | 2 | shoulder_lift, elbow_flex | 6.47 |
| 16 | -1.71 | +4.90 | -8.97 | +0.98 | +0.28 | +0.69 | 1 | elbow_flex | 5.39 |
| 17 | -1.39 | +1.54 | -4.82 | -0.66 | +0.07 | -0.17 | 0 | - | 2.18 |
| 18 | +0.93 | +1.31 | -6.75 | +0.31 | -0.59 | +0.47 | 1 | elbow_flex | 4.00 |
| 19 | +0.54 | +0.20 | -2.00 | -2.13 | +0.92 | +1.11 | 0 | - | 4.74 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.51 | 0.80 | -1.71 | +0.93 | 0/20 | 0% |
| shoulder_lift | +2.13 | 2.88 | -3.90 | +9.85 | 2/20 | 10% |
| elbow_flex | -6.88 | 2.22 | -10.95 | -2.00 | 14/20 | 70% |
| wrist_flex | +0.10 | 0.71 | -2.13 | +0.98 | 0/20 | 0% |
| wrist_roll | +0.19 | 0.35 | -0.59 | +0.92 | 0/20 | 0% |
| gripper | +0.24 | 0.40 | -0.34 | +1.11 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 6/20 ([2, 8, 12, 13, 17, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 6, '1': 12, '2': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.70, std=1.51, min=2.18, max=8.41
