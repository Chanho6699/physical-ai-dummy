# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v3_v4_combined69/smolvla_pick_drop_v3_v4_combined69_uniform_fresh/checkpoints/007500/pretrained_model`
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
| 0 | +0.74 | +1.36 | -8.89 | -0.35 | +0.22 | +0.01 | 1 | elbow_flex | 5.22 |
| 1 | +0.90 | +9.56 | -6.99 | -1.04 | +0.39 | -0.01 | 2 | shoulder_lift, elbow_flex | 6.77 |
| 2 | +2.49 | -1.21 | -2.66 | -0.46 | +0.33 | +0.02 | 0 | - | 6.03 |
| 3 | +2.43 | -0.80 | -6.01 | +0.37 | +0.22 | +0.38 | 1 | elbow_flex | 5.80 |
| 4 | -0.14 | -0.53 | -4.74 | -0.56 | +0.66 | +1.00 | 0 | - | 4.40 |
| 5 | +2.29 | +1.82 | -5.22 | +0.28 | +0.42 | -0.36 | 0 | - | 3.80 |
| 6 | +2.29 | -1.67 | -4.59 | -0.06 | +0.07 | +0.03 | 0 | - | 6.10 |
| 7 | +1.84 | +3.48 | -6.20 | +0.22 | +0.24 | -0.52 | 1 | elbow_flex | 3.37 |
| 8 | +1.82 | -0.85 | -1.67 | -0.08 | +0.82 | -0.05 | 0 | - | 5.90 |
| 9 | +1.74 | -0.77 | -4.08 | -0.92 | +0.12 | -0.38 | 0 | - | 4.95 |
| 10 | +1.43 | +4.05 | -6.08 | -0.58 | +0.15 | +0.26 | 1 | elbow_flex | 2.86 |
| 11 | +1.83 | -0.72 | -4.64 | -0.22 | +0.53 | +0.43 | 0 | - | 5.11 |
| 12 | +1.95 | -6.51 | -1.15 | -0.97 | -0.12 | +0.80 | 1 | shoulder_lift | 10.95 |
| 13 | +3.34 | +0.06 | -4.12 | -0.11 | +0.57 | +0.18 | 0 | - | 5.48 |
| 14 | +1.02 | +0.80 | -4.85 | -0.89 | +0.30 | +0.07 | 0 | - | 3.29 |
| 15 | +0.73 | +5.12 | -6.74 | -0.92 | +0.38 | -0.33 | 1 | elbow_flex | 3.17 |
| 16 | +0.37 | +4.17 | -7.84 | +0.99 | +0.43 | +0.60 | 1 | elbow_flex | 4.32 |
| 17 | +0.97 | +1.44 | -2.61 | -1.41 | +0.35 | -0.32 | 0 | - | 3.26 |
| 18 | +2.28 | -0.54 | -3.71 | -0.35 | -0.27 | +0.47 | 0 | - | 5.23 |
| 19 | +2.54 | -0.57 | +0.64 | -2.62 | +1.15 | +0.83 | 0 | - | 7.61 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | +1.64 | 0.85 | -0.14 | +3.34 | 0/20 | 0% |
| shoulder_lift | +0.88 | 3.21 | -6.51 | +9.56 | 2/20 | 10% |
| elbow_flex | -4.61 | 2.27 | -8.89 | +0.64 | 7/20 | 35% |
| wrist_flex | -0.49 | 0.75 | -2.62 | +0.99 | 0/20 | 0% |
| wrist_roll | +0.35 | 0.31 | -0.27 | +1.15 | 0/20 | 0% |
| gripper | +0.16 | 0.42 | -0.52 | +1.00 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 12/20 ([2, 4, 5, 6, 8, 9, 11, 13, 14, 17, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 12, '1': 7, '2': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.18, std=1.84, min=2.86, max=10.95
