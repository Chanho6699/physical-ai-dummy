# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T06` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T06/shadow_synthetic_T06.json`)
Checkpoint: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model`
Task: `Pick up the cube and place it in the target area.`
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
| shoulder_pan | -0.7912 |
| shoulder_lift | +0.5275 |
| elbow_flex | -1.7143 |
| wrist_flex | -1.0989 |
| wrist_roll | +0.0879 |
| gripper | +0.0313 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -1.98 | +6.76 | -7.95 | +0.40 | +0.28 | +0.79 | 2 | shoulder_lift, elbow_flex | 9.06 |
| 1 | -0.37 | +14.00 | -5.02 | -0.53 | +0.39 | +0.55 | 1 | shoulder_lift | 13.91 |
| 2 | +0.14 | +3.40 | -0.35 | +0.01 | +0.38 | +1.19 | 0 | - | 3.69 |
| 3 | +0.23 | +4.23 | -4.17 | +1.48 | +0.28 | +1.13 | 0 | - | 5.36 |
| 4 | -1.69 | +5.16 | -3.06 | -0.19 | +0.50 | +1.79 | 1 | shoulder_lift | 5.31 |
| 5 | -0.04 | +6.86 | -2.87 | +1.31 | +0.31 | +0.52 | 1 | shoulder_lift | 6.93 |
| 6 | -0.43 | +3.12 | -2.31 | +1.31 | +0.14 | +0.84 | 0 | - | 3.69 |
| 7 | -0.81 | +8.04 | -3.41 | +0.70 | +0.23 | -0.30 | 1 | shoulder_lift | 7.92 |
| 8 | -0.80 | +4.93 | +0.27 | +0.77 | +0.51 | +0.68 | 0 | - | 5.24 |
| 9 | -0.72 | +4.82 | -3.12 | +0.42 | +0.29 | +0.36 | 0 | - | 4.78 |
| 10 | -0.33 | +9.36 | -4.22 | -0.22 | +0.26 | +1.52 | 1 | shoulder_lift | 9.36 |
| 11 | -0.30 | +3.74 | -2.56 | +0.34 | +0.46 | +1.39 | 0 | - | 3.91 |
| 12 | -0.50 | -1.74 | -1.12 | -0.01 | +0.14 | +1.89 | 0 | - | 3.20 |
| 13 | +0.53 | +5.02 | -2.23 | +0.64 | +0.44 | +1.03 | 0 | - | 5.13 |
| 14 | -0.94 | +6.47 | -3.21 | -0.62 | +0.33 | +0.77 | 1 | shoulder_lift | 6.20 |
| 15 | -1.52 | +10.07 | -4.47 | -0.66 | +0.34 | +0.74 | 1 | shoulder_lift | 10.00 |
| 16 | -2.32 | +9.41 | -6.01 | +1.97 | +0.36 | +1.39 | 2 | shoulder_lift, elbow_flex | 10.53 |
| 17 | -1.41 | +6.12 | -0.06 | -1.05 | +0.31 | +0.23 | 1 | shoulder_lift | 5.88 |
| 18 | +0.98 | +4.68 | -2.66 | +0.84 | +0.00 | +1.15 | 0 | - | 5.12 |
| 19 | +0.43 | +5.22 | +2.53 | -2.16 | +0.78 | +1.94 | 1 | shoulder_lift | 6.84 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.59 | 0.85 | -2.32 | +0.98 | 0/20 | 0% |
| shoulder_lift | +5.98 | 3.15 | -1.74 | +14.00 | 11/20 | 55% |
| elbow_flex | -2.80 | 2.28 | -7.95 | +2.53 | 2/20 | 10% |
| wrist_flex | +0.24 | 0.95 | -2.16 | +1.97 | 0/20 | 0% |
| wrist_roll | +0.34 | 0.16 | +0.00 | +0.78 | 0/20 | 0% |
| gripper | +0.98 | 0.56 | -0.30 | +1.94 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 9/20 ([2, 3, 6, 8, 9, 11, 12, 13, 18])
- clamp-joint-count distribution (count -> #seeds): {'0': 9, '1': 9, '2': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.60, std=2.69, min=3.20, max=13.91
