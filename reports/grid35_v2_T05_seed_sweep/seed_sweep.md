# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T05` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T05/shadow_synthetic_T05.json`)
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
| shoulder_lift | +0.1758 |
| elbow_flex | -2.0659 |
| wrist_flex | +0.1319 |
| wrist_roll | +0.2637 |
| gripper | +0.1635 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -1.80 | +6.59 | -8.49 | +0.65 | +0.32 | +0.68 | 2 | shoulder_lift, elbow_flex | 9.16 |
| 1 | -0.12 | +13.91 | -4.93 | -0.23 | +0.44 | +0.51 | 1 | shoulder_lift | 14.06 |
| 2 | +0.50 | +3.40 | -0.61 | +0.27 | +0.42 | +1.22 | 0 | - | 3.92 |
| 3 | +0.58 | +4.15 | -4.52 | +1.74 | +0.32 | +1.09 | 0 | - | 5.21 |
| 4 | -1.39 | +4.82 | -3.20 | +0.03 | +0.54 | +1.73 | 0 | - | 5.07 |
| 5 | +0.29 | +6.93 | -3.11 | +1.52 | +0.36 | +0.49 | 1 | shoulder_lift | 7.07 |
| 6 | -0.05 | +2.97 | -2.65 | +1.59 | +0.19 | +0.83 | 0 | - | 3.36 |
| 7 | -0.42 | +8.09 | -3.74 | +0.95 | +0.27 | -0.27 | 1 | shoulder_lift | 8.15 |
| 8 | -0.36 | +4.92 | +0.17 | +1.01 | +0.55 | +0.68 | 0 | - | 5.37 |
| 9 | -0.46 | +4.82 | -3.49 | +0.67 | +0.34 | +0.31 | 0 | - | 4.90 |
| 10 | +0.01 | +9.15 | -4.44 | +0.00 | +0.31 | +1.47 | 1 | shoulder_lift | 9.41 |
| 11 | +0.12 | +3.90 | -2.88 | +0.64 | +0.51 | +1.35 | 0 | - | 4.13 |
| 12 | -0.05 | -2.06 | -1.36 | +0.20 | +0.19 | +1.76 | 0 | - | 2.93 |
| 13 | +0.96 | +4.75 | -2.30 | +0.82 | +0.49 | +0.98 | 0 | - | 5.03 |
| 14 | -0.58 | +6.18 | -3.52 | -0.36 | +0.38 | +0.71 | 1 | shoulder_lift | 6.23 |
| 15 | -1.13 | +9.99 | -4.70 | -0.41 | +0.40 | +0.71 | 1 | shoulder_lift | 10.20 |
| 16 | -1.99 | +9.56 | -6.31 | +2.20 | +0.41 | +1.32 | 2 | shoulder_lift, elbow_flex | 10.64 |
| 17 | -1.03 | +6.08 | -0.20 | -0.79 | +0.35 | +0.22 | 1 | shoulder_lift | 6.27 |
| 18 | +1.37 | +4.41 | -2.81 | +1.08 | +0.05 | +1.02 | 0 | - | 4.98 |
| 19 | +0.71 | +5.12 | +2.64 | -2.01 | +0.83 | +1.82 | 0 | - | 7.52 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.24 | 0.87 | -1.99 | +1.37 | 0/20 | 0% |
| shoulder_lift | +5.88 | 3.19 | -2.06 | +13.91 | 9/20 | 45% |
| elbow_flex | -3.02 | 2.37 | -8.49 | +2.64 | 2/20 | 10% |
| wrist_flex | +0.48 | 0.96 | -2.01 | +2.20 | 0/20 | 0% |
| wrist_roll | +0.38 | 0.16 | +0.05 | +0.83 | 0/20 | 0% |
| gripper | +0.93 | 0.54 | -0.27 | +1.82 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 11/20 ([2, 3, 4, 6, 8, 9, 11, 12, 13, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 11, '1': 7, '2': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.68, std=2.77, min=2.93, max=14.06
