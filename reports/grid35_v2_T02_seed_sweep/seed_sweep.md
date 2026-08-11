# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T02/shadow_synthetic_T02.json`)
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
| shoulder_pan | +0.7033 |
| shoulder_lift | +0.3516 |
| elbow_flex | -1.8901 |
| wrist_flex | +0.3077 |
| wrist_roll | +0.1758 |
| gripper | +0.3686 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -1.74 | +5.68 | -7.73 | +0.70 | +0.24 | +0.82 | 2 | shoulder_lift, elbow_flex | 8.30 |
| 1 | -0.03 | +13.19 | -4.29 | -0.17 | +0.35 | +0.58 | 1 | shoulder_lift | 13.10 |
| 2 | +0.38 | +2.58 | +0.03 | +0.38 | +0.34 | +1.27 | 0 | - | 3.10 |
| 3 | +0.59 | +3.18 | -3.71 | +1.83 | +0.23 | +1.27 | 0 | - | 3.80 |
| 4 | -1.57 | +4.42 | -2.83 | +0.18 | +0.47 | +1.93 | 0 | - | 5.02 |
| 5 | +0.20 | +6.23 | -2.13 | +1.67 | +0.28 | +0.46 | 1 | shoulder_lift | 6.06 |
| 6 | -0.08 | +2.01 | -1.70 | +1.62 | +0.08 | +0.94 | 0 | - | 2.34 |
| 7 | -0.56 | +7.08 | -2.85 | +1.03 | +0.16 | -0.24 | 1 | shoulder_lift | 6.98 |
| 8 | -0.49 | +3.98 | +1.15 | +1.07 | +0.47 | +0.82 | 0 | - | 4.97 |
| 9 | -0.47 | +3.99 | -2.71 | +0.79 | +0.25 | +0.38 | 0 | - | 3.94 |
| 10 | -0.02 | +8.64 | -3.68 | +0.08 | +0.21 | +1.63 | 1 | shoulder_lift | 8.61 |
| 11 | -0.05 | +3.13 | -2.29 | +0.74 | +0.43 | +1.56 | 0 | - | 3.19 |
| 12 | -0.31 | -2.59 | -0.91 | +0.34 | +0.11 | +2.06 | 0 | - | 3.67 |
| 13 | +1.04 | +4.01 | -1.59 | +0.96 | +0.41 | +1.08 | 0 | - | 3.82 |
| 14 | -0.71 | +5.57 | -2.81 | -0.28 | +0.29 | +0.93 | 1 | shoulder_lift | 5.55 |
| 15 | -1.20 | +9.27 | -3.85 | -0.33 | +0.30 | +0.82 | 1 | shoulder_lift | 9.36 |
| 16 | -2.06 | +8.72 | -5.38 | +2.31 | +0.32 | +1.45 | 1 | shoulder_lift | 9.75 |
| 17 | -1.22 | +5.53 | +0.62 | -0.76 | +0.26 | +0.32 | 1 | shoulder_lift | 6.16 |
| 18 | +1.44 | +3.82 | -2.16 | +1.21 | -0.05 | +1.30 | 0 | - | 3.80 |
| 19 | +0.85 | +4.72 | +3.24 | -1.83 | +0.77 | +2.04 | 0 | - | 7.29 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.30 | 0.91 | -2.06 | +1.44 | 0/20 | 0% |
| shoulder_lift | +5.16 | 3.18 | -2.59 | +13.19 | 9/20 | 45% |
| elbow_flex | -2.28 | 2.34 | -7.73 | +3.24 | 1/20 | 5% |
| wrist_flex | +0.58 | 0.95 | -1.83 | +2.31 | 0/20 | 0% |
| wrist_roll | +0.30 | 0.17 | -0.05 | +0.77 | 0/20 | 0% |
| gripper | +1.07 | 0.59 | -0.24 | +2.06 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 11/20 ([2, 3, 4, 6, 8, 9, 11, 12, 13, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 11, '1': 8, '2': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.94, std=2.71, min=2.34, max=13.10
