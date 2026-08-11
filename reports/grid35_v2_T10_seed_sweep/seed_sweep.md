# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T10` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T10/shadow_synthetic_T10.json`)
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
| 0 | -1.36 | +6.50 | -8.51 | +0.69 | +0.21 | +0.49 | 2 | shoulder_lift, elbow_flex | 9.27 |
| 1 | +0.23 | +13.17 | -5.19 | -0.18 | +0.31 | +0.19 | 1 | shoulder_lift | 13.26 |
| 2 | +0.62 | +3.19 | -0.93 | +0.32 | +0.31 | +0.84 | 0 | - | 3.04 |
| 3 | +0.93 | +4.05 | -4.54 | +1.71 | +0.21 | +0.83 | 0 | - | 4.79 |
| 4 | -1.08 | +4.95 | -3.80 | +0.13 | +0.42 | +1.41 | 0 | - | 5.40 |
| 5 | +0.40 | +6.54 | -3.40 | +1.56 | +0.24 | +0.17 | 1 | shoulder_lift | 6.50 |
| 6 | +0.12 | +2.89 | -2.76 | +1.58 | +0.08 | +0.57 | 0 | - | 3.03 |
| 7 | -0.17 | +7.52 | -3.88 | +1.00 | +0.15 | -0.55 | 1 | shoulder_lift | 7.58 |
| 8 | -0.06 | +4.45 | -0.13 | +1.02 | +0.41 | +0.32 | 0 | - | 4.58 |
| 9 | -0.16 | +4.88 | -3.72 | +0.76 | +0.22 | +0.01 | 0 | - | 5.00 |
| 10 | +0.32 | +8.89 | -4.60 | +0.05 | +0.19 | +1.21 | 1 | shoulder_lift | 9.01 |
| 11 | +0.19 | +3.85 | -3.20 | +0.72 | +0.38 | +1.08 | 0 | - | 3.86 |
| 12 | +0.04 | -1.75 | -1.72 | +0.33 | +0.10 | +1.50 | 0 | - | 2.48 |
| 13 | +1.03 | +4.71 | -2.80 | +0.90 | +0.35 | +0.69 | 0 | - | 4.51 |
| 14 | -0.23 | +6.06 | -3.55 | -0.23 | +0.26 | +0.46 | 1 | shoulder_lift | 6.04 |
| 15 | -0.82 | +9.49 | -4.73 | -0.34 | +0.27 | +0.40 | 1 | shoulder_lift | 9.72 |
| 16 | -1.64 | +9.05 | -6.60 | +2.19 | +0.27 | +1.01 | 2 | shoulder_lift, elbow_flex | 10.36 |
| 17 | -0.77 | +5.78 | -0.55 | -0.67 | +0.24 | -0.07 | 1 | shoulder_lift | 5.88 |
| 18 | +1.61 | +4.43 | -3.13 | +1.17 | -0.05 | +0.80 | 0 | - | 4.47 |
| 19 | +0.98 | +5.06 | +1.97 | -1.72 | +0.71 | +1.42 | 0 | - | 6.53 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | +0.01 | 0.82 | -1.64 | +1.61 | 0/20 | 0% |
| shoulder_lift | +5.68 | 2.98 | -1.75 | +13.17 | 9/20 | 45% |
| elbow_flex | -3.29 | 2.26 | -8.51 | +1.97 | 2/20 | 10% |
| wrist_flex | +0.55 | 0.90 | -1.72 | +2.19 | 0/20 | 0% |
| wrist_roll | +0.26 | 0.15 | -0.05 | +0.71 | 0/20 | 0% |
| gripper | +0.64 | 0.53 | -0.55 | +1.50 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 11/20 ([2, 3, 4, 6, 8, 9, 11, 12, 13, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 11, '1': 7, '2': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.27, std=2.75, min=2.48, max=13.26
