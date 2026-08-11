# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T09` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T09/shadow_synthetic_T09.json`)
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
| shoulder_pan | -0.6154 |
| shoulder_lift | +0.1758 |
| elbow_flex | -2.0659 |
| wrist_flex | +0.1319 |
| wrist_roll | +0.3516 |
| gripper | +0.1635 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -1.94 | +7.29 | -9.38 | +1.14 | +0.24 | -0.05 | 2 | shoulder_lift, elbow_flex | 10.34 |
| 1 | -0.48 | +13.93 | -5.95 | +0.23 | +0.35 | -0.35 | 2 | shoulder_lift, elbow_flex | 14.30 |
| 2 | +0.28 | +4.05 | -1.91 | +0.80 | +0.32 | +0.48 | 0 | - | 4.04 |
| 3 | +0.33 | +4.83 | -5.61 | +2.17 | +0.24 | +0.31 | 0 | - | 6.27 |
| 4 | -1.62 | +5.50 | -4.51 | +0.57 | +0.46 | +0.98 | 1 | shoulder_lift | 6.02 |
| 5 | +0.03 | +7.45 | -4.25 | +2.00 | +0.27 | -0.33 | 1 | shoulder_lift | 7.86 |
| 6 | -0.26 | +3.71 | -3.90 | +2.04 | +0.10 | +0.05 | 0 | - | 4.44 |
| 7 | -0.62 | +8.62 | -4.87 | +1.45 | +0.19 | -1.03 | 1 | shoulder_lift | 9.08 |
| 8 | -0.58 | +5.40 | -1.17 | +1.54 | +0.45 | -0.12 | 1 | shoulder_lift | 5.49 |
| 9 | -0.66 | +5.49 | -4.52 | +1.13 | +0.25 | -0.46 | 1 | shoulder_lift | 5.97 |
| 10 | -0.30 | +9.60 | -5.62 | +0.51 | +0.23 | +0.66 | 1 | shoulder_lift | 10.09 |
| 11 | -0.22 | +4.64 | -4.15 | +1.19 | +0.41 | +0.62 | 0 | - | 5.07 |
| 12 | -0.38 | -0.92 | -2.79 | +0.78 | +0.11 | +1.03 | 0 | - | 1.74 |
| 13 | +0.66 | +5.46 | -3.53 | +1.37 | +0.39 | +0.15 | 1 | shoulder_lift | 5.76 |
| 14 | -0.82 | +6.72 | -4.60 | +0.14 | +0.30 | -0.00 | 1 | shoulder_lift | 7.02 |
| 15 | -1.46 | +10.29 | -5.77 | +0.10 | +0.31 | -0.12 | 2 | shoulder_lift, elbow_flex | 10.81 |
| 16 | -2.23 | +10.11 | -7.45 | +2.69 | +0.33 | +0.53 | 2 | shoulder_lift, elbow_flex | 11.71 |
| 17 | -1.21 | +6.62 | -1.52 | -0.24 | +0.25 | -0.47 | 1 | shoulder_lift | 6.54 |
| 18 | +1.08 | +5.04 | -4.10 | +1.47 | -0.04 | +0.17 | 0 | - | 5.71 |
| 19 | +0.21 | +5.62 | +1.10 | -1.31 | +0.72 | +0.94 | 1 | shoulder_lift | 6.57 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.51 | 0.84 | -2.23 | +1.08 | 0/20 | 0% |
| shoulder_lift | +6.47 | 3.01 | -0.92 | +13.93 | 14/20 | 70% |
| elbow_flex | -4.22 | 2.24 | -9.38 | +1.10 | 4/20 | 20% |
| wrist_flex | +0.99 | 0.92 | -1.31 | +2.69 | 0/20 | 0% |
| wrist_roll | +0.30 | 0.15 | -0.04 | +0.72 | 0/20 | 0% |
| gripper | +0.15 | 0.53 | -1.03 | +1.03 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 6/20 ([2, 3, 6, 11, 12, 18])
- clamp-joint-count distribution (count -> #seeds): {'0': 6, '1': 10, '2': 4}
- L2 error vs nearest-demo immediate GT delta (deg): mean=7.24, std=2.91, min=1.74, max=14.30
