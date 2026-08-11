# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T03` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T03/shadow_synthetic_T03.json`)
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
| shoulder_pan | +0.1758 |
| shoulder_lift | +0.6154 |
| elbow_flex | -2.3297 |
| wrist_flex | +0.1319 |
| wrist_roll | +0.0000 |
| gripper | -0.3216 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -3.15 | +6.53 | -9.35 | +0.78 | +0.34 | +0.24 | 2 | shoulder_lift, elbow_flex | 9.81 |
| 1 | -1.49 | +13.53 | -6.00 | -0.10 | +0.46 | +0.04 | 2 | shoulder_lift, elbow_flex | 13.54 |
| 2 | -0.97 | +3.09 | -1.42 | +0.37 | +0.44 | +0.81 | 0 | - | 3.13 |
| 3 | -0.83 | +3.98 | -5.19 | +1.81 | +0.34 | +0.64 | 0 | - | 4.94 |
| 4 | -2.84 | +4.82 | -4.05 | +0.26 | +0.55 | +1.31 | 0 | - | 5.72 |
| 5 | -1.16 | +6.64 | -3.90 | +1.68 | +0.38 | +0.02 | 1 | shoulder_lift | 6.57 |
| 6 | -1.54 | +2.73 | -3.31 | +1.64 | +0.20 | +0.40 | 0 | - | 3.35 |
| 7 | -1.88 | +7.69 | -4.42 | +1.04 | +0.28 | -0.76 | 1 | shoulder_lift | 7.73 |
| 8 | -1.81 | +4.56 | -0.77 | +1.14 | +0.56 | +0.26 | 0 | - | 4.86 |
| 9 | -1.83 | +4.77 | -4.26 | +0.82 | +0.35 | -0.12 | 0 | - | 5.06 |
| 10 | -1.41 | +8.92 | -5.35 | +0.16 | +0.33 | +1.02 | 1 | shoulder_lift | 9.08 |
| 11 | -1.33 | +3.68 | -3.66 | +0.75 | +0.52 | +0.95 | 0 | - | 3.97 |
| 12 | -1.59 | -2.01 | -2.19 | +0.32 | +0.21 | +1.42 | 0 | - | 3.62 |
| 13 | -0.46 | +4.68 | -3.20 | +0.97 | +0.50 | +0.60 | 0 | - | 4.41 |
| 14 | -2.05 | +6.12 | -4.34 | -0.16 | +0.39 | +0.28 | 1 | shoulder_lift | 6.32 |
| 15 | -2.68 | +9.71 | -5.67 | -0.23 | +0.41 | +0.24 | 1 | shoulder_lift | 10.13 |
| 16 | -3.37 | +9.16 | -7.09 | +2.31 | +0.42 | +0.87 | 2 | shoulder_lift, elbow_flex | 10.70 |
| 17 | -2.52 | +5.99 | -1.06 | -0.65 | +0.36 | -0.21 | 1 | shoulder_lift | 6.21 |
| 18 | +0.00 | +4.25 | -3.60 | +1.15 | +0.07 | +0.64 | 0 | - | 4.10 |
| 19 | -0.69 | +4.91 | +1.61 | -1.74 | +0.84 | +1.36 | 0 | - | 6.47 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.68 | 0.87 | -3.37 | +0.00 | 0/20 | 0% |
| shoulder_lift | +5.69 | 3.12 | -2.01 | +13.53 | 9/20 | 45% |
| elbow_flex | -3.86 | 2.35 | -9.35 | +1.61 | 3/20 | 15% |
| wrist_flex | +0.62 | 0.92 | -1.74 | +2.31 | 0/20 | 0% |
| wrist_roll | +0.40 | 0.16 | +0.07 | +0.84 | 0/20 | 0% |
| gripper | +0.50 | 0.56 | -0.76 | +1.42 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 11/20 ([2, 3, 4, 6, 8, 9, 11, 12, 13, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 11, '1': 6, '2': 3}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.49, std=2.77, min=3.13, max=13.54
