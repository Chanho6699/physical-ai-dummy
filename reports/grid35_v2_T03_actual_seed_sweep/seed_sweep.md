# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T03_actual/shadow_patched.json`)
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
| shoulder_lift | +3.8681 |
| elbow_flex | -4.6154 |
| wrist_flex | +0.2198 |
| wrist_roll | -0.0879 |
| gripper | -0.3216 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -2.67 | +6.46 | -12.12 | +0.93 | +0.35 | +0.42 | 2 | shoulder_lift, elbow_flex | 8.51 |
| 1 | -1.13 | +13.49 | -8.86 | +0.05 | +0.48 | +0.16 | 2 | shoulder_lift, elbow_flex | 10.62 |
| 2 | -0.61 | +3.61 | -4.64 | +0.59 | +0.45 | +0.92 | 0 | - | 1.63 |
| 3 | -0.52 | +4.11 | -8.13 | +1.93 | +0.34 | +0.82 | 1 | elbow_flex | 4.16 |
| 4 | -2.34 | +5.18 | -7.27 | +0.42 | +0.56 | +1.52 | 2 | shoulder_lift, elbow_flex | 4.35 |
| 5 | -0.78 | +6.73 | -7.19 | +1.86 | +0.39 | +0.16 | 2 | shoulder_lift, elbow_flex | 4.35 |
| 6 | -1.20 | +2.99 | -6.41 | +1.76 | +0.21 | +0.54 | 1 | elbow_flex | 3.02 |
| 7 | -1.51 | +7.78 | -7.56 | +1.22 | +0.30 | -0.56 | 2 | shoulder_lift, elbow_flex | 5.29 |
| 8 | -1.39 | +4.93 | -3.88 | +1.33 | +0.57 | +0.39 | 0 | - | 2.51 |
| 9 | -1.38 | +4.67 | -7.27 | +0.90 | +0.36 | -0.01 | 1 | elbow_flex | 3.30 |
| 10 | -1.09 | +8.87 | -8.27 | +0.25 | +0.34 | +1.12 | 2 | shoulder_lift, elbow_flex | 6.50 |
| 11 | -1.00 | +3.73 | -6.81 | +0.94 | +0.52 | +1.04 | 1 | elbow_flex | 3.00 |
| 12 | -1.13 | -1.44 | -5.26 | +0.55 | +0.22 | +1.53 | 0 | - | 5.82 |
| 13 | -0.06 | +4.75 | -6.39 | +1.16 | +0.51 | +0.70 | 1 | elbow_flex | 2.51 |
| 14 | -1.58 | +6.07 | -7.24 | -0.07 | +0.40 | +0.44 | 2 | shoulder_lift, elbow_flex | 3.96 |
| 15 | -2.11 | +9.72 | -8.48 | -0.14 | +0.43 | +0.35 | 2 | shoulder_lift, elbow_flex | 7.43 |
| 16 | -2.97 | +9.32 | -10.16 | +2.53 | +0.43 | +1.02 | 2 | shoulder_lift, elbow_flex | 8.82 |
| 17 | -1.95 | +6.09 | -4.24 | -0.42 | +0.37 | -0.05 | 1 | shoulder_lift | 3.21 |
| 18 | +0.34 | +4.56 | -6.76 | +1.33 | +0.09 | +0.80 | 1 | elbow_flex | 2.76 |
| 19 | -0.30 | +5.33 | -1.80 | -1.41 | +0.84 | +1.42 | 1 | shoulder_lift | 4.11 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.27 | 0.83 | -2.97 | +0.34 | 0/20 | 0% |
| shoulder_lift | +5.85 | 2.99 | -1.44 | +13.49 | 11/20 | 55% |
| elbow_flex | -6.94 | 2.23 | -12.12 | -1.80 | 15/20 | 75% |
| wrist_flex | +0.79 | 0.91 | -1.41 | +2.53 | 0/20 | 0% |
| wrist_roll | +0.41 | 0.15 | +0.09 | +0.84 | 0/20 | 0% |
| gripper | +0.64 | 0.54 | -0.56 | +1.53 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 3/20 ([2, 8, 12])
- clamp-joint-count distribution (count -> #seeds): {'0': 3, '1': 8, '2': 9}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.79, std=2.38, min=1.63, max=10.62
