# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T02_real_final/shadow_patched.json`)
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
| 0 | -2.66 | +7.25 | -12.41 | +1.11 | +0.27 | +0.15 | 2 | shoulder_lift, elbow_flex | 9.02 |
| 1 | -1.33 | +13.60 | -9.62 | +0.21 | +0.37 | -0.16 | 2 | shoulder_lift, elbow_flex | 11.06 |
| 2 | -0.74 | +4.35 | -5.42 | +0.75 | +0.35 | +0.66 | 0 | - | 1.78 |
| 3 | -0.66 | +5.09 | -8.85 | +2.09 | +0.26 | +0.54 | 1 | elbow_flex | 4.95 |
| 4 | -2.42 | +5.80 | -8.22 | +0.61 | +0.46 | +1.14 | 2 | shoulder_lift, elbow_flex | 5.10 |
| 5 | -0.94 | +7.32 | -8.09 | +1.93 | +0.30 | -0.03 | 2 | shoulder_lift, elbow_flex | 5.33 |
| 6 | -1.33 | +3.83 | -7.23 | +1.94 | +0.14 | +0.29 | 1 | elbow_flex | 3.53 |
| 7 | -1.56 | +8.37 | -8.21 | +1.35 | +0.22 | -0.78 | 2 | shoulder_lift, elbow_flex | 6.15 |
| 8 | -1.47 | +5.49 | -4.79 | +1.49 | +0.47 | +0.15 | 1 | shoulder_lift | 2.74 |
| 9 | -1.51 | +5.53 | -8.09 | +1.01 | +0.28 | -0.22 | 2 | shoulder_lift, elbow_flex | 4.30 |
| 10 | -1.27 | +9.28 | -8.92 | +0.39 | +0.25 | +0.84 | 2 | shoulder_lift, elbow_flex | 7.17 |
| 11 | -1.19 | +4.63 | -7.56 | +1.14 | +0.42 | +0.81 | 1 | elbow_flex | 3.68 |
| 12 | -1.30 | -0.09 | -6.13 | +0.85 | +0.17 | +1.18 | 1 | elbow_flex | 4.78 |
| 13 | -0.43 | +5.50 | -7.47 | +1.37 | +0.41 | +0.43 | 2 | shoulder_lift, elbow_flex | 3.65 |
| 14 | -1.67 | +6.81 | -7.97 | +0.17 | +0.32 | +0.13 | 2 | shoulder_lift, elbow_flex | 4.87 |
| 15 | -2.22 | +10.16 | -9.13 | +0.07 | +0.33 | +0.05 | 2 | shoulder_lift, elbow_flex | 8.12 |
| 16 | -3.05 | +9.77 | -10.95 | +2.65 | +0.34 | +0.70 | 2 | shoulder_lift, elbow_flex | 9.62 |
| 17 | -2.05 | +6.51 | -5.19 | -0.23 | +0.29 | -0.26 | 1 | shoulder_lift | 3.55 |
| 18 | -0.11 | +5.38 | -7.62 | +1.48 | +0.04 | +0.46 | 2 | shoulder_lift, elbow_flex | 3.69 |
| 19 | -0.65 | +5.75 | -2.56 | -1.32 | +0.75 | +1.10 | 1 | shoulder_lift | 3.68 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.43 | 0.74 | -3.05 | -0.11 | 0/20 | 0% |
| shoulder_lift | +6.52 | 2.77 | -0.09 | +13.60 | 15/20 | 75% |
| elbow_flex | -7.72 | 2.14 | -12.41 | -2.56 | 16/20 | 80% |
| wrist_flex | +0.95 | 0.90 | -1.32 | +2.65 | 0/20 | 0% |
| wrist_roll | +0.32 | 0.14 | +0.04 | +0.75 | 0/20 | 0% |
| gripper | +0.36 | 0.51 | -0.78 | +1.18 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([2])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 7, '2': 12}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.34, std=2.40, min=1.78, max=11.06
