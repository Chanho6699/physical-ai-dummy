# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T09` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T09_real_final/shadow_patched.json`)
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
| 0 | -2.82 | +8.02 | -14.28 | +0.78 | +0.17 | +0.00 | 2 | shoulder_lift, elbow_flex | 10.96 |
| 1 | -1.35 | +14.48 | -11.21 | -0.15 | +0.29 | -0.34 | 2 | shoulder_lift, elbow_flex | 12.60 |
| 2 | -0.82 | +4.75 | -6.70 | +0.33 | +0.26 | +0.52 | 1 | elbow_flex | 2.64 |
| 3 | -0.66 | +5.67 | -10.19 | +1.68 | +0.17 | +0.37 | 2 | shoulder_lift, elbow_flex | 6.14 |
| 4 | -2.49 | +6.59 | -9.53 | +0.20 | +0.36 | +1.04 | 2 | shoulder_lift, elbow_flex | 6.38 |
| 5 | -0.96 | +7.82 | -9.26 | +1.54 | +0.20 | -0.22 | 2 | shoulder_lift, elbow_flex | 6.35 |
| 6 | -1.32 | +4.41 | -8.39 | +1.47 | +0.04 | +0.14 | 1 | elbow_flex | 4.31 |
| 7 | -1.68 | +9.08 | -9.59 | +0.96 | +0.13 | -0.92 | 2 | shoulder_lift, elbow_flex | 7.51 |
| 8 | -1.38 | +5.94 | -6.11 | +1.02 | +0.37 | -0.02 | 2 | shoulder_lift, elbow_flex | 3.14 |
| 9 | -1.61 | +6.29 | -9.51 | +0.66 | +0.18 | -0.37 | 2 | shoulder_lift, elbow_flex | 5.77 |
| 10 | -1.30 | +10.17 | -10.49 | +0.07 | +0.16 | +0.71 | 2 | shoulder_lift, elbow_flex | 8.81 |
| 11 | -1.27 | +5.29 | -8.97 | +0.75 | +0.33 | +0.64 | 2 | shoulder_lift, elbow_flex | 4.94 |
| 12 | -1.32 | +0.11 | -7.18 | +0.42 | +0.06 | +1.00 | 1 | elbow_flex | 4.97 |
| 13 | -0.39 | +6.11 | -8.68 | +0.95 | +0.32 | +0.27 | 2 | shoulder_lift, elbow_flex | 4.79 |
| 14 | -1.65 | +7.58 | -9.45 | -0.22 | +0.22 | +0.01 | 2 | shoulder_lift, elbow_flex | 6.39 |
| 15 | -2.33 | +10.99 | -10.76 | -0.26 | +0.24 | -0.11 | 2 | shoulder_lift, elbow_flex | 9.75 |
| 16 | -3.18 | +10.65 | -12.43 | +2.30 | +0.25 | +0.56 | 2 | shoulder_lift, elbow_flex | 11.11 |
| 17 | -1.98 | +7.13 | -6.44 | -0.66 | +0.19 | -0.36 | 2 | shoulder_lift, elbow_flex | 4.41 |
| 18 | -0.01 | +5.54 | -8.50 | +1.04 | -0.07 | +0.31 | 2 | shoulder_lift, elbow_flex | 4.36 |
| 19 | -0.63 | +6.30 | -4.13 | -1.60 | +0.64 | +0.93 | 1 | shoulder_lift | 3.50 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.46 | 0.79 | -3.18 | -0.01 | 0/20 | 0% |
| shoulder_lift | +7.15 | 2.92 | +0.11 | +14.48 | 17/20 | 85% |
| elbow_flex | -9.09 | 2.24 | -14.28 | -4.13 | 19/20 | 95% |
| wrist_flex | +0.56 | 0.87 | -1.60 | +2.30 | 0/20 | 0% |
| wrist_roll | +0.22 | 0.14 | -0.07 | +0.64 | 0/20 | 0% |
| gripper | +0.21 | 0.51 | -0.92 | +1.04 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 4, '2': 16}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.44, std=2.76, min=2.64, max=12.60
