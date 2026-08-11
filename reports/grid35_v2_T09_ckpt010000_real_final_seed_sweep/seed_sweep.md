# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T09` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T09_real_final/shadow_patched.json`)
Checkpoint: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/010000/pretrained_model`
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
| 0 | -2.72 | +6.68 | -14.41 | +0.40 | +0.16 | -0.17 | 2 | shoulder_lift, elbow_flex | 10.60 |
| 1 | -1.08 | +12.81 | -12.53 | -0.31 | +0.29 | -0.78 | 2 | shoulder_lift, elbow_flex | 12.03 |
| 2 | -0.78 | +3.81 | -7.77 | +0.11 | +0.25 | +0.23 | 1 | elbow_flex | 3.36 |
| 3 | -0.83 | +4.32 | -10.41 | +1.21 | +0.18 | +0.01 | 1 | elbow_flex | 6.00 |
| 4 | -2.23 | +5.49 | -10.62 | -0.04 | +0.35 | +0.59 | 2 | shoulder_lift, elbow_flex | 6.75 |
| 5 | -0.94 | +6.46 | -10.38 | +1.14 | +0.22 | -0.42 | 2 | shoulder_lift, elbow_flex | 6.49 |
| 6 | -1.45 | +3.39 | -9.32 | +1.04 | +0.08 | -0.08 | 1 | elbow_flex | 5.08 |
| 7 | -1.55 | +7.60 | -10.70 | +0.85 | +0.17 | -0.96 | 2 | shoulder_lift, elbow_flex | 7.41 |
| 8 | -1.30 | +5.19 | -8.19 | +0.63 | +0.34 | -0.24 | 2 | shoulder_lift, elbow_flex | 4.13 |
| 9 | -1.59 | +4.88 | -10.02 | +0.31 | +0.18 | -0.53 | 1 | elbow_flex | 5.79 |
| 10 | -1.15 | +8.41 | -11.21 | -0.15 | +0.16 | +0.30 | 2 | shoulder_lift, elbow_flex | 8.15 |
| 11 | -1.13 | +4.67 | -9.88 | +0.50 | +0.32 | +0.31 | 1 | elbow_flex | 5.54 |
| 12 | -1.38 | +0.38 | -8.44 | +0.18 | +0.07 | +0.69 | 1 | elbow_flex | 5.50 |
| 13 | -0.51 | +4.69 | -9.49 | +0.52 | +0.31 | +0.08 | 1 | elbow_flex | 5.03 |
| 14 | -1.37 | +6.42 | -10.39 | -0.41 | +0.21 | -0.24 | 2 | shoulder_lift, elbow_flex | 6.54 |
| 15 | -1.89 | +9.68 | -12.09 | -0.41 | +0.23 | -0.49 | 2 | shoulder_lift, elbow_flex | 9.72 |
| 16 | -2.92 | +8.99 | -13.44 | +1.97 | +0.26 | +0.31 | 2 | shoulder_lift, elbow_flex | 10.84 |
| 17 | -1.60 | +5.72 | -8.25 | -0.73 | +0.18 | -0.50 | 2 | shoulder_lift, elbow_flex | 4.56 |
| 18 | -0.35 | +4.46 | -9.53 | +0.88 | -0.02 | -0.19 | 1 | elbow_flex | 5.02 |
| 19 | -0.79 | +5.48 | -6.68 | -1.33 | +0.58 | +0.47 | 2 | shoulder_lift, elbow_flex | 3.36 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.38 | 0.65 | -2.92 | -0.35 | 0/20 | 0% |
| shoulder_lift | +5.98 | 2.57 | +0.38 | +12.81 | 12/20 | 60% |
| elbow_flex | -10.19 | 1.87 | -14.41 | -6.68 | 20/20 | 100% |
| wrist_flex | +0.32 | 0.75 | -1.33 | +1.97 | 0/20 | 0% |
| wrist_roll | +0.23 | 0.12 | -0.02 | +0.58 | 0/20 | 0% |
| gripper | -0.08 | 0.44 | -0.96 | +0.69 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 8, '2': 12}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.59, std=2.43, min=3.36, max=12.03
