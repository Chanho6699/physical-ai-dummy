# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T09` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T09_real_final/shadow_patched.json`)
Checkpoint: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/005000/pretrained_model`
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
| 0 | -4.99 | +6.35 | -19.83 | +1.90 | +0.05 | -0.04 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 16.35 |
| 1 | -2.08 | +16.67 | -15.44 | +0.24 | +0.33 | +0.00 | 2 | shoulder_lift, elbow_flex | 16.92 |
| 2 | -2.01 | +1.32 | -8.47 | +1.24 | +0.21 | +0.49 | 1 | elbow_flex | 5.28 |
| 3 | -1.15 | +3.75 | -14.23 | +3.24 | +0.07 | +0.21 | 1 | elbow_flex | 10.18 |
| 4 | -5.18 | +6.25 | -13.01 | +1.37 | +0.40 | +1.85 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 10.54 |
| 5 | -2.09 | +7.50 | -12.38 | +3.31 | +0.20 | -0.39 | 2 | shoulder_lift, elbow_flex | 9.40 |
| 6 | -1.97 | +3.36 | -12.48 | +2.94 | -0.13 | +0.16 | 1 | elbow_flex | 8.62 |
| 7 | -2.98 | +9.39 | -14.71 | +2.38 | +0.03 | -1.06 | 2 | shoulder_lift, elbow_flex | 12.15 |
| 8 | -3.56 | +5.41 | -9.36 | +2.37 | +0.37 | +0.46 | 2 | shoulder_lift, elbow_flex | 6.66 |
| 9 | -2.69 | +3.29 | -12.95 | +1.88 | +0.04 | -0.51 | 1 | elbow_flex | 8.99 |
| 10 | -2.29 | +10.93 | -13.90 | +0.79 | +0.13 | +0.40 | 2 | shoulder_lift, elbow_flex | 11.96 |
| 11 | -1.96 | +4.01 | -13.01 | +1.82 | +0.33 | +0.81 | 1 | elbow_flex | 8.90 |
| 12 | -3.69 | -5.08 | -9.53 | +1.78 | -0.21 | +1.90 | 1 | elbow_flex | 11.25 |
| 13 | -0.74 | +4.38 | -11.00 | +2.18 | +0.28 | +0.47 | 1 | elbow_flex | 6.82 |
| 14 | -3.13 | +7.00 | -13.86 | +0.54 | +0.17 | +0.04 | 2 | shoulder_lift, elbow_flex | 10.32 |
| 15 | -3.47 | +12.79 | -16.27 | +0.53 | +0.25 | -0.01 | 2 | shoulder_lift, elbow_flex | 15.14 |
| 16 | -4.97 | +10.68 | -17.29 | +4.27 | +0.17 | +1.26 | 4 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex | 15.89 |
| 17 | -4.26 | +7.21 | -8.88 | -0.59 | +0.13 | -0.49 | 2 | shoulder_lift, elbow_flex | 7.05 |
| 18 | -0.05 | +3.60 | -11.23 | +1.96 | -0.32 | +0.61 | 1 | elbow_flex | 6.91 |
| 19 | -0.89 | +4.37 | -5.11 | -1.46 | +0.76 | +1.82 | 0 | - | 3.12 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.71 | 1.42 | -5.18 | -0.05 | 3/20 | 15% |
| shoulder_lift | +6.16 | 4.48 | -5.08 | +16.67 | 11/20 | 55% |
| elbow_flex | -12.65 | 3.30 | -19.83 | -5.11 | 19/20 | 95% |
| wrist_flex | +1.63 | 1.33 | -1.46 | +4.27 | 1/20 | 5% |
| wrist_roll | +0.16 | 0.23 | -0.32 | +0.76 | 0/20 | 0% |
| gripper | +0.40 | 0.79 | -1.06 | +1.90 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 8, '2': 8, '3': 2, '4': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=10.12, std=3.69, min=3.12, max=16.92
