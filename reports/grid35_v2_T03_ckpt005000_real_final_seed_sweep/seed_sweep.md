# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T03` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T03_real_final/shadow_patched.json`)
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
| 0 | -5.16 | +6.23 | -18.38 | +2.03 | +0.09 | +0.28 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 15.07 |
| 1 | -2.31 | +16.42 | -14.03 | +0.32 | +0.36 | +0.35 | 2 | shoulder_lift, elbow_flex | 15.91 |
| 2 | -2.17 | +1.47 | -7.55 | +1.43 | +0.24 | +0.78 | 1 | elbow_flex | 4.75 |
| 3 | -1.37 | +3.83 | -13.23 | +3.40 | +0.10 | +0.50 | 1 | elbow_flex | 9.35 |
| 4 | -5.33 | +6.15 | -11.85 | +1.47 | +0.43 | +2.15 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 9.79 |
| 5 | -2.24 | +7.40 | -11.38 | +3.42 | +0.24 | -0.08 | 2 | shoulder_lift, elbow_flex | 8.63 |
| 6 | -2.20 | +3.42 | -11.55 | +3.10 | -0.09 | +0.48 | 1 | elbow_flex | 7.93 |
| 7 | -3.07 | +9.22 | -13.57 | +2.55 | +0.07 | -0.75 | 2 | shoulder_lift, elbow_flex | 11.18 |
| 8 | -3.71 | +5.56 | -8.32 | +2.51 | +0.41 | +0.76 | 2 | shoulder_lift, elbow_flex | 6.19 |
| 9 | -2.85 | +3.19 | -11.74 | +1.96 | +0.09 | -0.20 | 1 | elbow_flex | 7.96 |
| 10 | -2.53 | +10.84 | -12.79 | +0.86 | +0.16 | +0.70 | 2 | shoulder_lift, elbow_flex | 11.15 |
| 11 | -2.21 | +4.15 | -12.01 | +1.98 | +0.36 | +1.10 | 1 | elbow_flex | 8.11 |
| 12 | -3.78 | -5.01 | -8.34 | +1.89 | -0.15 | +2.24 | 1 | elbow_flex | 10.84 |
| 13 | -1.01 | +4.38 | -10.03 | +2.27 | +0.31 | +0.77 | 1 | elbow_flex | 6.04 |
| 14 | -3.33 | +6.99 | -12.65 | +0.60 | +0.20 | +0.30 | 2 | shoulder_lift, elbow_flex | 9.34 |
| 15 | -3.66 | +12.65 | -14.88 | +0.58 | +0.29 | +0.32 | 2 | shoulder_lift, elbow_flex | 14.07 |
| 16 | -5.01 | +10.57 | -16.21 | +4.40 | +0.20 | +1.61 | 4 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex | 15.09 |
| 17 | -4.40 | +7.03 | -7.63 | -0.50 | +0.18 | -0.22 | 2 | shoulder_lift, elbow_flex | 6.37 |
| 18 | -0.26 | +3.75 | -10.14 | +2.05 | -0.27 | +0.95 | 1 | elbow_flex | 5.98 |
| 19 | -1.04 | +4.20 | -3.74 | -1.44 | +0.80 | +2.13 | 0 | - | 3.46 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.88 | 1.39 | -5.33 | -0.26 | 3/20 | 15% |
| shoulder_lift | +6.12 | 4.40 | -5.01 | +16.42 | 11/20 | 55% |
| elbow_flex | -11.50 | 3.26 | -18.38 | -3.74 | 19/20 | 95% |
| wrist_flex | +1.74 | 1.36 | -1.44 | +4.40 | 1/20 | 5% |
| wrist_roll | +0.20 | 0.22 | -0.27 | +0.80 | 0/20 | 0% |
| gripper | +0.71 | 0.80 | -0.75 | +2.24 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 8, '2': 8, '3': 2, '4': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=9.36, std=3.49, min=3.46, max=15.91
