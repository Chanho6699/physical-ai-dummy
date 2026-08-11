# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T01` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01_real_final/shadow_patched.json`)
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
| 0 | -4.86 | +5.16 | -17.82 | +1.75 | +0.10 | +0.28 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 14.28 |
| 1 | -1.93 | +15.58 | -13.25 | -0.00 | +0.37 | +0.34 | 2 | shoulder_lift, elbow_flex | 14.73 |
| 2 | -1.89 | +0.32 | -6.57 | +1.18 | +0.25 | +0.79 | 1 | elbow_flex | 4.79 |
| 3 | -1.03 | +2.64 | -12.48 | +3.19 | +0.10 | +0.53 | 1 | elbow_flex | 8.62 |
| 4 | -5.29 | +5.06 | -11.11 | +1.23 | +0.45 | +2.17 | 2 | shoulder_pan, elbow_flex | 9.00 |
| 5 | -1.91 | +6.30 | -10.49 | +3.22 | +0.24 | -0.14 | 2 | shoulder_lift, elbow_flex | 7.35 |
| 6 | -1.82 | +2.28 | -10.61 | +2.90 | -0.10 | +0.45 | 1 | elbow_flex | 7.08 |
| 7 | -2.72 | +8.21 | -12.70 | +2.29 | +0.05 | -0.83 | 2 | shoulder_lift, elbow_flex | 9.86 |
| 8 | -3.58 | +4.44 | -7.38 | +2.31 | +0.42 | +0.74 | 1 | elbow_flex | 5.28 |
| 9 | -2.52 | +2.09 | -10.87 | +1.72 | +0.09 | -0.25 | 1 | elbow_flex | 7.20 |
| 10 | -2.22 | +9.84 | -11.93 | +0.57 | +0.15 | +0.70 | 2 | shoulder_lift, elbow_flex | 9.80 |
| 11 | -1.85 | +2.95 | -11.12 | +1.71 | +0.38 | +1.11 | 1 | elbow_flex | 7.19 |
| 12 | -3.64 | -6.13 | -7.67 | +1.67 | -0.16 | +2.29 | 2 | shoulder_lift, elbow_flex | 11.53 |
| 13 | -0.64 | +3.24 | -9.02 | +2.04 | +0.32 | +0.75 | 1 | elbow_flex | 5.00 |
| 14 | -3.03 | +5.82 | -11.79 | +0.29 | +0.21 | +0.31 | 2 | shoulder_lift, elbow_flex | 8.13 |
| 15 | -3.31 | +11.62 | -14.09 | +0.26 | +0.29 | +0.32 | 2 | shoulder_lift, elbow_flex | 12.75 |
| 16 | -4.83 | +9.50 | -15.45 | +4.20 | +0.21 | +1.59 | 4 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex | 13.92 |
| 17 | -4.22 | +6.19 | -6.75 | -0.79 | +0.19 | -0.28 | 2 | shoulder_lift, elbow_flex | 5.51 |
| 18 | +0.08 | +2.70 | -9.39 | +1.87 | -0.29 | +0.94 | 1 | elbow_flex | 5.35 |
| 19 | -0.70 | +3.41 | -2.85 | -1.72 | +0.84 | +2.27 | 0 | - | 3.93 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.59 | 1.46 | -5.29 | +0.08 | 3/20 | 15% |
| shoulder_lift | +5.06 | 4.45 | -6.13 | +15.58 | 10/20 | 50% |
| elbow_flex | -10.67 | 3.32 | -17.82 | -2.85 | 19/20 | 95% |
| wrist_flex | +1.49 | 1.39 | -1.72 | +4.20 | 1/20 | 5% |
| wrist_roll | +0.21 | 0.24 | -0.29 | +0.84 | 0/20 | 0% |
| gripper | +0.70 | 0.83 | -0.83 | +2.29 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 8, '2': 9, '3': 1, '4': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=8.56, std=3.28, min=3.93, max=14.73
