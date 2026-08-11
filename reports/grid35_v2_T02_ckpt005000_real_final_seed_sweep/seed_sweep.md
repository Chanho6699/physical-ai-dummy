# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T02_real_final/shadow_patched.json`)
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
| 0 | -4.64 | +5.12 | -17.43 | +2.15 | +0.12 | +0.09 | 2 | shoulder_pan, elbow_flex | 13.89 |
| 1 | -1.87 | +15.54 | -13.27 | +0.49 | +0.38 | +0.17 | 2 | shoulder_lift, elbow_flex | 14.70 |
| 2 | -1.86 | +0.53 | -6.74 | +1.59 | +0.27 | +0.62 | 1 | elbow_flex | 4.76 |
| 3 | -0.99 | +2.83 | -12.40 | +3.56 | +0.13 | +0.39 | 1 | elbow_flex | 8.65 |
| 4 | -5.06 | +4.96 | -10.91 | +1.63 | +0.47 | +1.94 | 2 | shoulder_pan, elbow_flex | 8.70 |
| 5 | -1.91 | +6.41 | -10.49 | +3.56 | +0.27 | -0.27 | 2 | shoulder_lift, elbow_flex | 7.52 |
| 6 | -1.81 | +2.38 | -10.77 | +3.30 | -0.06 | +0.30 | 1 | elbow_flex | 7.34 |
| 7 | -2.73 | +8.19 | -12.79 | +2.67 | +0.08 | -0.97 | 2 | shoulder_lift, elbow_flex | 10.02 |
| 8 | -3.45 | +4.61 | -7.42 | +2.71 | +0.44 | +0.64 | 1 | elbow_flex | 5.38 |
| 9 | -2.47 | +2.17 | -10.93 | +2.14 | +0.12 | -0.39 | 1 | elbow_flex | 7.32 |
| 10 | -2.14 | +9.78 | -11.90 | +1.03 | +0.18 | +0.50 | 2 | shoulder_lift, elbow_flex | 9.74 |
| 11 | -1.78 | +3.03 | -11.10 | +2.11 | +0.39 | +0.96 | 1 | elbow_flex | 7.21 |
| 12 | -3.54 | -5.72 | -7.90 | +2.11 | -0.13 | +2.11 | 2 | shoulder_lift, elbow_flex | 11.22 |
| 13 | -0.66 | +3.35 | -9.07 | +2.44 | +0.34 | +0.60 | 1 | elbow_flex | 5.18 |
| 14 | -2.98 | +5.96 | -11.87 | +0.82 | +0.23 | +0.16 | 2 | shoulder_lift, elbow_flex | 8.23 |
| 15 | -3.23 | +11.66 | -14.12 | +0.73 | +0.31 | +0.16 | 2 | shoulder_lift, elbow_flex | 12.78 |
| 16 | -4.64 | +9.49 | -15.33 | +4.52 | +0.24 | +1.39 | 4 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex | 13.83 |
| 17 | -4.15 | +6.18 | -6.96 | -0.30 | +0.21 | -0.39 | 2 | shoulder_lift, elbow_flex | 5.47 |
| 18 | -0.02 | +3.00 | -9.74 | +2.32 | -0.25 | +0.76 | 1 | elbow_flex | 5.72 |
| 19 | -0.72 | +3.46 | -2.71 | -1.29 | +0.85 | +2.00 | 0 | - | 3.63 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.53 | 1.39 | -5.06 | -0.02 | 3/20 | 15% |
| shoulder_lift | +5.15 | 4.36 | -5.72 | +15.54 | 9/20 | 45% |
| elbow_flex | -10.69 | 3.25 | -17.43 | -2.71 | 19/20 | 95% |
| wrist_flex | +1.91 | 1.36 | -1.29 | +4.52 | 1/20 | 5% |
| wrist_roll | +0.23 | 0.23 | -0.25 | +0.85 | 0/20 | 0% |
| gripper | +0.54 | 0.80 | -0.97 | +2.11 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 8, '2': 10, '4': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=8.56, std=3.21, min=3.63, max=14.70
