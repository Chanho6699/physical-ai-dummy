# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T04` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T04_real_final/shadow_patched.json`)
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
| 0 | -4.58 | +5.35 | -17.61 | +2.07 | +0.09 | -0.06 | 2 | shoulder_lift, elbow_flex | 14.04 |
| 1 | -1.70 | +15.55 | -13.24 | +0.52 | +0.36 | +0.02 | 2 | shoulder_lift, elbow_flex | 14.65 |
| 2 | -1.62 | +0.63 | -6.74 | +1.55 | +0.24 | +0.51 | 1 | elbow_flex | 4.56 |
| 3 | -0.85 | +2.90 | -12.40 | +3.46 | +0.10 | +0.22 | 1 | elbow_flex | 8.57 |
| 4 | -4.75 | +5.02 | -10.90 | +1.56 | +0.43 | +1.79 | 2 | shoulder_pan, elbow_flex | 8.46 |
| 5 | -1.75 | +6.49 | -10.51 | +3.53 | +0.24 | -0.39 | 2 | shoulder_lift, elbow_flex | 7.52 |
| 6 | -1.64 | +2.49 | -10.74 | +3.20 | -0.10 | +0.14 | 1 | elbow_flex | 7.20 |
| 7 | -2.58 | +8.30 | -12.75 | +2.65 | +0.05 | -1.09 | 2 | shoulder_lift, elbow_flex | 10.00 |
| 8 | -3.22 | +4.69 | -7.46 | +2.67 | +0.41 | +0.46 | 1 | elbow_flex | 5.21 |
| 9 | -2.33 | +2.36 | -10.93 | +2.05 | +0.09 | -0.52 | 1 | elbow_flex | 7.20 |
| 10 | -1.93 | +9.81 | -11.81 | +1.02 | +0.15 | +0.38 | 2 | shoulder_lift, elbow_flex | 9.63 |
| 11 | -1.59 | +3.14 | -11.15 | +2.07 | +0.36 | +0.82 | 1 | elbow_flex | 7.16 |
| 12 | -3.24 | -5.44 | -7.99 | +2.03 | -0.15 | +1.92 | 2 | shoulder_lift, elbow_flex | 10.86 |
| 13 | -0.43 | +3.36 | -8.97 | +2.39 | +0.31 | +0.44 | 1 | elbow_flex | 5.00 |
| 14 | -2.79 | +5.89 | -11.71 | +0.72 | +0.20 | +0.02 | 2 | shoulder_lift, elbow_flex | 7.98 |
| 15 | -3.03 | +11.65 | -13.96 | +0.73 | +0.28 | +0.01 | 2 | shoulder_lift, elbow_flex | 12.59 |
| 16 | -4.59 | +9.54 | -15.21 | +4.47 | +0.21 | +1.28 | 4 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex | 13.71 |
| 17 | -4.00 | +6.22 | -6.91 | -0.31 | +0.17 | -0.50 | 2 | shoulder_lift, elbow_flex | 5.35 |
| 18 | +0.25 | +3.02 | -9.64 | +2.22 | -0.28 | +0.64 | 1 | elbow_flex | 5.56 |
| 19 | -0.48 | +3.61 | -2.92 | -1.21 | +0.81 | +1.82 | 0 | - | 3.29 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.34 | 1.40 | -4.75 | +0.25 | 2/20 | 10% |
| shoulder_lift | +5.23 | 4.31 | -5.44 | +15.55 | 10/20 | 50% |
| elbow_flex | -10.68 | 3.22 | -17.61 | -2.92 | 19/20 | 95% |
| wrist_flex | +1.87 | 1.33 | -1.21 | +4.47 | 1/20 | 5% |
| wrist_roll | +0.20 | 0.23 | -0.28 | +0.81 | 0/20 | 0% |
| gripper | +0.39 | 0.79 | -1.09 | +1.92 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 8, '2': 10, '4': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=8.43, std=3.26, min=3.29, max=14.65
