# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T05` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T05_real_final/shadow_patched.json`)
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
| 0 | -5.22 | +5.18 | -18.52 | +2.00 | +0.09 | +0.20 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 15.09 |
| 1 | -2.41 | +15.24 | -13.93 | +0.38 | +0.34 | +0.24 | 2 | shoulder_lift, elbow_flex | 14.94 |
| 2 | -2.20 | +0.39 | -7.51 | +1.46 | +0.24 | +0.69 | 1 | elbow_flex | 5.37 |
| 3 | -1.42 | +2.77 | -13.25 | +3.42 | +0.10 | +0.46 | 1 | elbow_flex | 9.45 |
| 4 | -5.53 | +5.13 | -11.76 | +1.52 | +0.43 | +2.03 | 2 | shoulder_pan, elbow_flex | 9.62 |
| 5 | -2.43 | +6.22 | -11.31 | +3.46 | +0.23 | -0.19 | 2 | shoulder_lift, elbow_flex | 8.23 |
| 6 | -2.19 | +2.32 | -11.53 | +3.15 | -0.09 | +0.41 | 1 | elbow_flex | 8.06 |
| 7 | -3.14 | +8.09 | -13.52 | +2.57 | +0.05 | -0.86 | 2 | shoulder_lift, elbow_flex | 10.67 |
| 8 | -3.85 | +4.39 | -8.22 | +2.55 | +0.40 | +0.62 | 1 | elbow_flex | 6.00 |
| 9 | -2.90 | +2.17 | -11.77 | +1.97 | +0.08 | -0.30 | 1 | elbow_flex | 8.17 |
| 10 | -2.58 | +9.65 | -12.63 | +0.94 | +0.15 | +0.58 | 2 | shoulder_lift, elbow_flex | 10.33 |
| 11 | -2.18 | +3.02 | -11.90 | +2.01 | +0.36 | +0.99 | 1 | elbow_flex | 8.03 |
| 12 | -3.85 | -5.91 | -8.40 | +1.88 | -0.15 | +2.12 | 2 | shoulder_lift, elbow_flex | 11.61 |
| 13 | -1.06 | +3.22 | -9.84 | +2.29 | +0.30 | +0.66 | 1 | elbow_flex | 5.89 |
| 14 | -3.36 | +5.82 | -12.52 | +0.65 | +0.20 | +0.20 | 2 | shoulder_lift, elbow_flex | 8.91 |
| 15 | -3.74 | +11.46 | -14.77 | +0.64 | +0.28 | +0.20 | 2 | shoulder_lift, elbow_flex | 13.29 |
| 16 | -5.26 | +9.53 | -16.23 | +4.49 | +0.20 | +1.51 | 4 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex | 14.77 |
| 17 | -4.48 | +5.87 | -7.50 | -0.45 | +0.17 | -0.34 | 2 | shoulder_lift, elbow_flex | 5.88 |
| 18 | -0.22 | +2.72 | -10.14 | +2.13 | -0.28 | +0.88 | 1 | elbow_flex | 6.10 |
| 19 | -1.07 | +3.02 | -3.52 | -1.37 | +0.80 | +1.93 | 0 | - | 3.44 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.96 | 1.44 | -5.53 | -0.22 | 3/20 | 15% |
| shoulder_lift | +5.01 | 4.35 | -5.91 | +15.24 | 10/20 | 50% |
| elbow_flex | -11.44 | 3.31 | -18.52 | -3.52 | 19/20 | 95% |
| wrist_flex | +1.78 | 1.36 | -1.37 | +4.49 | 1/20 | 5% |
| wrist_roll | +0.19 | 0.23 | -0.28 | +0.80 | 0/20 | 0% |
| gripper | +0.60 | 0.78 | -0.86 | +2.12 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 8, '2': 9, '3': 1, '4': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=9.19, std=3.31, min=3.44, max=15.09
