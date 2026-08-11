# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T06` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T06_real_final/shadow_patched.json`)
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
| 0 | -5.20 | +5.36 | -18.67 | +1.98 | +0.07 | +0.18 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 15.23 |
| 1 | -2.33 | +15.58 | -14.46 | +0.35 | +0.34 | +0.25 | 2 | shoulder_lift, elbow_flex | 15.52 |
| 2 | -2.21 | +0.61 | -7.84 | +1.44 | +0.22 | +0.71 | 1 | elbow_flex | 5.42 |
| 3 | -1.43 | +2.96 | -13.46 | +3.34 | +0.09 | +0.44 | 1 | elbow_flex | 9.59 |
| 4 | -5.42 | +5.34 | -12.27 | +1.49 | +0.41 | +2.06 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 9.98 |
| 5 | -2.32 | +6.58 | -11.57 | +3.41 | +0.22 | -0.17 | 2 | shoulder_lift, elbow_flex | 8.50 |
| 6 | -2.23 | +2.59 | -11.77 | +3.09 | -0.10 | +0.40 | 1 | elbow_flex | 8.21 |
| 7 | -3.16 | +8.49 | -13.93 | +2.52 | +0.04 | -0.84 | 2 | shoulder_lift, elbow_flex | 11.17 |
| 8 | -3.70 | +4.65 | -8.65 | +2.51 | +0.38 | +0.67 | 1 | elbow_flex | 6.19 |
| 9 | -2.92 | +2.46 | -12.11 | +2.01 | +0.06 | -0.29 | 1 | elbow_flex | 8.42 |
| 10 | -2.50 | +10.02 | -13.06 | +0.92 | +0.14 | +0.61 | 2 | shoulder_lift, elbow_flex | 10.85 |
| 11 | -2.20 | +3.35 | -12.30 | +1.98 | +0.34 | +1.00 | 1 | elbow_flex | 8.37 |
| 12 | -3.92 | -5.82 | -8.79 | +1.94 | -0.18 | +2.13 | 2 | shoulder_lift, elbow_flex | 11.71 |
| 13 | -1.01 | +3.48 | -10.21 | +2.27 | +0.29 | +0.71 | 1 | elbow_flex | 6.19 |
| 14 | -3.30 | +6.19 | -13.00 | +0.69 | +0.18 | +0.23 | 2 | shoulder_lift, elbow_flex | 9.40 |
| 15 | -3.66 | +11.75 | -15.29 | +0.64 | +0.26 | +0.20 | 2 | shoulder_lift, elbow_flex | 13.84 |
| 16 | -5.14 | +9.72 | -16.48 | +4.37 | +0.18 | +1.51 | 4 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex | 14.96 |
| 17 | -4.42 | +6.32 | -8.04 | -0.39 | +0.16 | -0.30 | 2 | shoulder_lift, elbow_flex | 6.27 |
| 18 | -0.30 | +2.72 | -10.38 | +2.14 | -0.30 | +0.85 | 1 | elbow_flex | 6.32 |
| 19 | -1.06 | +3.29 | -4.02 | -1.27 | +0.77 | +2.05 | 0 | - | 3.28 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.92 | 1.40 | -5.42 | -0.30 | 3/20 | 15% |
| shoulder_lift | +5.28 | 4.40 | -5.82 | +15.58 | 11/20 | 55% |
| elbow_flex | -11.82 | 3.27 | -18.67 | -4.02 | 19/20 | 95% |
| wrist_flex | +1.77 | 1.32 | -1.27 | +4.37 | 1/20 | 5% |
| wrist_roll | +0.18 | 0.23 | -0.30 | +0.77 | 0/20 | 0% |
| gripper | +0.62 | 0.79 | -0.84 | +2.13 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 8, '2': 8, '3': 2, '4': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=9.47, std=3.39, min=3.28, max=15.52
