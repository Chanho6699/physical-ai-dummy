# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T10` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T10_real_final/shadow_patched.json`)
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
| 0 | -4.52 | +5.77 | -17.86 | +2.26 | +0.09 | +0.19 | 2 | shoulder_lift, elbow_flex | 14.34 |
| 1 | -1.78 | +15.95 | -13.71 | +0.69 | +0.36 | +0.33 | 2 | shoulder_lift, elbow_flex | 15.28 |
| 2 | -1.75 | +1.05 | -7.35 | +1.72 | +0.25 | +0.78 | 1 | elbow_flex | 4.76 |
| 3 | -0.89 | +3.28 | -12.86 | +3.62 | +0.10 | +0.47 | 1 | elbow_flex | 9.04 |
| 4 | -4.74 | +5.59 | -11.41 | +1.73 | +0.45 | +2.08 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 9.03 |
| 5 | -1.85 | +6.91 | -11.16 | +3.72 | +0.24 | -0.11 | 2 | shoulder_lift, elbow_flex | 8.28 |
| 6 | -1.74 | +2.98 | -11.42 | +3.40 | -0.08 | +0.43 | 1 | elbow_flex | 7.84 |
| 7 | -2.62 | +8.77 | -13.27 | +2.82 | +0.06 | -0.79 | 2 | shoulder_lift, elbow_flex | 10.67 |
| 8 | -3.26 | +5.00 | -7.96 | +2.83 | +0.41 | +0.75 | 1 | elbow_flex | 5.70 |
| 9 | -2.39 | +2.76 | -11.50 | +2.27 | +0.09 | -0.25 | 1 | elbow_flex | 7.72 |
| 10 | -2.05 | +10.22 | -12.37 | +1.18 | +0.15 | +0.65 | 2 | shoulder_lift, elbow_flex | 10.36 |
| 11 | -1.74 | +3.55 | -11.57 | +2.24 | +0.37 | +1.08 | 1 | elbow_flex | 7.64 |
| 12 | -3.39 | -4.93 | -8.60 | +2.25 | -0.14 | +2.14 | 1 | elbow_flex | 10.78 |
| 13 | -0.56 | +3.77 | -9.54 | +2.57 | +0.31 | +0.72 | 1 | elbow_flex | 5.62 |
| 14 | -2.84 | +6.38 | -12.22 | +0.93 | +0.20 | +0.30 | 2 | shoulder_lift, elbow_flex | 8.61 |
| 15 | -3.09 | +12.08 | -14.43 | +0.92 | +0.28 | +0.25 | 2 | shoulder_lift, elbow_flex | 13.24 |
| 16 | -4.52 | +9.98 | -15.74 | +4.64 | +0.21 | +1.55 | 3 | shoulder_lift, elbow_flex, wrist_flex | 14.37 |
| 17 | -3.99 | +6.70 | -7.52 | -0.12 | +0.18 | -0.20 | 2 | shoulder_lift, elbow_flex | 5.83 |
| 18 | +0.18 | +3.50 | -10.35 | +2.47 | -0.27 | +0.89 | 1 | elbow_flex | 6.29 |
| 19 | -0.62 | +4.02 | -3.66 | -1.04 | +0.82 | +2.09 | 0 | - | 3.13 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.41 | 1.36 | -4.74 | +0.18 | 1/20 | 5% |
| shoulder_lift | +5.67 | 4.30 | -4.93 | +15.95 | 10/20 | 50% |
| elbow_flex | -11.23 | 3.14 | -17.86 | -3.66 | 19/20 | 95% |
| wrist_flex | +2.06 | 1.33 | -1.04 | +4.64 | 1/20 | 5% |
| wrist_roll | +0.20 | 0.23 | -0.27 | +0.82 | 0/20 | 0% |
| gripper | +0.67 | 0.78 | -0.79 | +2.14 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 9, '2': 8, '3': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=8.93, std=3.32, min=3.13, max=15.28
