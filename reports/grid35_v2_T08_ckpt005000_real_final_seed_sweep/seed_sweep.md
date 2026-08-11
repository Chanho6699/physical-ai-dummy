# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T08` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T08_real_final/shadow_patched.json`)
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
| 0 | -4.45 | +6.09 | -18.25 | +2.02 | +0.11 | +0.20 | 2 | shoulder_lift, elbow_flex | 14.68 |
| 1 | -1.66 | +16.50 | -13.85 | +0.38 | +0.39 | +0.30 | 2 | shoulder_lift, elbow_flex | 15.78 |
| 2 | -1.50 | +1.41 | -7.29 | +1.49 | +0.26 | +0.72 | 1 | elbow_flex | 4.33 |
| 3 | -0.75 | +3.63 | -13.08 | +3.47 | +0.12 | +0.38 | 1 | elbow_flex | 9.14 |
| 4 | -4.82 | +6.11 | -11.58 | +1.50 | +0.47 | +2.07 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 9.28 |
| 5 | -1.74 | +7.34 | -11.12 | +3.54 | +0.26 | -0.17 | 2 | shoulder_lift, elbow_flex | 8.32 |
| 6 | -1.48 | +3.35 | -11.40 | +3.19 | -0.08 | +0.37 | 1 | elbow_flex | 7.64 |
| 7 | -2.44 | +9.30 | -13.42 | +2.62 | +0.07 | -0.86 | 2 | shoulder_lift, elbow_flex | 10.95 |
| 8 | -3.22 | +5.43 | -8.02 | +2.60 | +0.44 | +0.70 | 2 | shoulder_lift, elbow_flex | 5.71 |
| 9 | -2.22 | +3.16 | -11.58 | +2.05 | +0.11 | -0.35 | 1 | elbow_flex | 7.63 |
| 10 | -1.88 | +10.87 | -12.64 | +0.96 | +0.17 | +0.63 | 2 | shoulder_lift, elbow_flex | 10.91 |
| 11 | -1.52 | +4.12 | -11.76 | +2.03 | +0.40 | +1.02 | 1 | elbow_flex | 7.70 |
| 12 | -3.21 | -4.90 | -8.50 | +1.89 | -0.14 | +2.13 | 1 | elbow_flex | 10.60 |
| 13 | -0.32 | +4.22 | -9.58 | +2.38 | +0.34 | +0.64 | 1 | elbow_flex | 5.55 |
| 14 | -2.72 | +6.84 | -12.45 | +0.65 | +0.22 | +0.21 | 2 | shoulder_lift, elbow_flex | 8.90 |
| 15 | -3.09 | +12.73 | -14.79 | +0.63 | +0.31 | +0.25 | 2 | shoulder_lift, elbow_flex | 13.91 |
| 16 | -4.47 | +10.55 | -16.04 | +4.51 | +0.24 | +1.52 | 3 | shoulder_lift, elbow_flex, wrist_flex | 14.79 |
| 17 | -3.82 | +7.18 | -7.44 | -0.43 | +0.20 | -0.28 | 2 | shoulder_lift, elbow_flex | 5.95 |
| 18 | +0.38 | +3.75 | -10.30 | +2.15 | -0.27 | +0.86 | 1 | elbow_flex | 6.13 |
| 19 | -0.49 | +4.45 | -3.50 | -1.40 | +0.85 | +2.12 | 0 | - | 3.39 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.27 | 1.42 | -4.82 | +0.38 | 1/20 | 5% |
| shoulder_lift | +6.11 | 4.41 | -4.90 | +16.50 | 11/20 | 55% |
| elbow_flex | -11.33 | 3.28 | -18.25 | -3.50 | 19/20 | 95% |
| wrist_flex | +1.81 | 1.38 | -1.40 | +4.51 | 1/20 | 5% |
| wrist_roll | +0.22 | 0.23 | -0.27 | +0.85 | 0/20 | 0% |
| gripper | +0.62 | 0.80 | -0.86 | +2.13 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 8, '2': 9, '3': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=9.06, std=3.50, min=3.39, max=15.78
