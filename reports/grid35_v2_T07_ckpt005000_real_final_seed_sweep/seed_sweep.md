# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T07` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T07_real_final/shadow_patched.json`)
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
| 0 | -5.07 | +5.55 | -18.49 | +2.01 | +0.10 | +0.22 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 15.04 |
| 1 | -2.32 | +16.26 | -14.40 | +0.40 | +0.37 | +0.36 | 2 | shoulder_lift, elbow_flex | 16.01 |
| 2 | -2.29 | +0.80 | -7.70 | +1.51 | +0.24 | +0.83 | 1 | elbow_flex | 5.30 |
| 3 | -1.46 | +3.23 | -13.46 | +3.46 | +0.11 | +0.51 | 1 | elbow_flex | 9.62 |
| 4 | -5.50 | +5.71 | -12.17 | +1.60 | +0.46 | +2.18 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 10.05 |
| 5 | -2.31 | +6.91 | -11.46 | +3.54 | +0.25 | -0.13 | 2 | shoulder_lift, elbow_flex | 8.57 |
| 6 | -2.28 | +2.73 | -11.79 | +3.21 | -0.10 | +0.47 | 1 | elbow_flex | 8.27 |
| 7 | -3.18 | +8.66 | -13.76 | +2.57 | +0.06 | -0.79 | 2 | shoulder_lift, elbow_flex | 11.12 |
| 8 | -3.88 | +5.00 | -8.35 | +2.61 | +0.42 | +0.82 | 1 | elbow_flex | 6.24 |
| 9 | -2.90 | +2.50 | -11.87 | +2.01 | +0.09 | -0.23 | 1 | elbow_flex | 8.20 |
| 10 | -2.57 | +10.45 | -13.10 | +0.99 | +0.16 | +0.73 | 2 | shoulder_lift, elbow_flex | 11.16 |
| 11 | -2.19 | +3.48 | -12.22 | +2.09 | +0.38 | +1.13 | 1 | elbow_flex | 8.33 |
| 12 | -4.03 | -5.37 | -9.10 | +2.06 | -0.16 | +2.29 | 2 | shoulder_lift, elbow_flex | 11.55 |
| 13 | -1.04 | +3.78 | -10.08 | +2.43 | +0.33 | +0.76 | 1 | elbow_flex | 6.13 |
| 14 | -3.45 | +6.44 | -12.97 | +0.69 | +0.21 | +0.36 | 2 | shoulder_lift, elbow_flex | 9.50 |
| 15 | -3.68 | +12.36 | -15.21 | +0.64 | +0.30 | +0.35 | 2 | shoulder_lift, elbow_flex | 14.14 |
| 16 | -5.10 | +10.20 | -16.37 | +4.51 | +0.22 | +1.59 | 4 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex | 15.10 |
| 17 | -4.59 | +6.68 | -7.88 | -0.46 | +0.19 | -0.18 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 6.47 |
| 18 | -0.49 | +3.35 | -10.72 | +2.24 | -0.29 | +0.97 | 1 | elbow_flex | 6.62 |
| 19 | -1.17 | +3.96 | -3.77 | -1.36 | +0.84 | +2.25 | 0 | - | 3.53 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.97 | 1.39 | -5.50 | -0.49 | 4/20 | 20% |
| shoulder_lift | +5.63 | 4.47 | -5.37 | +16.26 | 11/20 | 55% |
| elbow_flex | -11.74 | 3.27 | -18.49 | -3.77 | 19/20 | 95% |
| wrist_flex | +1.84 | 1.37 | -1.36 | +4.51 | 1/20 | 5% |
| wrist_roll | +0.21 | 0.24 | -0.29 | +0.84 | 0/20 | 0% |
| gripper | +0.73 | 0.82 | -0.79 | +2.29 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 8, '2': 7, '3': 3, '4': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=9.55, std=3.42, min=3.53, max=16.01
