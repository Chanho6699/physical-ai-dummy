# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T08` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T08_real_final/shadow_patched.json`)
Checkpoint: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/010000/pretrained_model`
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
| 0 | -2.23 | +6.19 | -12.30 | +0.51 | +0.23 | -0.02 | 2 | shoulder_lift, elbow_flex | 8.39 |
| 1 | -0.67 | +12.44 | -10.56 | -0.18 | +0.35 | -0.57 | 2 | shoulder_lift, elbow_flex | 10.48 |
| 2 | -0.31 | +3.77 | -6.29 | +0.31 | +0.32 | +0.38 | 1 | elbow_flex | 1.93 |
| 3 | -0.47 | +3.92 | -8.66 | +1.35 | +0.24 | +0.12 | 1 | elbow_flex | 4.28 |
| 4 | -1.77 | +5.15 | -8.82 | +0.08 | +0.42 | +0.72 | 1 | elbow_flex | 4.95 |
| 5 | -0.64 | +6.13 | -8.75 | +1.30 | +0.30 | -0.21 | 2 | shoulder_lift, elbow_flex | 4.92 |
| 6 | -1.05 | +3.13 | -7.83 | +1.24 | +0.15 | +0.04 | 1 | elbow_flex | 3.69 |
| 7 | -1.06 | +7.27 | -8.95 | +1.02 | +0.23 | -0.80 | 2 | shoulder_lift, elbow_flex | 5.73 |
| 8 | -0.93 | +4.92 | -6.58 | +0.86 | +0.42 | -0.10 | 1 | elbow_flex | 2.62 |
| 9 | -1.18 | +4.39 | -8.27 | +0.44 | +0.26 | -0.43 | 1 | elbow_flex | 3.95 |
| 10 | -0.68 | +7.99 | -9.41 | -0.06 | +0.21 | +0.42 | 2 | shoulder_lift, elbow_flex | 6.44 |
| 11 | -0.71 | +4.48 | -8.13 | +0.63 | +0.39 | +0.46 | 1 | elbow_flex | 3.81 |
| 12 | -0.93 | +0.45 | -7.00 | +0.27 | +0.15 | +0.81 | 1 | elbow_flex | 4.46 |
| 13 | -0.11 | +4.40 | -7.73 | +0.67 | +0.38 | +0.21 | 1 | elbow_flex | 3.28 |
| 14 | -0.95 | +5.97 | -8.56 | -0.30 | +0.27 | -0.15 | 2 | shoulder_lift, elbow_flex | 4.65 |
| 15 | -1.48 | +9.34 | -10.10 | -0.31 | +0.30 | -0.31 | 2 | shoulder_lift, elbow_flex | 7.95 |
| 16 | -2.47 | +8.56 | -11.72 | +2.07 | +0.33 | +0.51 | 2 | shoulder_lift, elbow_flex | 9.16 |
| 17 | -1.18 | +5.45 | -6.71 | -0.55 | +0.25 | -0.35 | 2 | shoulder_lift, elbow_flex | 3.07 |
| 18 | +0.05 | +4.40 | -8.20 | +1.03 | +0.05 | -0.06 | 1 | elbow_flex | 3.73 |
| 19 | -0.43 | +5.31 | -4.94 | -1.21 | +0.67 | +0.61 | 1 | shoulder_lift | 2.46 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.96 | 0.63 | -2.47 | +0.05 | 0/20 | 0% |
| shoulder_lift | +5.68 | 2.49 | +0.45 | +12.44 | 10/20 | 50% |
| elbow_flex | -8.48 | 1.74 | -12.30 | -4.94 | 19/20 | 95% |
| wrist_flex | +0.46 | 0.76 | -1.21 | +2.07 | 0/20 | 0% |
| wrist_roll | +0.30 | 0.13 | +0.05 | +0.67 | 0/20 | 0% |
| gripper | +0.06 | 0.43 | -0.80 | +0.81 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 11, '2': 9}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.00, std=2.29, min=1.93, max=10.48
