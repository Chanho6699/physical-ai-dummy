# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T04` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T04_real_final/shadow_patched.json`)
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
| 0 | -2.54 | +6.01 | -12.64 | +0.86 | +0.23 | -0.18 | 2 | shoulder_lift, elbow_flex | 8.76 |
| 1 | -0.92 | +12.15 | -10.75 | +0.24 | +0.35 | -0.73 | 2 | shoulder_lift, elbow_flex | 10.39 |
| 2 | -0.66 | +3.64 | -6.57 | +0.66 | +0.32 | +0.29 | 1 | elbow_flex | 2.30 |
| 3 | -0.79 | +3.82 | -8.97 | +1.68 | +0.25 | +0.05 | 1 | elbow_flex | 4.72 |
| 4 | -2.07 | +4.78 | -9.09 | +0.45 | +0.42 | +0.54 | 1 | elbow_flex | 5.19 |
| 5 | -0.84 | +6.02 | -9.03 | +1.66 | +0.30 | -0.37 | 2 | shoulder_lift, elbow_flex | 5.23 |
| 6 | -1.32 | +2.98 | -8.06 | +1.58 | +0.15 | -0.09 | 1 | elbow_flex | 4.11 |
| 7 | -1.42 | +7.05 | -9.29 | +1.43 | +0.24 | -0.91 | 2 | shoulder_lift, elbow_flex | 6.04 |
| 8 | -1.23 | +4.85 | -6.81 | +1.22 | +0.41 | -0.24 | 1 | elbow_flex | 3.00 |
| 9 | -1.49 | +4.27 | -8.57 | +0.78 | +0.26 | -0.53 | 1 | elbow_flex | 4.37 |
| 10 | -0.94 | +7.61 | -9.54 | +0.33 | +0.22 | +0.28 | 2 | shoulder_lift, elbow_flex | 6.32 |
| 11 | -0.99 | +4.17 | -8.49 | +1.02 | +0.39 | +0.36 | 1 | elbow_flex | 4.22 |
| 12 | -1.13 | +0.42 | -7.30 | +0.66 | +0.15 | +0.77 | 1 | elbow_flex | 4.72 |
| 13 | -0.37 | +4.25 | -8.05 | +1.04 | +0.39 | +0.05 | 1 | elbow_flex | 3.65 |
| 14 | -1.26 | +5.73 | -8.74 | +0.07 | +0.27 | -0.20 | 2 | shoulder_lift, elbow_flex | 4.77 |
| 15 | -1.65 | +9.07 | -10.33 | +0.08 | +0.30 | -0.45 | 2 | shoulder_lift, elbow_flex | 7.96 |
| 16 | -2.74 | +8.40 | -11.90 | +2.40 | +0.34 | +0.31 | 2 | shoulder_lift, elbow_flex | 9.35 |
| 17 | -1.52 | +5.24 | -6.92 | -0.16 | +0.25 | -0.48 | 2 | shoulder_lift, elbow_flex | 3.22 |
| 18 | -0.24 | +4.35 | -8.39 | +1.42 | +0.07 | -0.17 | 1 | elbow_flex | 4.01 |
| 19 | -0.63 | +5.18 | -5.18 | -0.81 | +0.66 | +0.52 | 1 | shoulder_lift | 2.24 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.24 | 0.64 | -2.74 | -0.24 | 0/20 | 0% |
| shoulder_lift | +5.50 | 2.44 | +0.42 | +12.15 | 10/20 | 50% |
| elbow_flex | -8.73 | 1.74 | -12.64 | -5.18 | 19/20 | 95% |
| wrist_flex | +0.83 | 0.74 | -0.81 | +2.40 | 0/20 | 0% |
| wrist_roll | +0.30 | 0.12 | +0.07 | +0.66 | 0/20 | 0% |
| gripper | -0.06 | 0.44 | -0.91 | +0.77 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 11, '2': 9}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.23, std=2.23, min=2.24, max=10.39
