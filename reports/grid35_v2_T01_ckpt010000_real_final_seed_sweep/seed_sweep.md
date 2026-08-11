# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T01` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01_real_final/shadow_patched.json`)
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
| 0 | -2.66 | +6.26 | -12.95 | +0.15 | +0.23 | +0.20 | 2 | shoulder_lift, elbow_flex | 9.15 |
| 1 | -1.09 | +12.64 | -11.09 | -0.64 | +0.35 | -0.40 | 2 | shoulder_lift, elbow_flex | 11.01 |
| 2 | -0.77 | +3.79 | -6.52 | -0.06 | +0.32 | +0.58 | 1 | elbow_flex | 2.36 |
| 3 | -0.78 | +4.01 | -9.14 | +1.02 | +0.24 | +0.38 | 1 | elbow_flex | 4.76 |
| 4 | -2.33 | +5.10 | -9.35 | -0.34 | +0.42 | +0.98 | 1 | elbow_flex | 5.70 |
| 5 | -0.86 | +6.21 | -9.12 | +0.96 | +0.29 | -0.11 | 2 | shoulder_lift, elbow_flex | 5.25 |
| 6 | -1.38 | +3.19 | -8.10 | +0.90 | +0.14 | +0.26 | 1 | elbow_flex | 3.98 |
| 7 | -1.37 | +7.36 | -9.39 | +0.68 | +0.23 | -0.62 | 2 | shoulder_lift, elbow_flex | 6.15 |
| 8 | -1.38 | +5.14 | -7.04 | +0.52 | +0.42 | +0.07 | 1 | elbow_flex | 3.22 |
| 9 | -1.51 | +4.43 | -8.66 | +0.05 | +0.25 | -0.21 | 1 | elbow_flex | 4.44 |
| 10 | -1.09 | +8.14 | -9.78 | -0.47 | +0.20 | +0.63 | 2 | shoulder_lift, elbow_flex | 6.93 |
| 11 | -1.12 | +4.28 | -8.56 | +0.27 | +0.40 | +0.66 | 1 | elbow_flex | 4.31 |
| 12 | -1.42 | +0.13 | -7.28 | -0.04 | +0.14 | +1.10 | 1 | elbow_flex | 5.08 |
| 13 | -0.49 | +4.52 | -8.25 | +0.30 | +0.38 | +0.39 | 1 | elbow_flex | 3.84 |
| 14 | -1.38 | +6.07 | -9.02 | -0.74 | +0.27 | +0.09 | 2 | shoulder_lift, elbow_flex | 5.28 |
| 15 | -1.84 | +9.39 | -10.57 | -0.75 | +0.30 | -0.11 | 2 | shoulder_lift, elbow_flex | 8.44 |
| 16 | -2.87 | +8.75 | -12.22 | +1.76 | +0.33 | +0.67 | 2 | shoulder_lift, elbow_flex | 9.72 |
| 17 | -1.61 | +5.62 | -6.97 | -1.00 | +0.26 | -0.19 | 2 | shoulder_lift, elbow_flex | 3.66 |
| 18 | -0.27 | +4.53 | -8.37 | +0.69 | +0.05 | +0.18 | 1 | elbow_flex | 3.90 |
| 19 | -0.71 | +5.38 | -5.24 | -1.65 | +0.67 | +0.92 | 1 | shoulder_lift | 3.01 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.35 | 0.66 | -2.87 | -0.27 | 0/20 | 0% |
| shoulder_lift | +5.75 | 2.56 | +0.13 | +12.64 | 10/20 | 50% |
| elbow_flex | -8.88 | 1.83 | -12.95 | -5.24 | 19/20 | 95% |
| wrist_flex | +0.08 | 0.80 | -1.65 | +1.76 | 0/20 | 0% |
| wrist_roll | +0.29 | 0.13 | +0.05 | +0.67 | 0/20 | 0% |
| gripper | +0.27 | 0.46 | -0.62 | +1.10 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 11, '2': 9}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.51, std=2.33, min=2.36, max=11.01
