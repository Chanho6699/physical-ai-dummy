# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T05` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T05_real_final/shadow_patched.json`)
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
| 0 | -3.14 | +6.14 | -13.49 | +0.54 | +0.23 | +0.12 | 2 | shoulder_lift, elbow_flex | 9.76 |
| 1 | -1.50 | +12.14 | -11.52 | -0.12 | +0.34 | -0.49 | 2 | shoulder_lift, elbow_flex | 10.91 |
| 2 | -1.11 | +3.52 | -7.13 | +0.32 | +0.32 | +0.47 | 1 | elbow_flex | 2.99 |
| 3 | -1.23 | +3.87 | -9.67 | +1.40 | +0.25 | +0.29 | 1 | elbow_flex | 5.42 |
| 4 | -2.66 | +5.00 | -9.74 | +0.10 | +0.41 | +0.82 | 1 | elbow_flex | 6.09 |
| 5 | -1.44 | +5.89 | -9.69 | +1.36 | +0.29 | -0.13 | 2 | shoulder_lift, elbow_flex | 5.82 |
| 6 | -1.77 | +2.96 | -8.65 | +1.26 | +0.15 | +0.19 | 1 | elbow_flex | 4.72 |
| 7 | -1.87 | +7.00 | -9.94 | +1.09 | +0.23 | -0.65 | 2 | shoulder_lift, elbow_flex | 6.58 |
| 8 | -1.67 | +4.72 | -7.50 | +0.85 | +0.41 | -0.03 | 1 | elbow_flex | 3.63 |
| 9 | -1.93 | +4.21 | -9.21 | +0.42 | +0.25 | -0.29 | 1 | elbow_flex | 5.08 |
| 10 | -1.46 | +7.65 | -10.24 | -0.00 | +0.21 | +0.52 | 2 | shoulder_lift, elbow_flex | 7.03 |
| 11 | -1.53 | +4.20 | -9.07 | +0.71 | +0.39 | +0.59 | 1 | elbow_flex | 4.92 |
| 12 | -1.70 | +0.10 | -7.76 | +0.29 | +0.15 | +0.96 | 1 | elbow_flex | 5.42 |
| 13 | -0.89 | +4.20 | -8.72 | +0.64 | +0.37 | +0.31 | 1 | elbow_flex | 4.35 |
| 14 | -1.73 | +5.78 | -9.48 | -0.29 | +0.27 | -0.04 | 2 | shoulder_lift, elbow_flex | 5.61 |
| 15 | -2.32 | +9.03 | -11.01 | -0.24 | +0.29 | -0.25 | 2 | shoulder_lift, elbow_flex | 8.61 |
| 16 | -3.42 | +8.46 | -12.68 | +2.16 | +0.32 | +0.61 | 2 | shoulder_lift, elbow_flex | 10.19 |
| 17 | -1.91 | +5.19 | -7.33 | -0.56 | +0.25 | -0.30 | 2 | shoulder_lift, elbow_flex | 3.77 |
| 18 | -0.65 | +4.23 | -8.88 | +1.08 | +0.05 | +0.10 | 1 | elbow_flex | 4.46 |
| 19 | -1.08 | +4.82 | -5.69 | -1.21 | +0.64 | +0.67 | 0 | - | 2.68 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.75 | 0.68 | -3.42 | -0.65 | 0/20 | 0% |
| shoulder_lift | +5.45 | 2.48 | +0.10 | +12.14 | 9/20 | 45% |
| elbow_flex | -9.37 | 1.83 | -13.49 | -5.69 | 19/20 | 95% |
| wrist_flex | +0.49 | 0.77 | -1.21 | +2.16 | 0/20 | 0% |
| wrist_roll | +0.29 | 0.12 | +0.05 | +0.64 | 0/20 | 0% |
| gripper | +0.17 | 0.44 | -0.65 | +0.96 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 10, '2': 9}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.90, std=2.28, min=2.68, max=10.91
