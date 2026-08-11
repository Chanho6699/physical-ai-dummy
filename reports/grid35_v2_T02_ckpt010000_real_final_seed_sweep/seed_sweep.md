# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T02_real_final/shadow_patched.json`)
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
| 0 | -2.66 | +5.86 | -12.42 | +0.81 | +0.25 | -0.04 | 2 | shoulder_lift, elbow_flex | 8.57 |
| 1 | -1.08 | +12.02 | -10.83 | +0.09 | +0.37 | -0.57 | 2 | shoulder_lift, elbow_flex | 10.34 |
| 2 | -0.80 | +3.36 | -6.52 | +0.59 | +0.33 | +0.38 | 1 | elbow_flex | 2.38 |
| 3 | -0.91 | +3.72 | -8.99 | +1.66 | +0.27 | +0.20 | 1 | elbow_flex | 4.77 |
| 4 | -2.26 | +4.67 | -9.14 | +0.38 | +0.44 | +0.68 | 1 | elbow_flex | 5.32 |
| 5 | -0.99 | +5.85 | -9.07 | +1.57 | +0.32 | -0.25 | 2 | shoulder_lift, elbow_flex | 5.21 |
| 6 | -1.56 | +2.79 | -8.13 | +1.59 | +0.17 | +0.04 | 1 | elbow_flex | 4.31 |
| 7 | -1.53 | +6.90 | -9.36 | +1.34 | +0.25 | -0.80 | 2 | shoulder_lift, elbow_flex | 6.02 |
| 8 | -1.37 | +4.73 | -6.82 | +1.13 | +0.43 | -0.06 | 1 | elbow_flex | 3.03 |
| 9 | -1.64 | +4.02 | -8.56 | +0.72 | +0.27 | -0.40 | 1 | elbow_flex | 4.39 |
| 10 | -1.13 | +7.50 | -9.62 | +0.20 | +0.24 | +0.44 | 2 | shoulder_lift, elbow_flex | 6.37 |
| 11 | -1.13 | +3.97 | -8.41 | +0.94 | +0.40 | +0.51 | 1 | elbow_flex | 4.19 |
| 12 | -1.45 | +0.08 | -7.33 | +0.65 | +0.16 | +0.87 | 1 | elbow_flex | 5.10 |
| 13 | -0.59 | +4.11 | -8.21 | +0.98 | +0.40 | +0.23 | 1 | elbow_flex | 3.83 |
| 14 | -1.46 | +5.66 | -8.91 | +0.04 | +0.30 | -0.11 | 2 | shoulder_lift, elbow_flex | 4.96 |
| 15 | -1.88 | +8.94 | -10.42 | -0.03 | +0.32 | -0.29 | 2 | shoulder_lift, elbow_flex | 8.00 |
| 16 | -2.84 | +8.16 | -11.91 | +2.32 | +0.35 | +0.45 | 2 | shoulder_lift, elbow_flex | 9.27 |
| 17 | -1.72 | +5.10 | -7.02 | -0.25 | +0.27 | -0.36 | 1 | elbow_flex | 3.35 |
| 18 | -0.49 | +4.23 | -8.60 | +1.38 | +0.08 | -0.03 | 1 | elbow_flex | 4.24 |
| 19 | -0.85 | +4.95 | -5.08 | -0.96 | +0.68 | +0.64 | 0 | - | 2.32 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.42 | 0.62 | -2.84 | -0.49 | 0/20 | 0% |
| shoulder_lift | +5.33 | 2.46 | +0.08 | +12.02 | 8/20 | 40% |
| elbow_flex | -8.77 | 1.73 | -12.42 | -5.08 | 19/20 | 95% |
| wrist_flex | +0.76 | 0.76 | -0.96 | +2.32 | 0/20 | 0% |
| wrist_roll | +0.32 | 0.12 | +0.08 | +0.68 | 0/20 | 0% |
| gripper | +0.08 | 0.44 | -0.80 | +0.87 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([19])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 11, '2': 8}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.30, std=2.17, min=2.32, max=10.34
