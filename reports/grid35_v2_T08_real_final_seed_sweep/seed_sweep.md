# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T08` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T08_real_final/shadow_patched.json`)
Checkpoint: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model`
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
| 0 | -2.27 | +8.12 | -12.67 | +0.91 | +0.24 | +0.16 | 2 | shoulder_lift, elbow_flex | 9.47 |
| 1 | -0.89 | +14.53 | -9.63 | +0.02 | +0.35 | -0.15 | 2 | shoulder_lift, elbow_flex | 11.84 |
| 2 | -0.31 | +5.14 | -5.44 | +0.53 | +0.32 | +0.65 | 0 | - | 1.94 |
| 3 | -0.25 | +5.73 | -8.78 | +1.86 | +0.23 | +0.49 | 2 | shoulder_lift, elbow_flex | 4.94 |
| 4 | -1.95 | +6.72 | -8.12 | +0.36 | +0.43 | +1.14 | 2 | shoulder_lift, elbow_flex | 5.23 |
| 5 | -0.55 | +7.96 | -8.09 | +1.75 | +0.27 | +0.00 | 2 | shoulder_lift, elbow_flex | 5.64 |
| 6 | -0.85 | +4.69 | -7.27 | +1.69 | +0.10 | +0.29 | 1 | elbow_flex | 3.36 |
| 7 | -1.11 | +9.13 | -8.16 | +1.15 | +0.19 | -0.78 | 2 | shoulder_lift, elbow_flex | 6.56 |
| 8 | -1.01 | +6.25 | -4.80 | +1.25 | +0.44 | +0.13 | 1 | shoulder_lift | 2.94 |
| 9 | -1.12 | +6.35 | -8.11 | +0.80 | +0.25 | -0.27 | 2 | shoulder_lift, elbow_flex | 4.53 |
| 10 | -0.79 | +10.29 | -9.10 | +0.23 | +0.22 | +0.84 | 2 | shoulder_lift, elbow_flex | 7.98 |
| 11 | -0.79 | +5.58 | -7.63 | +0.93 | +0.40 | +0.77 | 2 | shoulder_lift, elbow_flex | 3.86 |
| 12 | -0.79 | +0.68 | -6.23 | +0.53 | +0.14 | +1.11 | 1 | elbow_flex | 3.99 |
| 13 | +0.01 | +6.21 | -7.32 | +1.14 | +0.38 | +0.43 | 2 | shoulder_lift, elbow_flex | 3.80 |
| 14 | -1.15 | +7.74 | -8.00 | -0.08 | +0.29 | +0.11 | 2 | shoulder_lift, elbow_flex | 5.35 |
| 15 | -1.81 | +11.11 | -9.15 | -0.13 | +0.31 | +0.07 | 2 | shoulder_lift, elbow_flex | 8.80 |
| 16 | -2.71 | +10.73 | -11.09 | +2.50 | +0.32 | +0.73 | 2 | shoulder_lift, elbow_flex | 10.19 |
| 17 | -1.54 | +7.32 | -5.16 | -0.47 | +0.26 | -0.25 | 1 | shoulder_lift | 3.97 |
| 18 | +0.39 | +5.99 | -7.57 | +1.20 | -0.00 | +0.44 | 2 | shoulder_lift, elbow_flex | 3.85 |
| 19 | -0.22 | +6.60 | -2.77 | -1.48 | +0.72 | +1.11 | 1 | shoulder_lift | 4.07 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.99 | 0.75 | -2.71 | +0.39 | 0/20 | 0% |
| shoulder_lift | +7.34 | 2.82 | +0.68 | +14.53 | 17/20 | 85% |
| elbow_flex | -7.75 | 2.15 | -12.67 | -2.77 | 16/20 | 80% |
| wrist_flex | +0.74 | 0.89 | -1.48 | +2.50 | 0/20 | 0% |
| wrist_roll | +0.29 | 0.15 | -0.00 | +0.72 | 0/20 | 0% |
| gripper | +0.35 | 0.50 | -0.78 | +1.14 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([2])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 5, '2': 14}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.62, std=2.61, min=1.94, max=11.84
