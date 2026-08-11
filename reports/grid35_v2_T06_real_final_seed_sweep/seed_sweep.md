# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T06` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T06_real_final/shadow_patched.json`)
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
| 0 | -3.07 | +8.18 | -14.25 | +0.92 | +0.20 | +0.24 | 2 | shoulder_lift, elbow_flex | 11.08 |
| 1 | -1.70 | +14.33 | -11.31 | +0.02 | +0.31 | -0.05 | 2 | shoulder_lift, elbow_flex | 12.57 |
| 2 | -1.11 | +5.16 | -7.06 | +0.56 | +0.29 | +0.79 | 1 | elbow_flex | 3.28 |
| 3 | -0.94 | +5.98 | -10.47 | +1.84 | +0.20 | +0.66 | 2 | shoulder_lift, elbow_flex | 6.61 |
| 4 | -2.70 | +6.79 | -9.97 | +0.37 | +0.39 | +1.29 | 2 | shoulder_lift, elbow_flex | 6.95 |
| 5 | -1.25 | +8.07 | -9.66 | +1.72 | +0.23 | +0.09 | 2 | shoulder_lift, elbow_flex | 6.90 |
| 6 | -1.62 | +4.70 | -8.77 | +1.64 | +0.07 | +0.46 | 1 | elbow_flex | 4.89 |
| 7 | -1.91 | +9.22 | -9.74 | +1.11 | +0.16 | -0.61 | 2 | shoulder_lift, elbow_flex | 7.76 |
| 8 | -1.64 | +6.33 | -6.54 | +1.23 | +0.39 | +0.23 | 2 | shoulder_lift, elbow_flex | 3.82 |
| 9 | -1.84 | +6.48 | -9.70 | +0.80 | +0.21 | -0.09 | 2 | shoulder_lift, elbow_flex | 6.10 |
| 10 | -1.53 | +10.16 | -10.69 | +0.24 | +0.19 | +0.95 | 2 | shoulder_lift, elbow_flex | 9.00 |
| 11 | -1.58 | +5.60 | -9.33 | +0.94 | +0.36 | +0.87 | 2 | shoulder_lift, elbow_flex | 5.52 |
| 12 | -1.59 | +0.61 | -7.66 | +0.62 | +0.10 | +1.24 | 1 | elbow_flex | 5.06 |
| 13 | -0.69 | +6.27 | -8.99 | +1.10 | +0.34 | +0.54 | 2 | shoulder_lift, elbow_flex | 5.23 |
| 14 | -1.86 | +7.84 | -9.72 | -0.04 | +0.25 | +0.22 | 2 | shoulder_lift, elbow_flex | 6.82 |
| 15 | -2.58 | +10.99 | -10.80 | -0.11 | +0.26 | +0.12 | 2 | shoulder_lift, elbow_flex | 9.84 |
| 16 | -3.42 | +10.66 | -12.66 | +2.42 | +0.27 | +0.85 | 2 | shoulder_lift, elbow_flex | 11.41 |
| 17 | -2.19 | +7.39 | -6.77 | -0.46 | +0.22 | -0.12 | 2 | shoulder_lift, elbow_flex | 4.82 |
| 18 | -0.32 | +5.85 | -8.84 | +1.28 | -0.03 | +0.57 | 2 | shoulder_lift, elbow_flex | 4.89 |
| 19 | -0.85 | +6.22 | -4.21 | -1.38 | +0.66 | +1.16 | 1 | shoulder_lift | 3.47 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.72 | 0.77 | -3.42 | -0.32 | 0/20 | 0% |
| shoulder_lift | +7.34 | 2.78 | +0.61 | +14.33 | 17/20 | 85% |
| elbow_flex | -9.36 | 2.18 | -14.25 | -4.21 | 19/20 | 95% |
| wrist_flex | +0.74 | 0.86 | -1.38 | +2.42 | 0/20 | 0% |
| wrist_roll | +0.25 | 0.14 | -0.03 | +0.66 | 0/20 | 0% |
| gripper | +0.47 | 0.50 | -0.61 | +1.29 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 4, '2': 16}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.80, std=2.64, min=3.28, max=12.57
