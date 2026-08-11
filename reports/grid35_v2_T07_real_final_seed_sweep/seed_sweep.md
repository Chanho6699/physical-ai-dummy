# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T07` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T07_real_final/shadow_patched.json`)
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
| 0 | -2.90 | +8.37 | -14.04 | +1.28 | +0.25 | +0.30 | 2 | shoulder_lift, elbow_flex | 10.97 |
| 1 | -1.64 | +14.84 | -11.26 | +0.41 | +0.35 | +0.00 | 2 | shoulder_lift, elbow_flex | 12.96 |
| 2 | -1.03 | +5.37 | -7.05 | +0.95 | +0.33 | +0.86 | 2 | shoulder_lift, elbow_flex | 3.43 |
| 3 | -0.93 | +6.19 | -10.39 | +2.23 | +0.24 | +0.71 | 2 | shoulder_lift, elbow_flex | 6.72 |
| 4 | -2.67 | +7.11 | -9.89 | +0.82 | +0.45 | +1.35 | 2 | shoulder_lift, elbow_flex | 7.06 |
| 5 | -1.23 | +8.45 | -9.69 | +2.15 | +0.28 | +0.11 | 2 | shoulder_lift, elbow_flex | 7.26 |
| 6 | -1.65 | +4.90 | -8.87 | +2.11 | +0.11 | +0.50 | 1 | elbow_flex | 5.17 |
| 7 | -1.87 | +9.49 | -9.84 | +1.52 | +0.19 | -0.61 | 2 | shoulder_lift, elbow_flex | 8.06 |
| 8 | -1.69 | +6.63 | -6.36 | +1.67 | +0.45 | +0.33 | 2 | shoulder_lift, elbow_flex | 4.12 |
| 9 | -1.82 | +6.53 | -9.57 | +1.16 | +0.25 | -0.04 | 2 | shoulder_lift, elbow_flex | 6.05 |
| 10 | -1.50 | +10.53 | -10.70 | +0.62 | +0.23 | +1.05 | 2 | shoulder_lift, elbow_flex | 9.29 |
| 11 | -1.49 | +5.75 | -9.25 | +1.36 | +0.40 | +0.98 | 2 | shoulder_lift, elbow_flex | 5.57 |
| 12 | -1.55 | +0.95 | -7.82 | +1.04 | +0.13 | +1.38 | 1 | elbow_flex | 5.04 |
| 13 | -0.70 | +6.58 | -9.01 | +1.60 | +0.39 | +0.57 | 2 | shoulder_lift, elbow_flex | 5.51 |
| 14 | -1.91 | +8.00 | -9.65 | +0.35 | +0.29 | +0.36 | 2 | shoulder_lift, elbow_flex | 6.88 |
| 15 | -2.46 | +11.46 | -10.84 | +0.24 | +0.31 | +0.25 | 2 | shoulder_lift, elbow_flex | 10.20 |
| 16 | -3.29 | +11.21 | -12.56 | +2.86 | +0.33 | +0.89 | 2 | shoulder_lift, elbow_flex | 11.73 |
| 17 | -2.31 | +7.53 | -6.69 | -0.13 | +0.26 | -0.03 | 2 | shoulder_lift, elbow_flex | 4.92 |
| 18 | -0.37 | +6.39 | -9.14 | +1.67 | -0.00 | +0.67 | 2 | shoulder_lift, elbow_flex | 5.50 |
| 19 | -0.87 | +6.81 | -4.16 | -1.07 | +0.72 | +1.31 | 1 | shoulder_lift | 3.86 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.69 | 0.73 | -3.29 | -0.37 | 0/20 | 0% |
| shoulder_lift | +7.65 | 2.84 | +0.95 | +14.84 | 18/20 | 90% |
| elbow_flex | -9.34 | 2.16 | -14.04 | -4.16 | 19/20 | 95% |
| wrist_flex | +1.14 | 0.89 | -1.07 | +2.86 | 0/20 | 0% |
| wrist_roll | +0.30 | 0.15 | -0.00 | +0.72 | 0/20 | 0% |
| gripper | +0.55 | 0.52 | -0.61 | +1.38 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 3, '2': 17}
- L2 error vs nearest-demo immediate GT delta (deg): mean=7.02, std=2.65, min=3.43, max=12.96
