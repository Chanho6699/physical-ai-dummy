# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T10` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T10_real_final/shadow_patched.json`)
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
| 0 | -2.45 | +7.92 | -12.57 | +1.15 | +0.22 | +0.19 | 2 | shoulder_lift, elbow_flex | 9.37 |
| 1 | -1.17 | +14.15 | -9.67 | +0.28 | +0.32 | -0.08 | 2 | shoulder_lift, elbow_flex | 11.54 |
| 2 | -0.58 | +5.01 | -5.66 | +0.78 | +0.32 | +0.74 | 0 | - | 2.14 |
| 3 | -0.48 | +5.62 | -8.96 | +2.06 | +0.21 | +0.59 | 2 | shoulder_lift, elbow_flex | 5.16 |
| 4 | -2.12 | +6.38 | -8.16 | +0.60 | +0.41 | +1.18 | 2 | shoulder_lift, elbow_flex | 5.18 |
| 5 | -0.77 | +7.83 | -8.30 | +1.96 | +0.25 | +0.05 | 2 | shoulder_lift, elbow_flex | 5.78 |
| 6 | -1.17 | +4.48 | -7.49 | +1.94 | +0.09 | +0.37 | 1 | elbow_flex | 3.73 |
| 7 | -1.39 | +8.87 | -8.25 | +1.37 | +0.18 | -0.69 | 2 | shoulder_lift, elbow_flex | 6.49 |
| 8 | -1.28 | +6.05 | -4.94 | +1.48 | +0.42 | +0.18 | 1 | shoulder_lift | 3.01 |
| 9 | -1.36 | +6.19 | -8.25 | +1.06 | +0.24 | -0.20 | 2 | shoulder_lift, elbow_flex | 4.67 |
| 10 | -1.08 | +9.80 | -9.17 | +0.46 | +0.20 | +0.91 | 2 | shoulder_lift, elbow_flex | 7.70 |
| 11 | -1.10 | +5.25 | -7.71 | +1.18 | +0.38 | +0.87 | 2 | shoulder_lift, elbow_flex | 3.96 |
| 12 | -1.01 | +0.67 | -6.47 | +0.88 | +0.13 | +1.18 | 1 | elbow_flex | 4.22 |
| 13 | -0.29 | +6.03 | -7.56 | +1.36 | +0.37 | +0.47 | 2 | shoulder_lift, elbow_flex | 3.96 |
| 14 | -1.41 | +7.39 | -7.97 | +0.18 | +0.27 | +0.21 | 2 | shoulder_lift, elbow_flex | 5.15 |
| 15 | -2.01 | +10.75 | -9.13 | +0.13 | +0.28 | +0.11 | 2 | shoulder_lift, elbow_flex | 8.53 |
| 16 | -2.79 | +10.39 | -11.06 | +2.62 | +0.30 | +0.77 | 2 | shoulder_lift, elbow_flex | 10.00 |
| 17 | -1.88 | +7.12 | -5.35 | -0.18 | +0.24 | -0.16 | 1 | shoulder_lift | 3.95 |
| 18 | +0.11 | +5.88 | -7.82 | +1.53 | +0.00 | +0.47 | 2 | shoulder_lift, elbow_flex | 4.09 |
| 19 | -0.43 | +6.34 | -2.92 | -1.18 | +0.70 | +1.15 | 1 | shoulder_lift | 3.76 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.23 | 0.72 | -2.79 | +0.11 | 0/20 | 0% |
| shoulder_lift | +7.11 | 2.73 | +0.67 | +14.15 | 17/20 | 85% |
| elbow_flex | -7.87 | 2.09 | -12.57 | -2.92 | 16/20 | 80% |
| wrist_flex | +0.98 | 0.87 | -1.18 | +2.62 | 0/20 | 0% |
| wrist_roll | +0.28 | 0.14 | +0.00 | +0.70 | 0/20 | 0% |
| gripper | +0.42 | 0.50 | -0.69 | +1.18 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([2])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 5, '2': 14}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.62, std=2.47, min=2.14, max=11.54
