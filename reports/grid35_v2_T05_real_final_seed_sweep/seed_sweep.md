# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T05` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T05_real_final/shadow_patched.json`)
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
| 0 | -3.21 | +7.87 | -13.66 | +0.97 | +0.23 | +0.37 | 2 | shoulder_lift, elbow_flex | 10.50 |
| 1 | -1.83 | +14.09 | -10.57 | +0.14 | +0.33 | +0.01 | 2 | shoulder_lift, elbow_flex | 12.01 |
| 2 | -1.16 | +4.83 | -6.24 | +0.59 | +0.33 | +0.82 | 1 | elbow_flex | 2.64 |
| 3 | -1.05 | +5.60 | -9.72 | +1.91 | +0.23 | +0.73 | 2 | shoulder_lift, elbow_flex | 5.88 |
| 4 | -2.91 | +6.41 | -8.92 | +0.41 | +0.42 | +1.31 | 2 | shoulder_lift, elbow_flex | 6.12 |
| 5 | -1.45 | +7.59 | -8.92 | +1.83 | +0.27 | +0.14 | 2 | shoulder_lift, elbow_flex | 6.16 |
| 6 | -1.71 | +4.42 | -8.03 | +1.76 | +0.11 | +0.53 | 1 | elbow_flex | 4.32 |
| 7 | -1.96 | +8.75 | -9.10 | +1.24 | +0.19 | -0.59 | 2 | shoulder_lift, elbow_flex | 7.05 |
| 8 | -1.86 | +5.90 | -5.67 | +1.30 | +0.43 | +0.27 | 1 | shoulder_lift | 3.34 |
| 9 | -1.89 | +6.02 | -8.89 | +0.82 | +0.24 | -0.03 | 2 | shoulder_lift, elbow_flex | 5.26 |
| 10 | -1.67 | +9.77 | -9.82 | +0.32 | +0.21 | +1.00 | 2 | shoulder_lift, elbow_flex | 8.20 |
| 11 | -1.67 | +5.18 | -8.47 | +1.02 | +0.39 | +0.98 | 2 | shoulder_lift, elbow_flex | 4.75 |
| 12 | -1.63 | +0.32 | -6.77 | +0.61 | +0.14 | +1.34 | 1 | elbow_flex | 4.84 |
| 13 | -0.83 | +5.93 | -8.17 | +1.13 | +0.37 | +0.58 | 2 | shoulder_lift, elbow_flex | 4.44 |
| 14 | -2.02 | +7.33 | -8.79 | -0.03 | +0.28 | +0.28 | 2 | shoulder_lift, elbow_flex | 5.89 |
| 15 | -2.71 | +10.61 | -10.01 | -0.04 | +0.30 | +0.22 | 2 | shoulder_lift, elbow_flex | 9.13 |
| 16 | -3.67 | +10.45 | -12.08 | +2.62 | +0.31 | +0.96 | 2 | shoulder_lift, elbow_flex | 11.02 |
| 17 | -2.32 | +6.80 | -5.70 | -0.45 | +0.25 | -0.10 | 1 | shoulder_lift | 4.08 |
| 18 | -0.34 | +5.71 | -8.20 | +1.28 | +0.00 | +0.70 | 2 | shoulder_lift, elbow_flex | 4.32 |
| 19 | -0.92 | +5.96 | -3.33 | -1.45 | +0.69 | +1.23 | 1 | shoulder_lift | 3.61 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.84 | 0.80 | -3.67 | -0.34 | 0/20 | 0% |
| shoulder_lift | +6.98 | 2.78 | +0.32 | +14.09 | 17/20 | 85% |
| elbow_flex | -8.55 | 2.25 | -13.66 | -3.33 | 17/20 | 85% |
| wrist_flex | +0.80 | 0.90 | -1.45 | +2.62 | 0/20 | 0% |
| wrist_roll | +0.29 | 0.14 | +0.00 | +0.69 | 0/20 | 0% |
| gripper | +0.54 | 0.51 | -0.59 | +1.34 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 6, '2': 14}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.18, std=2.61, min=2.64, max=12.01
