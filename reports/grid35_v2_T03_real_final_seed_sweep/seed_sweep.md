# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T03` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T03_real_final/shadow_patched.json`)
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
| 0 | -2.94 | +8.55 | -13.31 | +0.91 | +0.22 | +0.33 | 2 | shoulder_lift, elbow_flex | 10.40 |
| 1 | -1.49 | +14.82 | -10.28 | -0.10 | +0.32 | +0.00 | 2 | shoulder_lift, elbow_flex | 12.46 |
| 2 | -0.86 | +5.50 | -6.06 | +0.46 | +0.31 | +0.83 | 2 | shoulder_lift, elbow_flex | 2.71 |
| 3 | -0.82 | +6.37 | -9.52 | +1.79 | +0.22 | +0.68 | 2 | shoulder_lift, elbow_flex | 5.90 |
| 4 | -2.55 | +7.11 | -8.79 | +0.25 | +0.41 | +1.34 | 2 | shoulder_lift, elbow_flex | 6.20 |
| 5 | -1.04 | +8.41 | -8.75 | +1.67 | +0.25 | +0.13 | 2 | shoulder_lift, elbow_flex | 6.45 |
| 6 | -1.46 | +5.12 | -7.90 | +1.62 | +0.10 | +0.49 | 1 | elbow_flex | 4.20 |
| 7 | -1.68 | +9.62 | -8.85 | +1.10 | +0.18 | -0.60 | 2 | shoulder_lift, elbow_flex | 7.44 |
| 8 | -1.53 | +6.61 | -5.47 | +1.16 | +0.42 | +0.28 | 1 | shoulder_lift | 3.56 |
| 9 | -1.64 | +6.78 | -8.68 | +0.73 | +0.22 | -0.04 | 2 | shoulder_lift, elbow_flex | 5.36 |
| 10 | -1.38 | +10.63 | -9.74 | +0.09 | +0.21 | +1.02 | 2 | shoulder_lift, elbow_flex | 8.73 |
| 11 | -1.40 | +5.91 | -8.42 | +0.88 | +0.38 | +0.95 | 2 | shoulder_lift, elbow_flex | 4.84 |
| 12 | -1.35 | +0.82 | -6.44 | +0.49 | +0.12 | +1.34 | 1 | elbow_flex | 4.23 |
| 13 | -0.59 | +6.70 | -8.07 | +1.00 | +0.36 | +0.60 | 2 | shoulder_lift, elbow_flex | 4.71 |
| 14 | -1.77 | +8.10 | -8.67 | -0.21 | +0.27 | +0.28 | 2 | shoulder_lift, elbow_flex | 6.23 |
| 15 | -2.43 | +11.43 | -9.85 | -0.24 | +0.29 | +0.20 | 2 | shoulder_lift, elbow_flex | 9.59 |
| 16 | -3.27 | +11.10 | -11.72 | +2.41 | +0.29 | +0.93 | 2 | shoulder_lift, elbow_flex | 11.01 |
| 17 | -2.06 | +7.60 | -5.67 | -0.61 | +0.24 | -0.10 | 1 | shoulder_lift | 4.57 |
| 18 | -0.06 | +6.30 | -7.84 | +1.10 | -0.00 | +0.65 | 2 | shoulder_lift, elbow_flex | 4.25 |
| 19 | -0.70 | +6.65 | -3.20 | -1.66 | +0.69 | +1.28 | 1 | shoulder_lift | 4.15 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.55 | 0.78 | -3.27 | -0.06 | 0/20 | 0% |
| shoulder_lift | +7.71 | 2.83 | +0.82 | +14.82 | 18/20 | 90% |
| elbow_flex | -8.36 | 2.21 | -13.31 | -3.20 | 17/20 | 85% |
| wrist_flex | +0.64 | 0.92 | -1.66 | +2.41 | 0/20 | 0% |
| wrist_roll | +0.27 | 0.14 | -0.00 | +0.69 | 0/20 | 0% |
| gripper | +0.53 | 0.51 | -0.60 | +1.34 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 5, '2': 15}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.35, std=2.66, min=2.71, max=12.46
