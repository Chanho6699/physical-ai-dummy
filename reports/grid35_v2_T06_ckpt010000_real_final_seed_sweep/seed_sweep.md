# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T06` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T06_real_final/shadow_patched.json`)
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
| 0 | -2.98 | +6.52 | -13.99 | +0.48 | +0.19 | +0.10 | 2 | shoulder_lift, elbow_flex | 10.26 |
| 1 | -1.35 | +12.47 | -12.35 | -0.20 | +0.32 | -0.46 | 2 | shoulder_lift, elbow_flex | 11.69 |
| 2 | -1.06 | +4.02 | -7.99 | +0.27 | +0.29 | +0.52 | 1 | elbow_flex | 3.72 |
| 3 | -1.15 | +4.29 | -10.38 | +1.33 | +0.22 | +0.30 | 1 | elbow_flex | 6.07 |
| 4 | -2.50 | +5.40 | -10.75 | +0.07 | +0.38 | +0.85 | 2 | shoulder_lift, elbow_flex | 6.98 |
| 5 | -1.23 | +6.31 | -10.39 | +1.26 | +0.26 | -0.15 | 2 | shoulder_lift, elbow_flex | 6.52 |
| 6 | -1.77 | +3.43 | -9.38 | +1.19 | +0.13 | +0.19 | 1 | elbow_flex | 5.29 |
| 7 | -1.79 | +7.57 | -10.73 | +1.00 | +0.21 | -0.60 | 2 | shoulder_lift, elbow_flex | 7.47 |
| 8 | -1.54 | +5.20 | -8.37 | +0.78 | +0.37 | +0.00 | 2 | shoulder_lift, elbow_flex | 4.41 |
| 9 | -1.89 | +4.81 | -10.04 | +0.44 | +0.22 | -0.25 | 1 | elbow_flex | 5.89 |
| 10 | -1.40 | +8.09 | -11.06 | -0.05 | +0.19 | +0.54 | 2 | shoulder_lift, elbow_flex | 7.92 |
| 11 | -1.47 | +4.72 | -9.95 | +0.65 | +0.35 | +0.58 | 1 | elbow_flex | 5.75 |
| 12 | -1.64 | +0.55 | -8.57 | +0.32 | +0.12 | +0.94 | 1 | elbow_flex | 5.62 |
| 13 | -0.82 | +4.66 | -9.56 | +0.61 | +0.34 | +0.39 | 1 | elbow_flex | 5.19 |
| 14 | -1.60 | +6.36 | -10.38 | -0.27 | +0.24 | -0.02 | 2 | shoulder_lift, elbow_flex | 6.56 |
| 15 | -2.16 | +9.38 | -11.85 | -0.30 | +0.26 | -0.24 | 2 | shoulder_lift, elbow_flex | 9.41 |
| 16 | -3.16 | +8.71 | -13.35 | +2.02 | +0.29 | +0.60 | 2 | shoulder_lift, elbow_flex | 10.73 |
| 17 | -1.86 | +5.68 | -8.33 | -0.57 | +0.22 | -0.22 | 2 | shoulder_lift, elbow_flex | 4.69 |
| 18 | -0.65 | +4.54 | -9.57 | +1.08 | +0.03 | +0.08 | 1 | elbow_flex | 5.15 |
| 19 | -1.08 | +5.27 | -6.45 | -1.15 | +0.61 | +0.70 | 2 | shoulder_lift, elbow_flex | 3.21 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.66 | 0.64 | -3.16 | -0.65 | 0/20 | 0% |
| shoulder_lift | +5.90 | 2.45 | +0.55 | +12.47 | 12/20 | 60% |
| elbow_flex | -10.17 | 1.77 | -13.99 | -6.45 | 20/20 | 100% |
| wrist_flex | +0.45 | 0.74 | -1.15 | +2.02 | 0/20 | 0% |
| wrist_roll | +0.26 | 0.12 | +0.03 | +0.61 | 0/20 | 0% |
| gripper | +0.19 | 0.43 | -0.60 | +0.94 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 8, '2': 12}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.63, std=2.27, min=3.21, max=11.69
