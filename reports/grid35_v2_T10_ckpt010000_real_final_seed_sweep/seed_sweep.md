# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T10` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T10_real_final/shadow_patched.json`)
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
| 0 | -2.55 | +6.75 | -12.82 | +0.87 | +0.22 | +0.09 | 2 | shoulder_lift, elbow_flex | 9.15 |
| 1 | -0.99 | +12.81 | -11.27 | +0.23 | +0.34 | -0.46 | 2 | shoulder_lift, elbow_flex | 11.22 |
| 2 | -0.73 | +4.29 | -7.09 | +0.69 | +0.32 | +0.56 | 1 | elbow_flex | 2.87 |
| 3 | -0.86 | +4.53 | -9.38 | +1.71 | +0.24 | +0.31 | 1 | elbow_flex | 5.19 |
| 4 | -2.01 | +5.58 | -9.53 | +0.45 | +0.42 | +0.82 | 2 | shoulder_lift, elbow_flex | 5.79 |
| 5 | -0.92 | +6.66 | -9.63 | +1.68 | +0.29 | -0.09 | 2 | shoulder_lift, elbow_flex | 6.04 |
| 6 | -1.43 | +3.68 | -8.69 | +1.64 | +0.15 | +0.19 | 1 | elbow_flex | 4.64 |
| 7 | -1.43 | +7.85 | -9.78 | +1.43 | +0.24 | -0.64 | 2 | shoulder_lift, elbow_flex | 6.84 |
| 8 | -1.24 | +5.52 | -7.35 | +1.21 | +0.40 | +0.04 | 2 | shoulder_lift, elbow_flex | 3.69 |
| 9 | -1.54 | +4.99 | -9.07 | +0.80 | +0.25 | -0.28 | 1 | elbow_flex | 4.95 |
| 10 | -1.07 | +8.35 | -10.09 | +0.33 | +0.21 | +0.57 | 2 | shoulder_lift, elbow_flex | 7.25 |
| 11 | -1.12 | +4.89 | -8.82 | +1.05 | +0.38 | +0.65 | 1 | elbow_flex | 4.72 |
| 12 | -1.24 | +1.26 | -7.92 | +0.76 | +0.16 | +0.96 | 1 | elbow_flex | 4.66 |
| 13 | -0.54 | +4.92 | -8.61 | +1.04 | +0.37 | +0.33 | 1 | elbow_flex | 4.35 |
| 14 | -1.33 | +6.48 | -9.23 | +0.11 | +0.27 | +0.05 | 2 | shoulder_lift, elbow_flex | 5.54 |
| 15 | -1.73 | +9.69 | -10.78 | +0.10 | +0.29 | -0.19 | 2 | shoulder_lift, elbow_flex | 8.70 |
| 16 | -2.68 | +9.02 | -12.29 | +2.36 | +0.33 | +0.58 | 2 | shoulder_lift, elbow_flex | 9.96 |
| 17 | -1.61 | +5.90 | -7.53 | -0.13 | +0.25 | -0.19 | 2 | shoulder_lift, elbow_flex | 4.01 |
| 18 | -0.37 | +5.03 | -9.04 | +1.50 | +0.06 | +0.08 | 1 | elbow_flex | 4.81 |
| 19 | -0.74 | +5.88 | -5.77 | -0.78 | +0.66 | +0.77 | 2 | shoulder_lift, elbow_flex | 2.99 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.31 | 0.59 | -2.68 | -0.37 | 0/20 | 0% |
| shoulder_lift | +6.20 | 2.41 | +1.26 | +12.81 | 12/20 | 60% |
| elbow_flex | -9.24 | 1.67 | -12.82 | -5.77 | 20/20 | 100% |
| wrist_flex | +0.85 | 0.74 | -0.78 | +2.36 | 0/20 | 0% |
| wrist_roll | +0.29 | 0.12 | +0.06 | +0.66 | 0/20 | 0% |
| gripper | +0.21 | 0.43 | -0.64 | +0.96 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 8, '2': 12}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.87, std=2.26, min=2.87, max=11.22
