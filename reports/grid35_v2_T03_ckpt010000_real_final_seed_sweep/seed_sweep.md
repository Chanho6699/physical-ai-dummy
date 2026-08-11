# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T03` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T03_real_final/shadow_patched.json`)
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
| 0 | -2.89 | +6.96 | -13.17 | +0.41 | +0.20 | +0.13 | 2 | shoulder_lift, elbow_flex | 9.61 |
| 1 | -1.22 | +13.07 | -11.35 | -0.39 | +0.32 | -0.45 | 2 | shoulder_lift, elbow_flex | 11.51 |
| 2 | -0.91 | +4.35 | -6.91 | +0.10 | +0.30 | +0.52 | 1 | elbow_flex | 2.75 |
| 3 | -1.04 | +4.74 | -9.47 | +1.19 | +0.22 | +0.29 | 1 | elbow_flex | 5.21 |
| 4 | -2.35 | +5.78 | -9.58 | -0.16 | +0.39 | +0.86 | 2 | shoulder_lift, elbow_flex | 6.04 |
| 5 | -1.07 | +6.86 | -9.43 | +1.14 | +0.27 | -0.13 | 2 | shoulder_lift, elbow_flex | 5.89 |
| 6 | -1.62 | +3.83 | -8.46 | +1.05 | +0.13 | +0.18 | 1 | elbow_flex | 4.36 |
| 7 | -1.63 | +8.04 | -9.78 | +0.88 | +0.22 | -0.66 | 2 | shoulder_lift, elbow_flex | 6.93 |
| 8 | -1.44 | +5.64 | -7.28 | +0.63 | +0.38 | +0.02 | 2 | shoulder_lift, elbow_flex | 3.66 |
| 9 | -1.67 | +5.11 | -8.95 | +0.23 | +0.23 | -0.26 | 1 | elbow_flex | 4.88 |
| 10 | -1.32 | +8.70 | -10.17 | -0.28 | +0.20 | +0.57 | 2 | shoulder_lift, elbow_flex | 7.59 |
| 11 | -1.36 | +5.10 | -8.96 | +0.48 | +0.37 | +0.59 | 1 | elbow_flex | 4.88 |
| 12 | -1.48 | +0.84 | -7.41 | +0.11 | +0.13 | +1.02 | 1 | elbow_flex | 4.65 |
| 13 | -0.61 | +5.17 | -8.62 | +0.47 | +0.35 | +0.37 | 2 | shoulder_lift, elbow_flex | 4.36 |
| 14 | -1.61 | +6.70 | -9.36 | -0.53 | +0.25 | +0.01 | 2 | shoulder_lift, elbow_flex | 5.87 |
| 15 | -2.06 | +9.99 | -10.93 | -0.51 | +0.27 | -0.20 | 2 | shoulder_lift, elbow_flex | 9.11 |
| 16 | -3.03 | +9.29 | -12.45 | +1.89 | +0.30 | +0.64 | 2 | shoulder_lift, elbow_flex | 10.25 |
| 17 | -1.71 | +6.09 | -7.18 | -0.83 | +0.23 | -0.25 | 2 | shoulder_lift, elbow_flex | 4.03 |
| 18 | -0.51 | +5.03 | -8.55 | +0.81 | +0.04 | +0.12 | 1 | elbow_flex | 4.22 |
| 19 | -0.86 | +5.73 | -5.43 | -1.52 | +0.63 | +0.76 | 1 | shoulder_lift | 3.15 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.52 | 0.65 | -3.03 | -0.51 | 0/20 | 0% |
| shoulder_lift | +6.35 | 2.52 | +0.84 | +13.07 | 13/20 | 65% |
| elbow_flex | -9.17 | 1.83 | -13.17 | -5.43 | 19/20 | 95% |
| wrist_flex | +0.26 | 0.79 | -1.52 | +1.89 | 0/20 | 0% |
| wrist_roll | +0.27 | 0.12 | +0.04 | +0.63 | 0/20 | 0% |
| gripper | +0.21 | 0.45 | -0.66 | +1.02 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 8, '2': 12}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.95, std=2.41, min=2.75, max=11.51
