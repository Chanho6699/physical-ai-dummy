# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T02_actual/shadow_patched.json`)
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
| 0 | -1.57 | +6.48 | -11.01 | +0.75 | +0.20 | +1.03 | 2 | shoulder_lift, elbow_flex | 7.28 |
| 1 | +0.20 | +13.07 | -8.15 | -0.14 | +0.32 | +0.64 | 2 | shoulder_lift, elbow_flex | 9.92 |
| 2 | +0.29 | +3.41 | -3.92 | +0.41 | +0.32 | +1.32 | 0 | - | 1.90 |
| 3 | +0.48 | +4.08 | -7.38 | +1.79 | +0.21 | +1.33 | 1 | elbow_flex | 3.61 |
| 4 | -1.15 | +5.13 | -6.64 | +0.21 | +0.43 | +1.93 | 1 | elbow_flex | 3.57 |
| 5 | +0.12 | +6.50 | -6.54 | +1.70 | +0.24 | +0.73 | 2 | shoulder_lift, elbow_flex | 3.75 |
| 6 | -0.28 | +2.93 | -5.69 | +1.62 | +0.08 | +1.06 | 0 | - | 2.47 |
| 7 | -0.51 | +7.35 | -6.56 | +1.08 | +0.16 | -0.08 | 2 | shoulder_lift, elbow_flex | 4.15 |
| 8 | -0.24 | +4.55 | -3.21 | +1.09 | +0.42 | +0.84 | 0 | - | 2.23 |
| 9 | -0.47 | +4.60 | -6.41 | +0.71 | +0.23 | +0.53 | 1 | elbow_flex | 2.29 |
| 10 | +0.14 | +8.83 | -7.58 | +0.10 | +0.19 | +1.66 | 2 | shoulder_lift, elbow_flex | 6.12 |
| 11 | -0.01 | +3.77 | -6.14 | +0.79 | +0.38 | +1.60 | 1 | elbow_flex | 2.57 |
| 12 | -0.03 | -0.99 | -4.72 | +0.41 | +0.11 | +1.98 | 0 | - | 5.38 |
| 13 | +0.68 | +4.65 | -5.88 | +0.96 | +0.36 | +1.23 | 1 | elbow_flex | 2.37 |
| 14 | -0.38 | +5.92 | -6.26 | -0.35 | +0.26 | +0.93 | 2 | shoulder_lift, elbow_flex | 3.03 |
| 15 | -0.82 | +9.37 | -7.48 | -0.34 | +0.27 | +0.86 | 2 | shoulder_lift, elbow_flex | 6.43 |
| 16 | -1.82 | +9.05 | -9.45 | +2.34 | +0.29 | +1.55 | 2 | shoulder_lift, elbow_flex | 7.89 |
| 17 | -0.82 | +5.61 | -3.51 | -0.65 | +0.23 | +0.43 | 1 | shoulder_lift | 2.58 |
| 18 | +1.34 | +4.70 | -6.28 | +1.16 | -0.03 | +1.28 | 1 | elbow_flex | 2.87 |
| 19 | +0.82 | +5.39 | -1.52 | -1.60 | +0.72 | +1.91 | 1 | shoulder_lift | 4.61 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.20 | 0.76 | -1.82 | +1.34 | 0/20 | 0% |
| shoulder_lift | +5.72 | 2.84 | -0.99 | +13.07 | 10/20 | 50% |
| elbow_flex | -6.22 | 2.11 | -11.01 | -1.52 | 14/20 | 70% |
| wrist_flex | +0.60 | 0.92 | -1.60 | +2.34 | 0/20 | 0% |
| wrist_roll | +0.27 | 0.15 | -0.03 | +0.72 | 0/20 | 0% |
| gripper | +1.14 | 0.53 | -0.08 | +1.98 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([2, 6, 8, 12])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 8, '2': 8}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.25, std=2.17, min=1.90, max=9.92
