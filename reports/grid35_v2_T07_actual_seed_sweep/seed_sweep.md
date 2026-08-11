# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T07_actual/shadow_patched.json`)
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
| 0 | -2.00 | +6.33 | -11.72 | +1.20 | +0.20 | +0.65 | 2 | shoulder_lift, elbow_flex | 7.95 |
| 1 | -0.33 | +13.12 | -8.66 | +0.30 | +0.31 | +0.25 | 2 | shoulder_lift, elbow_flex | 10.13 |
| 2 | -0.01 | +3.16 | -4.23 | +0.83 | +0.30 | +1.00 | 0 | - | 1.72 |
| 3 | +0.15 | +3.83 | -7.82 | +2.26 | +0.20 | +0.97 | 1 | elbow_flex | 4.02 |
| 4 | -1.76 | +5.04 | -7.06 | +0.67 | +0.43 | +1.60 | 1 | elbow_flex | 3.91 |
| 5 | -0.22 | +6.40 | -6.79 | +2.18 | +0.24 | +0.28 | 2 | shoulder_lift, elbow_flex | 3.95 |
| 6 | -0.53 | +2.58 | -5.97 | +2.09 | +0.05 | +0.65 | 1 | elbow_flex | 2.91 |
| 7 | -0.88 | +7.25 | -6.92 | +1.52 | +0.14 | -0.48 | 2 | shoulder_lift, elbow_flex | 4.43 |
| 8 | -0.81 | +4.46 | -3.56 | +1.55 | +0.42 | +0.48 | 0 | - | 2.26 |
| 9 | -0.83 | +4.38 | -6.85 | +1.18 | +0.21 | +0.13 | 1 | elbow_flex | 2.74 |
| 10 | -0.32 | +8.70 | -7.98 | +0.54 | +0.18 | +1.27 | 2 | shoulder_lift, elbow_flex | 6.13 |
| 11 | -0.50 | +3.60 | -6.54 | +1.25 | +0.38 | +1.24 | 1 | elbow_flex | 2.82 |
| 12 | -0.61 | -1.61 | -5.10 | +0.81 | +0.07 | +1.64 | 0 | - | 5.93 |
| 13 | +0.42 | +4.39 | -6.17 | +1.48 | +0.36 | +0.81 | 1 | elbow_flex | 2.41 |
| 14 | -0.90 | +5.82 | -6.92 | +0.11 | +0.25 | +0.59 | 2 | shoulder_lift, elbow_flex | 3.35 |
| 15 | -1.46 | +9.35 | -8.15 | +0.11 | +0.27 | +0.51 | 2 | shoulder_lift, elbow_flex | 6.79 |
| 16 | -2.35 | +9.05 | -9.81 | +2.84 | +0.28 | +1.10 | 2 | shoulder_lift, elbow_flex | 8.32 |
| 17 | -1.36 | +5.55 | -3.84 | -0.28 | +0.22 | +0.05 | 1 | shoulder_lift | 2.50 |
| 18 | +0.91 | +4.35 | -6.47 | +1.61 | -0.08 | +0.90 | 1 | elbow_flex | 2.76 |
| 19 | +0.36 | +4.88 | -1.48 | -1.25 | +0.71 | +1.60 | 0 | - | 4.16 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.65 | 0.82 | -2.35 | +0.91 | 0/20 | 0% |
| shoulder_lift | +5.53 | 2.97 | -1.61 | +13.12 | 9/20 | 45% |
| elbow_flex | -6.60 | 2.22 | -11.72 | -1.48 | 15/20 | 75% |
| wrist_flex | +1.05 | 0.95 | -1.25 | +2.84 | 0/20 | 0% |
| wrist_roll | +0.26 | 0.16 | -0.08 | +0.71 | 0/20 | 0% |
| gripper | +0.76 | 0.55 | -0.48 | +1.64 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([2, 8, 12, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 8, '2': 8}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.46, std=2.27, min=1.72, max=10.13
