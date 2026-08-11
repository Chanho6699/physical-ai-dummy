# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T06_actual/shadow_patched.json`)
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
| 0 | -1.89 | +6.77 | -11.43 | +1.13 | +0.24 | +0.80 | 2 | shoulder_lift, elbow_flex | 7.83 |
| 1 | -0.27 | +13.64 | -8.38 | +0.27 | +0.36 | +0.50 | 2 | shoulder_lift, elbow_flex | 10.52 |
| 2 | +0.08 | +3.69 | -3.98 | +0.84 | +0.34 | +1.20 | 0 | - | 1.82 |
| 3 | +0.30 | +4.31 | -7.57 | +2.24 | +0.24 | +1.12 | 1 | elbow_flex | 3.90 |
| 4 | -1.63 | +5.49 | -6.78 | +0.61 | +0.48 | +1.86 | 2 | shoulder_lift, elbow_flex | 3.97 |
| 5 | -0.09 | +6.89 | -6.50 | +2.17 | +0.28 | +0.51 | 2 | shoulder_lift, elbow_flex | 4.17 |
| 6 | -0.42 | +3.14 | -5.82 | +2.08 | +0.09 | +0.84 | 1 | elbow_flex | 2.68 |
| 7 | -0.69 | +7.85 | -6.73 | +1.55 | +0.19 | -0.28 | 2 | shoulder_lift, elbow_flex | 4.78 |
| 8 | -0.70 | +4.85 | -3.22 | +1.54 | +0.46 | +0.67 | 0 | - | 2.59 |
| 9 | -0.70 | +4.78 | -6.52 | +1.10 | +0.25 | +0.30 | 1 | elbow_flex | 2.55 |
| 10 | -0.22 | +9.22 | -7.58 | +0.48 | +0.23 | +1.55 | 2 | shoulder_lift, elbow_flex | 6.42 |
| 11 | -0.39 | +4.06 | -6.30 | +1.19 | +0.42 | +1.44 | 1 | elbow_flex | 2.74 |
| 12 | -0.41 | -1.34 | -4.60 | +0.69 | +0.11 | +1.78 | 0 | - | 5.67 |
| 13 | +0.54 | +5.01 | -6.06 | +1.43 | +0.41 | +0.98 | 1 | elbow_flex | 2.63 |
| 14 | -0.76 | +6.25 | -6.48 | +0.02 | +0.28 | +0.78 | 2 | shoulder_lift, elbow_flex | 3.38 |
| 15 | -1.34 | +9.76 | -7.74 | +0.03 | +0.31 | +0.72 | 2 | shoulder_lift, elbow_flex | 6.93 |
| 16 | -2.24 | +9.63 | -9.68 | +2.88 | +0.34 | +1.33 | 2 | shoulder_lift, elbow_flex | 8.64 |
| 17 | -1.23 | +5.88 | -3.48 | -0.30 | +0.26 | +0.24 | 1 | shoulder_lift | 2.83 |
| 18 | +1.04 | +4.77 | -6.14 | +1.53 | -0.03 | +1.09 | 1 | elbow_flex | 2.76 |
| 19 | +0.36 | +5.10 | -1.11 | -1.31 | +0.73 | +1.87 | 0 | - | 4.66 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.53 | 0.81 | -2.24 | +1.04 | 0/20 | 0% |
| shoulder_lift | +5.99 | 3.01 | -1.34 | +13.64 | 10/20 | 50% |
| elbow_flex | -6.30 | 2.24 | -11.43 | -1.11 | 15/20 | 75% |
| wrist_flex | +1.01 | 0.97 | -1.31 | +2.88 | 0/20 | 0% |
| wrist_roll | +0.30 | 0.16 | -0.03 | +0.73 | 0/20 | 0% |
| gripper | +0.97 | 0.56 | -0.28 | +1.87 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([2, 8, 12, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 7, '2': 9}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.57, std=2.32, min=1.82, max=10.52
