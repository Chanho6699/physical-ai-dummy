# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T08_actual/shadow_patched.json`)
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
| 0 | -1.59 | +6.12 | -11.56 | +0.97 | +0.32 | +0.25 | 2 | shoulder_lift, elbow_flex | 7.58 |
| 1 | +0.12 | +13.15 | -8.30 | -0.02 | +0.44 | +0.09 | 2 | shoulder_lift, elbow_flex | 10.01 |
| 2 | +0.44 | +3.22 | -4.03 | +0.62 | +0.42 | +0.76 | 0 | - | 1.55 |
| 3 | +0.45 | +3.69 | -7.61 | +1.97 | +0.32 | +0.65 | 1 | elbow_flex | 3.64 |
| 4 | -1.15 | +4.70 | -6.62 | +0.40 | +0.54 | +1.34 | 1 | elbow_flex | 3.11 |
| 5 | +0.28 | +6.35 | -6.57 | +1.87 | +0.36 | +0.06 | 2 | shoulder_lift, elbow_flex | 3.62 |
| 6 | -0.18 | +2.58 | -5.85 | +1.82 | +0.18 | +0.39 | 1 | elbow_flex | 2.54 |
| 7 | -0.47 | +7.33 | -6.81 | +1.26 | +0.27 | -0.69 | 2 | shoulder_lift, elbow_flex | 4.31 |
| 8 | -0.20 | +4.35 | -3.23 | +1.30 | +0.53 | +0.28 | 0 | - | 2.05 |
| 9 | -0.46 | +4.22 | -6.67 | +0.95 | +0.33 | -0.16 | 1 | elbow_flex | 2.35 |
| 10 | +0.11 | +8.67 | -7.79 | +0.24 | +0.31 | +1.06 | 2 | shoulder_lift, elbow_flex | 5.94 |
| 11 | +0.04 | +3.39 | -6.13 | +0.98 | +0.49 | +0.92 | 1 | elbow_flex | 2.23 |
| 12 | -0.14 | -1.71 | -4.91 | +0.67 | +0.20 | +1.25 | 0 | - | 5.84 |
| 13 | +0.97 | +4.27 | -5.76 | +1.15 | +0.48 | +0.57 | 1 | elbow_flex | 2.02 |
| 14 | -0.51 | +5.81 | -6.68 | -0.09 | +0.38 | +0.34 | 2 | shoulder_lift, elbow_flex | 3.04 |
| 15 | -0.99 | +9.39 | -8.02 | -0.12 | +0.40 | +0.24 | 2 | shoulder_lift, elbow_flex | 6.64 |
| 16 | -1.77 | +8.93 | -9.63 | +2.52 | +0.40 | +0.86 | 2 | shoulder_lift, elbow_flex | 7.84 |
| 17 | -0.81 | +5.49 | -3.57 | -0.46 | +0.34 | -0.13 | 1 | shoulder_lift | 2.32 |
| 18 | +1.36 | +4.27 | -6.38 | +1.38 | +0.05 | +0.61 | 1 | elbow_flex | 2.63 |
| 19 | +0.76 | +4.94 | -1.32 | -1.45 | +0.82 | +1.31 | 0 | - | 4.32 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.19 | 0.79 | -1.77 | +1.36 | 0/20 | 0% |
| shoulder_lift | +5.46 | 3.00 | -1.71 | +13.15 | 9/20 | 45% |
| elbow_flex | -6.37 | 2.23 | -11.56 | -1.32 | 15/20 | 75% |
| wrist_flex | +0.80 | 0.92 | -1.45 | +2.52 | 0/20 | 0% |
| wrist_roll | +0.38 | 0.15 | +0.05 | +0.82 | 0/20 | 0% |
| gripper | +0.50 | 0.52 | -0.69 | +1.34 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([2, 8, 12, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 8, '2': 8}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.18, std=2.30, min=1.55, max=10.01
