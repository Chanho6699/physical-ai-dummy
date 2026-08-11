# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T10_actual/shadow_patched.json`)
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
| 0 | -0.91 | +7.09 | -11.27 | +1.01 | +0.17 | +0.87 | 2 | shoulder_lift, elbow_flex | 7.61 |
| 1 | +0.83 | +13.67 | -8.33 | +0.07 | +0.29 | +0.58 | 2 | shoulder_lift, elbow_flex | 10.55 |
| 2 | +1.02 | +4.15 | -4.14 | +0.63 | +0.28 | +1.29 | 0 | - | 1.97 |
| 3 | +1.20 | +4.67 | -7.58 | +2.04 | +0.17 | +1.25 | 1 | elbow_flex | 4.04 |
| 4 | -0.59 | +6.01 | -6.90 | +0.46 | +0.39 | +1.92 | 2 | shoulder_lift, elbow_flex | 3.97 |
| 5 | +0.80 | +7.23 | -6.71 | +1.94 | +0.21 | +0.64 | 2 | shoulder_lift, elbow_flex | 4.48 |
| 6 | +0.56 | +3.49 | -5.86 | +1.82 | +0.04 | +0.97 | 1 | elbow_flex | 2.47 |
| 7 | +0.21 | +7.88 | -6.68 | +1.28 | +0.12 | -0.16 | 2 | shoulder_lift, elbow_flex | 4.64 |
| 8 | +0.40 | +5.19 | -3.40 | +1.32 | +0.39 | +0.78 | 1 | shoulder_lift | 2.43 |
| 9 | +0.20 | +5.18 | -6.53 | +0.95 | +0.19 | +0.39 | 2 | shoulder_lift, elbow_flex | 2.55 |
| 10 | +0.75 | +9.48 | -7.69 | +0.32 | +0.15 | +1.62 | 2 | shoulder_lift, elbow_flex | 6.72 |
| 11 | +0.49 | +4.54 | -6.38 | +1.04 | +0.34 | +1.52 | 1 | elbow_flex | 2.82 |
| 12 | +0.61 | -0.68 | -4.85 | +0.65 | +0.06 | +1.92 | 0 | - | 5.11 |
| 13 | +1.44 | +5.32 | -6.14 | +1.23 | +0.33 | +1.16 | 2 | shoulder_lift, elbow_flex | 3.07 |
| 14 | +0.31 | +6.56 | -6.48 | -0.11 | +0.22 | +0.87 | 2 | shoulder_lift, elbow_flex | 3.52 |
| 15 | -0.29 | +10.01 | -7.65 | -0.10 | +0.23 | +0.78 | 2 | shoulder_lift, elbow_flex | 6.96 |
| 16 | -1.22 | +9.76 | -9.67 | +2.59 | +0.25 | +1.45 | 2 | shoulder_lift, elbow_flex | 8.43 |
| 17 | -0.13 | +6.11 | -3.63 | -0.49 | +0.19 | +0.34 | 1 | shoulder_lift | 2.67 |
| 18 | +2.03 | +5.10 | -6.33 | +1.37 | -0.09 | +1.19 | 1 | elbow_flex | 3.39 |
| 19 | +1.44 | +5.74 | -1.45 | -1.46 | +0.67 | +1.84 | 1 | shoulder_lift | 4.82 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | +0.46 | 0.79 | -1.22 | +2.03 | 0/20 | 0% |
| shoulder_lift | +6.33 | 2.88 | -0.68 | +13.67 | 14/20 | 70% |
| elbow_flex | -6.38 | 2.15 | -11.27 | -1.45 | 15/20 | 75% |
| wrist_flex | +0.83 | 0.94 | -1.46 | +2.59 | 0/20 | 0% |
| wrist_roll | +0.23 | 0.15 | -0.09 | +0.67 | 0/20 | 0% |
| gripper | +1.06 | 0.55 | -0.16 | +1.92 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 2/20 ([2, 12])
- clamp-joint-count distribution (count -> #seeds): {'0': 2, '1': 7, '2': 11}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.61, std=2.26, min=1.97, max=10.55
