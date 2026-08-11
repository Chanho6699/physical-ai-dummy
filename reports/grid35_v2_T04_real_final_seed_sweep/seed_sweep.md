# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T04` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T04_real_final/shadow_patched.json`)
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
| 0 | -2.54 | +7.58 | -12.87 | +1.25 | +0.23 | +0.02 | 2 | shoulder_lift, elbow_flex | 9.52 |
| 1 | -1.18 | +13.83 | -9.74 | +0.43 | +0.34 | -0.28 | 2 | shoulder_lift, elbow_flex | 11.30 |
| 2 | -0.59 | +4.68 | -5.67 | +0.90 | +0.33 | +0.58 | 0 | - | 1.96 |
| 3 | -0.54 | +5.33 | -8.98 | +2.18 | +0.23 | +0.43 | 2 | shoulder_lift, elbow_flex | 5.12 |
| 4 | -2.21 | +6.02 | -8.24 | +0.70 | +0.43 | +1.03 | 2 | shoulder_lift, elbow_flex | 5.08 |
| 5 | -0.78 | +7.54 | -8.24 | +2.07 | +0.27 | -0.14 | 2 | shoulder_lift, elbow_flex | 5.58 |
| 6 | -1.21 | +4.18 | -7.37 | +2.03 | +0.11 | +0.21 | 1 | elbow_flex | 3.64 |
| 7 | -1.42 | +8.50 | -8.34 | +1.52 | +0.19 | -0.86 | 2 | shoulder_lift, elbow_flex | 6.32 |
| 8 | -1.28 | +5.76 | -4.96 | +1.61 | +0.44 | +0.01 | 1 | shoulder_lift | 2.85 |
| 9 | -1.36 | +5.84 | -8.27 | +1.14 | +0.25 | -0.34 | 2 | shoulder_lift, elbow_flex | 4.54 |
| 10 | -1.00 | +9.49 | -9.10 | +0.61 | +0.22 | +0.74 | 2 | shoulder_lift, elbow_flex | 7.38 |
| 11 | -1.02 | +4.97 | -7.89 | +1.31 | +0.39 | +0.69 | 1 | elbow_flex | 3.97 |
| 12 | -1.04 | +0.38 | -6.43 | +0.93 | +0.14 | +1.08 | 1 | elbow_flex | 4.41 |
| 13 | -0.25 | +5.71 | -7.54 | +1.49 | +0.39 | +0.27 | 2 | shoulder_lift, elbow_flex | 3.78 |
| 14 | -1.45 | +7.03 | -8.05 | +0.31 | +0.28 | +0.08 | 2 | shoulder_lift, elbow_flex | 4.98 |
| 15 | -2.01 | +10.37 | -9.29 | +0.26 | +0.30 | -0.05 | 2 | shoulder_lift, elbow_flex | 8.31 |
| 16 | -2.94 | +10.09 | -11.14 | +2.76 | +0.32 | +0.60 | 2 | shoulder_lift, elbow_flex | 9.92 |
| 17 | -1.88 | +6.71 | -5.28 | -0.08 | +0.26 | -0.31 | 1 | shoulder_lift | 3.60 |
| 18 | +0.09 | +5.62 | -7.69 | +1.62 | +0.02 | +0.36 | 2 | shoulder_lift, elbow_flex | 3.87 |
| 19 | -0.39 | +6.05 | -2.80 | -1.08 | +0.71 | +0.99 | 1 | shoulder_lift | 3.53 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.25 | 0.75 | -2.94 | +0.09 | 0/20 | 0% |
| shoulder_lift | +6.78 | 2.72 | +0.38 | +13.83 | 16/20 | 80% |
| elbow_flex | -7.89 | 2.16 | -12.87 | -2.80 | 16/20 | 80% |
| wrist_flex | +1.10 | 0.87 | -1.08 | +2.76 | 0/20 | 0% |
| wrist_roll | +0.29 | 0.14 | +0.02 | +0.71 | 0/20 | 0% |
| gripper | +0.26 | 0.51 | -0.86 | +1.08 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([2])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 6, '2': 13}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.48, std=2.48, min=1.96, max=11.30
