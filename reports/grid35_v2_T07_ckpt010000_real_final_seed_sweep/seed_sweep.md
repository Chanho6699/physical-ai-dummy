# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T07` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T07_real_final/shadow_patched.json`)
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
| 0 | -2.89 | +6.81 | -13.80 | +0.93 | +0.24 | +0.20 | 2 | shoulder_lift, elbow_flex | 10.16 |
| 1 | -1.39 | +13.06 | -12.27 | +0.25 | +0.36 | -0.36 | 2 | shoulder_lift, elbow_flex | 12.07 |
| 2 | -1.11 | +4.20 | -8.01 | +0.78 | +0.31 | +0.61 | 1 | elbow_flex | 3.82 |
| 3 | -1.22 | +4.62 | -10.42 | +1.82 | +0.25 | +0.35 | 1 | elbow_flex | 6.27 |
| 4 | -2.54 | +5.86 | -10.71 | +0.60 | +0.43 | +0.93 | 2 | shoulder_lift, elbow_flex | 7.10 |
| 5 | -1.25 | +6.86 | -10.41 | +1.78 | +0.30 | -0.09 | 2 | shoulder_lift, elbow_flex | 6.87 |
| 6 | -1.87 | +3.71 | -9.52 | +1.73 | +0.15 | +0.28 | 1 | elbow_flex | 5.57 |
| 7 | -1.83 | +7.91 | -10.79 | +1.50 | +0.24 | -0.60 | 2 | shoulder_lift, elbow_flex | 7.77 |
| 8 | -1.65 | +5.62 | -8.23 | +1.30 | +0.41 | +0.16 | 2 | shoulder_lift, elbow_flex | 4.60 |
| 9 | -1.87 | +4.85 | -9.88 | +0.87 | +0.25 | -0.20 | 1 | elbow_flex | 5.78 |
| 10 | -1.41 | +8.62 | -11.17 | +0.44 | +0.22 | +0.67 | 2 | shoulder_lift, elbow_flex | 8.32 |
| 11 | -1.45 | +4.94 | -9.89 | +1.18 | +0.40 | +0.69 | 1 | elbow_flex | 5.81 |
| 12 | -1.80 | +1.04 | -8.89 | +0.84 | +0.14 | +1.12 | 1 | elbow_flex | 5.71 |
| 13 | -0.88 | +5.05 | -9.59 | +1.19 | +0.39 | +0.44 | 1 | elbow_flex | 5.39 |
| 14 | -1.70 | +6.65 | -10.39 | +0.20 | +0.28 | +0.16 | 2 | shoulder_lift, elbow_flex | 6.70 |
| 15 | -2.09 | +10.06 | -11.93 | +0.15 | +0.31 | -0.06 | 2 | shoulder_lift, elbow_flex | 9.86 |
| 16 | -3.14 | +9.35 | -13.33 | +2.52 | +0.34 | +0.66 | 2 | shoulder_lift, elbow_flex | 11.11 |
| 17 | -1.93 | +6.02 | -8.39 | -0.10 | +0.25 | -0.13 | 2 | shoulder_lift, elbow_flex | 4.86 |
| 18 | -0.83 | +5.12 | -10.01 | +1.56 | +0.05 | +0.22 | 1 | elbow_flex | 5.81 |
| 19 | -1.14 | +5.91 | -6.55 | -0.75 | +0.66 | +0.90 | 2 | shoulder_lift, elbow_flex | 3.56 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.70 | 0.60 | -3.14 | -0.83 | 0/20 | 0% |
| shoulder_lift | +6.31 | 2.52 | +1.04 | +13.06 | 12/20 | 60% |
| elbow_flex | -10.21 | 1.73 | -13.80 | -6.55 | 20/20 | 100% |
| wrist_flex | +0.94 | 0.76 | -0.75 | +2.52 | 0/20 | 0% |
| wrist_roll | +0.30 | 0.13 | +0.05 | +0.66 | 0/20 | 0% |
| gripper | +0.30 | 0.45 | -0.60 | +1.12 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 8, '2': 12}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.86, std=2.31, min=3.56, max=12.07
