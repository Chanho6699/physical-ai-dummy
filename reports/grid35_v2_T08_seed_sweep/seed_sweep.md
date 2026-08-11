# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T08` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T08/shadow_synthetic_T08.json`)
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
| shoulder_pan | +0.2637 |
| shoulder_lift | +0.2637 |
| elbow_flex | -1.8022 |
| wrist_flex | -0.8352 |
| wrist_roll | +0.2637 |
| gripper | +0.1615 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -1.86 | +8.42 | -10.19 | +0.62 | +0.28 | +0.83 | 2 | shoulder_lift, elbow_flex | 11.99 |
| 1 | -0.15 | +15.36 | -6.48 | -0.41 | +0.40 | +0.75 | 2 | shoulder_lift, elbow_flex | 15.82 |
| 2 | +0.27 | +5.02 | -2.28 | +0.19 | +0.38 | +1.36 | 0 | - | 5.03 |
| 3 | +0.39 | +6.02 | -6.13 | +1.64 | +0.28 | +1.18 | 2 | shoulder_lift, elbow_flex | 7.69 |
| 4 | -1.65 | +6.72 | -4.91 | -0.00 | +0.50 | +1.96 | 1 | shoulder_lift | 7.69 |
| 5 | +0.19 | +8.64 | -4.68 | +1.50 | +0.32 | +0.60 | 1 | shoulder_lift | 9.17 |
| 6 | -0.21 | +4.73 | -4.24 | +1.48 | +0.14 | +0.93 | 0 | - | 5.67 |
| 7 | -0.59 | +9.83 | -5.33 | +0.85 | +0.23 | -0.16 | 1 | shoulder_lift | 10.38 |
| 8 | -0.57 | +6.38 | -1.24 | +0.86 | +0.50 | +0.91 | 1 | shoulder_lift | 6.48 |
| 9 | -0.58 | +6.59 | -5.11 | +0.63 | +0.29 | +0.40 | 1 | shoulder_lift | 7.34 |
| 10 | -0.08 | +10.95 | -6.00 | -0.10 | +0.27 | +1.64 | 2 | shoulder_lift, elbow_flex | 11.60 |
| 11 | -0.18 | +5.60 | -4.62 | +0.58 | +0.46 | +1.49 | 1 | shoulder_lift | 6.36 |
| 12 | -0.48 | -0.30 | -3.00 | +0.12 | +0.16 | +1.89 | 0 | - | 2.49 |
| 13 | +0.90 | +6.31 | -3.83 | +0.74 | +0.44 | +1.12 | 1 | shoulder_lift | 6.67 |
| 14 | -0.83 | +8.06 | -5.15 | -0.39 | +0.34 | +0.90 | 1 | shoulder_lift | 8.60 |
| 15 | -1.40 | +11.75 | -6.34 | -0.45 | +0.35 | +0.89 | 2 | shoulder_lift, elbow_flex | 12.49 |
| 16 | -2.07 | +11.16 | -7.86 | +2.09 | +0.36 | +1.45 | 2 | shoulder_lift, elbow_flex | 13.08 |
| 17 | -1.24 | +7.58 | -1.58 | -0.93 | +0.31 | +0.43 | 1 | shoulder_lift | 7.48 |
| 18 | +1.17 | +6.15 | -4.45 | +0.93 | +0.01 | +1.20 | 1 | shoulder_lift | 6.83 |
| 19 | +0.52 | +6.40 | +1.20 | -2.07 | +0.78 | +1.95 | 1 | shoulder_lift | 7.19 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.42 | 0.87 | -2.07 | +1.17 | 0/20 | 0% |
| shoulder_lift | +7.57 | 3.17 | -0.30 | +15.36 | 17/20 | 85% |
| elbow_flex | -4.61 | 2.41 | -10.19 | +1.20 | 6/20 | 30% |
| wrist_flex | +0.39 | 0.95 | -2.07 | +2.09 | 0/20 | 0% |
| wrist_roll | +0.34 | 0.16 | +0.01 | +0.78 | 0/20 | 0% |
| gripper | +1.09 | 0.54 | -0.16 | +1.96 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 3/20 ([2, 6, 12])
- clamp-joint-count distribution (count -> #seeds): {'0': 3, '1': 11, '2': 6}
- L2 error vs nearest-demo immediate GT delta (deg): mean=8.50, std=3.10, min=2.49, max=15.82
