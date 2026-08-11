# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T07` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T07/shadow_synthetic_T07.json`)
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
| shoulder_pan | -1.0549 |
| shoulder_lift | +0.2637 |
| elbow_flex | -1.8901 |
| wrist_flex | -0.8352 |
| wrist_roll | +0.0879 |
| gripper | -0.1106 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -1.59 | +7.55 | -9.80 | +0.40 | +0.18 | +0.72 | 2 | shoulder_lift, elbow_flex | 10.87 |
| 1 | +0.15 | +14.85 | -6.31 | -0.49 | +0.31 | +0.40 | 2 | shoulder_lift, elbow_flex | 15.30 |
| 2 | +0.67 | +4.42 | -2.11 | +0.08 | +0.30 | +1.10 | 0 | - | 4.76 |
| 3 | +0.73 | +5.11 | -6.02 | +1.55 | +0.19 | +1.05 | 1 | elbow_flex | 7.13 |
| 4 | -1.38 | +6.30 | -4.98 | -0.10 | +0.42 | +1.70 | 1 | shoulder_lift | 7.07 |
| 5 | +0.43 | +7.98 | -4.50 | +1.37 | +0.24 | +0.37 | 1 | shoulder_lift | 8.59 |
| 6 | +0.07 | +4.04 | -4.15 | +1.39 | +0.04 | +0.78 | 0 | - | 5.14 |
| 7 | -0.29 | +8.81 | -5.09 | +0.80 | +0.13 | -0.38 | 1 | shoulder_lift | 9.31 |
| 8 | -0.36 | +5.94 | -1.34 | +0.81 | +0.42 | +0.57 | 1 | shoulder_lift | 6.02 |
| 9 | -0.29 | +5.71 | -4.87 | +0.47 | +0.20 | +0.25 | 1 | shoulder_lift | 6.40 |
| 10 | +0.24 | +10.43 | -5.87 | -0.20 | +0.18 | +1.39 | 2 | shoulder_lift, elbow_flex | 11.11 |
| 11 | +0.09 | +5.03 | -4.59 | +0.54 | +0.38 | +1.33 | 0 | - | 5.94 |
| 12 | -0.10 | -0.54 | -3.10 | +0.04 | +0.07 | +1.76 | 0 | - | 2.69 |
| 13 | +1.09 | +5.93 | -3.88 | +0.69 | +0.36 | +0.89 | 1 | shoulder_lift | 6.64 |
| 14 | -0.42 | +7.26 | -4.97 | -0.55 | +0.25 | +0.66 | 1 | shoulder_lift | 7.72 |
| 15 | -1.04 | +11.01 | -6.03 | -0.64 | +0.27 | +0.64 | 2 | shoulder_lift, elbow_flex | 11.55 |
| 16 | -1.98 | +10.60 | -7.73 | +2.04 | +0.28 | +1.25 | 2 | shoulder_lift, elbow_flex | 12.33 |
| 17 | -1.00 | +7.21 | -1.79 | -0.96 | +0.22 | +0.16 | 1 | shoulder_lift | 6.95 |
| 18 | +1.46 | +5.64 | -4.44 | +0.89 | -0.09 | +0.97 | 1 | shoulder_lift | 6.78 |
| 19 | +0.90 | +6.50 | +0.91 | -2.05 | +0.72 | +1.72 | 1 | shoulder_lift | 7.47 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.13 | 0.89 | -1.98 | +1.46 | 0/20 | 0% |
| shoulder_lift | +6.99 | 3.11 | -0.54 | +14.85 | 15/20 | 75% |
| elbow_flex | -4.53 | 2.28 | -9.80 | +0.91 | 6/20 | 30% |
| wrist_flex | +0.30 | 0.95 | -2.05 | +2.04 | 0/20 | 0% |
| wrist_roll | +0.25 | 0.16 | -0.09 | +0.72 | 0/20 | 0% |
| gripper | +0.87 | 0.55 | -0.38 | +1.76 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([2, 6, 11, 12])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 11, '2': 5}
- L2 error vs nearest-demo immediate GT delta (deg): mean=7.99, std=2.90, min=2.69, max=15.30
