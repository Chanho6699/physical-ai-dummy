# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T04` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T04/shadow_synthetic_T04.json`)
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
| shoulder_lift | +0.6154 |
| elbow_flex | -2.3297 |
| wrist_flex | +0.1319 |
| wrist_roll | +0.0000 |
| gripper | -0.3216 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -2.69 | +6.83 | -9.98 | +0.81 | +0.19 | +0.75 | 2 | shoulder_lift, elbow_flex | 10.34 |
| 1 | -1.02 | +14.01 | -6.40 | -0.08 | +0.30 | +0.47 | 2 | shoulder_lift, elbow_flex | 14.08 |
| 2 | -0.54 | +3.37 | -1.87 | +0.48 | +0.30 | +1.13 | 0 | - | 3.26 |
| 3 | -0.33 | +4.25 | -5.81 | +1.92 | +0.19 | +1.15 | 1 | elbow_flex | 5.57 |
| 4 | -2.47 | +5.18 | -4.88 | +0.26 | +0.41 | +1.76 | 1 | shoulder_lift | 6.23 |
| 5 | -0.76 | +7.06 | -4.34 | +1.77 | +0.22 | +0.41 | 1 | shoulder_lift | 7.05 |
| 6 | -0.97 | +2.98 | -3.70 | +1.74 | +0.05 | +0.86 | 0 | - | 3.58 |
| 7 | -1.45 | +8.06 | -4.92 | +1.18 | +0.13 | -0.36 | 1 | shoulder_lift | 8.11 |
| 8 | -1.37 | +4.81 | -0.99 | +1.13 | +0.42 | +0.66 | 0 | - | 4.89 |
| 9 | -1.41 | +5.00 | -4.77 | +0.87 | +0.21 | +0.25 | 0 | - | 5.35 |
| 10 | -0.92 | +9.50 | -5.75 | +0.18 | +0.17 | +1.49 | 2 | shoulder_lift, elbow_flex | 9.75 |
| 11 | -0.99 | +3.99 | -4.41 | +0.85 | +0.37 | +1.37 | 0 | - | 4.53 |
| 12 | -1.01 | -1.99 | -2.62 | +0.45 | +0.08 | +1.88 | 0 | - | 3.64 |
| 13 | -0.07 | +4.95 | -3.76 | +1.06 | +0.34 | +0.99 | 0 | - | 4.86 |
| 14 | -1.49 | +6.38 | -4.76 | -0.21 | +0.25 | +0.81 | 1 | shoulder_lift | 6.58 |
| 15 | -2.08 | +10.09 | -5.89 | -0.29 | +0.26 | +0.67 | 2 | shoulder_lift, elbow_flex | 10.43 |
| 16 | -2.99 | +9.64 | -7.84 | +2.42 | +0.26 | +1.31 | 2 | shoulder_lift, elbow_flex | 11.39 |
| 17 | -2.05 | +6.23 | -1.34 | -0.67 | +0.22 | +0.19 | 1 | shoulder_lift | 6.20 |
| 18 | +0.62 | +4.60 | -4.01 | +1.26 | -0.07 | +1.15 | 0 | - | 4.73 |
| 19 | -0.14 | +5.10 | +1.34 | -1.77 | +0.72 | +1.74 | 0 | - | 6.48 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.21 | 0.90 | -2.99 | +0.62 | 0/20 | 0% |
| shoulder_lift | +6.00 | 3.21 | -1.99 | +14.01 | 10/20 | 50% |
| elbow_flex | -4.33 | 2.43 | -9.98 | +1.34 | 6/20 | 30% |
| wrist_flex | +0.67 | 0.96 | -1.77 | +2.42 | 0/20 | 0% |
| wrist_roll | +0.25 | 0.16 | -0.07 | +0.72 | 0/20 | 0% |
| gripper | +0.93 | 0.57 | -0.36 | +1.88 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 9/20 ([2, 6, 8, 9, 11, 12, 13, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 9, '1': 6, '2': 5}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.85, std=2.87, min=3.26, max=14.08
