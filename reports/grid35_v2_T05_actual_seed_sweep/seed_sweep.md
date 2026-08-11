# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T05_actual/shadow_patched.json`)
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
| 0 | -2.13 | +7.01 | -12.05 | +0.99 | +0.28 | +0.38 | 2 | shoulder_lift, elbow_flex | 8.47 |
| 1 | -0.60 | +14.14 | -8.73 | +0.11 | +0.41 | +0.17 | 2 | shoulder_lift, elbow_flex | 11.12 |
| 2 | -0.09 | +3.99 | -4.35 | +0.63 | +0.39 | +0.96 | 0 | - | 1.48 |
| 3 | -0.02 | +4.56 | -7.99 | +2.02 | +0.28 | +0.79 | 1 | elbow_flex | 4.07 |
| 4 | -1.91 | +5.60 | -7.16 | +0.43 | +0.50 | +1.49 | 2 | shoulder_lift, elbow_flex | 4.18 |
| 5 | -0.32 | +7.21 | -6.88 | +1.91 | +0.33 | +0.18 | 2 | shoulder_lift, elbow_flex | 4.45 |
| 6 | -0.65 | +3.34 | -6.23 | +1.83 | +0.14 | +0.56 | 1 | elbow_flex | 2.64 |
| 7 | -1.01 | +8.25 | -7.26 | +1.31 | +0.24 | -0.58 | 2 | shoulder_lift, elbow_flex | 5.38 |
| 8 | -0.87 | +5.28 | -3.64 | +1.35 | +0.51 | +0.41 | 1 | shoulder_lift | 2.49 |
| 9 | -0.84 | +5.01 | -7.00 | +0.89 | +0.29 | -0.03 | 1 | elbow_flex | 2.95 |
| 10 | -0.54 | +9.43 | -8.13 | +0.36 | +0.27 | +1.20 | 2 | shoulder_lift, elbow_flex | 6.80 |
| 11 | -0.59 | +4.31 | -6.57 | +0.96 | +0.46 | +1.11 | 1 | elbow_flex | 2.74 |
| 12 | -0.57 | -1.14 | -5.11 | +0.53 | +0.15 | +1.52 | 0 | - | 5.43 |
| 13 | +0.37 | +5.17 | -6.19 | +1.19 | +0.44 | +0.70 | 2 | shoulder_lift, elbow_flex | 2.54 |
| 14 | -1.04 | +6.47 | -6.99 | -0.11 | +0.32 | +0.44 | 2 | shoulder_lift, elbow_flex | 3.84 |
| 15 | -1.64 | +10.20 | -8.28 | -0.08 | +0.36 | +0.37 | 2 | shoulder_lift, elbow_flex | 7.59 |
| 16 | -2.50 | +9.82 | -10.03 | +2.61 | +0.37 | +1.00 | 2 | shoulder_lift, elbow_flex | 8.92 |
| 17 | -1.44 | +6.40 | -3.98 | -0.52 | +0.30 | -0.08 | 1 | shoulder_lift | 3.19 |
| 18 | +0.88 | +4.83 | -6.59 | +1.35 | +0.01 | +0.74 | 1 | elbow_flex | 2.78 |
| 19 | +0.20 | +5.51 | -1.50 | -1.56 | +0.77 | +1.44 | 1 | shoulder_lift | 4.40 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.77 | 0.83 | -2.50 | +0.88 | 0/20 | 0% |
| shoulder_lift | +6.27 | 3.06 | -1.14 | +14.14 | 13/20 | 65% |
| elbow_flex | -6.73 | 2.27 | -12.05 | -1.50 | 15/20 | 75% |
| wrist_flex | +0.81 | 0.95 | -1.56 | +2.61 | 0/20 | 0% |
| wrist_roll | +0.34 | 0.16 | +0.01 | +0.77 | 0/20 | 0% |
| gripper | +0.64 | 0.55 | -0.58 | +1.52 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 2/20 ([2, 12])
- clamp-joint-count distribution (count -> #seeds): {'0': 2, '1': 8, '2': 10}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.77, std=2.50, min=1.48, max=11.12
