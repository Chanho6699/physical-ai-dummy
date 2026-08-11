# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T09_actual/shadow_patched.json`)
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
| 0 | -1.96 | +5.44 | -10.64 | +1.16 | +0.16 | +0.77 | 2 | shoulder_lift, elbow_flex | 6.74 |
| 1 | -0.26 | +12.11 | -7.67 | +0.28 | +0.27 | +0.52 | 2 | shoulder_lift, elbow_flex | 8.84 |
| 2 | +0.03 | +2.39 | -3.35 | +0.85 | +0.25 | +1.15 | 0 | - | 2.55 |
| 3 | +0.24 | +3.02 | -6.94 | +2.26 | +0.16 | +1.05 | 1 | elbow_flex | 3.50 |
| 4 | -1.66 | +4.21 | -6.10 | +0.59 | +0.38 | +1.78 | 1 | elbow_flex | 3.24 |
| 5 | -0.12 | +5.52 | -5.99 | +2.16 | +0.19 | +0.54 | 2 | shoulder_lift, elbow_flex | 3.04 |
| 6 | -0.45 | +1.91 | -5.34 | +2.10 | +0.02 | +0.78 | 0 | - | 3.08 |
| 7 | -0.85 | +6.42 | -6.10 | +1.53 | +0.10 | -0.34 | 2 | shoulder_lift, elbow_flex | 3.40 |
| 8 | -0.67 | +3.72 | -2.78 | +1.54 | +0.37 | +0.61 | 0 | - | 2.63 |
| 9 | -0.80 | +3.59 | -6.00 | +1.16 | +0.16 | +0.31 | 1 | elbow_flex | 2.08 |
| 10 | -0.23 | +7.83 | -6.94 | +0.49 | +0.15 | +1.52 | 2 | shoulder_lift, elbow_flex | 4.97 |
| 11 | -0.42 | +2.80 | -5.52 | +1.19 | +0.33 | +1.36 | 0 | - | 2.50 |
| 12 | -0.51 | -2.36 | -4.25 | +0.82 | +0.03 | +1.69 | 0 | - | 6.62 |
| 13 | +0.47 | +3.69 | -5.37 | +1.43 | +0.32 | +0.97 | 0 | - | 2.00 |
| 14 | -0.88 | +5.07 | -5.92 | +0.09 | +0.21 | +0.67 | 1 | elbow_flex | 2.31 |
| 15 | -1.42 | +8.52 | -7.17 | +0.08 | +0.22 | +0.69 | 2 | shoulder_lift, elbow_flex | 5.65 |
| 16 | -2.29 | +8.20 | -9.04 | +2.84 | +0.25 | +1.36 | 2 | shoulder_lift, elbow_flex | 7.36 |
| 17 | -1.30 | +4.69 | -3.02 | -0.25 | +0.17 | +0.21 | 0 | - | 2.44 |
| 18 | +0.98 | +3.58 | -5.62 | +1.65 | -0.11 | +1.01 | 0 | - | 2.36 |
| 19 | +0.24 | +4.13 | -0.61 | -1.25 | +0.64 | +1.87 | 0 | - | 4.86 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.59 | 0.81 | -2.29 | +0.98 | 0/20 | 0% |
| shoulder_lift | +4.72 | 2.92 | -2.36 | +12.11 | 7/20 | 35% |
| elbow_flex | -5.72 | 2.18 | -10.64 | -0.61 | 11/20 | 55% |
| wrist_flex | +1.04 | 0.95 | -1.25 | +2.84 | 0/20 | 0% |
| wrist_roll | +0.21 | 0.15 | -0.11 | +0.64 | 0/20 | 0% |
| gripper | +0.93 | 0.55 | -0.34 | +1.87 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 9/20 ([2, 6, 8, 11, 12, 13, 17, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 9, '1': 4, '2': 7}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.01, std=1.98, min=2.00, max=8.84
