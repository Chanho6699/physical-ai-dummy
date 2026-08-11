# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model`
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
| 0 | -2.35 | +5.65 | -12.51 | +0.88 | +0.37 | +0.46 | 2 | shoulder_lift, elbow_flex | 8.55 |
| 1 | -0.78 | +12.78 | -9.20 | -0.06 | +0.49 | +0.15 | 2 | shoulder_lift, elbow_flex | 10.10 |
| 2 | -0.27 | +2.92 | -4.72 | +0.47 | +0.46 | +0.96 | 0 | - | 1.77 |
| 3 | -0.15 | +3.32 | -8.25 | +1.82 | +0.36 | +0.86 | 1 | elbow_flex | 4.22 |
| 4 | -2.20 | +4.43 | -7.36 | +0.26 | +0.59 | +1.57 | 1 | elbow_flex | 4.19 |
| 5 | -0.46 | +6.11 | -7.34 | +1.76 | +0.41 | +0.20 | 2 | shoulder_lift, elbow_flex | 3.97 |
| 6 | -0.83 | +2.25 | -6.43 | +1.63 | +0.22 | +0.57 | 1 | elbow_flex | 3.13 |
| 7 | -1.18 | +6.97 | -7.62 | +1.12 | +0.32 | -0.55 | 2 | shoulder_lift, elbow_flex | 4.64 |
| 8 | -1.12 | +4.21 | -3.94 | +1.20 | +0.59 | +0.45 | 0 | - | 2.06 |
| 9 | -1.04 | +3.82 | -7.38 | +0.77 | +0.37 | +0.03 | 1 | elbow_flex | 3.12 |
| 10 | -0.75 | +8.20 | -8.43 | +0.10 | +0.35 | +1.20 | 2 | shoulder_lift, elbow_flex | 6.06 |
| 11 | -0.64 | +3.17 | -7.01 | +0.82 | +0.54 | +1.16 | 1 | elbow_flex | 3.14 |
| 12 | -0.89 | -2.18 | -5.47 | +0.48 | +0.23 | +1.59 | 0 | - | 6.51 |
| 13 | +0.26 | +4.02 | -6.48 | +1.02 | +0.53 | +0.70 | 1 | elbow_flex | 2.36 |
| 14 | -1.34 | +5.34 | -7.40 | -0.22 | +0.42 | +0.47 | 2 | shoulder_lift, elbow_flex | 3.64 |
| 15 | -1.84 | +9.07 | -8.80 | -0.23 | +0.45 | +0.36 | 2 | shoulder_lift, elbow_flex | 7.04 |
| 16 | -2.79 | +8.76 | -10.44 | +2.48 | +0.46 | +1.06 | 2 | shoulder_lift, elbow_flex | 8.60 |
| 17 | -1.65 | +5.33 | -4.25 | -0.66 | +0.38 | -0.03 | 1 | shoulder_lift | 2.59 |
| 18 | +0.67 | +3.77 | -6.72 | +1.23 | +0.10 | +0.85 | 1 | elbow_flex | 2.66 |
| 19 | +0.03 | +4.61 | -1.87 | -1.61 | +0.86 | +1.56 | 0 | - | 3.99 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.97 | 0.87 | -2.79 | +0.67 | 0/20 | 0% |
| shoulder_lift | +5.13 | 3.01 | -2.18 | +12.78 | 9/20 | 45% |
| elbow_flex | -7.08 | 2.30 | -12.51 | -1.87 | 15/20 | 75% |
| wrist_flex | +0.66 | 0.93 | -1.61 | +2.48 | 0/20 | 0% |
| wrist_roll | +0.42 | 0.16 | +0.10 | +0.86 | 0/20 | 0% |
| gripper | +0.68 | 0.57 | -0.55 | +1.59 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([2, 8, 12, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 8, '2': 8}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.62, std=2.34, min=1.77, max=10.10
