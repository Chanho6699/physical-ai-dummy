# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65/smolvla_pick_drop_combined65_fresh/checkpoints/005000/pretrained_model`
Task: `Pick up the cube and drop it into the bin.`
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
| 0 | -4.19 | +6.19 | -16.21 | +1.18 | +0.35 | +0.52 | 2 | shoulder_lift, elbow_flex | 12.68 |
| 1 | -2.94 | +15.43 | -11.71 | -0.47 | +0.53 | +0.03 | 2 | shoulder_lift, elbow_flex | 13.95 |
| 2 | -1.35 | +1.99 | -5.42 | +0.42 | +0.44 | +0.13 | 0 | - | 2.65 |
| 3 | -0.58 | +3.80 | -11.13 | +2.80 | +0.33 | +0.50 | 1 | elbow_flex | 7.11 |
| 4 | -5.25 | +4.71 | -8.92 | +0.36 | +0.70 | +2.00 | 2 | shoulder_pan, elbow_flex | 7.40 |
| 5 | -0.98 | +7.36 | -7.25 | +2.54 | +0.52 | -1.02 | 2 | shoulder_lift, elbow_flex | 5.17 |
| 6 | -0.88 | +3.05 | -8.10 | +2.02 | +0.14 | -0.01 | 1 | elbow_flex | 4.17 |
| 7 | -1.96 | +7.73 | -8.77 | +1.50 | +0.27 | -1.12 | 2 | shoulder_lift, elbow_flex | 6.26 |
| 8 | -3.98 | +2.88 | -4.85 | +0.91 | +0.65 | +0.42 | 0 | - | 4.45 |
| 9 | -2.06 | +2.12 | -7.20 | +0.50 | +0.31 | -0.91 | 1 | elbow_flex | 3.91 |
| 10 | -2.29 | +10.20 | -9.81 | -0.04 | +0.36 | +0.72 | 2 | shoulder_lift, elbow_flex | 8.63 |
| 11 | -2.12 | +3.45 | -8.45 | +1.15 | +0.60 | +1.11 | 1 | elbow_flex | 4.85 |
| 12 | -2.51 | -4.82 | -5.30 | -0.51 | +0.07 | +1.78 | 0 | - | 9.38 |
| 13 | -0.50 | +5.06 | -7.02 | +1.30 | +0.60 | +0.18 | 1 | elbow_flex | 3.09 |
| 14 | -3.50 | +5.21 | -9.05 | -0.80 | +0.38 | +0.59 | 2 | shoulder_lift, elbow_flex | 6.09 |
| 15 | -4.07 | +10.46 | -11.39 | -0.42 | +0.47 | +0.35 | 2 | shoulder_lift, elbow_flex | 10.42 |
| 16 | -4.85 | +9.82 | -12.25 | +3.25 | +0.53 | +1.00 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 11.42 |
| 17 | -4.82 | +5.54 | -4.32 | -1.75 | +0.41 | -0.98 | 2 | shoulder_pan, shoulder_lift | 5.70 |
| 18 | -0.48 | +2.99 | -6.55 | +0.26 | -0.06 | +0.32 | 1 | elbow_flex | 2.32 |
| 19 | -2.60 | +2.07 | -0.47 | -3.83 | +0.93 | +1.25 | 0 | - | 6.93 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.60 | 1.51 | -5.25 | -0.48 | 3/20 | 15% |
| shoulder_lift | +5.26 | 4.13 | -4.82 | +15.43 | 9/20 | 45% |
| elbow_flex | -8.21 | 3.35 | -16.21 | -0.47 | 15/20 | 75% |
| wrist_flex | +0.52 | 1.59 | -3.83 | +3.25 | 0/20 | 0% |
| wrist_roll | +0.43 | 0.22 | -0.06 | +0.93 | 0/20 | 0% |
| gripper | +0.34 | 0.85 | -1.12 | +2.00 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([2, 8, 12, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 6, '2': 9, '3': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.83, std=3.25, min=2.32, max=13.95
