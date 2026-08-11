# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v4/smolvla_pick_drop_v4_fresh/checkpoints/005000/pretrained_model`
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
| shoulder_pan | -0.7033 |
| shoulder_lift | +3.5165 |
| elbow_flex | -4.4396 |
| wrist_flex | -1.0989 |
| wrist_roll | +0.0879 |
| gripper | -0.3945 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -2.82 | +5.36 | -11.91 | -1.63 | -0.01 | -0.25 | 2 | shoulder_lift, elbow_flex | 8.00 |
| 1 | -2.17 | +16.44 | -10.10 | -2.62 | +0.40 | -0.21 | 2 | shoulder_lift, elbow_flex | 14.27 |
| 2 | -0.39 | +2.52 | -2.96 | -2.43 | +0.38 | +0.28 | 0 | - | 2.36 |
| 3 | -0.25 | +4.34 | -8.13 | -0.77 | +0.02 | +0.35 | 1 | elbow_flex | 3.90 |
| 4 | -2.98 | +4.73 | -6.85 | -2.50 | +0.76 | +1.57 | 1 | elbow_flex | 4.33 |
| 5 | -0.79 | +9.16 | -7.49 | -0.67 | +0.47 | -0.18 | 2 | shoulder_lift, elbow_flex | 6.44 |
| 6 | -0.71 | +4.16 | -6.17 | -0.86 | -0.36 | -0.21 | 1 | elbow_flex | 1.92 |
| 7 | -1.71 | +8.26 | -7.70 | -1.04 | -0.07 | -0.50 | 2 | shoulder_lift, elbow_flex | 5.84 |
| 8 | -2.42 | +4.14 | -2.88 | -1.64 | +0.83 | +0.41 | 0 | - | 2.69 |
| 9 | -0.80 | +2.92 | -5.69 | -2.04 | -0.07 | -0.54 | 0 | - | 1.69 |
| 10 | -1.74 | +10.71 | -8.87 | -2.25 | -0.00 | +0.73 | 2 | shoulder_lift, elbow_flex | 8.66 |
| 11 | -0.71 | +4.30 | -6.72 | -1.95 | +0.60 | +0.63 | 1 | elbow_flex | 2.80 |
| 12 | -1.04 | -4.07 | -2.73 | -3.76 | -0.83 | +2.03 | 0 | - | 8.63 |
| 13 | +0.73 | +5.56 | -5.22 | -1.50 | +0.73 | +0.10 | 1 | shoulder_lift | 2.77 |
| 14 | -2.03 | +5.90 | -6.97 | -2.86 | -0.08 | +0.48 | 2 | shoulder_lift, elbow_flex | 4.21 |
| 15 | -2.91 | +11.88 | -10.06 | -2.49 | +0.29 | -0.18 | 2 | shoulder_lift, elbow_flex | 10.41 |
| 16 | -2.61 | +9.36 | -9.48 | -0.42 | +0.30 | +1.33 | 2 | shoulder_lift, elbow_flex | 8.17 |
| 17 | -2.84 | +6.32 | -3.37 | -3.63 | +0.03 | -0.21 | 1 | shoulder_lift | 4.47 |
| 18 | +0.46 | +4.62 | -5.85 | -2.37 | -1.07 | +0.99 | 1 | elbow_flex | 3.07 |
| 19 | +0.55 | +4.84 | +0.00 | -6.23 | +1.40 | +1.92 | 2 | wrist_flex, wrist_roll | 7.51 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.36 | 1.19 | -2.98 | +0.73 | 0/20 | 0% |
| shoulder_lift | +6.07 | 4.08 | -4.07 | +16.44 | 10/20 | 50% |
| elbow_flex | -6.46 | 2.91 | -11.91 | +0.00 | 13/20 | 65% |
| wrist_flex | -2.18 | 1.29 | -6.23 | -0.42 | 1/20 | 5% |
| wrist_roll | +0.18 | 0.55 | -1.07 | +1.40 | 1/20 | 5% |
| gripper | +0.43 | 0.77 | -0.54 | +2.03 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([2, 8, 9, 12])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 7, '2': 9}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.61, std=3.25, min=1.69, max=14.27
