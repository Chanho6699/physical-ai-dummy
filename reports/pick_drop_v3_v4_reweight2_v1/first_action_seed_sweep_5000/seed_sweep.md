# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v3_v4_reweight2/smolvla_pick_drop_v3_v4_reweight2_fresh/checkpoints/005000/pretrained_model`
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
| 0 | -0.65 | +1.46 | -12.17 | -0.77 | +0.73 | +2.22 | 1 | elbow_flex | 8.44 |
| 1 | +0.08 | +10.27 | -9.44 | -1.99 | +0.89 | +1.57 | 2 | shoulder_lift, elbow_flex | 8.75 |
| 2 | +1.60 | -1.73 | -3.74 | -0.97 | +0.98 | +1.66 | 0 | - | 6.20 |
| 3 | +1.79 | +0.30 | -7.96 | +0.72 | +0.77 | +1.96 | 1 | elbow_flex | 6.19 |
| 4 | -2.36 | +0.12 | -5.76 | -1.19 | +1.26 | +2.82 | 2 | elbow_flex, wrist_roll | 5.27 |
| 5 | +1.41 | +3.46 | -6.98 | +0.85 | +0.96 | +1.27 | 1 | elbow_flex | 4.27 |
| 6 | +1.05 | -0.63 | -6.47 | +0.57 | +0.39 | +1.45 | 1 | elbow_flex | 5.54 |
| 7 | +0.20 | +4.27 | -7.86 | -0.19 | +0.69 | +0.93 | 1 | elbow_flex | 4.01 |
| 8 | -0.13 | +0.82 | -3.50 | -0.32 | +1.50 | +1.85 | 1 | wrist_roll | 4.02 |
| 9 | +1.23 | -1.03 | -5.30 | -1.08 | +0.65 | +1.23 | 0 | - | 5.31 |
| 10 | +0.01 | +4.84 | -7.90 | -1.42 | +0.77 | +2.09 | 1 | elbow_flex | 4.58 |
| 11 | +0.76 | -0.96 | -5.88 | -0.40 | +1.25 | +2.12 | 2 | elbow_flex, wrist_roll | 5.70 |
| 12 | +0.98 | -6.54 | -1.83 | -1.97 | +0.37 | +2.77 | 1 | shoulder_lift | 11.03 |
| 13 | +2.78 | +0.70 | -5.33 | -0.02 | +1.10 | +1.70 | 0 | - | 5.23 |
| 14 | +0.13 | +0.57 | -5.62 | -2.20 | +0.80 | +1.97 | 0 | - | 4.25 |
| 15 | -0.88 | +5.95 | -9.54 | -1.58 | +1.04 | +1.56 | 2 | shoulder_lift, elbow_flex | 6.08 |
| 16 | -1.89 | +6.61 | -11.19 | +1.19 | +0.89 | +2.29 | 2 | shoulder_lift, elbow_flex | 8.35 |
| 17 | -0.99 | +0.86 | -2.49 | -2.72 | +0.81 | +1.04 | 0 | - | 4.02 |
| 18 | +1.58 | -0.67 | -5.35 | -1.40 | +0.05 | +1.88 | 0 | - | 5.37 |
| 19 | +2.66 | -1.69 | +0.87 | -5.11 | +1.88 | +2.27 | 2 | wrist_flex, wrist_roll | 9.64 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | +0.47 | 1.35 | -2.36 | +2.78 | 0/20 | 0% |
| shoulder_lift | +1.35 | 3.59 | -6.54 | +10.27 | 4/20 | 20% |
| elbow_flex | -6.17 | 3.09 | -12.17 | +0.87 | 11/20 | 55% |
| wrist_flex | -1.00 | 1.41 | -5.11 | +1.19 | 1/20 | 5% |
| wrist_roll | +0.89 | 0.39 | +0.05 | +1.88 | 4/20 | 20% |
| gripper | +1.83 | 0.50 | +0.93 | +2.82 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 6/20 ([2, 9, 13, 14, 17, 18])
- clamp-joint-count distribution (count -> #seeds): {'0': 6, '1': 8, '2': 6}
- L2 error vs nearest-demo immediate GT delta (deg): mean=6.11, std=2.00, min=4.01, max=11.03
