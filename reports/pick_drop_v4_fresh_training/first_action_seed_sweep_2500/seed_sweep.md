# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v4/smolvla_pick_drop_v4_fresh/checkpoints/002500/pretrained_model`
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
| 0 | -3.05 | +6.41 | -23.37 | +1.39 | +0.66 | +0.67 | 2 | shoulder_lift, elbow_flex | 19.49 |
| 1 | -2.24 | +20.56 | -15.13 | -0.89 | +1.72 | +1.41 | 3 | shoulder_lift, elbow_flex, wrist_roll | 20.32 |
| 2 | +0.68 | -1.34 | -5.67 | -0.75 | +1.29 | +0.55 | 1 | wrist_roll | 5.43 |
| 3 | +0.87 | +1.49 | -14.55 | +2.12 | +0.81 | +0.69 | 1 | elbow_flex | 10.99 |
| 4 | -4.45 | +4.53 | -13.29 | -0.85 | +2.62 | +2.79 | 2 | elbow_flex, wrist_roll | 10.48 |
| 5 | +0.05 | +7.69 | -9.31 | +2.18 | +1.82 | -1.10 | 3 | shoulder_lift, elbow_flex, wrist_roll | 7.48 |
| 6 | +0.58 | +0.31 | -9.22 | +1.31 | -0.15 | -0.28 | 1 | elbow_flex | 6.37 |
| 7 | -0.37 | +6.77 | -11.16 | +1.09 | +0.49 | -1.12 | 2 | shoulder_lift, elbow_flex | 7.83 |
| 8 | -2.50 | +3.04 | -7.09 | +0.92 | +2.65 | +1.14 | 2 | elbow_flex, wrist_roll | 4.85 |
| 9 | +0.22 | -0.18 | -9.30 | -0.70 | +0.78 | -1.02 | 1 | elbow_flex | 6.26 |
| 10 | -0.27 | +10.03 | -13.53 | -0.77 | +0.75 | +1.15 | 2 | shoulder_lift, elbow_flex | 11.32 |
| 11 | +0.49 | +0.23 | -10.75 | -0.17 | +2.24 | +0.42 | 2 | elbow_flex, wrist_roll | 7.63 |
| 12 | -0.26 | -12.72 | -5.16 | -2.74 | -0.59 | +2.51 | 1 | shoulder_lift | 16.61 |
| 13 | +2.65 | +4.31 | -8.15 | +0.67 | +1.86 | +0.98 | 2 | elbow_flex, wrist_roll | 5.82 |
| 14 | -1.96 | +3.46 | -12.80 | -2.22 | +0.94 | +1.07 | 1 | elbow_flex | 8.70 |
| 15 | -3.39 | +12.78 | -14.87 | -1.89 | +1.49 | +0.80 | 3 | shoulder_lift, elbow_flex, wrist_roll | 14.35 |
| 16 | -3.73 | +10.98 | -15.36 | +2.88 | +1.68 | +2.12 | 3 | shoulder_lift, elbow_flex, wrist_roll | 14.45 |
| 17 | -3.20 | +4.95 | -4.96 | -3.36 | +1.00 | -1.17 | 0 | - | 3.88 |
| 18 | +1.95 | -0.68 | -8.17 | -0.69 | -1.73 | +1.09 | 2 | elbow_flex, wrist_roll | 6.65 |
| 19 | +1.84 | +0.72 | +0.18 | -7.43 | +3.52 | +2.07 | 2 | wrist_flex, wrist_roll | 9.68 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.81 | 2.04 | -4.45 | +2.65 | 0/20 | 0% |
| shoulder_lift | +4.17 | 6.55 | -12.72 | +20.56 | 8/20 | 40% |
| elbow_flex | -10.58 | 4.96 | -23.37 | +0.18 | 16/20 | 80% |
| wrist_flex | -0.50 | 2.29 | -7.43 | +2.88 | 1/20 | 5% |
| wrist_roll | +1.19 | 1.16 | -1.73 | +3.52 | 11/20 | 55% |
| gripper | +0.74 | 1.16 | -1.17 | +2.79 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([17])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 6, '2': 9, '3': 4}
- L2 error vs nearest-demo immediate GT delta (deg): mean=9.93, std=4.70, min=3.88, max=20.32
