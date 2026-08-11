# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v3_v4_reweight2/smolvla_pick_drop_v3_v4_reweight2_fresh/checkpoints/002500/pretrained_model`
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
| 0 | -6.07 | +2.42 | -15.88 | +0.73 | +0.33 | +2.95 | 2 | shoulder_pan, elbow_flex | 13.25 |
| 1 | -6.00 | +19.47 | -6.56 | -1.35 | +1.46 | +3.46 | 4 | shoulder_pan, shoulder_lift, elbow_flex, wrist_roll | 17.44 |
| 2 | -1.26 | -3.25 | +1.41 | -0.17 | +0.61 | +3.08 | 0 | - | 9.68 |
| 3 | +0.68 | -1.23 | -6.61 | +2.15 | +0.18 | +3.06 | 1 | elbow_flex | 7.18 |
| 4 | -8.58 | +2.41 | -4.96 | -1.07 | +1.29 | +5.37 | 2 | shoulder_pan, wrist_roll | 9.91 |
| 5 | -0.76 | +4.88 | -0.55 | +2.94 | +0.95 | +0.82 | 0 | - | 5.96 |
| 6 | -0.38 | -4.02 | -0.33 | +1.92 | -0.42 | +2.29 | 0 | - | 9.51 |
| 7 | -3.22 | +5.45 | -4.25 | +1.56 | -0.01 | +0.83 | 1 | shoulder_lift | 4.33 |
| 8 | -6.58 | +0.52 | +3.37 | +0.75 | +1.76 | +2.67 | 2 | shoulder_pan, wrist_roll | 10.96 |
| 9 | -1.76 | -3.04 | -1.37 | +0.27 | +0.43 | +1.05 | 0 | - | 7.59 |
| 10 | -3.43 | +8.80 | -5.10 | -0.75 | +0.44 | +3.90 | 1 | shoulder_lift | 7.39 |
| 11 | -1.46 | -1.73 | -3.54 | +0.40 | +1.10 | +3.88 | 0 | - | 7.10 |
| 12 | -2.44 | -13.74 | +2.07 | -1.79 | -0.73 | +5.29 | 1 | shoulder_lift | 19.41 |
| 13 | +1.05 | +0.35 | +0.95 | +1.52 | +1.04 | +2.37 | 0 | - | 7.58 |
| 14 | -5.14 | +1.89 | -4.36 | -2.33 | +0.48 | +3.39 | 1 | shoulder_pan | 6.19 |
| 15 | -6.88 | +10.39 | -5.92 | -1.88 | +1.15 | +2.77 | 4 | shoulder_pan, shoulder_lift, elbow_flex, wrist_roll | 9.97 |
| 16 | -5.87 | +10.79 | -10.89 | +3.26 | +0.91 | +3.83 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 12.60 |
| 17 | -8.19 | +2.84 | +4.90 | -3.40 | +0.48 | +0.62 | 1 | shoulder_pan | 12.25 |
| 18 | +0.81 | -3.55 | -0.76 | -0.81 | -0.96 | +4.38 | 0 | - | 9.47 |
| 19 | -1.41 | +1.99 | +6.29 | -5.93 | +2.45 | +5.72 | 3 | elbow_flex, wrist_flex, wrist_roll | 13.57 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -3.34 | 3.01 | -8.58 | +1.05 | 8/20 | 40% |
| shoulder_lift | +2.08 | 6.77 | -13.74 | +19.47 | 6/20 | 30% |
| elbow_flex | -2.60 | 5.21 | -15.88 | +6.29 | 6/20 | 30% |
| wrist_flex | -0.20 | 2.19 | -5.93 | +3.26 | 1/20 | 5% |
| wrist_roll | +0.65 | 0.80 | -0.96 | +2.45 | 5/20 | 25% |
| gripper | +3.09 | 1.46 | +0.62 | +5.72 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 7/20 ([2, 5, 6, 9, 11, 13, 18])
- clamp-joint-count distribution (count -> #seeds): {'0': 7, '1': 6, '2': 3, '3': 2, '4': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=10.07, std=3.73, min=4.33, max=19.41
