# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v3_v4_combined69/smolvla_pick_drop_v3_v4_combined69_uniform_fresh/checkpoints/002500/pretrained_model`
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
| 0 | -4.06 | +8.01 | -18.43 | +0.71 | +0.18 | +3.42 | 2 | shoulder_lift, elbow_flex | 15.65 |
| 1 | -2.85 | +19.94 | -8.60 | -1.53 | +0.85 | +3.68 | 2 | shoulder_lift, elbow_flex | 17.58 |
| 2 | -0.54 | -0.35 | -0.41 | -0.40 | +0.73 | +4.73 | 0 | - | 7.65 |
| 3 | -0.88 | +2.39 | -9.64 | +2.48 | +0.18 | +4.57 | 1 | elbow_flex | 8.11 |
| 4 | -6.49 | +3.79 | -6.37 | -0.54 | +1.76 | +7.07 | 3 | shoulder_pan, elbow_flex, wrist_roll | 9.80 |
| 5 | +0.14 | +8.35 | -5.03 | +2.19 | +1.19 | +2.08 | 2 | shoulder_lift, wrist_roll | 6.53 |
| 6 | -0.76 | +1.78 | -5.47 | +1.16 | -0.61 | +3.60 | 0 | - | 5.07 |
| 7 | -2.40 | +10.47 | -8.45 | +1.03 | -0.26 | +2.15 | 2 | shoulder_lift, elbow_flex | 8.86 |
| 8 | -3.39 | +3.17 | -0.20 | +0.27 | +1.82 | +4.38 | 1 | wrist_roll | 7.28 |
| 9 | -1.00 | +2.61 | -6.07 | +0.30 | +0.35 | +2.65 | 1 | elbow_flex | 3.86 |
| 10 | -1.59 | +11.70 | -7.53 | -1.38 | +0.31 | +4.97 | 2 | shoulder_lift, elbow_flex | 10.31 |
| 11 | -0.17 | +0.96 | -4.90 | +0.25 | +1.29 | +4.50 | 1 | wrist_roll | 5.85 |
| 12 | -3.87 | -11.34 | -0.17 | -2.28 | -1.39 | +7.87 | 2 | shoulder_lift, wrist_roll | 17.91 |
| 13 | +3.67 | +3.74 | -3.06 | +1.03 | +1.42 | +3.92 | 1 | wrist_roll | 6.79 |
| 14 | -4.46 | +5.10 | -7.23 | -2.14 | +0.47 | +4.43 | 1 | elbow_flex | 7.00 |
| 15 | -4.01 | +12.35 | -8.56 | -2.37 | +0.92 | +3.08 | 2 | shoulder_lift, elbow_flex | 10.97 |
| 16 | -5.82 | +14.47 | -12.02 | +2.90 | +1.14 | +5.83 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 16.11 |
| 17 | -4.01 | +4.91 | +0.53 | -3.43 | +0.78 | +2.79 | 0 | - | 7.32 |
| 18 | -1.35 | +3.11 | -4.21 | -0.53 | -2.26 | +5.86 | 1 | wrist_roll | 6.75 |
| 19 | +0.50 | +2.31 | +5.50 | -6.53 | +2.73 | +8.19 | 2 | wrist_flex, wrist_roll | 14.56 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -2.17 | 2.36 | -6.49 | +3.67 | 2/20 | 10% |
| shoulder_lift | +5.37 | 6.37 | -11.34 | +19.94 | 8/20 | 40% |
| elbow_flex | -5.52 | 5.05 | -18.43 | +5.50 | 10/20 | 50% |
| wrist_flex | -0.44 | 2.17 | -6.53 | +2.90 | 1/20 | 5% |
| wrist_roll | +0.58 | 1.10 | -2.26 | +2.73 | 8/20 | 40% |
| gripper | +4.49 | 1.71 | +2.08 | +8.19 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 3/20 ([2, 6, 17])
- clamp-joint-count distribution (count -> #seeds): {'0': 3, '1': 7, '2': 8, '3': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=9.70, std=4.22, min=3.86, max=17.91
