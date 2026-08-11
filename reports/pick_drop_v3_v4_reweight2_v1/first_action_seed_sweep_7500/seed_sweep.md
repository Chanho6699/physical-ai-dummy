# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v3_v4_reweight2/smolvla_pick_drop_v3_v4_reweight2_fresh/checkpoints/007500/pretrained_model`
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
| 0 | +0.43 | +1.32 | -9.68 | -0.17 | +0.04 | +0.59 | 1 | elbow_flex | 5.95 |
| 1 | +0.30 | +9.48 | -7.46 | -0.87 | +0.31 | +0.54 | 2 | shoulder_lift, elbow_flex | 6.83 |
| 2 | +2.54 | -0.76 | -2.93 | -0.56 | +0.26 | +0.71 | 0 | - | 5.71 |
| 3 | +2.93 | -0.14 | -6.05 | +0.69 | +0.07 | +0.79 | 1 | elbow_flex | 5.81 |
| 4 | -0.88 | +0.80 | -5.04 | -0.82 | +0.53 | +1.70 | 0 | - | 3.53 |
| 5 | +2.00 | +2.64 | -4.63 | +0.84 | +0.33 | +0.08 | 0 | - | 3.48 |
| 6 | +2.30 | -0.92 | -4.50 | +0.49 | -0.15 | +0.36 | 0 | - | 5.65 |
| 7 | +1.25 | +3.74 | -5.67 | +0.05 | +0.10 | -0.22 | 0 | - | 2.60 |
| 8 | +0.97 | +1.18 | -2.38 | +0.30 | +0.72 | +0.72 | 0 | - | 4.02 |
| 9 | +1.91 | -0.43 | -4.47 | -0.76 | +0.03 | -0.16 | 0 | - | 4.75 |
| 10 | +1.24 | +4.37 | -5.39 | -0.66 | +0.12 | +0.88 | 0 | - | 2.69 |
| 11 | +1.93 | -0.52 | -4.66 | -0.08 | +0.46 | +0.73 | 0 | - | 5.08 |
| 12 | +1.54 | -5.71 | -2.22 | -1.14 | -0.35 | +1.68 | 1 | shoulder_lift | 9.98 |
| 13 | +3.59 | +0.14 | -3.18 | +0.16 | +0.43 | +0.49 | 0 | - | 5.82 |
| 14 | +0.98 | +0.97 | -4.69 | -1.21 | +0.15 | +0.51 | 0 | - | 3.19 |
| 15 | -0.02 | +5.11 | -6.77 | -1.06 | +0.39 | +0.27 | 1 | elbow_flex | 2.99 |
| 16 | -0.93 | +5.26 | -8.01 | +1.11 | +0.30 | +1.37 | 2 | shoulder_lift, elbow_flex | 4.88 |
| 17 | +0.54 | +1.87 | -2.22 | -1.67 | +0.20 | -0.09 | 0 | - | 3.10 |
| 18 | +2.33 | -0.07 | -3.99 | -0.73 | -0.45 | +1.12 | 0 | - | 4.99 |
| 19 | +2.22 | +0.82 | +0.25 | -3.78 | +1.13 | +1.45 | 0 | - | 7.04 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | +1.36 | 1.17 | -0.93 | +3.59 | 0/20 | 0% |
| shoulder_lift | +1.46 | 3.04 | -5.71 | +9.48 | 3/20 | 15% |
| elbow_flex | -4.68 | 2.23 | -9.68 | +0.25 | 5/20 | 25% |
| wrist_flex | -0.49 | 1.05 | -3.78 | +1.11 | 0/20 | 0% |
| wrist_roll | +0.23 | 0.34 | -0.45 | +1.13 | 0/20 | 0% |
| gripper | +0.67 | 0.56 | -0.22 | +1.70 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 14/20 ([2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 14, '1': 4, '2': 2}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.90, std=1.77, min=2.60, max=9.98
