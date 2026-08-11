# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65_reweight3/smolvla_pick_drop_combined65_reweight3_fresh/checkpoints/010000/pretrained_model`
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
| 0 | -1.38 | +4.55 | -10.79 | +0.50 | +0.23 | -0.59 | 1 | elbow_flex | 6.42 |
| 1 | -1.61 | +10.59 | -8.30 | -1.11 | +0.33 | -0.75 | 2 | shoulder_lift, elbow_flex | 8.01 |
| 2 | +0.49 | +1.60 | -4.85 | -0.63 | +0.26 | -0.50 | 0 | - | 2.49 |
| 3 | +0.87 | +1.50 | -6.94 | +0.86 | +0.20 | -0.42 | 1 | elbow_flex | 3.46 |
| 4 | -1.56 | +2.27 | -6.11 | -0.46 | +0.43 | +0.00 | 1 | elbow_flex | 2.94 |
| 5 | +0.09 | +4.44 | -7.08 | +0.97 | +0.33 | -0.91 | 1 | elbow_flex | 2.74 |
| 6 | -0.07 | +0.74 | -5.88 | +0.56 | +0.09 | -0.62 | 1 | elbow_flex | 3.42 |
| 7 | -0.25 | +5.54 | -7.67 | +0.28 | +0.20 | -1.02 | 2 | shoulder_lift, elbow_flex | 3.59 |
| 8 | -0.65 | +2.01 | -4.06 | +0.15 | +0.41 | -0.56 | 0 | - | 2.18 |
| 9 | -0.45 | +2.19 | -6.41 | -0.19 | +0.20 | -0.70 | 1 | elbow_flex | 2.61 |
| 10 | -0.20 | +5.45 | -7.07 | -0.60 | +0.23 | -0.30 | 2 | shoulder_lift, elbow_flex | 3.07 |
| 11 | -0.13 | +2.40 | -6.66 | +0.23 | +0.37 | -0.33 | 1 | elbow_flex | 2.58 |
| 12 | -0.63 | -2.14 | -3.88 | -0.93 | +0.04 | -0.04 | 0 | - | 6.22 |
| 13 | +1.19 | +2.63 | -5.86 | +0.38 | +0.37 | -0.58 | 1 | elbow_flex | 2.10 |
| 14 | -0.49 | +2.79 | -6.09 | -0.86 | +0.23 | -0.67 | 1 | elbow_flex | 2.28 |
| 15 | -1.23 | +7.05 | -7.80 | -0.93 | +0.30 | -0.90 | 2 | shoulder_lift, elbow_flex | 4.90 |
| 16 | -1.53 | +5.95 | -8.57 | +1.21 | +0.34 | -0.37 | 2 | shoulder_lift, elbow_flex | 4.90 |
| 17 | -0.62 | +2.72 | -4.42 | -1.57 | +0.24 | -0.96 | 0 | - | 2.39 |
| 18 | +0.29 | +2.14 | -5.56 | -0.55 | -0.02 | -0.46 | 0 | - | 2.12 |
| 19 | +0.02 | +2.22 | -1.36 | -2.94 | +0.58 | +0.18 | 0 | - | 4.90 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.39 | 0.77 | -1.61 | +1.19 | 0/20 | 0% |
| shoulder_lift | +3.33 | 2.62 | -2.14 | +10.59 | 5/20 | 25% |
| elbow_flex | -6.27 | 1.96 | -10.79 | -1.36 | 14/20 | 70% |
| wrist_flex | -0.28 | 0.96 | -2.94 | +1.21 | 0/20 | 0% |
| wrist_roll | +0.27 | 0.13 | -0.02 | +0.58 | 0/20 | 0% |
| gripper | -0.52 | 0.32 | -1.02 | +0.18 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 6/20 ([2, 8, 12, 17, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 6, '1': 9, '2': 5}
- L2 error vs nearest-demo immediate GT delta (deg): mean=3.67, std=1.64, min=2.10, max=8.01
