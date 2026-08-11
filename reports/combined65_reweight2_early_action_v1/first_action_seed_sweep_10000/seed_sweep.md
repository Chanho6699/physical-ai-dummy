# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65_reweight2_early/smolvla_pick_drop_combined65_reweight2_early_fresh/checkpoints/010000/pretrained_model`
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
| 0 | -2.18 | +4.45 | -11.95 | +0.33 | +0.30 | -0.07 | 1 | elbow_flex | 7.74 |
| 1 | -1.31 | +10.76 | -7.84 | -0.78 | +0.38 | -0.30 | 2 | shoulder_lift, elbow_flex | 7.83 |
| 2 | -0.03 | +2.23 | -5.16 | -0.24 | +0.33 | -0.19 | 0 | - | 1.85 |
| 3 | -0.02 | +2.07 | -7.32 | +1.08 | +0.28 | +0.08 | 1 | elbow_flex | 3.41 |
| 4 | -2.18 | +3.12 | -7.00 | +0.17 | +0.44 | +0.73 | 1 | elbow_flex | 3.63 |
| 5 | -0.17 | +4.32 | -6.12 | +1.16 | +0.37 | -0.80 | 1 | elbow_flex | 1.97 |
| 6 | -0.23 | +1.67 | -6.18 | +1.11 | +0.19 | -0.34 | 1 | elbow_flex | 2.88 |
| 7 | -1.05 | +5.36 | -7.49 | +0.47 | +0.27 | -0.86 | 2 | shoulder_lift, elbow_flex | 3.53 |
| 8 | -1.33 | +2.75 | -4.56 | +0.44 | +0.45 | -0.10 | 0 | - | 1.98 |
| 9 | -0.58 | +2.34 | -6.43 | +0.06 | +0.28 | -0.79 | 1 | elbow_flex | 2.57 |
| 10 | -0.38 | +5.40 | -6.88 | -0.12 | +0.29 | +0.12 | 2 | shoulder_lift, elbow_flex | 2.87 |
| 11 | -0.68 | +2.71 | -7.14 | +0.44 | +0.44 | +0.22 | 1 | elbow_flex | 3.01 |
| 12 | -1.10 | +0.27 | -5.49 | -0.02 | +0.12 | +0.39 | 0 | - | 3.99 |
| 13 | +1.21 | +2.38 | -5.27 | +0.66 | +0.42 | -0.33 | 0 | - | 2.04 |
| 14 | -0.97 | +4.08 | -6.89 | -0.44 | +0.30 | -0.19 | 1 | elbow_flex | 2.67 |
| 15 | -1.68 | +7.40 | -8.14 | -0.39 | +0.37 | -0.29 | 2 | shoulder_lift, elbow_flex | 5.38 |
| 16 | -2.16 | +6.41 | -9.10 | +1.84 | +0.37 | +0.16 | 2 | shoulder_lift, elbow_flex | 5.92 |
| 17 | -1.27 | +3.32 | -5.22 | -0.99 | +0.29 | -0.61 | 0 | - | 2.10 |
| 18 | +0.03 | +2.69 | -5.71 | +0.33 | +0.06 | +0.11 | 0 | - | 1.68 |
| 19 | -0.37 | +2.38 | -2.32 | -2.00 | +0.60 | +0.70 | 0 | - | 3.77 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.82 | 0.85 | -2.18 | +1.21 | 0/20 | 0% |
| shoulder_lift | +3.80 | 2.31 | +0.27 | +10.76 | 5/20 | 25% |
| elbow_flex | -6.61 | 1.89 | -11.95 | -2.32 | 13/20 | 65% |
| wrist_flex | +0.16 | 0.83 | -2.00 | +1.84 | 0/20 | 0% |
| wrist_roll | +0.33 | 0.12 | +0.06 | +0.60 | 0/20 | 0% |
| gripper | -0.12 | 0.44 | -0.86 | +0.73 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 7/20 ([2, 8, 12, 13, 17, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 7, '1': 8, '2': 5}
- L2 error vs nearest-demo immediate GT delta (deg): mean=3.54, std=1.79, min=1.68, max=7.83
