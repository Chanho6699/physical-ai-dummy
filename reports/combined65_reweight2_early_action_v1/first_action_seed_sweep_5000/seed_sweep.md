# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65_reweight2_early/smolvla_pick_drop_combined65_reweight2_early_fresh/checkpoints/005000/pretrained_model`
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
| 0 | -1.37 | +6.42 | -14.72 | +0.60 | +0.44 | +1.34 | 2 | shoulder_lift, elbow_flex | 10.68 |
| 1 | +0.17 | +12.53 | -7.38 | -1.66 | +0.54 | +0.64 | 2 | shoulder_lift, elbow_flex | 9.36 |
| 2 | +2.33 | +0.93 | -2.82 | -0.24 | +0.44 | +0.66 | 0 | - | 4.24 |
| 3 | +2.23 | +3.01 | -8.36 | +1.68 | +0.41 | +0.98 | 1 | elbow_flex | 4.80 |
| 4 | -1.69 | +3.26 | -7.06 | +0.20 | +0.58 | +2.41 | 1 | elbow_flex | 4.21 |
| 5 | +1.50 | +5.95 | -6.43 | +1.52 | +0.49 | -0.40 | 2 | shoulder_lift, elbow_flex | 3.37 |
| 6 | +1.53 | +1.97 | -5.77 | +1.38 | +0.23 | +0.60 | 1 | elbow_flex | 3.01 |
| 7 | +0.69 | +6.81 | -8.00 | +0.67 | +0.35 | -0.59 | 2 | shoulder_lift, elbow_flex | 4.57 |
| 8 | +0.29 | +1.85 | -2.02 | +0.08 | +0.61 | +1.08 | 0 | - | 3.64 |
| 9 | +1.16 | +1.55 | -5.85 | +0.29 | +0.38 | -0.06 | 1 | elbow_flex | 2.86 |
| 10 | +1.39 | +7.16 | -6.99 | -0.62 | +0.43 | +0.94 | 2 | shoulder_lift, elbow_flex | 4.53 |
| 11 | +0.81 | +1.75 | -6.27 | +0.07 | +0.55 | +1.06 | 1 | elbow_flex | 3.15 |
| 12 | +1.39 | -3.66 | -3.51 | -0.39 | +0.14 | +1.85 | 0 | - | 8.03 |
| 13 | +3.06 | +2.20 | -4.40 | +0.76 | +0.52 | +0.73 | 0 | - | 3.59 |
| 14 | +0.79 | +4.43 | -7.13 | -1.07 | +0.41 | +0.79 | 1 | elbow_flex | 3.18 |
| 15 | -0.89 | +8.87 | -7.99 | -1.30 | +0.51 | +0.53 | 2 | shoulder_lift, elbow_flex | 6.39 |
| 16 | -1.64 | +8.98 | -10.57 | +2.51 | +0.49 | +1.46 | 2 | shoulder_lift, elbow_flex | 8.58 |
| 17 | -0.22 | +3.00 | -3.42 | -2.03 | +0.39 | +0.33 | 0 | - | 2.83 |
| 18 | +3.37 | +0.94 | -3.94 | -0.13 | +0.05 | +1.27 | 0 | - | 4.68 |
| 19 | +2.20 | +0.91 | +1.72 | -4.05 | +0.76 | +2.09 | 1 | wrist_flex | 8.82 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | +0.85 | 1.44 | -1.69 | +3.37 | 0/20 | 0% |
| shoulder_lift | +3.94 | 3.61 | -3.66 | +12.53 | 7/20 | 35% |
| elbow_flex | -6.05 | 3.33 | -14.72 | +1.72 | 13/20 | 65% |
| wrist_flex | -0.09 | 1.44 | -4.05 | +2.51 | 1/20 | 5% |
| wrist_roll | +0.44 | 0.16 | +0.05 | +0.76 | 0/20 | 0% |
| gripper | +0.89 | 0.74 | -0.59 | +2.41 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 6/20 ([2, 8, 12, 13, 17, 18])
- clamp-joint-count distribution (count -> #seeds): {'0': 6, '1': 7, '2': 7}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.23, std=2.42, min=2.83, max=10.68
