# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/reweight_ablation/combined65_reweight_new2_old1_v1/checkpoints/005000/pretrained_model`
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
| 0 | -0.28 | +6.67 | -13.81 | +0.42 | +0.45 | +1.51 | 2 | shoulder_lift, elbow_flex | 9.81 |
| 1 | +0.69 | +14.14 | -8.34 | -1.56 | +0.51 | +0.89 | 2 | shoulder_lift, elbow_flex | 11.16 |
| 2 | +2.78 | +2.10 | -3.08 | -0.27 | +0.43 | +0.95 | 0 | - | 3.79 |
| 3 | +2.84 | +3.73 | -8.27 | +1.65 | +0.42 | +1.13 | 1 | elbow_flex | 4.99 |
| 4 | -1.34 | +4.81 | -7.43 | +0.15 | +0.60 | +2.95 | 1 | elbow_flex | 4.72 |
| 5 | +1.92 | +7.54 | -6.68 | +1.63 | +0.49 | -0.11 | 2 | shoulder_lift, elbow_flex | 4.81 |
| 6 | +1.80 | +2.78 | -6.21 | +1.32 | +0.23 | +0.92 | 1 | elbow_flex | 3.04 |
| 7 | +1.28 | +8.17 | -8.28 | +0.87 | +0.33 | -0.32 | 2 | shoulder_lift, elbow_flex | 5.81 |
| 8 | +0.74 | +2.91 | -1.87 | +0.01 | +0.61 | +1.45 | 0 | - | 3.53 |
| 9 | +1.33 | +2.99 | -6.13 | +0.26 | +0.40 | +0.16 | 1 | elbow_flex | 2.20 |
| 10 | +1.64 | +9.06 | -7.51 | -0.76 | +0.43 | +1.45 | 2 | shoulder_lift, elbow_flex | 6.47 |
| 11 | +1.06 | +2.99 | -6.35 | +0.11 | +0.52 | +1.61 | 1 | elbow_flex | 2.95 |
| 12 | +1.73 | -2.74 | -3.59 | -0.56 | +0.14 | +2.44 | 0 | - | 7.44 |
| 13 | +3.49 | +4.24 | -4.68 | +0.72 | +0.52 | +1.14 | 0 | - | 3.72 |
| 14 | +1.08 | +5.32 | -7.09 | -1.31 | +0.40 | +1.05 | 2 | shoulder_lift, elbow_flex | 3.68 |
| 15 | -0.82 | +10.39 | -8.82 | -1.33 | +0.47 | +0.95 | 2 | shoulder_lift, elbow_flex | 8.09 |
| 16 | -1.28 | +10.98 | -11.19 | +2.60 | +0.50 | +1.87 | 2 | shoulder_lift, elbow_flex | 10.33 |
| 17 | +0.16 | +4.38 | -3.19 | -2.27 | +0.39 | +0.45 | 0 | - | 3.06 |
| 18 | +3.86 | +1.80 | -4.37 | -0.29 | +0.07 | +1.54 | 0 | - | 4.65 |
| 19 | +2.41 | +2.00 | +1.16 | -4.44 | +0.76 | +2.40 | 1 | wrist_flex | 8.47 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | +1.25 | 1.43 | -1.34 | +3.86 | 0/20 | 0% |
| shoulder_lift | +5.21 | 3.81 | -2.74 | +14.14 | 8/20 | 40% |
| elbow_flex | -6.29 | 3.25 | -13.81 | +1.16 | 13/20 | 65% |
| wrist_flex | -0.15 | 1.53 | -4.44 | +2.60 | 1/20 | 5% |
| wrist_roll | +0.43 | 0.15 | +0.07 | +0.76 | 0/20 | 0% |
| gripper | +1.22 | 0.81 | -0.32 | +2.95 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 6/20 ([2, 8, 12, 13, 17, 18])
- clamp-joint-count distribution (count -> #seeds): {'0': 6, '1': 6, '2': 8}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.64, std=2.63, min=2.20, max=11.16
