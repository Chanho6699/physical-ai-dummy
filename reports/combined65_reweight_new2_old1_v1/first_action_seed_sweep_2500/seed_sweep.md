# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/reweight_ablation/combined65_reweight_new2_old1_v1/checkpoints/002500/pretrained_model`
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
| 0 | -6.25 | +1.82 | -22.81 | +0.74 | +0.50 | -1.11 | 2 | shoulder_pan, elbow_flex | 19.44 |
| 1 | -5.19 | +16.35 | -14.74 | +0.05 | +0.75 | -1.06 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 16.98 |
| 2 | -1.51 | -6.04 | -3.03 | -1.84 | +0.50 | -0.69 | 1 | shoulder_lift | 10.40 |
| 3 | -1.25 | -1.76 | -13.20 | +1.75 | +0.44 | -0.59 | 1 | elbow_flex | 10.49 |
| 4 | -8.81 | -0.67 | -9.86 | -1.37 | +0.81 | +0.90 | 2 | shoulder_pan, elbow_flex | 11.56 |
| 5 | -1.20 | +5.36 | -8.35 | +2.64 | +0.57 | -3.16 | 2 | shoulder_lift, elbow_flex | 5.69 |
| 6 | -1.49 | -1.71 | -8.32 | +1.06 | +0.14 | -1.17 | 1 | elbow_flex | 7.00 |
| 7 | -3.31 | +5.42 | -10.18 | +0.88 | +0.34 | -2.74 | 2 | shoulder_lift, elbow_flex | 7.21 |
| 8 | -6.22 | -1.00 | -3.73 | +0.45 | +0.80 | -0.45 | 1 | shoulder_pan | 8.14 |
| 9 | -2.08 | -3.92 | -8.33 | -0.75 | +0.36 | -3.07 | 1 | elbow_flex | 9.39 |
| 10 | -2.48 | +5.37 | -9.67 | -1.29 | +0.50 | +0.57 | 2 | shoulder_lift, elbow_flex | 6.19 |
| 11 | -2.72 | -4.02 | -9.46 | -0.22 | +0.63 | -0.17 | 1 | elbow_flex | 9.74 |
| 12 | -5.06 | -15.49 | -4.80 | -3.54 | -0.04 | +1.71 | 2 | shoulder_pan, shoulder_lift | 20.51 |
| 13 | +1.81 | +1.67 | -7.17 | +0.89 | +0.60 | -1.13 | 1 | elbow_flex | 3.96 |
| 14 | -5.37 | -1.28 | -10.08 | -3.09 | +0.52 | -0.81 | 2 | shoulder_pan, elbow_flex | 9.94 |
| 15 | -7.26 | +8.33 | -13.31 | -1.21 | +0.63 | -0.92 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 12.40 |
| 16 | -6.61 | +10.40 | -15.72 | +3.53 | +0.61 | -0.26 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 14.95 |
| 17 | -6.73 | +0.49 | -1.86 | -4.52 | +0.44 | -2.76 | 2 | shoulder_pan, wrist_flex | 9.77 |
| 18 | -0.36 | -3.52 | -7.32 | -1.19 | -0.15 | +0.22 | 1 | elbow_flex | 8.03 |
| 19 | -2.68 | -3.68 | +3.35 | -7.97 | +1.07 | +0.52 | 1 | wrist_flex | 14.06 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -3.74 | 2.70 | -8.81 | +1.81 | 9/20 | 45% |
| shoulder_lift | +0.61 | 6.61 | -15.49 | +16.35 | 8/20 | 40% |
| elbow_flex | -8.93 | 5.48 | -22.81 | +3.35 | 15/20 | 75% |
| wrist_flex | -0.75 | 2.56 | -7.97 | +3.53 | 2/20 | 10% |
| wrist_roll | +0.50 | 0.28 | -0.15 | +1.07 | 0/20 | 0% |
| gripper | -0.81 | 1.29 | -3.16 | +1.71 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 9, '2': 8, '3': 3}
- L2 error vs nearest-demo immediate GT delta (deg): mean=10.79, std=4.36, min=3.96, max=20.51
