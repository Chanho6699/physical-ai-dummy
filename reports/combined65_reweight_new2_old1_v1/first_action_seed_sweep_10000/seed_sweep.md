# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/reweight_ablation/combined65_reweight_new2_old1_v1/checkpoints/010000/pretrained_model`
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
| 0 | -1.58 | +4.10 | -10.50 | +0.03 | +0.30 | -0.17 | 1 | elbow_flex | 6.17 |
| 1 | -1.12 | +11.38 | -7.91 | -0.95 | +0.36 | -0.49 | 2 | shoulder_lift, elbow_flex | 8.40 |
| 2 | +0.44 | +2.47 | -4.87 | -0.47 | +0.33 | -0.41 | 0 | - | 1.66 |
| 3 | +0.52 | +1.83 | -6.30 | +0.82 | +0.29 | -0.05 | 1 | elbow_flex | 2.77 |
| 4 | -1.44 | +3.22 | -6.38 | -0.17 | +0.45 | +0.48 | 1 | elbow_flex | 2.69 |
| 5 | +0.27 | +4.46 | -5.55 | +0.86 | +0.37 | -0.91 | 0 | - | 1.49 |
| 6 | +0.25 | +1.22 | -5.80 | +0.76 | +0.17 | -0.40 | 1 | elbow_flex | 2.97 |
| 7 | -0.39 | +5.66 | -7.33 | +0.29 | +0.25 | -0.95 | 2 | shoulder_lift, elbow_flex | 3.38 |
| 8 | -0.73 | +2.77 | -4.03 | +0.15 | +0.45 | -0.33 | 0 | - | 1.63 |
| 9 | -0.21 | +2.46 | -5.96 | -0.28 | +0.29 | -0.88 | 1 | elbow_flex | 2.16 |
| 10 | +0.01 | +6.48 | -6.68 | -0.35 | +0.30 | +0.06 | 2 | shoulder_lift, elbow_flex | 3.43 |
| 11 | -0.27 | +2.77 | -6.44 | +0.08 | +0.43 | +0.08 | 1 | elbow_flex | 2.27 |
| 12 | -0.43 | -0.10 | -4.85 | -0.37 | +0.12 | +0.23 | 0 | - | 4.10 |
| 13 | +1.63 | +2.91 | -4.66 | +0.27 | +0.41 | -0.35 | 0 | - | 1.81 |
| 14 | -0.50 | +3.86 | -6.15 | -0.82 | +0.29 | -0.44 | 1 | elbow_flex | 2.02 |
| 15 | -1.46 | +7.92 | -8.14 | -0.61 | +0.35 | -0.41 | 2 | shoulder_lift, elbow_flex | 5.70 |
| 16 | -1.47 | +7.22 | -9.06 | +1.60 | +0.38 | +0.11 | 2 | shoulder_lift, elbow_flex | 6.00 |
| 17 | -0.58 | +3.31 | -4.34 | -1.42 | +0.28 | -0.82 | 0 | - | 2.01 |
| 18 | +0.73 | +2.69 | -5.29 | +0.08 | +0.05 | -0.06 | 0 | - | 1.51 |
| 19 | +0.03 | +2.57 | -2.05 | -2.52 | +0.59 | +0.48 | 0 | - | 4.11 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.32 | 0.82 | -1.58 | +1.63 | 0/20 | 0% |
| shoulder_lift | +3.96 | 2.57 | -0.10 | +11.38 | 5/20 | 25% |
| elbow_flex | -6.11 | 1.84 | -10.50 | -2.05 | 12/20 | 60% |
| wrist_flex | -0.15 | 0.87 | -2.52 | +1.60 | 0/20 | 0% |
| wrist_roll | +0.32 | 0.12 | +0.05 | +0.59 | 0/20 | 0% |
| gripper | -0.26 | 0.42 | -0.95 | +0.48 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 8/20 ([2, 5, 8, 12, 13, 17, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 8, '1': 7, '2': 5}
- L2 error vs nearest-demo immediate GT delta (deg): mean=3.31, std=1.86, min=1.49, max=8.40
