# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/reweight_ablation/combined65_reweight_new2_old1_v1/checkpoints/007500/pretrained_model`
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
| 0 | -1.28 | +4.28 | -11.10 | -0.18 | +0.34 | +0.09 | 1 | elbow_flex | 6.70 |
| 1 | -1.43 | +11.02 | -8.02 | -1.05 | +0.42 | -0.32 | 2 | shoulder_lift, elbow_flex | 8.19 |
| 2 | +1.15 | +2.30 | -5.19 | -0.47 | +0.39 | -0.11 | 0 | - | 2.12 |
| 3 | +0.84 | +1.61 | -6.50 | +0.70 | +0.34 | +0.27 | 1 | elbow_flex | 3.14 |
| 4 | -1.25 | +3.55 | -7.08 | -0.18 | +0.49 | +0.64 | 1 | elbow_flex | 3.11 |
| 5 | +0.14 | +4.58 | -6.17 | +0.98 | +0.41 | -0.59 | 1 | elbow_flex | 1.95 |
| 6 | +0.60 | +1.09 | -6.14 | +0.65 | +0.22 | +0.03 | 1 | elbow_flex | 3.26 |
| 7 | +0.26 | +5.55 | -7.47 | +0.06 | +0.29 | -0.74 | 2 | shoulder_lift, elbow_flex | 3.37 |
| 8 | -0.44 | +2.73 | -4.18 | -0.01 | +0.49 | -0.12 | 0 | - | 1.51 |
| 9 | +0.02 | +2.42 | -6.74 | -0.21 | +0.34 | -0.64 | 1 | elbow_flex | 2.66 |
| 10 | +0.17 | +6.57 | -7.28 | -0.32 | +0.36 | +0.51 | 2 | shoulder_lift, elbow_flex | 3.95 |
| 11 | +0.13 | +2.88 | -6.83 | +0.01 | +0.50 | +0.43 | 1 | elbow_flex | 2.62 |
| 12 | +0.08 | -0.16 | -4.92 | -0.46 | +0.18 | +0.37 | 0 | - | 4.17 |
| 13 | +1.83 | +2.86 | -5.41 | +0.41 | +0.46 | -0.00 | 0 | - | 2.20 |
| 14 | -0.19 | +4.05 | -6.61 | -0.85 | +0.35 | -0.24 | 1 | elbow_flex | 2.35 |
| 15 | -1.45 | +8.12 | -8.47 | -0.75 | +0.40 | -0.03 | 2 | shoulder_lift, elbow_flex | 6.07 |
| 16 | -1.46 | +6.96 | -9.15 | +1.43 | +0.40 | +0.38 | 2 | shoulder_lift, elbow_flex | 5.91 |
| 17 | -0.34 | +3.69 | -5.16 | -1.47 | +0.36 | -0.52 | 0 | - | 1.92 |
| 18 | +1.09 | +2.04 | -5.01 | -0.12 | +0.10 | +0.28 | 0 | - | 2.20 |
| 19 | +0.42 | +2.52 | -2.61 | -2.70 | +0.65 | +0.65 | 0 | - | 3.99 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.06 | 0.92 | -1.46 | +1.83 | 0/20 | 0% |
| shoulder_lift | +3.93 | 2.57 | -0.16 | +11.02 | 5/20 | 25% |
| elbow_flex | -6.50 | 1.83 | -11.10 | -2.61 | 13/20 | 65% |
| wrist_flex | -0.23 | 0.88 | -2.70 | +1.43 | 0/20 | 0% |
| wrist_roll | +0.37 | 0.12 | +0.10 | +0.65 | 0/20 | 0% |
| gripper | +0.02 | 0.42 | -0.74 | +0.65 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 7/20 ([2, 8, 12, 13, 17, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 7, '1': 8, '2': 5}
- L2 error vs nearest-demo immediate GT delta (deg): mean=3.57, std=1.77, min=1.51, max=8.19
