# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65_reweight3/smolvla_pick_drop_combined65_reweight3_fresh/checkpoints/007500/pretrained_model`
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
| 0 | -2.03 | +5.03 | -12.15 | +0.02 | +0.23 | -0.00 | 1 | elbow_flex | 7.95 |
| 1 | -2.61 | +11.91 | -10.30 | -1.57 | +0.34 | -0.13 | 2 | shoulder_lift, elbow_flex | 10.40 |
| 2 | -0.29 | +2.11 | -6.21 | -1.32 | +0.29 | -0.16 | 1 | elbow_flex | 2.90 |
| 3 | +0.19 | +2.24 | -8.87 | +0.48 | +0.21 | +0.04 | 1 | elbow_flex | 4.59 |
| 4 | -2.54 | +3.32 | -8.05 | -0.95 | +0.46 | +0.38 | 1 | elbow_flex | 4.65 |
| 5 | -0.68 | +5.40 | -9.12 | +0.72 | +0.35 | -0.43 | 2 | shoulder_lift, elbow_flex | 4.88 |
| 6 | -0.83 | +1.21 | -7.44 | -0.03 | +0.11 | -0.09 | 1 | elbow_flex | 4.03 |
| 7 | -1.06 | +6.37 | -9.16 | -0.34 | +0.23 | -0.81 | 2 | shoulder_lift, elbow_flex | 5.39 |
| 8 | -1.63 | +2.96 | -5.63 | -0.42 | +0.45 | -0.07 | 0 | - | 2.43 |
| 9 | -1.26 | +2.89 | -7.99 | -0.63 | +0.22 | -0.16 | 1 | elbow_flex | 3.90 |
| 10 | -1.05 | +6.41 | -8.89 | -1.12 | +0.23 | +0.01 | 2 | shoulder_lift, elbow_flex | 5.31 |
| 11 | -1.02 | +3.18 | -8.08 | -0.43 | +0.40 | +0.11 | 1 | elbow_flex | 3.84 |
| 12 | -1.29 | -1.43 | -5.43 | -1.57 | +0.05 | +0.39 | 0 | - | 5.88 |
| 13 | +0.31 | +3.19 | -7.46 | -0.05 | +0.41 | -0.11 | 1 | elbow_flex | 2.99 |
| 14 | -1.51 | +3.87 | -8.12 | -1.44 | +0.25 | -0.21 | 1 | elbow_flex | 4.24 |
| 15 | -2.12 | +8.09 | -9.55 | -1.57 | +0.32 | -0.30 | 2 | shoulder_lift, elbow_flex | 7.13 |
| 16 | -2.45 | +6.85 | -10.17 | +0.99 | +0.37 | +0.06 | 2 | shoulder_lift, elbow_flex | 6.90 |
| 17 | -1.44 | +3.55 | -6.04 | -2.18 | +0.26 | -0.62 | 1 | elbow_flex | 3.28 |
| 18 | -0.43 | +2.91 | -7.21 | -1.13 | -0.04 | -0.03 | 1 | elbow_flex | 3.15 |
| 19 | -0.98 | +3.60 | -3.18 | -3.57 | +0.64 | +0.74 | 0 | - | 4.41 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.24 | 0.82 | -2.61 | +0.31 | 0/20 | 0% |
| shoulder_lift | +4.18 | 2.75 | -1.43 | +11.91 | 6/20 | 30% |
| elbow_flex | -7.95 | 1.97 | -12.15 | -3.18 | 17/20 | 85% |
| wrist_flex | -0.81 | 1.04 | -3.57 | +0.99 | 0/20 | 0% |
| wrist_roll | +0.29 | 0.15 | -0.04 | +0.64 | 0/20 | 0% |
| gripper | -0.07 | 0.33 | -0.81 | +0.74 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 3/20 ([8, 12, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 3, '1': 11, '2': 6}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.91, std=1.91, min=2.43, max=10.40
