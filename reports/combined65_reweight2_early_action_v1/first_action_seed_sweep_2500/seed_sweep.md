# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65_reweight2_early/smolvla_pick_drop_combined65_reweight2_early_fresh/checkpoints/002500/pretrained_model`
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
| 0 | -4.32 | +4.55 | -26.18 | +1.46 | +0.58 | -0.70 | 1 | elbow_flex | 22.09 |
| 1 | -3.42 | +17.60 | -15.16 | +0.37 | +0.88 | -0.74 | 2 | shoulder_lift, elbow_flex | 17.72 |
| 2 | +0.40 | -1.67 | -6.05 | -0.55 | +0.68 | -0.68 | 1 | elbow_flex | 5.84 |
| 3 | +1.05 | +1.52 | -16.37 | +2.12 | +0.54 | -0.18 | 1 | elbow_flex | 12.18 |
| 4 | -6.64 | +3.29 | -13.27 | +0.19 | +0.94 | +1.23 | 2 | shoulder_pan, elbow_flex | 11.19 |
| 5 | +0.55 | +7.81 | -10.40 | +2.85 | +0.69 | -2.98 | 2 | shoulder_lift, elbow_flex | 7.99 |
| 6 | +0.56 | +1.27 | -10.78 | +1.70 | +0.29 | -1.22 | 1 | elbow_flex | 6.94 |
| 7 | -1.52 | +7.15 | -11.49 | +1.37 | +0.49 | -2.80 | 2 | shoulder_lift, elbow_flex | 8.29 |
| 8 | -3.79 | +1.95 | -6.08 | +1.02 | +0.90 | -0.15 | 1 | elbow_flex | 4.81 |
| 9 | +0.12 | -0.90 | -11.19 | +0.43 | +0.51 | -3.07 | 1 | elbow_flex | 8.60 |
| 10 | -0.81 | +7.31 | -11.68 | -0.70 | +0.61 | +0.80 | 2 | shoulder_lift, elbow_flex | 8.08 |
| 11 | -0.96 | -0.81 | -12.35 | +0.63 | +0.78 | -0.15 | 1 | elbow_flex | 9.16 |
| 12 | -2.16 | -10.12 | -8.77 | -1.96 | +0.13 | +1.46 | 2 | shoulder_lift, elbow_flex | 15.05 |
| 13 | +3.45 | +4.35 | -9.86 | +1.67 | +0.76 | -1.08 | 1 | elbow_flex | 6.47 |
| 14 | -3.17 | +2.69 | -13.65 | -1.63 | +0.63 | -0.73 | 1 | elbow_flex | 9.91 |
| 15 | -4.97 | +9.64 | -14.67 | -0.67 | +0.75 | -0.63 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 12.74 |
| 16 | -4.22 | +12.34 | -17.46 | +3.92 | +0.74 | -0.01 | 2 | shoulder_lift, elbow_flex | 16.45 |
| 17 | -4.68 | +3.02 | -4.01 | -3.24 | +0.56 | -2.71 | 1 | shoulder_pan | 6.54 |
| 18 | +1.12 | +0.48 | -9.98 | -0.09 | -0.01 | +0.41 | 1 | elbow_flex | 6.46 |
| 19 | -0.92 | -0.47 | +0.47 | -5.88 | +1.12 | +0.32 | 1 | wrist_flex | 9.22 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.72 | 2.54 | -6.64 | +3.45 | 3/20 | 15% |
| shoulder_lift | +3.55 | 5.70 | -10.12 | +17.60 | 7/20 | 35% |
| elbow_flex | -11.45 | 5.38 | -26.18 | +0.47 | 18/20 | 90% |
| wrist_flex | +0.15 | 2.14 | -5.88 | +3.92 | 1/20 | 5% |
| wrist_roll | +0.63 | 0.26 | -0.01 | +1.12 | 0/20 | 0% |
| gripper | -0.68 | 1.30 | -3.07 | +1.46 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 12, '2': 7, '3': 1}
- L2 error vs nearest-demo immediate GT delta (deg): mean=10.29, std=4.41, min=4.81, max=22.09
