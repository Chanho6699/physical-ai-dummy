# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65_reweight3/smolvla_pick_drop_combined65_reweight3_fresh/checkpoints/002500/pretrained_model`
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
| 0 | -5.71 | +9.14 | -22.98 | +0.72 | +0.33 | -0.60 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 20.00 |
| 1 | -5.83 | +21.74 | -14.27 | -0.82 | +0.56 | -1.10 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 21.23 |
| 2 | -2.47 | +2.62 | -6.79 | -1.08 | +0.36 | -0.10 | 1 | elbow_flex | 3.91 |
| 3 | -1.60 | +2.25 | -13.83 | +1.14 | +0.23 | -0.60 | 1 | elbow_flex | 9.58 |
| 4 | -7.65 | +6.61 | -10.97 | -0.57 | +0.66 | +1.51 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 10.66 |
| 5 | -1.30 | +9.20 | -8.00 | +1.89 | +0.47 | -2.09 | 2 | shoulder_lift, elbow_flex | 6.95 |
| 6 | -1.15 | +2.03 | -8.56 | +0.60 | +0.07 | -0.84 | 1 | elbow_flex | 4.60 |
| 7 | -4.12 | +10.95 | -12.11 | +0.75 | +0.23 | -2.16 | 2 | shoulder_lift, elbow_flex | 11.34 |
| 8 | -5.06 | +5.93 | -5.13 | +0.23 | +0.68 | -0.45 | 2 | shoulder_pan, shoulder_lift | 5.70 |
| 9 | -2.61 | +4.32 | -11.13 | -0.24 | +0.28 | -1.53 | 1 | elbow_flex | 7.22 |
| 10 | -2.91 | +11.24 | -12.84 | -0.55 | +0.31 | +0.25 | 2 | shoulder_lift, elbow_flex | 11.51 |
| 11 | -2.25 | +3.69 | -11.89 | -0.09 | +0.49 | +0.09 | 1 | elbow_flex | 7.71 |
| 12 | -4.47 | -7.48 | -5.80 | -1.91 | -0.14 | +2.20 | 2 | shoulder_lift, elbow_flex | 12.75 |
| 13 | +1.79 | +6.99 | -7.25 | +0.42 | +0.52 | -0.61 | 2 | shoulder_lift, elbow_flex | 4.44 |
| 14 | -4.34 | +7.50 | -13.24 | -1.82 | +0.36 | -0.11 | 2 | shoulder_lift, elbow_flex | 10.60 |
| 15 | -5.86 | +16.23 | -15.26 | -1.38 | +0.52 | -1.19 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 17.50 |
| 16 | -6.41 | +13.68 | -13.35 | +2.59 | +0.52 | +0.24 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 14.91 |
| 17 | -5.09 | +9.63 | -5.97 | -3.38 | +0.41 | -1.15 | 3 | shoulder_pan, shoulder_lift, elbow_flex | 8.75 |
| 18 | -1.36 | +3.27 | -8.15 | -1.30 | -0.17 | +0.74 | 1 | elbow_flex | 4.32 |
| 19 | -2.12 | +4.68 | +0.96 | -6.26 | +0.92 | +2.56 | 1 | wrist_flex | 9.40 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -3.53 | 2.24 | -7.65 | +1.79 | 7/20 | 35% |
| shoulder_lift | +7.21 | 5.96 | -7.48 | +21.74 | 13/20 | 65% |
| elbow_flex | -10.33 | 4.86 | -22.98 | +0.96 | 18/20 | 90% |
| wrist_flex | -0.55 | 1.88 | -6.26 | +2.59 | 1/20 | 5% |
| wrist_roll | +0.38 | 0.25 | -0.17 | +0.92 | 0/20 | 0% |
| gripper | -0.25 | 1.23 | -2.16 | +2.56 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 0/20 ([])
- clamp-joint-count distribution (count -> #seeds): {'1': 7, '2': 7, '3': 6}
- L2 error vs nearest-demo immediate GT delta (deg): mean=10.15, std=4.95, min=3.91, max=21.23
