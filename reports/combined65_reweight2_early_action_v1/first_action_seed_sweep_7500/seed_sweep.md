# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_combined65_reweight2_early/smolvla_pick_drop_combined65_reweight2_early_fresh/checkpoints/007500/pretrained_model`
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
| 0 | -1.97 | +3.88 | -12.45 | +0.10 | +0.36 | +0.28 | 1 | elbow_flex | 8.16 |
| 1 | -1.58 | +9.77 | -7.61 | -0.99 | +0.46 | -0.13 | 2 | shoulder_lift, elbow_flex | 6.98 |
| 2 | +0.64 | +1.05 | -4.86 | -0.37 | +0.42 | +0.23 | 0 | - | 3.02 |
| 3 | +0.19 | +1.05 | -7.26 | +0.90 | +0.37 | +0.58 | 1 | elbow_flex | 4.05 |
| 4 | -2.14 | +2.62 | -7.37 | +0.04 | +0.52 | +1.02 | 1 | elbow_flex | 4.09 |
| 5 | -0.34 | +3.74 | -6.50 | +1.20 | +0.45 | -0.31 | 1 | elbow_flex | 2.26 |
| 6 | +0.02 | +0.83 | -6.19 | +0.95 | +0.26 | +0.25 | 1 | elbow_flex | 3.57 |
| 7 | -0.47 | +4.65 | -7.20 | +0.19 | +0.35 | -0.55 | 1 | elbow_flex | 2.82 |
| 8 | -1.14 | +1.85 | -4.33 | +0.24 | +0.54 | +0.15 | 0 | - | 2.55 |
| 9 | -0.36 | +1.55 | -6.84 | +0.01 | +0.37 | -0.36 | 1 | elbow_flex | 3.30 |
| 10 | -0.27 | +4.83 | -7.07 | -0.20 | +0.37 | +0.72 | 1 | elbow_flex | 2.94 |
| 11 | -0.23 | +1.82 | -7.15 | +0.23 | +0.53 | +0.70 | 1 | elbow_flex | 3.50 |
| 12 | -0.67 | -0.82 | -5.17 | -0.23 | +0.21 | +0.69 | 0 | - | 4.93 |
| 13 | +1.42 | +1.56 | -5.62 | +0.66 | +0.51 | +0.17 | 0 | - | 2.95 |
| 14 | -0.72 | +3.30 | -6.94 | -0.60 | +0.38 | +0.10 | 1 | elbow_flex | 2.76 |
| 15 | -1.74 | +7.10 | -8.33 | -0.60 | +0.45 | +0.12 | 2 | shoulder_lift, elbow_flex | 5.39 |
| 16 | -2.25 | +5.72 | -9.13 | +1.65 | +0.44 | +0.64 | 2 | shoulder_lift, elbow_flex | 5.74 |
| 17 | -1.11 | +2.85 | -5.56 | -1.14 | +0.39 | -0.22 | 0 | - | 2.38 |
| 18 | +0.33 | +1.32 | -5.18 | +0.06 | +0.13 | +0.63 | 0 | - | 2.79 |
| 19 | +0.06 | +1.94 | -2.69 | -2.16 | +0.71 | +0.98 | 0 | - | 3.93 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.62 | 0.95 | -2.25 | +1.42 | 0/20 | 0% |
| shoulder_lift | +3.03 | 2.40 | -0.82 | +9.77 | 3/20 | 15% |
| elbow_flex | -6.67 | 1.95 | -12.45 | -2.69 | 13/20 | 65% |
| wrist_flex | -0.00 | 0.85 | -2.16 | +1.65 | 0/20 | 0% |
| wrist_roll | +0.41 | 0.12 | +0.13 | +0.71 | 0/20 | 0% |
| gripper | +0.28 | 0.44 | -0.55 | +1.02 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 7/20 ([2, 8, 12, 13, 17, 18, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 7, '1': 10, '2': 3}
- L2 error vs nearest-demo immediate GT delta (deg): mean=3.91, std=1.55, min=2.26, max=8.16
