# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T04_actual/shadow_patched.json`)
Checkpoint: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model`
Task: `Pick up the cube and place it in the target area.`
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
| 0 | -1.62 | +5.91 | -11.54 | +0.89 | +0.21 | +0.80 | 2 | shoulder_lift, elbow_flex | 7.56 |
| 1 | +0.06 | +12.91 | -8.41 | -0.03 | +0.34 | +0.53 | 2 | shoulder_lift, elbow_flex | 9.86 |
| 2 | +0.33 | +2.69 | -3.88 | +0.50 | +0.31 | +1.25 | 0 | - | 2.16 |
| 3 | +0.56 | +3.40 | -7.54 | +1.93 | +0.21 | +1.21 | 1 | elbow_flex | 3.78 |
| 4 | -1.45 | +4.59 | -6.74 | +0.31 | +0.45 | +1.90 | 1 | elbow_flex | 3.59 |
| 5 | +0.21 | +6.07 | -6.38 | +1.84 | +0.26 | +0.52 | 2 | shoulder_lift, elbow_flex | 3.38 |
| 6 | -0.06 | +2.14 | -5.63 | +1.74 | +0.06 | +0.92 | 0 | - | 2.82 |
| 7 | -0.45 | +6.88 | -6.52 | +1.16 | +0.15 | -0.25 | 2 | shoulder_lift, elbow_flex | 3.74 |
| 8 | -0.43 | +4.04 | -3.03 | +1.20 | +0.43 | +0.75 | 0 | - | 2.30 |
| 9 | -0.45 | +3.96 | -6.45 | +0.85 | +0.22 | +0.37 | 1 | elbow_flex | 2.18 |
| 10 | +0.10 | +8.34 | -7.61 | +0.23 | +0.19 | +1.62 | 2 | shoulder_lift, elbow_flex | 5.73 |
| 11 | -0.10 | +3.12 | -6.21 | +0.84 | +0.39 | +1.53 | 1 | elbow_flex | 2.68 |
| 12 | -0.10 | -2.12 | -4.73 | +0.52 | +0.08 | +1.97 | 0 | - | 6.43 |
| 13 | +0.83 | +4.09 | -5.91 | +1.11 | +0.38 | +1.05 | 1 | elbow_flex | 2.25 |
| 14 | -0.52 | +5.53 | -6.60 | -0.19 | +0.26 | +0.86 | 2 | shoulder_lift, elbow_flex | 2.98 |
| 15 | -1.05 | +9.10 | -7.85 | -0.21 | +0.28 | +0.75 | 2 | shoulder_lift, elbow_flex | 6.38 |
| 16 | -2.04 | +8.67 | -9.52 | +2.49 | +0.30 | +1.37 | 2 | shoulder_lift, elbow_flex | 7.76 |
| 17 | -0.96 | +5.21 | -3.32 | -0.62 | +0.23 | +0.30 | 1 | shoulder_lift | 2.44 |
| 18 | +1.36 | +4.00 | -6.20 | +1.35 | -0.07 | +1.26 | 1 | elbow_flex | 2.78 |
| 19 | +0.82 | +4.60 | -1.17 | -1.55 | +0.73 | +1.97 | 0 | - | 4.67 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.25 | 0.85 | -2.04 | +1.36 | 0/20 | 0% |
| shoulder_lift | +5.16 | 3.03 | -2.12 | +12.91 | 9/20 | 45% |
| elbow_flex | -6.26 | 2.27 | -11.54 | -1.17 | 14/20 | 70% |
| wrist_flex | +0.72 | 0.94 | -1.55 | +2.49 | 0/20 | 0% |
| wrist_roll | +0.27 | 0.16 | -0.07 | +0.73 | 0/20 | 0% |
| gripper | +1.03 | 0.58 | -0.25 | +1.97 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 5/20 ([2, 6, 8, 12, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 5, '1': 7, '2': 8}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.27, std=2.19, min=2.16, max=9.86
