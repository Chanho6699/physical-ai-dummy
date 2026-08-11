# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `T01` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01_real_final/shadow_patched.json`)
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
| 0 | -2.78 | +7.66 | -13.07 | +0.55 | +0.24 | +0.29 | 2 | shoulder_lift, elbow_flex | 9.75 |
| 1 | -1.39 | +14.20 | -9.86 | -0.43 | +0.34 | -0.05 | 2 | shoulder_lift, elbow_flex | 11.72 |
| 2 | -0.82 | +4.63 | -5.44 | +0.15 | +0.33 | +0.77 | 0 | - | 1.90 |
| 3 | -0.61 | +5.28 | -9.05 | +1.52 | +0.23 | +0.69 | 2 | shoulder_lift, elbow_flex | 5.01 |
| 4 | -2.62 | +6.19 | -8.45 | -0.06 | +0.44 | +1.34 | 2 | shoulder_lift, elbow_flex | 5.57 |
| 5 | -0.91 | +7.53 | -8.22 | +1.39 | +0.26 | +0.06 | 2 | shoulder_lift, elbow_flex | 5.40 |
| 6 | -1.26 | +4.11 | -7.32 | +1.35 | +0.10 | +0.45 | 1 | elbow_flex | 3.37 |
| 7 | -1.53 | +8.76 | -8.30 | +0.79 | +0.18 | -0.68 | 2 | shoulder_lift, elbow_flex | 6.40 |
| 8 | -1.53 | +5.80 | -4.95 | +0.90 | +0.45 | +0.21 | 1 | shoulder_lift | 2.79 |
| 9 | -1.53 | +5.84 | -8.19 | +0.43 | +0.24 | -0.11 | 2 | shoulder_lift, elbow_flex | 4.45 |
| 10 | -1.21 | +9.87 | -9.17 | -0.24 | +0.21 | +0.98 | 2 | shoulder_lift, elbow_flex | 7.79 |
| 11 | -1.26 | +4.93 | -7.79 | +0.52 | +0.40 | +0.95 | 1 | elbow_flex | 3.90 |
| 12 | -1.33 | -0.16 | -6.08 | +0.20 | +0.13 | +1.33 | 1 | elbow_flex | 4.84 |
| 13 | -0.36 | +5.84 | -7.50 | +0.78 | +0.38 | +0.54 | 2 | shoulder_lift, elbow_flex | 3.70 |
| 14 | -1.69 | +7.13 | -8.14 | -0.54 | +0.28 | +0.27 | 2 | shoulder_lift, elbow_flex | 5.26 |
| 15 | -2.27 | +10.62 | -9.37 | -0.60 | +0.30 | +0.18 | 2 | shoulder_lift, elbow_flex | 8.67 |
| 16 | -3.17 | +10.30 | -11.37 | +2.15 | +0.31 | +0.86 | 2 | shoulder_lift, elbow_flex | 10.17 |
| 17 | -1.99 | +6.92 | -5.17 | -0.92 | +0.26 | -0.15 | 1 | shoulder_lift | 3.97 |
| 18 | +0.05 | +5.56 | -7.47 | +0.86 | -0.01 | +0.62 | 2 | shoulder_lift, elbow_flex | 3.51 |
| 19 | -0.59 | +6.05 | -2.71 | -1.95 | +0.73 | +1.31 | 1 | shoulder_lift | 4.12 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -1.44 | 0.80 | -3.17 | +0.05 | 0/20 | 0% |
| shoulder_lift | +6.85 | 2.90 | -0.16 | +14.20 | 16/20 | 80% |
| elbow_flex | -7.88 | 2.25 | -13.07 | -2.71 | 16/20 | 80% |
| wrist_flex | +0.34 | 0.94 | -1.95 | +2.15 | 0/20 | 0% |
| wrist_roll | +0.29 | 0.15 | -0.01 | +0.73 | 0/20 | 0% |
| gripper | +0.49 | 0.53 | -0.68 | +1.34 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 1/20 ([2])
- clamp-joint-count distribution (count -> #seeds): {'0': 1, '1': 6, '2': 13}
- L2 error vs nearest-demo immediate GT delta (deg): mean=5.62, std=2.60, min=1.90, max=11.72
