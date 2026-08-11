# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep

Reference Shadow observation: `V2_F02` (`/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`)
Checkpoint: `outputs/pick_drop_v3_v4_reweight2/smolvla_pick_drop_v3_v4_reweight2_fresh/checkpoints/010000/pretrained_model`
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
| shoulder_pan | -0.7033 |
| shoulder_lift | +3.5165 |
| elbow_flex | -4.4396 |
| wrist_flex | -1.0989 |
| wrist_roll | +0.0879 |
| gripper | -0.3945 |

## Per-seed chunk[0] delta table

| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | clamp joint count | clamped joints | L2 err vs GT (deg) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | -0.73 | +3.49 | -11.05 | +0.14 | -0.05 | +0.83 | 1 | elbow_flex | 6.84 |
| 1 | -0.59 | +10.74 | -10.05 | -0.55 | +0.17 | +0.89 | 2 | shoulder_lift, elbow_flex | 9.26 |
| 2 | +1.45 | +1.53 | -6.04 | -0.20 | +0.12 | +0.95 | 1 | elbow_flex | 3.71 |
| 3 | +1.38 | +2.08 | -7.81 | +0.82 | +0.02 | +1.01 | 1 | elbow_flex | 4.84 |
| 4 | -1.32 | +2.91 | -7.51 | -0.38 | +0.36 | +1.71 | 1 | elbow_flex | 3.90 |
| 5 | +0.88 | +4.42 | -7.64 | +0.96 | +0.20 | +0.45 | 1 | elbow_flex | 4.30 |
| 6 | +0.83 | +1.57 | -7.17 | +0.72 | -0.18 | +0.71 | 1 | elbow_flex | 4.27 |
| 7 | +0.10 | +5.56 | -8.51 | +0.37 | +0.01 | +0.20 | 2 | shoulder_lift, elbow_flex | 4.89 |
| 8 | +0.19 | +3.50 | -5.41 | +0.57 | +0.49 | +0.97 | 0 | - | 2.55 |
| 9 | +0.59 | +2.26 | -7.09 | -0.31 | -0.08 | +0.33 | 1 | elbow_flex | 3.38 |
| 10 | +0.12 | +5.69 | -8.26 | -0.29 | +0.04 | +1.12 | 2 | shoulder_lift, elbow_flex | 4.79 |
| 11 | +0.88 | +1.67 | -7.32 | +0.24 | +0.27 | +1.04 | 1 | elbow_flex | 4.26 |
| 12 | +0.78 | -2.30 | -5.25 | -0.40 | -0.35 | +1.59 | 0 | - | 6.43 |
| 13 | +2.33 | +2.56 | -6.33 | +0.34 | +0.27 | +0.76 | 1 | elbow_flex | 4.14 |
| 14 | -0.03 | +3.01 | -7.32 | -0.57 | +0.02 | +0.75 | 1 | elbow_flex | 3.26 |
| 15 | -0.82 | +6.91 | -9.56 | -0.54 | +0.24 | +0.64 | 2 | shoulder_lift, elbow_flex | 6.26 |
| 16 | -1.32 | +6.94 | -10.17 | +1.19 | +0.17 | +1.28 | 2 | shoulder_lift, elbow_flex | 7.28 |
| 17 | -0.66 | +4.16 | -5.65 | -1.06 | +0.05 | +0.25 | 0 | - | 1.52 |
| 18 | +1.17 | +1.81 | -6.57 | -0.25 | -0.38 | +1.04 | 1 | elbow_flex | 3.74 |
| 19 | +0.93 | +3.12 | -3.18 | -2.82 | +0.85 | +1.70 | 0 | - | 3.51 |

## Summary statistics (seeds swept)

| joint | mean | std | min | max | clamp count | clamp rate |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | +0.31 | 0.96 | -1.32 | +2.33 | 0/20 | 0% |
| shoulder_lift | +3.58 | 2.63 | -2.30 | +10.74 | 5/20 | 25% |
| elbow_flex | -7.40 | 1.85 | -11.05 | -3.18 | 16/20 | 80% |
| wrist_flex | -0.10 | 0.85 | -2.82 | +1.19 | 0/20 | 0% |
| wrist_roll | +0.11 | 0.27 | -0.38 | +0.85 | 0/20 | 0% |
| gripper | +0.91 | 0.42 | +0.20 | +1.71 | 0/20 | 0% |

- Seeds with **zero** clamped joints: 4/20 ([8, 12, 17, 19])
- clamp-joint-count distribution (count -> #seeds): {'0': 4, '1': 11, '2': 5}
- L2 error vs nearest-demo immediate GT delta (deg): mean=4.66, std=1.74, min=1.52, max=9.26
