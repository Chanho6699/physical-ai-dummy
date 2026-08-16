# Grid35 Episode-Start Leader-Follower Sync Jump Analysis

- Dataset: `/home/sunglee/Projects/physical-ai-dummy/data/so101_cube_xy_grid35_v2_clean`
- Episodes analyzed: 3
- Joints (order from `meta/info.json`): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
- Start window: first 30 frames (1s @ 30 FPS)
- `delta = action[t] - observation.state[t]` (units as stored, project convention 'deg')

## 1. Schema actually used

- `data/chunk-000/file-{NNN}.parquet` -> one file per episode (confirmed: each file's
  `episode_index` column contains a single unique value == its file index).
- Columns read: `action` (float32[6]), `observation.state` (float32[6]), `timestamp`,
  `frame_index`, `episode_index`.
- Joint order: `['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']`

## 2. Representative episodes

### Episode 0 (n_frames=358)

| joint | state[0] | action[0] | delta[0] | max\|delta\| (first 30) | argmax frame |
|---|---:|---:|---:|---:|---:|
| shoulder_pan | -7.96 | -8.40 | -0.44 | 0.44 | 0 |
| shoulder_lift | -97.71 | -98.33 | -0.62 | 0.62 | 0 |
| elbow_flex | 96.48 | 94.33 | -2.15 | 2.24 | 28 |
| wrist_flex | 49.19 | 49.49 | +0.31 | 0.31 | 0 |
| wrist_roll | -1.54 | -1.10 | +0.44 | 0.53 | 25 |
| gripper | 0.62 | 0.85 | +0.23 | 0.23 | 0 |

### Episode 1 (n_frames=359)

| joint | state[0] | action[0] | delta[0] | max\|delta\| (first 30) | argmax frame |
|---|---:|---:|---:|---:|---:|
| shoulder_pan | -5.14 | -5.93 | -0.79 | 0.79 | 0 |
| shoulder_lift | -97.71 | -98.33 | -0.62 | 0.62 | 0 |
| elbow_flex | 96.57 | 94.42 | -2.15 | 2.59 | 29 |
| wrist_flex | 48.75 | 49.14 | +0.40 | 0.57 | 23 |
| wrist_roll | -1.45 | -1.27 | +0.18 | 0.62 | 29 |
| gripper | 1.52 | 1.14 | -0.38 | 0.38 | 0 |

### Episode 2 (n_frames=359)

| joint | state[0] | action[0] | delta[0] | max\|delta\| (first 30) | argmax frame |
|---|---:|---:|---:|---:|---:|
| shoulder_pan | -6.73 | -7.34 | -0.62 | 0.70 | 9 |
| shoulder_lift | -99.03 | -98.51 | +0.53 | 0.53 | 0 |
| elbow_flex | 96.04 | 94.42 | -1.63 | 1.80 | 24 |
| wrist_flex | 49.19 | 49.49 | +0.31 | 0.48 | 24 |
| wrist_roll | -1.27 | -0.92 | +0.35 | 0.35 | 0 |
| gripper | 1.45 | 1.14 | -0.31 | 0.31 | 0 |

## 3. Aggregate statistics (all episodes)

### 3a. Frame-0 delta (action[0] - state[0])

| joint | mean | median | std | max | min | max\|delta\| |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | -0.62 | -0.62 | 0.14 | -0.44 | -0.79 | 0.79 |
| shoulder_lift | -0.23 | -0.62 | 0.54 | +0.53 | -0.62 | 0.62 |
| elbow_flex | -1.98 | -2.15 | 0.25 | -1.63 | -2.15 | 2.15 |
| wrist_flex | +0.34 | +0.31 | 0.04 | +0.40 | +0.31 | 0.40 |
| wrist_roll | +0.32 | +0.35 | 0.11 | +0.44 | +0.18 | 0.44 |
| gripper | -0.16 | -0.31 | 0.27 | +0.23 | -0.38 | 0.38 |

### 3b. First-5-frames |delta| distribution

| joint | mean | median | std | p95 | max |
|---|---:|---:|---:|---:|---:|
| shoulder_pan | 0.62 | 0.62 | 0.14 | 0.79 | 0.79 |
| shoulder_lift | 0.59 | 0.62 | 0.04 | 0.62 | 0.62 |
| elbow_flex | 1.98 | 2.15 | 0.25 | 2.15 | 2.15 |
| wrist_flex | 0.34 | 0.31 | 0.04 | 0.40 | 0.40 |
| wrist_roll | 0.32 | 0.35 | 0.11 | 0.44 | 0.44 |
| gripper | 0.31 | 0.31 | 0.06 | 0.38 | 0.38 |

### 3c. First-30-frames max|delta| distribution (per episode, then aggregated)

| joint | mean | median | std | max | min |
|---|---:|---:|---:|---:|---:|
| shoulder_pan | 0.64 | 0.70 | 0.15 | 0.79 | 0.44 |
| shoulder_lift | 0.59 | 0.62 | 0.04 | 0.62 | 0.53 |
| elbow_flex | 2.21 | 2.24 | 0.32 | 2.59 | 1.80 |
| wrist_flex | 0.45 | 0.48 | 0.11 | 0.57 | 0.31 |
| wrist_roll | 0.50 | 0.53 | 0.11 | 0.62 | 0.35 |
| gripper | 0.31 | 0.31 | 0.06 | 0.38 | 0.23 |

### 3d. Episodes crossing |delta| thresholds

**At frame 0:**

| joint | >=5° | >=10° | >=15° | >=20° | >=25° |
|---|---:|---:|---:|---:|---:|
| shoulder_pan | 0 | 0 | 0 | 0 | 0 |
| shoulder_lift | 0 | 0 | 0 | 0 | 0 |
| elbow_flex | 0 | 0 | 0 | 0 | 0 |
| wrist_flex | 0 | 0 | 0 | 0 | 0 |
| wrist_roll | 0 | 0 | 0 | 0 | 0 |
| gripper | 0 | 0 | 0 | 0 | 0 |

**Anywhere in first 30 frames (max\|delta\| per episode):**

| joint | >=5° | >=10° | >=15° | >=20° | >=25° |
|---|---:|---:|---:|---:|---:|
| shoulder_pan | 0 | 0 | 0 | 0 | 0 |
| shoulder_lift | 0 | 0 | 0 | 0 | 0 |
| elbow_flex | 0 | 0 | 0 | 0 | 0 |
| wrist_flex | 0 | 0 | 0 | 0 | 0 |
| wrist_roll | 0 | 0 | 0 | 0 | 0 |
| gripper | 0 | 0 | 0 | 0 | 0 |

(out of 3 episodes)

### 3e. Sign consistency of frame-0 delta across episodes

| joint | n_positive | n_negative | n_zero | dominant sign | dominant fraction |
|---|---:|---:|---:|---|---:|
| shoulder_pan | 0 | 3 | 0 | negative | 100% |
| shoulder_lift | 1 | 2 | 0 | negative | 67% |
| elbow_flex | 0 | 3 | 0 | negative | 100% |
| wrist_flex | 3 | 0 | 0 | positive | 100% |
| wrist_roll | 3 | 0 | 0 | positive | 100% |
| gripper | 1 | 2 | 0 | negative | 67% |

### 3f. Timing of the peak |delta| within the first 30 frames

Distinguishes an *instantaneous* sync jump (max|delta| at frame 0-2) from a
*gradually building* trajectory (max|delta| near the end of the 1s window).

| joint | mean argmax frame | % episodes argmax at frame 0-2 | % episodes argmax in last 3 frames of window |
|---|---:|---:|---:|
| shoulder_pan | 3.0 | 67% | 0% |
| shoulder_lift | 0.0 | 100% | 0% |
| elbow_flex | 27.0 | 0% | 67% |
| wrist_flex | 15.7 | 33% | 0% |
| wrist_roll | 18.0 | 33% | 33% |
| gripper | 0.0 | 100% | 0% |

## 4. Early vs middle vs late |delta| comparison

Segment = 30 frames (early = first 30, late = last 30, middle = centered on episode midpoint),
stat = mean across episodes of that episode's mean|delta| in the segment.

| joint | early mean | middle mean | late mean | early max(of means) is largest? |
|---|---:|---:|---:|---|
| shoulder_pan | 0.64 | 0.74 | 0.58 | middle |
| shoulder_lift | 0.57 | 0.23 | 0.29 | early |
| elbow_flex | 2.02 | 1.61 | 2.32 | late |
| wrist_flex | 0.34 | 0.24 | 0.42 | late |
| wrist_roll | 0.35 | 0.30 | 0.18 | early |
| gripper | 0.31 | 5.14 | 2.21 | middle |

Pooled over the 6 joints (mean of per-episode mean|delta|, and mean of per-episode max|delta|):

| segment | mean of episode mean\|delta\| | mean of episode max\|delta\| | overall max\|delta\| |
|---|---:|---:|---:|
| early | 0.70 | 2.21 | 2.59 |
| middle | 1.38 | 10.13 | 11.70 |
| late | 1.00 | 7.08 | 13.03 |

## 5. Comparison to Shadow first-action bias

Shadow fixed-scene runs used (5): F01, F02, F03, F04, F05

| joint | Shadow fixed-scene mean delta | dataset frame-0 mean delta | same sign? | magnitude ratio (dataset/shadow) |
|---|---:|---:|---|---:|
| shoulder_lift | +16.03 | -0.23 | no | 0.01x |
| elbow_flex | -24.86 | -1.98 | YES | 0.08x |
| wrist_flex | +8.01 | +0.34 | YES | 0.04x |
| shoulder_pan | -5.05 | -0.62 | YES | 0.12x |
| wrist_roll | -0.85 | +0.32 | no | 0.38x |
| gripper | +3.21 | -0.16 | no | 0.05x |

T05 REJECT/rerun runs used (5): T05[REJECT], T05-R2[REJECT], T05-R3[WOULD_CLAMP], T05-R4[WOULD_CLAMP], T05-R5[WOULD_CLAMP]

| joint | T05 mean delta | T05 max\|delta\| |
|---|---:|---:|
| shoulder_lift | +20.07 | 23.77 |
| elbow_flex | -26.70 | 32.06 |
| wrist_flex | +7.61 | 9.43 |

## 6. Verdict

**C**

Dataset start-of-episode GT deltas are not unusually large, are not concentrated at frame 0, or do not match Shadow's first-action sign/magnitude pattern; look elsewhere (e.g. policy/normalization, action head init, control-loop timing).

- Same-sign match on 2/3 key joints (shoulder_lift, elbow_flex, wrist_flex)
- Comparable-magnitude match on 0/3 key joints
- Instant-jump timing (peak at frame 0-2) on 1/3 key joints
- Fully matching (sign + magnitude + timing) key joints: none
- Large (>=10deg) start-of-episode delta observed in dataset: False
