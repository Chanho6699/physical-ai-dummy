# SmolVLA training-target action-chunk alignment verification (V2-clean)

Dataset: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_xy_grid35_v2_clean`  
Temporal config source: checkpoint config.json: /home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model/config.json  
`chunk_size`=50, `n_action_steps`=50, `n_obs_steps`=1, `action_delta_indices`=`range(0,50)` (=[0, 1, 2]...[48, 49]), `observation_delta_indices`=[0]

## 1. How the training target chunk is actually built (code trail)

- `lerobot/policies/smolvla/configuration_smolvla.py` `SmolVLAConfig.action_delta_indices` -> `list(range(self.chunk_size))` (i.e. offsets `0, 1, 2, ..., chunk_size-1`); `observation_delta_indices` -> `[0]` (current frame only, no history/future for state).
- `lerobot/datasets/factory.py` `resolve_delta_timestamps(cfg, ds_meta)` turns those into `delta_timestamps['action'] = [k/fps for k in action_delta_indices]` and `delta_timestamps['observation.state'] = [0/fps]`.
- `lerobot/datasets/lerobot_dataset.py` `LeRobotDataset.__getitem__` delegates straight to `DatasetReader.get_item(idx)`.
- `lerobot/datasets/dataset_reader.py` `DatasetReader._get_query_indices(abs_idx, ep_idx)` computes, per key, `[max(ep_start, min(ep_end-1, abs_idx + delta)) for delta in delta_idx]` - for `action` this is exactly `[abs_idx+0, abs_idx+1, ..., abs_idx+chunk_size-1]`, clamped to the episode's own `[ep_start, ep_end)` range, with an `action_is_pad` mask marking any entry where `abs_idx+delta` fell outside that range (post-clamp duplicate).
- `lerobot/policies/smolvla/processor_smolvla.py` pre/post-processors only normalize/unnormalize + move device/dtype - no temporal shift anywhere in the processor pipeline.
- `lerobot/policies/smolvla/modeling_smolvla.py` `select_action()` extends the action queue with `actions.transpose(0,1)[:n_action_steps]` (chunk indices `0..n_action_steps-1` in order) and returns `self._queues[ACTION].popleft()` - i.e. the **first** dequeued action is chunk index 0, matching the training-target definition of chunk[0].

**Conclusion from code alone: chunk[0] is defined, at training-data-construction time, to equal `action(t)` itself (offset 0) - not a future action.** Sections 2-4 below verify this empirically against the real V2-clean dataset rather than trusting the reasoning above by itself.

## 2. Representative numeric trace (episode 0 / 17 / 34, frames 0/5/10/15/20/25)

| ep | t | abs_idx | ts(s) | state(t) SL/EF | raw action(t) SL/EF | target chunk[0] SL/EF | chunk0==action(t) | all 50 offsets exact | NN confirms t+k |
|---:|---:|---:|---:|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.000 | -95.43/96.40 | -95.16/94.59 | -95.16/94.59 | True | True | True |
| 0 | 5 | 5 | 0.167 | -95.43/96.40 | -95.16/94.59 | -95.16/94.59 | True | True | True |
| 0 | 10 | 10 | 0.333 | -95.43/96.40 | -95.16/94.59 | -95.16/94.59 | True | True | True |
| 0 | 15 | 15 | 0.500 | -95.43/96.40 | -95.16/94.59 | -95.16/94.59 | True | True | True |
| 0 | 20 | 20 | 0.667 | -95.43/96.40 | -93.76/93.80 | -93.76/93.80 | True | True | True |
| 0 | 25 | 25 | 0.833 | -91.47/94.90 | -86.64/90.20 | -86.64/90.20 | True | True | True |
| 17 | 0 | 5590 | 0.000 | -97.98/96.48 | -98.42/94.42 | -98.42/94.42 | True | True | True |
| 17 | 5 | 5595 | 0.167 | -97.98/96.48 | -98.42/94.42 | -98.42/94.42 | True | True | True |
| 17 | 10 | 5600 | 0.333 | -97.98/96.48 | -98.42/94.42 | -98.42/94.42 | True | True | True |
| 17 | 15 | 5605 | 0.500 | -97.98/96.48 | -98.42/94.42 | -98.42/94.42 | True | True | True |
| 17 | 20 | 5610 | 0.667 | -97.98/96.48 | -97.19/93.98 | -97.19/93.98 | True | True | True |
| 17 | 25 | 5615 | 0.833 | -94.55/95.43 | -90.15/90.81 | -90.15/90.81 | True | True | True |
| 34 | 0 | 11177 | 0.000 | -98.59/96.31 | -98.24/94.42 | -98.24/94.42 | True | True | True |
| 34 | 5 | 11182 | 0.167 | -98.59/96.31 | -98.24/94.42 | -98.24/94.42 | True | True | True |
| 34 | 10 | 11187 | 0.333 | -98.59/96.31 | -98.24/94.42 | -98.24/94.42 | True | True | True |
| 34 | 15 | 11192 | 0.500 | -98.59/96.31 | -98.15/94.42 | -98.15/94.42 | True | True | True |
| 34 | 20 | 11197 | 0.667 | -98.42/96.31 | -95.43/93.01 | -95.43/93.01 | True | True | True |
| 34 | 25 | 11202 | 0.833 | -92.70/94.46 | -87.16/88.97 | -87.16/88.97 | True | True | True |

Full per-joint numeric values: `representative_frames.csv`. Full per-chunk-index (k=0..chunk_size-1) detail for every representative frame: `chunk_index_detail.csv`.

## 3. Chunk[k] -> which raw frame, for a sample frame (episode 0, frame 10)

episode=0, t=10, chunk_size=50

| k | expected raw frame (t+k, clamped) | max\|diff\| vs expected (deg) | exact match | NN raw frame found | NN matches expected |
|---:|---:|---:|---|---:|---|
| 0 | 10 | 0.000000 | True | 0 | True |
| 1 | 11 | 0.000000 | True | 0 | True |
| 2 | 12 | 0.000000 | True | 0 | True |
| 5 | 15 | 0.000000 | True | 0 | True |
| 10 | 20 | 0.000000 | True | 20 | True |
| 20 | 30 | 0.000000 | True | 30 | True |
| 30 | 40 | 0.000000 | True | 40 | True |
| 40 | 50 | 0.000000 | True | 50 | True |
| 49 | 59 | 0.000000 | True | 59 | True |

## 4. Full-dataset (35 episodes) offset-law check

Method: DatasetReader._get_query_indices + DatasetReader._query_hf_dataset (lerobot.datasets.dataset_reader.DatasetReader - the same internals dataset[idx] calls), stride=5 frames per episode, all 35 episodes, all chunk_size=50 offsets per sampled frame

- frames sampled: **2310**, chunk entries checked: **115500**
- exact-match failures (`chunk[k] != raw_action[t+k]` beyond 1e-4 deg): **0**
- `action_is_pad` flag mismatches: **0**
- max\|diff\| observed anywhere: **0.000000 deg**
- **offset law holds exactly across the full dataset: True**

## 5. Episode-end boundary / padding behaviour

Episode 0 (length=329, chunk_size=50):

| frame t | n_pad entries in chunk | first pad chunk-index | pad entries == last raw action | chunk[0]==action(t) |
|---:|---:|---:|---|---|
| 324 | 45 | 5 | True | True |
| 325 | 46 | 4 | True | True |
| 326 | 47 | 3 | True | True |
| 327 | 48 | 2 | True | True |
| 328 | 49 | 1 | True | True |

Episode 17 (length=328, chunk_size=50):

| frame t | n_pad entries in chunk | first pad chunk-index | pad entries == last raw action | chunk[0]==action(t) |
|---:|---:|---:|---|---|
| 323 | 45 | 5 | True | True |
| 324 | 46 | 4 | True | True |
| 325 | 47 | 3 | True | True |
| 326 | 48 | 2 | True | True |
| 327 | 49 | 1 | True | True |

Episode 34 (length=328, chunk_size=50):

| frame t | n_pad entries in chunk | first pad chunk-index | pad entries == last raw action | chunk[0]==action(t) |
|---:|---:|---:|---|---|
| 323 | 45 | 5 | True | True |
| 324 | 46 | 4 | True | True |
| 325 | 47 | 3 | True | True |
| 326 | 48 | 2 | True | True |
| 327 | 49 | 1 | True | True |

Note: `action_delta_indices` only ever contains non-negative offsets (`0..chunk_size-1`), so **no clamping/padding can occur at episode start** - chunk[0] always equals `action(t)` regardless of how close `t` is to the episode's own start. Padding only occurs near the *end* of an episode, when `t + k` runs past the episode's last frame; those entries are clamped to (repeat) the episode's last raw action and flagged `action_is_pad=True`.

## 6. Live policy chunk[0] vs. training target chunk[0] (same observations)

Checkpoint: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model`, task=`Pick up the cube and place it in the target area.`, seeds=[0, 1, 2]

| ep | t | joint | state(t) | training target chunk[0] | target-state | policy chunk[0] (mean over seeds) | policy-state | policy-target |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | shoulder_lift | -95.43 | -95.16 | +0.26 | -90.17 | +5.26 | +5.00 |
| 0 | 0 | elbow_flex | 96.40 | 94.59 | -1.80 | 93.88 | -2.51 | -0.71 |
| 0 | 5 | shoulder_lift | -95.43 | -95.16 | +0.26 | -89.52 | +5.91 | +5.64 |
| 0 | 5 | elbow_flex | 96.40 | 94.59 | -1.80 | 93.35 | -3.04 | -1.24 |
| 0 | 10 | shoulder_lift | -95.43 | -95.16 | +0.26 | -90.05 | +5.38 | +5.12 |
| 0 | 10 | elbow_flex | 96.40 | 94.59 | -1.80 | 93.73 | -2.66 | -0.86 |
| 0 | 15 | shoulder_lift | -95.43 | -95.16 | +0.26 | -89.87 | +5.56 | +5.29 |
| 0 | 15 | elbow_flex | 96.40 | 94.59 | -1.80 | 93.58 | -2.82 | -1.02 |
| 0 | 20 | shoulder_lift | -95.43 | -93.76 | +1.67 | -90.06 | +5.37 | +3.70 |
| 0 | 20 | elbow_flex | 96.40 | 93.80 | -2.59 | 93.91 | -2.48 | +0.11 |
| 0 | 25 | shoulder_lift | -91.47 | -86.64 | +4.84 | -84.47 | +7.00 | +2.16 |
| 0 | 25 | elbow_flex | 94.90 | 90.20 | -4.70 | 89.45 | -5.45 | -0.75 |
| 17 | 0 | shoulder_lift | -97.98 | -98.42 | -0.44 | -91.54 | +6.43 | +6.87 |
| 17 | 0 | elbow_flex | 96.48 | 94.42 | -2.07 | 93.36 | -3.13 | -1.06 |
| 17 | 5 | shoulder_lift | -97.98 | -98.42 | -0.44 | -91.44 | +6.54 | +6.98 |
| 17 | 5 | elbow_flex | 96.48 | 94.42 | -2.07 | 93.48 | -3.00 | -0.93 |
| 17 | 10 | shoulder_lift | -97.98 | -98.42 | -0.44 | -91.39 | +6.59 | +7.03 |
| 17 | 10 | elbow_flex | 96.48 | 94.42 | -2.07 | 93.43 | -3.06 | -0.99 |
| 17 | 15 | shoulder_lift | -97.98 | -98.42 | -0.44 | -91.39 | +6.59 | +7.03 |
| 17 | 15 | elbow_flex | 96.48 | 94.42 | -2.07 | 93.65 | -2.83 | -0.76 |
| 17 | 20 | shoulder_lift | -97.98 | -97.19 | +0.79 | -91.49 | +6.49 | +5.70 |
| 17 | 20 | elbow_flex | 96.48 | 93.98 | -2.51 | 93.51 | -2.98 | -0.47 |
| 17 | 25 | shoulder_lift | -94.55 | -90.15 | +4.40 | -88.04 | +6.51 | +2.11 |
| 17 | 25 | elbow_flex | 95.43 | 90.81 | -4.62 | 91.50 | -3.93 | +0.69 |
| 34 | 0 | shoulder_lift | -98.59 | -98.24 | +0.35 | -91.34 | +7.25 | +6.90 |
| 34 | 0 | elbow_flex | 96.31 | 94.42 | -1.89 | 92.29 | -4.02 | -2.12 |
| 34 | 5 | shoulder_lift | -98.59 | -98.24 | +0.35 | -91.07 | +7.52 | +7.17 |
| 34 | 5 | elbow_flex | 96.31 | 94.42 | -1.89 | 91.74 | -4.57 | -2.68 |
| 34 | 10 | shoulder_lift | -98.59 | -98.24 | +0.35 | -90.52 | +8.07 | +7.72 |
| 34 | 10 | elbow_flex | 96.31 | 94.42 | -1.89 | 91.58 | -4.73 | -2.84 |
| 34 | 15 | shoulder_lift | -98.59 | -98.15 | +0.44 | -91.08 | +7.52 | +7.08 |
| 34 | 15 | elbow_flex | 96.31 | 94.42 | -1.89 | 91.97 | -4.33 | -2.44 |
| 34 | 20 | shoulder_lift | -98.42 | -95.43 | +2.99 | -89.71 | +8.71 | +5.72 |
| 34 | 20 | elbow_flex | 96.31 | 93.01 | -3.30 | 91.08 | -5.23 | -1.94 |
| 34 | 25 | shoulder_lift | -92.70 | -87.16 | +5.54 | -83.77 | +8.94 | +3.40 |
| 34 | 25 | elbow_flex | 94.46 | 88.97 | -5.49 | 87.42 | -7.04 | -1.55 |

`target-state` is the training-label's own immediate delta (what the model *should* learn to output as chunk[0] for this observation, per the offset law in section 1). `policy-state` is what the live checkpoint actually outputs as chunk[0]. `policy-target` is the direct discrepancy between the two, isolating whether any oversized first-action behaviour comes from the policy having learned something other than its own training target (large `policy-target`) vs. the training target itself already being large (large `target-state`, which section 1-4 rule out for this dataset).

Full per-seed values: `live_policy_vs_training_target.json`.

## 7. Answers

**A.** YES - empirically confirmed on 115500 chunk entries across all 35 episodes (section 4) and on every representative frame (section 2/3): `training target chunk[0] == raw dataset action(t)` exactly (max|diff| 0.00e+00 deg, i.e. float round-trip noise only).

**B.** NO future offset exists in the training-target construction itself - `action_delta_indices = range(chunk_size)` starts at 0, so chunk[0] is defined as the *current* frame's own action, not action(t+k) for any k>0.

**C.** N/A - offset is 0 (see A/B).

**D.** YES - chunk[1]=action(t+1), chunk[2]=action(t+2), ..., chunk[chunk_size-1]=action(t+chunk_size-1), confirmed both by exact value equality and by an independent nearest-neighbour search over each episode's full raw action trajectory (section 2/3: 'all 50 offsets exact' / 'NN confirms t+k' both True for every representative frame; section 4: 0 mismatches dataset-wide).

**E.** NO material shift at episode start: `action_delta_indices` only contains non-negative offsets (0..chunk_size-1), so clamping/padding is structurally impossible at t=0 or any other start-of-episode frame - chunk[0] always equals action(t) regardless of position in the episode. Padding *does* occur near the episode's *end* (see section 5): once t+k runs past the last frame, that chunk entry is clamped to (repeats) the episode's final raw action and is flagged `action_is_pad=True` - this is a well-defined, correctly-flagged boundary behaviour, not a silent misalignment.

**F.** `chunk_size`/`n_action_steps` set the *number* of future steps in the target/consumed window (both 50 for this checkpoint, i.e. 1.667s @ 30fps) but do not shift *where* offset-0 starts - `action_delta_indices = list(range(chunk_size))` always starts at 0 regardless of chunk_size. `delta_timestamps` (via `resolve_delta_timestamps`) is a direct, unmodified re-expression of those same indices in seconds (`k/fps`). The SmolVLA pre/post-processors (`processor_smolvla.py`) perform only normalization/unnormalization and device/dtype/tokenization steps - no temporal resampling, shifting, or delaying of the action target anywhere in that pipeline.

**G.** YES, consistent - `select_action()` (`modeling_smolvla.py`) extends its internal action queue with `actions.transpose(0,1)[:n_action_steps]`, i.e. chunk indices `0, 1, 2, ...` in that order, and pops from the *front* (`popleft()`) - so the very first action a fresh `select_action()` call returns is chunk index 0, matching the training-target definition of chunk[0] used above (built with a freshly-reset queue, one call per representative observation, so this comparison is apples-to-apples with training).

**6 (live checkpoint vs. training target).** See section 6: for each representative observation, the actually-loaded 7.5k checkpoint's delivered chunk[0] is reported side by side with that same observation's training target chunk[0] (== action(t) per A-E above). A large `policy-target` discrepancy with a small `target-state` (training label itself near-zero, matching the earlier start-segment audit) would point at the model (mis)learning something other than its own training target; a large `target-state` would instead point at the training data itself. Read the numbers in `live_policy_vs_training_target.json` / section 6 table before concluding either way - this script reports the comparison, it does not pre-judge which side is at fault.
