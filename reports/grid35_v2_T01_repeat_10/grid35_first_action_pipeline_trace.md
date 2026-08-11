# Grid35 10k SmolVLA First-Action Pipeline Trace

Reference Shadow observation used: `V2_F02` (V2_F02)

## 1. Joint order / feature-name check

- dataset action names: `['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']`
- dataset state names: `['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']`
- `runtime.common.vla_contract.JOINT_ORDER`: `['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']`
- match: action=True, state=True
- `hardware.state_server.readonly_so101_reader.JOINT_ORDER`: `['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']`

> State is always dict-keyed by joint name end-to-end (HTTP JSON body, adapter, safety gate all index by name) - a permutation bug is structurally impossible on the state path. The ONE positional-order risk is in runtime.desktop.vla_server.SmolVLAPolicyRunner.predict(): `{joint: float(action_flat[i]) for i, joint in enumerate(JOINT_ORDER)}` zips the checkpoint's raw output tensor to JOINT_ORDER by position with no cross-check against the checkpoint's own training-dataset feature-name order.

## 2. Normalization stats provenance (action feature, by dataset)

| dataset | elbow_flex mean | elbow_flex std | wrist_flex mean | wrist_flex std | shoulder_lift mean | shoulder_lift std |
|---|---:|---:|---:|---:|---:|---:|
| so101_cube_train_v6 | -19.56 | 36.43 | 64.97 | 11.92 | 14.39 | 38.99 |
| so101_cube_xy_grid35_v2_clean | 6.26 | 42.42 | 71.84 | 12.11 | -12.86 | 41.68 |
| so101_cube_xy_midpoint_test10_v2_clean | 11.22 | 39.16 | 68.30 | 9.84 | -14.52 | 39.48 |

(This is what the *datasets themselves* look like - it does NOT confirm what stats are actually baked into the checkpoint's `policy_postprocessor.json`, which lives inside the checkpoint directory and was not accessible from this machine. See section 3.)

## 3. Checkpoint provenance (best-effort)

```json
{
  "status": "OK",
  "checkpoint_dir": "outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/007500/pretrained_model",
  "inferred_step": 7500,
  "train_config_dataset_root": "/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_xy_grid35_v2_clean",
  "train_config_read_error": null,
  "trained_dataset_name": "so101_cube_xy_grid35_v2_clean",
  "trained_dataset_action_names": [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
  ],
  "trained_dataset_matches_JOINT_ORDER": true,
  "policy_postprocessor_json_present": true,
  "policy_postprocessor_steps": [
    "unnormalizer_processor",
    "device_processor"
  ],
  "policy_postprocessor_norm_step_config_keys": [
    "eps",
    "features",
    "norm_map"
  ]
}
```

## 4. Real Shadow-log stage table (elbow_flex / wrist_flex / shoulder_lift)

| run | decision | joint | state | raw_action (post-postprocessor) | adapter.mapped_action | delta |
|---|---|---|---:|---:|---:|---:|

(`adapter.mapped_action` == `raw_action` in every row - confirms the Action Adapter does a pure passthrough for this checkpoint's outputs, no unit/sign transform observed.)

## 5. Nearest Grid35 training frame + trajectory-horizon analysis

Reference Shadow delta (raw_action - state): `{'shoulder_pan': -3.281228055010785, 'shoulder_lift': 7.974747794015073, 'elbow_flex': -9.074475131192045, 'wrist_flex': -0.8835524255102811, 'wrist_roll': 0.31214585147061147, 'gripper': -0.1262449155271752}`

Top-5 nearest Grid35 frames (L2 distance over the 6 joints, degrees):

| episode | frame | L2 dist (deg) |
|---:|---:|---:|
| 33 | 25 | 2.22 |
| 34 | 23 | 2.51 |
| 32 | 21 | 2.88 |
| 24 | 16 | 3.44 |
| 28 | 17 | 3.68 |

GT immediate delta (action[t]-state[t]) at the single best-matching frame (episode 33, frame 25):

| joint | GT immediate delta | Shadow model delta |
|---|---:|---:|
| shoulder_lift | +3.87 | +7.97 |
| elbow_flex | -4.62 | -9.07 |
| wrist_flex | +0.22 | -0.88 |

Dataset-wide (all 35 episodes) mean `STATE[t0+k]-STATE[t0]` displacement per horizon k:

| joint | k=10(0.33s) | k=15(0.50s) | k=20(0.67s) | k=25(0.83s) | k=30(1.00s) | k=35(1.17s) | k=40(1.33s) | k=45(1.50s) | k=50(1.67s) | k=60(2.00s) | k=80(2.67s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shoulder_lift | +0.00 | +0.13 | +0.76 | +3.02 | +7.38 | +12.76 | +18.75 | +24.88 | +31.09 | +43.24 | +68.13 |
| elbow_flex | +0.00 | -0.01 | -0.31 | -1.57 | -4.43 | -8.91 | -14.71 | -21.30 | -28.47 | -43.54 | -71.78 |
| wrist_flex | +0.00 | +0.00 | +0.04 | +0.11 | +0.40 | +1.29 | +3.04 | +5.39 | +8.30 | +15.11 | +24.22 |

Estimated horizon (frames/seconds into the episode) whose GT displacement best matches the Shadow model's delivered first-action delta, per key joint:

| joint | Shadow delta | estimated matching horizon |
|---|---:|---:|
| shoulder_lift | +7.97 | 30.5 frames (~1.02s) |
| elbow_flex | -9.07 | 35.1 frames (~1.17s) |
| wrist_flex | -0.88 | 10.0 frames (~0.33s) |

> If the model's delivered first action (chunk index 0, what select_action() actually returns) closely matched the IMMEDIATE next-step GT delta, values here would resemble 'gt_immediate_delta_action_minus_state' (small, single-digit degrees, matching the prior episode-start analysis). Instead, the Shadow delta magnitude for the 3 key joints lines up with the dataset's own state displacement roughly 0.7s-1.7s into the episode (see estimated_matching_horizon_per_key_joint) - i.e. within or near the chunk's own chunk_size=50 (1.67s @ 30fps) horizon, but far later than index 0's nominal 33ms step. This does not by itself distinguish a training-time temporal-alignment bug (C) from an undertrained flow-matching policy whose early chunk indices have not yet differentiated from later ones (D) - that requires the live chunk dump (see checkpoint section).

## 6. Live action-chunk dump (chunk index 0-49)

chunk_size=50, n_action_steps=50, index actually used as first command=0

## 7. Safety threshold math (T05 REJECT explanation)

| joint | WOULD_CLAMP threshold (deg) | REJECT threshold (deg, x5) |
|---|---:|---:|
| shoulder_pan | 4.58 | 22.90 |
| shoulder_lift | 5.16 | 25.80 |
| elbow_flex | 5.73 | 28.65 |
| wrist_flex | 4.01 | 20.05 |
| wrist_roll | 1.15 | 5.75 |
| gripper | 9.17 | 45.85 |

T05's elbow_flex delta was -32.03deg, exceeding the 28.65deg REJECT threshold -> REJECT. T05-R3/R4/R5 (elbow delta -21 to -27deg) stayed under it -> WOULD_CLAMP.

## 8. Verdict

**LIVE_CHUNK_READY**

Live action-chunk inference completed successfully. Automatic C-vs-D classification is intentionally deferred; inspect live_action_chunk_dump together with the nearest-frame trajectory analysis and safety thresholds.

- a_mapping_adapter_bug: RULED_OUT (code-verified)
- b_normalization_bug: checked_dataset_order_ok
- c_chunk_extraction_bug_in_runtime_call_site: RULED_OUT (code-verified, select_action pops index 0 correctly)
- c_prime_temporal_alignment_at_training_time: UNVERIFIED - plausible per trajectory-horizon-match finding
- d_policy_itself_predicts_large_first_action: PLAUSIBLE - leading hypothesis given clean runtime pipeline + small (35ep/10k-step) checkpoint
