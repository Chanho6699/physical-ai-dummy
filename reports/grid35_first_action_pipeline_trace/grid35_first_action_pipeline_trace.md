# Grid35 10k SmolVLA First-Action Pipeline Trace

Reference Shadow observation used: `T05` (shadow_20260808_152102.json)

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
| so101_cube_smoke_v1 | 8.58 | 51.74 | 66.62 | 18.02 | -15.13 | 47.79 |
| so101_cube_train_v1 | 27.45 | 58.09 | 77.18 | 15.96 | -42.60 | 55.29 |
| so101_cube_train_v2 | -26.70 | 47.53 | 88.17 | 15.52 | 11.71 | 46.04 |
| so101_cube_train_v3 | -31.52 | 43.76 | 88.55 | 13.69 | 15.25 | 43.51 |
| so101_cube_train_v4 | -31.05 | 45.98 | 84.29 | 12.84 | 15.37 | 45.02 |
| so101_cube_train_v5 | -19.67 | 41.68 | 72.62 | 17.42 | 10.61 | 42.10 |
| so101_cube_train_v6 | -19.56 | 36.43 | 64.97 | 11.92 | 14.39 | 38.99 |
| so101_cube_trials_v1 | 52.07 | 58.37 | 64.55 | 17.00 | -61.95 | 53.93 |
| so101_cube_xy_grid35_v1 | -5.85 | 42.76 | 76.19 | 14.00 | -2.21 | 41.43 |
| so101_cube_xy_grid35_v1_backup_29ep | -9.32 | 42.96 | 75.30 | 13.79 | 0.60 | 41.70 |
| so101_cube_xy_midpoint_test10_v1 | 0.87 | 39.32 | 76.53 | 13.30 | -6.38 | 38.95 |
| so101_cube_xy_train_v1 | -27.23 | 45.05 | 83.41 | 16.30 | 13.23 | 44.24 |

(This is what the *datasets themselves* look like - it does NOT confirm what stats are actually baked into the checkpoint's `policy_postprocessor.json`, which lives inside the checkpoint directory and was not accessible from this machine. See section 3.)

## 3. Checkpoint provenance (best-effort)

```json
{
  "status": "SKIPPED",
  "reason": "--checkpoint not provided"
}
```

## 4. Real Shadow-log stage table (elbow_flex / wrist_flex / shoulder_lift)

| run | decision | joint | state | raw_action (post-postprocessor) | adapter.mapped_action | delta |
|---|---|---|---:|---:|---:|---:|
| F01 | WOULD_CLAMP | shoulder_lift | -99.56 | -81.16 | -81.16 | +18.40 |
| F01 | WOULD_CLAMP | elbow_flex | 96.84 | 69.97 | 69.97 | -26.87 |
| F01 | WOULD_CLAMP | wrist_flex | 55.43 | 63.65 | 63.65 | +8.22 |
| F02 | WOULD_CLAMP | shoulder_lift | -99.56 | -76.74 | -76.74 | +22.82 |
| F02 | WOULD_CLAMP | elbow_flex | 96.84 | 68.26 | 68.26 | -28.57 |
| F02 | WOULD_CLAMP | wrist_flex | 55.43 | 64.85 | 64.85 | +9.42 |
| F03 | WOULD_CLAMP | shoulder_lift | -99.56 | -86.95 | -86.95 | +12.61 |
| F03 | WOULD_CLAMP | elbow_flex | 96.84 | 76.00 | 76.00 | -20.84 |
| F03 | WOULD_CLAMP | wrist_flex | 55.43 | 61.67 | 61.67 | +6.24 |
| F04 | WOULD_CLAMP | shoulder_lift | -99.56 | -89.94 | -89.94 | +9.62 |
| F04 | WOULD_CLAMP | elbow_flex | 96.84 | 73.74 | 73.74 | -23.09 |
| F04 | WOULD_CLAMP | wrist_flex | 55.43 | 64.11 | 64.11 | +8.68 |
| F05 | WOULD_CLAMP | shoulder_lift | -99.56 | -82.84 | -82.84 | +16.72 |
| F05 | WOULD_CLAMP | elbow_flex | 96.84 | 71.88 | 71.88 | -24.95 |
| F05 | WOULD_CLAMP | wrist_flex | 55.43 | 62.91 | 62.91 | +7.48 |
| T01 | WOULD_CLAMP | shoulder_lift | -99.56 | -83.10 | -83.10 | +16.46 |
| T01 | WOULD_CLAMP | elbow_flex | 96.84 | 70.17 | 70.17 | -26.66 |
| T01 | WOULD_CLAMP | wrist_flex | 55.43 | 64.88 | 64.88 | +9.45 |
| T05 | REJECT | shoulder_lift | -99.56 | -80.25 | -80.25 | +19.31 |
| T05 | REJECT | elbow_flex | 96.84 | 64.81 | 64.81 | -32.03 |
| T05 | REJECT | wrist_flex | 55.43 | 63.79 | 63.79 | +8.36 |
| T08 | WOULD_CLAMP | shoulder_lift | -99.56 | -83.83 | -83.83 | +15.73 |
| T08 | WOULD_CLAMP | elbow_flex | 96.84 | 75.19 | 75.19 | -21.65 |
| T08 | WOULD_CLAMP | wrist_flex | 55.43 | 64.88 | 64.88 | +9.45 |
| T10 | WOULD_CLAMP | shoulder_lift | -99.56 | -84.36 | -84.36 | +15.20 |
| T10 | WOULD_CLAMP | elbow_flex | 96.84 | 72.24 | 72.24 | -24.60 |
| T10 | WOULD_CLAMP | wrist_flex | 55.43 | 63.11 | 63.11 | +7.68 |
| T05-R2 | REJECT | shoulder_lift | -99.56 | -75.79 | -75.79 | +23.77 |
| T05-R2 | REJECT | elbow_flex | 96.84 | 64.78 | 64.78 | -32.06 |
| T05-R2 | REJECT | wrist_flex | 55.43 | 64.85 | 64.85 | +9.43 |
| T05-R3 | WOULD_CLAMP | shoulder_lift | -99.56 | -82.67 | -82.67 | +16.89 |
| T05-R3 | WOULD_CLAMP | elbow_flex | 96.84 | 75.44 | 75.44 | -21.39 |
| T05-R3 | WOULD_CLAMP | wrist_flex | 55.43 | 62.32 | 62.32 | +6.89 |
| T05-R4 | WOULD_CLAMP | shoulder_lift | -99.56 | -76.89 | -76.89 | +22.67 |
| T05-R4 | WOULD_CLAMP | elbow_flex | 96.84 | 70.14 | 70.14 | -26.70 |
| T05-R4 | WOULD_CLAMP | wrist_flex | 55.43 | 62.05 | 62.05 | +6.62 |
| T05-R5 | WOULD_CLAMP | shoulder_lift | -99.56 | -81.84 | -81.84 | +17.73 |
| T05-R5 | WOULD_CLAMP | elbow_flex | 96.84 | 75.51 | 75.51 | -21.33 |
| T05-R5 | WOULD_CLAMP | wrist_flex | 55.43 | 62.19 | 62.19 | +6.76 |

(`adapter.mapped_action` == `raw_action` in every row - confirms the Action Adapter does a pure passthrough for this checkpoint's outputs, no unit/sign transform observed.)

## 5. Nearest Grid35 training frame + trajectory-horizon analysis

Reference Shadow delta (raw_action - state): `{'shoulder_pan': -5.963317567175562, 'shoulder_lift': 19.313193771865343, 'elbow_flex': -32.02935581416874, 'wrist_flex': 8.364794049944194, 'wrist_roll': -0.8952968762471125, 'gripper': 4.85760498046875}`

Top-5 nearest Grid35 frames (L2 distance over the 6 joints, degrees):

| episode | frame | L2 dist (deg) |
|---:|---:|---:|
| 15 | 5 | 3.75 |
| 24 | 4 | 3.90 |
| 14 | 4 | 4.06 |
| 25 | 0 | 4.07 |
| 1 | 5 | 4.28 |

GT immediate delta (action[t]-state[t]) at the single best-matching frame (episode 15, frame 5):

| joint | GT immediate delta | Shadow model delta |
|---|---:|---:|
| shoulder_lift | +2.20 | +19.31 |
| elbow_flex | -1.63 | -32.03 |
| wrist_flex | -5.67 | +8.36 |

Dataset-wide (all 35 episodes) mean `STATE[t0+k]-STATE[t0]` displacement per horizon k:

| joint | k=10(0.33s) | k=15(0.50s) | k=20(0.67s) | k=25(0.83s) | k=30(1.00s) | k=35(1.17s) | k=40(1.33s) | k=45(1.50s) | k=50(1.67s) | k=60(2.00s) | k=80(2.67s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shoulder_lift | +7.66 | +8.34 | +12.49 | +20.47 | +29.88 | +39.61 | +49.35 | +58.80 | +67.93 | +85.00 | +112.80 |
| elbow_flex | -0.45 | -0.75 | -4.50 | -11.09 | -19.52 | -29.30 | -39.83 | -50.44 | -60.71 | -77.63 | -102.08 |
| wrist_flex | -6.72 | -6.83 | -6.83 | -6.79 | -6.40 | -4.78 | -1.39 | +3.10 | +7.78 | +14.51 | +21.88 |

Estimated horizon (frames/seconds into the episode) whose GT displacement best matches the Shadow model's delivered first-action delta, per key joint:

| joint | Shadow delta | estimated matching horizon |
|---|---:|---:|
| shoulder_lift | +19.31 | 24.3 frames (~0.81s) |
| elbow_flex | -32.03 | 36.3 frames (~1.21s) |
| wrist_flex | +8.36 | 50.9 frames (~1.70s) |

> If the model's delivered first action (chunk index 0, what select_action() actually returns) closely matched the IMMEDIATE next-step GT delta, values here would resemble 'gt_immediate_delta_action_minus_state' (small, single-digit degrees, matching the prior episode-start analysis). Instead, the Shadow delta magnitude for the 3 key joints lines up with the dataset's own state displacement roughly 0.7s-1.7s into the episode (see estimated_matching_horizon_per_key_joint) - i.e. within or near the chunk's own chunk_size=50 (1.67s @ 30fps) horizon, but far later than index 0's nominal 33ms step. This does not by itself distinguish a training-time temporal-alignment bug (C) from an undertrained flow-matching policy whose early chunk indices have not yet differentiated from later ones (D) - that requires the live chunk dump (see checkpoint section).

## 6. Live action-chunk dump (chunk index 0-49)

**SKIPPED**: No usable --checkpoint directory. This sandbox has neither the checkpoint files nor a reachable VLA HTTP server (checked: filesystem search under $HOME for 'pretrained_model'/'train_config.json'/'*smolvla_grid35*' found nothing; no VLA_SERVER_URL set; localhost:9200/health unreachable). Re-run this script with --checkpoint pointing at the real outputs/grid35/smolvla_grid35_fresh_v1/checkpoints/010000/pretrained_model directory (e.g. on the Desktop machine) to get the actual chunk_index 0-49 dump.

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

**E**

Code review rules out A (adapter/client are pure passthrough, all state/action wiring is name-keyed not positional, the only positional-order risk - checkpoint raw tensor -> JOINT_ORDER - is architecturally low-risk since every dataset in this repo declares the same joint order) with high confidence, and rules out a bug in the RUNTIME chunk-extraction call site (select_action() correctly pops chunk index 0 per LeRobot's own API contract) with high confidence. B (normalization/denormalization) could not be verified - the checkpoint's baked-in policy_postprocessor.json is not present on this machine, and no VLA HTTP server was reachable, so its embedded stats couldn't be inspected. The strongest concrete finding is a dataset-wide, reproducible match between the delivered first-action delta and the Grid35 dataset's own state displacement roughly 0.7-1.7s into the episode (see nearest_frame_and_trajectory_analysis) for all 3 key joints - consistent with either a training-time temporal-alignment bug (C, unverifiable further without checkpoint/training config access) or an undertrained flow-matching policy whose early chunk indices have not differentiated from later ones (D, plausible given only 35 episodes / 10k training steps). Distinguishing C from D requires the live action-chunk dump (chunk index 0-49), which this script performs automatically once pointed at a real checkpoint via --checkpoint - it was SKIPPED here because no checkpoint files or reachable VLA server exist on this machine.

- a_mapping_adapter_bug: RULED_OUT (code-verified)
- b_normalization_bug: UNVERIFIED
- c_chunk_extraction_bug_in_runtime_call_site: RULED_OUT (code-verified, select_action pops index 0 correctly)
- c_prime_temporal_alignment_at_training_time: UNVERIFIED - plausible per trajectory-horizon-match finding
- d_policy_itself_predicts_large_first_action: PLAUSIBLE - leading hypothesis given clean runtime pipeline + small (35ep/10k-step) checkpoint
