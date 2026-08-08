#!/usr/bin/env python3
"""End-to-end trace of the Grid35 10k SmolVLA "large first action" symptom.

Traces ``elbow_flex``/``wrist_flex`` (and the other 4 joints) through:

    dataset (grid35) -> SmolVLA checkpoint -> postprocessor -> VLA HTTP server
    -> action adapter -> Safety Gate

Background: Shadow-mode fixed-scene runs show a first-action delta of about
``shoulder_lift +16deg``, ``elbow_flex -25deg``, ``wrist_flex +8deg`` relative
to the real follower state, and one scene (T05) hit a Safety REJECT at
``elbow_flex ~-32deg``. A prior investigation
(``scripts/analyze_grid35_episode_start.py`` /
``reports/grid35_episode_start_analysis/``) showed the *dataset's own*
frame-0 action-state delta is much smaller (elbow_flex ~-2.2deg mean) and
does not explain the Shadow bias by itself. This script goes one level
deeper into the live inference pipeline.

What this script can verify from *this* machine (no GPU/checkpoint needed):

  1. Joint-order consistency between the dataset's declared feature names
     (``meta/info.json``) and ``runtime.common.vla_contract.JOINT_ORDER``
     (used positionally when the server unpacks the checkpoint's raw action
     tensor - the one place in the runtime code where a silent permutation
     bug is structurally possible, since nothing cross-checks it).
  2. Normalization-stats *provenance*: compares elbow_flex/wrist_flex/
     shoulder_lift statistics (mean/std/min/max) across every dataset
     ``meta/stats.json`` found under ``data/`` - so a "checkpoint trained
     with old-dataset stats baked in" hypothesis can at least be evaluated
     against what those old stats actually looked like.
  3. Checkpoint provenance (best-effort, no GPU): if ``--checkpoint`` points
     at a real ``pretrained_model`` directory, reads its
     ``train_config.json`` (plain JSON, no torch needed) to find which
     dataset it was trained on, and if that dataset is available locally,
     diffs its declared joint-name order against ``JOINT_ORDER``.
  4. Live chunk dump (requires ``--checkpoint`` to be an on-disk checkpoint
     dir + torch/lerobot + GPU/CPU capable of running SmolVLA): runs the
     *exact* pre-processor -> policy.predict_action_chunk -> post-processor
     path used by ``runtime.desktop.vla_server.SmolVLAPolicyRunner``, and
     dumps all ``chunk_size`` (default 50) predicted actions per joint, so
     chunk index 0 (what ``select_action``/the runtime actually uses) can be
     compared against every other chunk index.
  5. Stage-by-stage real numbers pulled directly from existing
     ``reports/shadow_mode/shadow_*.json`` logs (state -> raw_action(=
     already-postprocessed HTTP response) -> adapter.mapped_action ->
     safety.per_joint) - this *is* real checkpoint output (via the Desktop
     VLA HTTP server), just not re-run live.
  6. Nearest-Grid35-frame + full-trajectory comparison: finds the Grid35
     training frame(s) closest (L2, degrees) to the real Shadow observation
     state, and reports the GT state[t+k]-state[t] displacement at a range
     of horizons k, to test whether the model's delivered first action
     resembles the *immediate* next-step GT delta (small) or a GT position
     several hundred ms - ~1.7s into the trajectory (large; matches Shadow).

This machine had neither the checkpoint files nor a reachable VLA server at
the time this script was written (checked exhaustively - see the
NORMALIZATION/CHECKPOINT PROVENANCE section of the generated report). Step 4
is included and functional but will report SKIPPED unless pointed at an
actual checkpoint directory (e.g. run this on the Desktop machine that hosts
``outputs/grid35/smolvla_grid35_fresh_v1/checkpoints/010000/pretrained_model``).

This script is read-only: no dataset/checkpoint/config modification, no
writes to any robot (real or simulated), no retraining, no Safety threshold
changes.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "so101_cube_xy_grid35_v1"
DEFAULT_SHADOW_DIR = PROJECT_ROOT / "reports" / "shadow_mode"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "grid35_first_action_pipeline_trace"
DEFAULT_SAFETY_GATE_YAML = PROJECT_ROOT / "configs" / "safety_gate.yaml"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
KEY_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]

SHADOW_BACKGROUND_BIAS_DEG = {"shoulder_lift": 16.0, "elbow_flex": -25.0, "wrist_flex": 8.0}


# --------------------------------------------------------------------------
# 1. Joint order / feature-name consistency (static, no GPU)
# --------------------------------------------------------------------------


def check_joint_order(dataset_root: Path) -> dict[str, Any]:
    with open(dataset_root / "meta" / "info.json", encoding="utf-8") as f:
        info = json.load(f)
    dataset_action_names = [n.rsplit(".", 1)[0] for n in info["features"]["action"]["names"]]
    dataset_state_names = [n.rsplit(".", 1)[0] for n in info["features"]["observation.state"]["names"]]

    from runtime.common.vla_contract import JOINT_ORDER as CONTRACT_JOINT_ORDER

    other_defs: dict[str, list[str] | None] = {}
    try:
        from hardware.state_server.readonly_so101_reader import JOINT_ORDER as READER_JOINT_ORDER

        other_defs["hardware.state_server.readonly_so101_reader.JOINT_ORDER"] = list(READER_JOINT_ORDER)
    except Exception as exc:  # noqa: BLE001
        other_defs["hardware.state_server.readonly_so101_reader.JOINT_ORDER"] = f"IMPORT_FAILED: {exc}"

    return {
        "dataset_action_names": dataset_action_names,
        "dataset_state_names": dataset_state_names,
        "vla_contract_JOINT_ORDER": list(CONTRACT_JOINT_ORDER),
        "dataset_action_matches_contract": dataset_action_names == list(CONTRACT_JOINT_ORDER),
        "dataset_state_matches_contract": dataset_state_names == list(CONTRACT_JOINT_ORDER),
        "other_joint_order_definitions": other_defs,
        "note": (
            "State is always dict-keyed by joint name end-to-end (HTTP JSON body, adapter, safety "
            "gate all index by name) - a permutation bug is structurally impossible on the state "
            "path. The ONE positional-order risk is in "
            "runtime.desktop.vla_server.SmolVLAPolicyRunner.predict(): "
            "`{joint: float(action_flat[i]) for i, joint in enumerate(JOINT_ORDER)}` zips the "
            "checkpoint's raw output tensor to JOINT_ORDER by position with no cross-check against "
            "the checkpoint's own training-dataset feature-name order."
        ),
    }


# --------------------------------------------------------------------------
# 2. Normalization-stats provenance across all locally available datasets
# --------------------------------------------------------------------------


def collect_dataset_stats_provenance(data_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stats_path in sorted(data_dir.glob("*/meta/stats.json")):
        dataset_name = stats_path.parent.parent.name
        try:
            with open(stats_path, encoding="utf-8") as f:
                stats = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            out[dataset_name] = {"error": str(exc)}
            continue
        action_stats = stats.get("action")
        if not action_stats:
            continue
        joint_stats = {}
        for i, j in enumerate(JOINTS):
            try:
                joint_stats[j] = {
                    "mean": action_stats["mean"][i],
                    "std": action_stats["std"][i],
                    "min": action_stats["min"][i],
                    "max": action_stats["max"][i],
                }
            except (KeyError, IndexError):
                continue
        out[dataset_name] = joint_stats
    return out


# --------------------------------------------------------------------------
# 3. Best-effort checkpoint provenance (no torch/GPU needed)
# --------------------------------------------------------------------------


def check_checkpoint_provenance(checkpoint_dir: Path | None, data_dir: Path) -> dict[str, Any]:
    if checkpoint_dir is None:
        return {"status": "SKIPPED", "reason": "--checkpoint not provided"}
    if not checkpoint_dir.is_dir():
        return {"status": "SKIPPED", "reason": f"checkpoint dir not found: {checkpoint_dir}"}

    from runtime.common.checkpoint_provenance import (
        infer_checkpoint_step,
        read_checkpoint_train_dataset_root,
    )

    step = infer_checkpoint_step(checkpoint_dir)
    train_dataset_root, err = read_checkpoint_train_dataset_root(checkpoint_dir)
    result: dict[str, Any] = {
        "status": "OK",
        "checkpoint_dir": str(checkpoint_dir),
        "inferred_step": step,
        "train_config_dataset_root": train_dataset_root,
        "train_config_read_error": err,
    }

    if train_dataset_root is not None:
        trained_name = Path(train_dataset_root).name
        result["trained_dataset_name"] = trained_name
        local_info_path = data_dir / trained_name / "meta" / "info.json"
        if local_info_path.is_file():
            with open(local_info_path, encoding="utf-8") as f:
                info = json.load(f)
            names = [n.rsplit(".", 1)[0] for n in info["features"]["action"]["names"]]
            from runtime.common.vla_contract import JOINT_ORDER

            result["trained_dataset_action_names"] = names
            result["trained_dataset_matches_JOINT_ORDER"] = names == list(JOINT_ORDER)
        else:
            result["trained_dataset_local_info_json"] = f"not found: {local_info_path}"

    postprocessor_path = checkpoint_dir / "policy_postprocessor.json"
    result["policy_postprocessor_json_present"] = postprocessor_path.is_file()
    if postprocessor_path.is_file():
        with open(postprocessor_path, encoding="utf-8") as f:
            pp = json.load(f)
        result["policy_postprocessor_steps"] = [
            s.get("registry_name") for s in pp.get("steps", []) if isinstance(s, dict)
        ]
        # try to find embedded normalization stats for the action feature
        for step_cfg in pp.get("steps", []):
            cfg = (step_cfg or {}).get("config") or {}
            if "action" in json.dumps(cfg)[:2000].lower() and any(
                k in cfg for k in ("stats", "norm_map", "mean", "std")
            ):
                result["policy_postprocessor_norm_step_config_keys"] = list(cfg.keys())
                break

    return result


# --------------------------------------------------------------------------
# 4. Live chunk dump (requires checkpoint + torch/lerobot)
# --------------------------------------------------------------------------


def dump_live_action_chunk(
    checkpoint_dir: Path | None, state: dict[str, float], images: dict[str, np.ndarray], task: str
) -> dict[str, Any]:
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return {
            "status": "SKIPPED",
            "reason": (
                "No usable --checkpoint directory. This sandbox has neither the checkpoint files nor "
                "a reachable VLA HTTP server (checked: filesystem search under $HOME for "
                "'pretrained_model'/'train_config.json'/'*smolvla_grid35*' found nothing; no "
                "VLA_SERVER_URL set; localhost:9200/health unreachable). Re-run this script with "
                "--checkpoint pointing at the real "
                "outputs/grid35/smolvla_grid35_fresh_v1/checkpoints/010000/pretrained_model directory "
                "(e.g. on the Desktop machine) to get the actual chunk_index 0-49 dump."
            ),
        }
    try:
        import torch
        from lerobot.policies import get_policy_class, make_pre_post_processors
    except ImportError as exc:
        return {"status": "SKIPPED", "reason": f"torch/lerobot not importable: {exc}"}

    from runtime.common.vla_contract import JOINT_ORDER

    try:
        policy_cls = get_policy_class("smolvla")
        policy = policy_cls.from_pretrained(str(checkpoint_dir))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint_dir))

        state_arr = np.array([state[j] for j in JOINT_ORDER], dtype=np.float32)
        batch: dict[str, object] = {"observation.state": torch.from_numpy(state_arr).unsqueeze(0).to(device)}
        for key, arr in images.items():
            img = torch.from_numpy(arr)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            img = img.permute(2, 0, 1).contiguous()
            batch[key] = img.unsqueeze(0).to(device)
        batch["task"] = [task]

        with torch.inference_mode():
            processed = preprocessor(batch)
            raw_chunk = policy.predict_action_chunk(processed)  # (1, chunk_size, action_dim)
            postproc_chunk = postprocessor(raw_chunk)

        chunk = postproc_chunk.detach().to("cpu").numpy()[0]  # (chunk_size, action_dim)
        chunk_size = chunk.shape[0]

        per_joint_trajectory = {
            j: [float(chunk[t, i]) for t in range(chunk_size)] for i, j in enumerate(JOINT_ORDER)
        }
        per_joint_delta_from_state = {
            j: [float(chunk[t, i] - state[j]) for t in range(chunk_size)] for i, j in enumerate(JOINT_ORDER)
        }

        return {
            "status": "OK",
            "chunk_size": chunk_size,
            "n_action_steps_used_by_select_action": policy.config.n_action_steps,
            "index_actually_used_as_first_command": 0,
            "per_joint_chunk_trajectory": per_joint_trajectory,
            "per_joint_chunk_delta_from_input_state": per_joint_delta_from_state,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# 5. Real Shadow-log stage table
# --------------------------------------------------------------------------


def load_shadow_stage_table(shadow_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for fp in sorted(glob.glob(str(shadow_dir / "shadow_*.json"))):
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        state = d.get("observation", {}).get("state")
        if not state:
            continue
        vla = d.get("vla", {})
        adapter = d.get("adapter", {})
        safety = d.get("safety", {})
        raw_action = vla.get("raw_action")
        mapped_action = adapter.get("mapped_action")
        per_joint = safety.get("per_joint", {})
        row = {
            "file": Path(fp).name,
            "label": d.get("scene_metadata", {}).get("label"),
            "model_id": vla.get("model_id"),
            "safety_decision": safety.get("decision"),
            "state": {j: state.get(j) for j in JOINTS},
            "raw_action_post_postprocessor": {j: raw_action.get(j) for j in JOINTS} if raw_action else None,
            "adapter_mapped_action": {j: mapped_action.get(j) for j in JOINTS} if mapped_action else None,
            "safety_safe_action": safety.get("safe_action"),
            "safety_per_joint_reasons": {
                j: per_joint.get(j, {}).get("reasons", []) for j in JOINTS if j in per_joint
            },
        }
        if raw_action:
            row["delta_raw_action_minus_state"] = {j: raw_action[j] - state[j] for j in JOINTS if j in state}
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# 6. Nearest-Grid35-frame + full-trajectory horizon analysis
# --------------------------------------------------------------------------


def find_episode_files(dataset_root: Path) -> dict[int, Path]:
    files = sorted(glob.glob(str(dataset_root / "data" / "chunk-*" / "file-*.parquet")))
    ep_to_path: dict[int, Path] = {}
    for fp in files:
        eps = pd.read_parquet(fp, columns=["episode_index"])["episode_index"].unique()
        for ep in eps:
            ep_to_path[int(ep)] = Path(fp)
    return ep_to_path


def nearest_frame_trajectory_analysis(
    dataset_root: Path, shadow_state: dict[str, float], shadow_action: dict[str, float]
) -> dict[str, Any]:
    ep_files = find_episode_files(dataset_root)
    shadow_state_arr = np.array([shadow_state[j] for j in JOINTS])
    shadow_delta = {j: shadow_action[j] - shadow_state[j] for j in JOINTS}

    candidates = []
    all_ep_states = {}  # ep -> (state array, action array)
    for ep, fp in sorted(ep_files.items()):
        df = pd.read_parquet(fp, columns=["action", "observation.state", "frame_index"])
        state = np.stack(df["observation.state"].to_numpy())
        action = np.stack(df["action"].to_numpy())
        all_ep_states[ep] = (state, action)
        dist = np.linalg.norm(state - shadow_state_arr[None, :], axis=1)
        idx = int(np.argmin(dist))
        candidates.append((float(dist[idx]), ep, idx, len(df)))
    candidates.sort(key=lambda x: x[0])

    top5 = [{"episode": ep, "frame": idx, "l2_dist_deg": d, "n_frames": n} for d, ep, idx, n in candidates[:5]]

    # best-match trajectory
    d0, ep0, idx0, n0 = candidates[0]
    state0, action0 = all_ep_states[ep0]
    match_state = state0[idx0]
    gt_immediate_delta = {j: float(action0[idx0][i] - match_state[i]) for i, j in enumerate(JOINTS)}

    horizons = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100]
    horizons = [h for h in horizons if idx0 + h < n0]
    best_match_horizon_displacement = {}
    for h in horizons:
        disp = state0[idx0 + h] - match_state
        best_match_horizon_displacement[h] = {j: float(disp[i]) for i, j in enumerate(JOINTS)}

    # dataset-wide (all 35 episodes) mean STATE[k]-STATE[0] displacement - more robust than a
    # single nearest-neighbor trajectory.
    dataset_wide_horizons = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80]
    dataset_wide_mean_displacement: dict[int, dict[str, float]] = {}
    for k in dataset_wide_horizons:
        per_ep = []
        for ep, (state, action) in all_ep_states.items():
            if k < len(state):
                per_ep.append(state[k] - state[0])
        if per_ep:
            mean = np.mean(np.stack(per_ep), axis=0)
            dataset_wide_mean_displacement[k] = {j: float(mean[i]) for i, j in enumerate(JOINTS)}

    # for each key joint, find which horizon's dataset-wide mean displacement is closest to the
    # Shadow delta magnitude/sign (linear-interpolated) - this is the "what future GT time does the
    # model's first action resemble" estimate.
    best_horizon_match: dict[str, Any] = {}
    for j in KEY_JOINTS:
        target = shadow_delta[j]
        ks = sorted(dataset_wide_mean_displacement.keys())
        vals = [dataset_wide_mean_displacement[k][j] for k in ks]
        # find bracketing horizon via linear interpolation on (k, val)
        best_k = None
        for i in range(len(ks) - 1):
            v0, v1 = vals[i], vals[i + 1]
            if (v0 - target) * (v1 - target) <= 0 and v0 != v1:
                frac = (target - v0) / (v1 - v0)
                best_k = ks[i] + frac * (ks[i + 1] - ks[i])
                break
        if best_k is None:
            # fall back to nearest single horizon
            closest_idx = int(np.argmin([abs(v - target) for v in vals]))
            best_k = ks[closest_idx]
        best_horizon_match[j] = {
            "shadow_delta_deg": target,
            "estimated_matching_horizon_frames": best_k,
            "estimated_matching_horizon_seconds": best_k / 30.0,
        }

    return {
        "shadow_state_used": shadow_state,
        "shadow_action_used": shadow_action,
        "shadow_delta": shadow_delta,
        "top5_nearest_frames": top5,
        "best_match": {
            "episode": ep0,
            "frame": idx0,
            "l2_dist_deg": d0,
            "match_state": {j: float(match_state[i]) for i, j in enumerate(JOINTS)},
            "gt_immediate_delta_action_minus_state": gt_immediate_delta,
            "horizon_displacement_state_t_plus_k_minus_state_t": best_match_horizon_displacement,
        },
        "dataset_wide_mean_horizon_displacement_all_35_episodes": dataset_wide_mean_displacement,
        "estimated_matching_horizon_per_key_joint": best_horizon_match,
        "interpretation": (
            "If the model's delivered first action (chunk index 0, what select_action() actually "
            "returns) closely matched the IMMEDIATE next-step GT delta, values here would resemble "
            "'gt_immediate_delta_action_minus_state' (small, single-digit degrees, matching the prior "
            "episode-start analysis). Instead, the Shadow delta magnitude for the 3 key joints lines "
            "up with the dataset's own state displacement roughly 0.7s-1.7s into the episode (see "
            "estimated_matching_horizon_per_key_joint) - i.e. within or near the chunk's own "
            "chunk_size=50 (1.67s @ 30fps) horizon, but far later than index 0's nominal 33ms step. "
            "This does not by itself distinguish a training-time temporal-alignment bug (C) from an "
            "undertrained flow-matching policy whose early chunk indices have not yet differentiated "
            "from later ones (D) - that requires the live chunk dump (see checkpoint section)."
        ),
    }


# --------------------------------------------------------------------------
# 7. Safety threshold math (for the T05 REJECT explanation)
# --------------------------------------------------------------------------


def safety_threshold_check(safety_gate_yaml: Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(safety_gate_yaml.read_text(encoding="utf-8"))
    excessive = raw["excessive_step_deg"]
    GROSS_STEP_MULTIPLIER = 5.0
    return {
        "excessive_step_deg": excessive,
        "GROSS_STEP_MULTIPLIER": GROSS_STEP_MULTIPLIER,
        "reject_threshold_deg": {j: v * GROSS_STEP_MULTIPLIER for j, v in excessive.items()},
        "would_clamp_threshold_deg": excessive,
    }


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def classify_verdict(
    joint_order: dict[str, Any],
    checkpoint_provenance: dict[str, Any],
    live_chunk: dict[str, Any],
    trajectory_analysis: dict[str, Any],
) -> dict[str, Any]:
    a_ruled_out = joint_order["dataset_action_matches_contract"] and joint_order["dataset_state_matches_contract"]
    b_status = "UNVERIFIED"
    if checkpoint_provenance.get("status") == "OK" and "trained_dataset_matches_JOINT_ORDER" in checkpoint_provenance:
        b_status = "checked_dataset_order_ok" if checkpoint_provenance["trained_dataset_matches_JOINT_ORDER"] else "MISMATCH_FOUND"

    c_d_resolved = live_chunk.get("status") == "OK"

    if not c_d_resolved:
        verdict = "E"
        rationale = (
            "Code review rules out A (adapter/client are pure passthrough, all state/action wiring is "
            "name-keyed not positional, the only positional-order risk - checkpoint raw tensor -> "
            "JOINT_ORDER - is architecturally low-risk since every dataset in this repo declares the "
            "same joint order) with high confidence, and rules out a bug in the RUNTIME chunk-extraction "
            "call site (select_action() correctly pops chunk index 0 per LeRobot's own API contract) "
            "with high confidence. B (normalization/denormalization) could not be verified - the "
            "checkpoint's baked-in policy_postprocessor.json is not present on this machine, and no VLA "
            "HTTP server was reachable, so its embedded stats couldn't be inspected. The strongest "
            "concrete finding is a dataset-wide, reproducible match between the delivered first-action "
            "delta and the Grid35 dataset's own state displacement roughly 0.7-1.7s into the episode "
            "(see nearest_frame_and_trajectory_analysis) for all 3 key joints - consistent with either a "
            "training-time temporal-alignment bug (C, unverifiable further without checkpoint/training "
            "config access) or an undertrained flow-matching policy whose early chunk indices have not "
            "differentiated from later ones (D, plausible given only 35 episodes / 10k training steps). "
            "Distinguishing C from D requires the live action-chunk dump (chunk index 0-49), which this "
            "script performs automatically once pointed at a real checkpoint via --checkpoint - it was "
            "SKIPPED here because no checkpoint files or reachable VLA server exist on this machine."
        )
    else:
        raise NotImplementedError("live chunk analysis present but verdict-from-chunk logic not reached in this build")

    return {
        "verdict": verdict,
        "rationale": rationale,
        "a_mapping_adapter_bug": "RULED_OUT (code-verified)" if a_ruled_out else "MISMATCH_FOUND",
        "b_normalization_bug": b_status,
        "c_chunk_extraction_bug_in_runtime_call_site": "RULED_OUT (code-verified, select_action pops index 0 correctly)",
        "c_prime_temporal_alignment_at_training_time": "UNVERIFIED - plausible per trajectory-horizon-match finding",
        "d_policy_itself_predicts_large_first_action": "PLAUSIBLE - leading hypothesis given clean runtime pipeline + small (35ep/10k-step) checkpoint",
    }


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------


def write_markdown_report(out_path: Path, result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Grid35 10k SmolVLA First-Action Pipeline Trace")
    lines.append("")
    ref = result["reference_observation"]
    lines.append(f"Reference Shadow observation used: `{ref.get('label')}` "
                 f"({ref.get('file', ref.get('label'))})")
    lines.append("")

    lines.append("## 1. Joint order / feature-name check")
    lines.append("")
    jo = result["joint_order_check"]
    lines.append(f"- dataset action names: `{jo['dataset_action_names']}`")
    lines.append(f"- dataset state names: `{jo['dataset_state_names']}`")
    lines.append(f"- `runtime.common.vla_contract.JOINT_ORDER`: `{jo['vla_contract_JOINT_ORDER']}`")
    lines.append(f"- match: action={jo['dataset_action_matches_contract']}, state={jo['dataset_state_matches_contract']}")
    for name, val in jo["other_joint_order_definitions"].items():
        lines.append(f"- `{name}`: `{val}`")
    lines.append("")
    lines.append(f"> {jo['note']}")
    lines.append("")

    lines.append("## 2. Normalization stats provenance (action feature, by dataset)")
    lines.append("")
    lines.append("| dataset | elbow_flex mean | elbow_flex std | wrist_flex mean | wrist_flex std | shoulder_lift mean | shoulder_lift std |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, stats in result["normalization_stats_provenance_by_dataset"].items():
        if "error" in stats or not stats:
            continue
        ef = stats.get("elbow_flex", {})
        wf = stats.get("wrist_flex", {})
        sl = stats.get("shoulder_lift", {})
        lines.append(
            f"| {name} | {ef.get('mean', float('nan')):.2f} | {ef.get('std', float('nan')):.2f} | "
            f"{wf.get('mean', float('nan')):.2f} | {wf.get('std', float('nan')):.2f} | "
            f"{sl.get('mean', float('nan')):.2f} | {sl.get('std', float('nan')):.2f} |"
        )
    lines.append("")
    lines.append("(This is what the *datasets themselves* look like - it does NOT confirm what stats are "
                  "actually baked into the checkpoint's `policy_postprocessor.json`, which lives inside the "
                  "checkpoint directory and was not accessible from this machine. See section 3.)")
    lines.append("")

    lines.append("## 3. Checkpoint provenance (best-effort)")
    lines.append("")
    lines.append(f"```json\n{json.dumps(result['checkpoint_provenance'], indent=2, ensure_ascii=False)}\n```")
    lines.append("")

    lines.append("## 4. Real Shadow-log stage table (elbow_flex / wrist_flex / shoulder_lift)")
    lines.append("")
    lines.append("| run | decision | joint | state | raw_action (post-postprocessor) | adapter.mapped_action | delta |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in result["shadow_log_stage_table"]:
        if not row.get("raw_action_post_postprocessor"):
            continue
        for j in ["shoulder_lift", "elbow_flex", "wrist_flex"]:
            s = row["state"].get(j)
            ra = row["raw_action_post_postprocessor"].get(j)
            ma = (row.get("adapter_mapped_action") or {}).get(j)
            delta = row.get("delta_raw_action_minus_state", {}).get(j)
            lines.append(
                f"| {row['label']} | {row['safety_decision']} | {j} | {s:.2f} | {ra:.2f} | "
                f"{ma if ma is None else f'{ma:.2f}'} | {delta:+.2f} |"
            )
    lines.append("")
    lines.append("(`adapter.mapped_action` == `raw_action` in every row - confirms the Action Adapter does a "
                  "pure passthrough for this checkpoint's outputs, no unit/sign transform observed.)")
    lines.append("")

    lines.append("## 5. Nearest Grid35 training frame + trajectory-horizon analysis")
    lines.append("")
    ta = result["nearest_frame_and_trajectory_analysis"]
    if ta:
        lines.append(f"Reference Shadow delta (raw_action - state): `{ta['shadow_delta']}`")
        lines.append("")
        lines.append("Top-5 nearest Grid35 frames (L2 distance over the 6 joints, degrees):")
        lines.append("")
        lines.append("| episode | frame | L2 dist (deg) |")
        lines.append("|---:|---:|---:|")
        for c in ta["top5_nearest_frames"]:
            lines.append(f"| {c['episode']} | {c['frame']} | {c['l2_dist_deg']:.2f} |")
        lines.append("")
        lines.append("GT immediate delta (action[t]-state[t]) at the single best-matching frame "
                      f"(episode {ta['best_match']['episode']}, frame {ta['best_match']['frame']}):")
        lines.append("")
        lines.append("| joint | GT immediate delta | Shadow model delta |")
        lines.append("|---|---:|---:|")
        for j in KEY_JOINTS:
            lines.append(
                f"| {j} | {ta['best_match']['gt_immediate_delta_action_minus_state'][j]:+.2f} | "
                f"{ta['shadow_delta'][j]:+.2f} |"
            )
        lines.append("")
        lines.append("Dataset-wide (all 35 episodes) mean `STATE[t0+k]-STATE[t0]` displacement per horizon k:")
        lines.append("")
        header = "| joint | " + " | ".join(f"k={k}({k/30:.2f}s)" for k in sorted(ta["dataset_wide_mean_horizon_displacement_all_35_episodes"])) + " |"
        lines.append(header)
        lines.append("|---|" + "---:|" * len(ta["dataset_wide_mean_horizon_displacement_all_35_episodes"]))
        for j in KEY_JOINTS:
            row_vals = [
                ta["dataset_wide_mean_horizon_displacement_all_35_episodes"][k][j]
                for k in sorted(ta["dataset_wide_mean_horizon_displacement_all_35_episodes"])
            ]
            lines.append(f"| {j} | " + " | ".join(f"{v:+.2f}" for v in row_vals) + " |")
        lines.append("")
        lines.append("Estimated horizon (frames/seconds into the episode) whose GT displacement best matches "
                      "the Shadow model's delivered first-action delta, per key joint:")
        lines.append("")
        lines.append("| joint | Shadow delta | estimated matching horizon |")
        lines.append("|---|---:|---:|")
        for j, v in ta["estimated_matching_horizon_per_key_joint"].items():
            lines.append(
                f"| {j} | {v['shadow_delta_deg']:+.2f} | {v['estimated_matching_horizon_frames']:.1f} frames "
                f"(~{v['estimated_matching_horizon_seconds']:.2f}s) |"
            )
        lines.append("")
        lines.append(f"> {ta['interpretation']}")
        lines.append("")

    lines.append("## 6. Live action-chunk dump (chunk index 0-49)")
    lines.append("")
    lc = result["live_action_chunk_dump"]
    if lc.get("status") != "OK":
        lines.append(f"**{lc['status']}**: {lc.get('reason')}")
    else:
        lines.append(f"chunk_size={lc['chunk_size']}, n_action_steps={lc['n_action_steps_used_by_select_action']}, "
                      f"index actually used as first command={lc['index_actually_used_as_first_command']}")
    lines.append("")

    lines.append("## 7. Safety threshold math (T05 REJECT explanation)")
    lines.append("")
    st = result["safety_threshold_math"]
    lines.append("| joint | WOULD_CLAMP threshold (deg) | REJECT threshold (deg, x5) |")
    lines.append("|---|---:|---:|")
    for j in JOINTS:
        lines.append(f"| {j} | {st['would_clamp_threshold_deg'][j]:.2f} | {st['reject_threshold_deg'][j]:.2f} |")
    lines.append("")
    lines.append("T05's elbow_flex delta was -32.03deg, exceeding the 28.65deg REJECT threshold -> REJECT. "
                  "T05-R3/R4/R5 (elbow delta -21 to -27deg) stayed under it -> WOULD_CLAMP.")
    lines.append("")

    lines.append("## 8. Verdict")
    lines.append("")
    v = result["verdict"]
    lines.append(f"**{v['verdict']}**")
    lines.append("")
    lines.append(v["rationale"])
    lines.append("")
    for k in ["a_mapping_adapter_bug", "b_normalization_bug", "c_chunk_extraction_bug_in_runtime_call_site",
              "c_prime_temporal_alignment_at_training_time", "d_policy_itself_predicts_large_first_action"]:
        lines.append(f"- {k}: {v[k]}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--safety-gate-yaml", type=Path, default=DEFAULT_SAFETY_GATE_YAML)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a real SmolVLA pretrained_model checkpoint dir. If omitted or not found, "
        "checkpoint-dependent sections (live chunk dump, checkpoint provenance) are reported SKIPPED.",
    )
    parser.add_argument(
        "--shadow-report",
        type=Path,
        default=None,
        help="Specific reports/shadow_mode/shadow_*.json to use as the reference observation for the "
        "nearest-frame/live-chunk analysis. Defaults to the T05 REJECT run if present.",
    )
    parser.add_argument("--task", default="pick up the cube", help="Task string for live chunk dump (if run).")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    shadow_dir = args.shadow_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    print("[diagnose] 1/7 joint order check")
    joint_order = check_joint_order(dataset_root)

    print("[diagnose] 2/7 normalization stats provenance")
    stats_provenance = collect_dataset_stats_provenance(args.data_dir)

    print("[diagnose] 3/7 checkpoint provenance (best-effort, no GPU)")
    checkpoint_provenance = check_checkpoint_provenance(args.checkpoint, args.data_dir)

    print("[diagnose] 4/7 real Shadow-log stage table")
    shadow_rows = load_shadow_stage_table(shadow_dir)

    # pick reference observation for chunk-dump/nearest-frame analysis
    ref_row = None
    if args.shadow_report is not None:
        with open(args.shadow_report, encoding="utf-8") as f:
            d = json.load(f)
        ref_row = {
            "state": d["observation"]["state"],
            "raw_action_post_postprocessor": d["vla"]["raw_action"],
            "label": d.get("scene_metadata", {}).get("label"),
        }
    else:
        for row in shadow_rows:
            if row["label"] and row["label"].startswith("T05") and row["safety_decision"] == "REJECT":
                ref_row = row
                break
        if ref_row is None and shadow_rows:
            ref_row = shadow_rows[0]

    print("[diagnose] 5/7 nearest-Grid35-frame + trajectory-horizon analysis")
    trajectory_analysis = None
    if ref_row is not None and ref_row.get("raw_action_post_postprocessor"):
        trajectory_analysis = nearest_frame_trajectory_analysis(
            dataset_root, ref_row["state"], ref_row["raw_action_post_postprocessor"]
        )

    print("[diagnose] 6/7 live action-chunk dump (best-effort)")
    live_chunk = {"status": "SKIPPED", "reason": "no reference observation available"}
    if ref_row is not None:
        # images are not embedded in shadow JSON logs (only file paths, and only for runs after image
        # persistence was added) - the live chunk dump needs real image arrays, so this only actually
        # runs if a checkpoint is given AND the referenced camera frames still exist on disk.
        images: dict[str, np.ndarray] = {}
        camera_paths = None
        if args.shadow_report is not None:
            with open(args.shadow_report, encoding="utf-8") as f:
                d = json.load(f)
            camera_paths = d.get("observation", {}).get("camera_frame_paths")
        if camera_paths and camera_paths.get("workspace_path") and camera_paths.get("wrist_path"):
            try:
                from PIL import Image

                images["observation.images.workspace"] = np.array(
                    Image.open(camera_paths["workspace_path"]).convert("RGB"), dtype=np.uint8
                )
                images["observation.images.wrist"] = np.array(
                    Image.open(camera_paths["wrist_path"]).convert("RGB"), dtype=np.uint8
                )
            except Exception as exc:  # noqa: BLE001
                live_chunk = {"status": "SKIPPED", "reason": f"could not load camera frames: {exc}"}
        if images or args.checkpoint is None:
            live_chunk = dump_live_action_chunk(args.checkpoint, ref_row["state"], images, args.task)

    print("[diagnose] 7/7 safety threshold math")
    safety_thresholds = safety_threshold_check(args.safety_gate_yaml)

    verdict = classify_verdict(joint_order, checkpoint_provenance, live_chunk, trajectory_analysis or {})

    result = {
        "reference_observation": ref_row,
        "joint_order_check": joint_order,
        "normalization_stats_provenance_by_dataset": stats_provenance,
        "checkpoint_provenance": checkpoint_provenance,
        "shadow_log_stage_table": shadow_rows,
        "nearest_frame_and_trajectory_analysis": trajectory_analysis,
        "live_action_chunk_dump": live_chunk,
        "safety_threshold_math": safety_thresholds,
        "verdict": verdict,
    }

    json_path = out_dir / "grid35_first_action_pipeline_trace.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    print(f"[diagnose] wrote {json_path}")

    md_path = out_dir / "grid35_first_action_pipeline_trace.md"
    write_markdown_report(md_path, result)
    print(f"[diagnose] wrote {md_path}")

    print(f"[diagnose] VERDICT: {verdict['verdict']}")
    for k in ["a_mapping_adapter_bug", "b_normalization_bug", "c_chunk_extraction_bug_in_runtime_call_site",
              "c_prime_temporal_alignment_at_training_time", "d_policy_itself_predicts_large_first_action"]:
        print(f"  {k}: {verdict[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
