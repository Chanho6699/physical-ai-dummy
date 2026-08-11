#!/usr/bin/env python3
"""Assemble reports/pick_drop_v4_fresh_training/ from the raw V4-fresh-training run artifacts
(training log, historical-test10 offline eval, V4-heldout6 offline eval + per-episode MAE,
first-action seed sweeps, temporal chunk-error diagnostics) plus the already-completed prior
experiments' own summaries (V2 baseline, combined65 uniform/early-weight, reinforcement30-only,
reweight 2:1/3:1) for direct comparison. Read-only aggregation - no training/eval performed here.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_pick_drop_combined65_training_report import parse_training_log  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STEPS = [2500, 5000, 7500, 10000]
REPORTS = PROJECT_ROOT / "reports"
OUT_DIR = REPORTS / "pick_drop_v4_fresh_training"


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- training log ----------------
    train_log_info = parse_training_log(OUT_DIR / "train.log")
    tr_by_step = {r["checkpoint_step"]: r for r in train_log_info["checkpoint_rows"]}

    # ---------------- offline eval: historical test10 + V4 heldout6 (kept separate, never pooled) ----------------
    hist_offline = load(REPORTS / "pick_drop_v4_historical_test10_offline_eval" / "summary.json")
    v4h6_offline = load(REPORTS / "pick_drop_v4_heldout6_offline_eval" / "summary.json")
    hist_by_step = {r["checkpoint_step"]: r for r in hist_offline["rows"]}
    v4h6_by_step = {r["checkpoint_step"]: r for r in v4h6_offline["rows"]}
    v4h6_per_episode = load(OUT_DIR / "v4_heldout6_per_episode_mae.json")

    # ---------------- comparison baselines (already-completed prior experiments) ----------------
    v2_offline = load(REPORTS / "grid35_v2_midpoint_eval" / "summary.json")
    c65_offline = load(REPORTS / "pick_drop_combined65_offline_eval" / "summary.json")
    r30_offline = load(REPORTS / "reinforcement30_only_v1_offline_eval" / "summary.json")
    v2_off_by_step = {r["checkpoint_step"]: r for r in v2_offline["rows"]}
    c65_off_by_step = {r["checkpoint_step"]: r for r in c65_offline["rows"]}
    r30_off_by_step = {r["checkpoint_step"]: r for r in r30_offline["rows"]}

    v2_sweep = load(REPORTS / "grid35_v2_T01_seed_sweep" / "seed_sweep.json")
    c65_sweep_7500 = load(REPORTS / "pick_drop_combined65_fresh_training" / "first_action_seed_sweep_7500" / "seed_sweep.json")
    c65_sweep_10000 = load(REPORTS / "pick_drop_combined65_fresh_training" / "first_action_seed_sweep_10000" / "seed_sweep.json")
    r30_sweep_7500 = load(REPORTS / "reinforcement30_only_v1" / "first_action_seed_sweep_7500" / "seed_sweep.json")
    reweight21_sweep_10000 = load(REPORTS / "combined65_reweight_new2_old1_v1" / "first_action_seed_sweep_10000" / "seed_sweep.json")
    reweight31_sweep_10000 = load(REPORTS / "combined65_reweight_new3_old1_v1" / "first_action_seed_sweep_10000" / "seed_sweep.json")

    v4_sweeps = {s: load(OUT_DIR / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    v4_temporal = {s: load(OUT_DIR / "temporal_chunk_error" / f"v4_fresh_{s}" / "temporal_chunk_error.json") for s in STEPS}

    # ================= historical_test10_metrics.csv =================
    hist_rows = []
    for s in STEPS:
        tr = tr_by_step.get(s, {})
        off = hist_by_step.get(s, {})
        hist_rows.append(
            {
                "checkpoint_step": s,
                "train_loss": tr.get("train_loss"),
                "wall_time_since_start_min": tr.get("wall_time_since_start_min"),
                "action_mae": off.get("action_mae"),
                "action_mae_per_joint": json.dumps(off.get("action_mae_per_joint")),
                "would_pass": off.get("would_pass"),
                "would_clamp": off.get("would_clamp"),
                "would_reject": off.get("would_reject"),
                "frames_evaluated": off.get("num_frames_evaluated"),
                "episodes_evaluated": off.get("num_episodes_evaluated"),
                "v2_baseline_action_mae": v2_off_by_step.get(s, {}).get("action_mae"),
                "combined65_uniform_action_mae": c65_off_by_step.get(s, {}).get("action_mae"),
                "reinforcement30_only_action_mae": r30_off_by_step.get(s, {}).get("action_mae"),
            }
        )
    with open(OUT_DIR / "historical_test10_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist_rows[0].keys()))
        w.writeheader()
        w.writerows(hist_rows)

    # ================= v4_heldout6_metrics.csv (checkpoint-level AND episode-level rows, clearly tagged, never averaged with test10) =================
    v4h6_rows = []
    for s in STEPS:
        off = v4h6_by_step.get(s, {})
        v4h6_rows.append(
            {
                "row_type": "checkpoint_level",
                "checkpoint_step": s,
                "episode": None,
                "action_mae": off.get("action_mae"),
                "action_mae_per_joint": json.dumps(off.get("action_mae_per_joint")),
                "would_pass": off.get("would_pass"),
                "would_clamp": off.get("would_clamp"),
                "would_reject": off.get("would_reject"),
                "n_frames": off.get("num_frames_evaluated"),
            }
        )
        for ep_str, ep_data in sorted(v4h6_per_episode[str(s)].items(), key=lambda kv: int(kv[0])):
            v4h6_rows.append(
                {
                    "row_type": "episode_level",
                    "checkpoint_step": s,
                    "episode": int(ep_str),
                    "action_mae": ep_data["action_mae_overall"],
                    "action_mae_per_joint": json.dumps(ep_data["action_mae_per_joint"]),
                    "would_pass": None,
                    "would_clamp": None,
                    "would_reject": None,
                    "n_frames": ep_data["n_frames"],
                }
            )
    with open(OUT_DIR / "v4_heldout6_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(v4h6_rows[0].keys()))
        w.writeheader()
        w.writerows(v4h6_rows)

    # ================= first_action_diagnostics.csv =================
    fa_rows = []

    def add_sweep_rows(exp_label: str, sweep: dict, step: int | None):
        summ = sweep["summary"]
        for j in JOINTS:
            pj = summ["per_joint"][j]
            fa_rows.append(
                {
                    "experiment": exp_label,
                    "checkpoint_step": step,
                    "joint": j,
                    "delta_mean_deg": pj["mean"],
                    "delta_std_deg": pj["std"],
                    "would_clamp_threshold_deg": pj["threshold_deg"],
                    "clamp_rate": pj["clamp_rate"],
                    "n_seeds": summ["n_seeds"],
                    "clamp_free_seed_count": summ["clamp_free_seed_count"],
                    "l2_error_vs_gt_mean": summ["l2_error_vs_gt"]["mean"],
                }
            )

    add_sweep_rows("V2_7500_baseline", v2_sweep, 7500)
    add_sweep_rows("combined65_uniform_7500", c65_sweep_7500, 7500)
    add_sweep_rows("combined65_uniform_10000", c65_sweep_10000, 10000)
    add_sweep_rows("reinforcement30_only_7500", r30_sweep_7500, 7500)
    add_sweep_rows("reweight_2to1_10000", reweight21_sweep_10000, 10000)
    add_sweep_rows("reweight_3to1_10000", reweight31_sweep_10000, 10000)
    for s in STEPS:
        add_sweep_rows("v4_fresh", v4_sweeps[s], s)

    with open(OUT_DIR / "first_action_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fa_rows[0].keys()))
        w.writeheader()
        w.writerows(fa_rows)

    # ================= temporal_chunk_error.csv =================
    temp_rows = []
    for s in STEPS:
        d = v4_temporal[s]
        for bucket_name, b in d["buckets"].items():
            temp_rows.append(
                {
                    "experiment": "v4_fresh",
                    "checkpoint_step": s,
                    "bucket": bucket_name,
                    "mae_shoulder_lift_deg": b["mae_per_joint"]["shoulder_lift"],
                    "mae_elbow_flex_deg": b["mae_per_joint"]["elbow_flex"],
                    "mae_key_joints_mean_deg": b["mae_key_joints_mean"],
                    "nearest_demo_episode": d["nearest_demo_match"]["episode"],
                    "nearest_demo_frame": d["nearest_demo_match"]["frame"],
                    "nearest_demo_l2_dist_deg": d["nearest_demo_match"]["l2_dist_deg"],
                }
            )
    temp_rows.sort(key=lambda r: (r["checkpoint_step"], r["bucket"]))
    with open(OUT_DIR / "temporal_chunk_error.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(temp_rows[0].keys()))
        w.writeheader()
        w.writerows(temp_rows)

    # ================= summary.json =================
    summary = {
        "train_log_info": {k: v for k, v in train_log_info.items() if k != "all_logged_steps"},
        "historical_test10_metrics": hist_rows,
        "v4_heldout6_metrics": v4h6_rows,
        "first_action_diagnostics": fa_rows,
        "temporal_chunk_error": temp_rows,
        "raw": {
            "hist_offline": hist_offline,
            "v4h6_offline": v4h6_offline,
            "v4_sweeps": v4_sweeps,
            "v4_temporal": v4_temporal,
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    for fname in ["historical_test10_metrics.csv", "v4_heldout6_metrics.csv", "first_action_diagnostics.csv", "temporal_chunk_error.csv", "summary.json"]:
        print(f"wrote {OUT_DIR / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
