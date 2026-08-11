#!/usr/bin/env python3
"""Assemble reports/pick_drop_combined65_early_weight_v1/ from the three experiments' raw
artifacts: V2 baseline, combined65 baseline (uniform loss), combined65_early_weight_v1
(early-action temporal loss weighting). Read-only aggregation - no training/eval performed here.
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
OUT_DIR = PROJECT_ROOT / "reports" / "pick_drop_combined65_early_weight_v1"


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v2_train_log_info = None  # not re-parsed (already reported earlier); numbers taken from existing reports
    v2_offline = load(PROJECT_ROOT / "reports" / "grid35_v2_midpoint_eval" / "summary.json")
    c65_offline = load(PROJECT_ROOT / "reports" / "pick_drop_combined65_offline_eval" / "summary.json")
    ew_offline = load(PROJECT_ROOT / "reports" / "pick_drop_combined65_early_weight_offline_eval" / "summary.json")

    v2_sweep_baseline_7500 = load(PROJECT_ROOT / "reports" / "grid35_v2_T01_seed_sweep" / "seed_sweep.json")
    c65_sweeps = {s: load(PROJECT_ROOT / "reports" / "pick_drop_combined65_fresh_training" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    ew_sweeps = {s: load(OUT_DIR / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}

    c65_train_log = parse_training_log(Path("/tmp/pick_drop_combined65_train.log"))
    ew_train_log = parse_training_log(Path("/tmp/pick_drop_combined65_early_weight_train.log"))

    temporal = {}
    for s in STEPS:
        for label in ["baseline", "early_weight"]:
            p = OUT_DIR / "temporal_chunk_error" / f"{label}_{s}" / "temporal_chunk_error.json"
            temporal[(label, s)] = load(p)

    # ---- checkpoint_metrics.csv ----
    c65_offline_by_step = {r["checkpoint_step"]: r for r in c65_offline["rows"]}
    ew_offline_by_step = {r["checkpoint_step"]: r for r in ew_offline["rows"]}
    v2_offline_by_step = {r["checkpoint_step"]: r for r in v2_offline["rows"]}
    c65_rows_by_step = {r["checkpoint_step"]: r for r in c65_train_log["checkpoint_rows"]}
    ew_rows_by_step = {r["checkpoint_step"]: r for r in ew_train_log["checkpoint_rows"]}

    ckpt_rows = []
    for s in STEPS:
        for exp, tr, off in [
            ("v2_baseline", None, v2_offline_by_step.get(s)),
            ("combined65_baseline", c65_rows_by_step.get(s), c65_offline_by_step.get(s)),
            ("combined65_early_weight_v1", ew_rows_by_step.get(s), ew_offline_by_step.get(s)),
        ]:
            ckpt_rows.append(
                {
                    "experiment": exp,
                    "checkpoint_step": s,
                    "train_loss": tr["train_loss"] if tr else None,
                    "lr": tr["lr"] if tr else None,
                    "wall_time_since_start_min": tr["wall_time_since_start_min"] if tr else None,
                    "gpu_mem_gb": tr["gpu_mem_gb"] if tr else None,
                    "offline_action_mae": off["action_mae"] if off else None,
                    "offline_shoulder_lift_mae": off["action_mae_per_joint"]["shoulder_lift"] if off else None,
                    "offline_elbow_flex_mae": off["action_mae_per_joint"]["elbow_flex"] if off else None,
                    "offline_would_pass": off.get("would_pass") if off else None,
                    "offline_would_clamp": off.get("would_clamp") if off else None,
                    "offline_would_reject": off.get("would_reject") if off else None,
                }
            )
    with open(OUT_DIR / "checkpoint_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ckpt_rows[0].keys()))
        w.writeheader()
        w.writerows(ckpt_rows)

    # ---- first_action_diagnostics.csv ----
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
                    "clamp_count": pj["clamp_count"],
                    "n_seeds": summ["n_seeds"],
                    "clamp_rate": pj["clamp_rate"],
                    "clamp_free_seed_count": summ["clamp_free_seed_count"],
                    "l2_error_vs_gt_mean": summ["l2_error_vs_gt"]["mean"],
                    "l2_error_vs_gt_std": summ["l2_error_vs_gt"]["std"],
                }
            )

    add_sweep_rows("v2_baseline", v2_sweep_baseline_7500, 7500)
    for s in STEPS:
        add_sweep_rows("combined65_baseline", c65_sweeps[s], s)
    for s in STEPS:
        add_sweep_rows("combined65_early_weight_v1", ew_sweeps[s], s)

    with open(OUT_DIR / "first_action_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fa_rows[0].keys()))
        w.writeheader()
        w.writerows(fa_rows)

    # ---- temporal_chunk_error.csv ----
    temp_rows = []
    for s in STEPS:
        for label in ["baseline", "early_weight"]:
            d = temporal[(label, s)]
            for bucket_name, b in d["buckets"].items():
                temp_rows.append(
                    {
                        "experiment": f"combined65_{label}",
                        "checkpoint_step": s,
                        "bucket": bucket_name,
                        "chunk_indices": f"{b['chunk_indices'][0]}-{b['chunk_indices'][-1]}",
                        "mae_overall_deg": b["mae_overall"],
                        "mae_shoulder_lift_deg": b["mae_per_joint"]["shoulder_lift"],
                        "mae_elbow_flex_deg": b["mae_per_joint"]["elbow_flex"],
                        "mae_key_joints_mean_deg": b["mae_key_joints_mean"],
                        "n_seeds": d["n_seeds"],
                    }
                )
    with open(OUT_DIR / "temporal_chunk_error.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(temp_rows[0].keys()))
        w.writeheader()
        w.writerows(temp_rows)

    # ---- summary.json ----
    summary = {
        "checkpoint_metrics": ckpt_rows,
        "first_action_diagnostics": fa_rows,
        "temporal_chunk_error": temp_rows,
        "raw": {
            "v2_offline_eval": v2_offline,
            "combined65_offline_eval": c65_offline,
            "early_weight_offline_eval": ew_offline,
            "v2_sweep_7500": v2_sweep_baseline_7500,
            "combined65_sweeps": c65_sweeps,
            "early_weight_sweeps": ew_sweeps,
            "combined65_train_log_info": {k: v for k, v in c65_train_log.items()},
            "early_weight_train_log_info": {k: v for k, v in ew_train_log.items()},
            "temporal_raw": {f"{label}_{s}": temporal[(label, s)] for s in STEPS for label in ["baseline", "early_weight"]},
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    print(f"wrote {OUT_DIR / 'checkpoint_metrics.csv'}")
    print(f"wrote {OUT_DIR / 'first_action_diagnostics.csv'}")
    print(f"wrote {OUT_DIR / 'temporal_chunk_error.csv'}")
    print(f"wrote {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
