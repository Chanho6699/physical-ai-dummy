#!/usr/bin/env python3
"""Assemble reports/combined65_reweight_new2_old1_v1/ - the 5-way comparison (V2 baseline /
combined65 baseline / combined65 early-weight / reinforcement30-only / combined65 reweight 2:1)
for the dataset-reweighting ablation. Read-only aggregation of already-completed run artifacts -
no training/eval performed here.
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
OUT_DIR = PROJECT_ROOT / "reports" / "combined65_reweight_new2_old1_v1"
REPORTS = PROJECT_ROOT / "reports"


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v2_offline = load(REPORTS / "grid35_v2_midpoint_eval" / "summary.json")
    c65_offline = load(REPORTS / "pick_drop_combined65_offline_eval" / "summary.json")
    ew_offline = load(REPORTS / "pick_drop_combined65_early_weight_offline_eval" / "summary.json")
    r30_offline = load(REPORTS / "reinforcement30_only_v1_offline_eval" / "summary.json")
    r30_insample = load(REPORTS / "reinforcement30_only_v1_offline_eval" / "in_sample_fit.json")
    rw_offline = load(OUT_DIR / "summary.json") if (OUT_DIR / "summary.json").exists() else None
    rw_offline = load(REPORTS / "combined65_reweight_new2_old1_v1_offline_eval" / "summary.json")
    rw_insample = load(REPORTS / "combined65_reweight_new2_old1_v1_offline_eval" / "in_sample_fit.json")

    v2_sweep = load(REPORTS / "grid35_v2_T01_seed_sweep" / "seed_sweep.json")
    c65_sweeps = {s: load(REPORTS / "pick_drop_combined65_fresh_training" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    ew_sweeps = {s: load(REPORTS / "pick_drop_combined65_early_weight_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    r30_sweeps = {s: load(REPORTS / "reinforcement30_only_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    rw_sweeps = {s: load(OUT_DIR / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}

    rw_train_log = parse_training_log(Path("/tmp/reweight_new2_old1_train.log"))
    c65_train_log = parse_training_log(Path("/tmp/pick_drop_combined65_train.log"))
    ew_train_log = parse_training_log(Path("/tmp/pick_drop_combined65_early_weight_train.log"))
    r30_train_log = parse_training_log(Path("/tmp/reinforcement30_only_train.log"))

    temporal_c65 = {s: load(REPORTS / "pick_drop_combined65_early_weight_v1" / "temporal_chunk_error" / f"baseline_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_ew = {s: load(REPORTS / "pick_drop_combined65_early_weight_v1" / "temporal_chunk_error" / f"early_weight_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_r30 = {s: load(REPORTS / "reinforcement30_only_v1" / "temporal_chunk_error" / f"reinforcement30_only_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_rw = {s: load(OUT_DIR / "temporal_chunk_error" / f"reweight_{s}" / "temporal_chunk_error.json") for s in STEPS}

    # ---- checkpoint_metrics.csv ----
    def by_step(offline):
        return {r["checkpoint_step"]: r for r in offline["rows"]}

    v2_off, c65_off, ew_off, r30_off, rw_off = (by_step(x) for x in (v2_offline, c65_offline, ew_offline, r30_offline, rw_offline))
    r30_ins_by = {r["checkpoint_step"]: r for r in r30_insample["rows"]}
    rw_ins_by = {r["checkpoint_step"]: r for r in rw_insample["rows"]}
    c65_tr = {r["checkpoint_step"]: r for r in c65_train_log["checkpoint_rows"]}
    ew_tr = {r["checkpoint_step"]: r for r in ew_train_log["checkpoint_rows"]}
    r30_tr = {r["checkpoint_step"]: r for r in r30_train_log["checkpoint_rows"]}
    rw_tr = {r["checkpoint_step"]: r for r in rw_train_log["checkpoint_rows"]}

    ckpt_rows = []
    for s in STEPS:
        for exp, tr, off, insample in [
            ("v2_baseline", None, v2_off.get(s), None),
            ("combined65_baseline", c65_tr.get(s), c65_off.get(s), None),
            ("combined65_early_weight_v1", ew_tr.get(s), ew_off.get(s), None),
            ("reinforcement30_only_v1", r30_tr.get(s), r30_off.get(s), r30_ins_by.get(s)),
            ("combined65_reweight_new2_old1_v1", rw_tr.get(s), rw_off.get(s), rw_ins_by.get(s)),
        ]:
            heldout = off["action_mae"] if off else None
            insamp = insample["in_sample_action_mae"] if insample else None
            ckpt_rows.append(
                {
                    "experiment": exp,
                    "checkpoint_step": s,
                    "train_loss": tr["train_loss"] if tr else None,
                    "wall_time_min": tr["wall_time_since_start_min"] if tr else None,
                    "heldout_mae": heldout,
                    "in_sample_mae": insamp,
                    "train_heldout_gap": (heldout - insamp) if (heldout is not None and insamp is not None) else None,
                }
            )
    with open(OUT_DIR / "checkpoint_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ckpt_rows[0].keys()))
        w.writeheader()
        w.writerows(ckpt_rows)

    # ---- first_action_diagnostics.csv ----
    fa_rows = []

    def add_sweep_rows(exp_label, sweep, step):
        summ = sweep["summary"]
        for j in JOINTS:
            pj = summ["per_joint"][j]
            fa_rows.append(
                {
                    "experiment": exp_label, "checkpoint_step": step, "joint": j,
                    "delta_mean_deg": pj["mean"], "delta_std_deg": pj["std"],
                    "would_clamp_threshold_deg": pj["threshold_deg"], "clamp_rate": pj["clamp_rate"],
                    "n_seeds": summ["n_seeds"], "clamp_free_seed_count": summ["clamp_free_seed_count"],
                    "l2_error_vs_gt_mean": summ["l2_error_vs_gt"]["mean"],
                }
            )

    add_sweep_rows("v2_baseline", v2_sweep, 7500)
    for s in STEPS:
        add_sweep_rows("combined65_baseline", c65_sweeps[s], s)
    for s in STEPS:
        add_sweep_rows("combined65_early_weight_v1", ew_sweeps[s], s)
    for s in STEPS:
        add_sweep_rows("reinforcement30_only_v1", r30_sweeps[s], s)
    for s in STEPS:
        add_sweep_rows("combined65_reweight_new2_old1_v1", rw_sweeps[s], s)

    with open(OUT_DIR / "first_action_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fa_rows[0].keys()))
        w.writeheader()
        w.writerows(fa_rows)

    # ---- temporal_chunk_error.csv ----
    temp_rows = []
    for label, temporal in [
        ("combined65_baseline", temporal_c65),
        ("combined65_early_weight_v1", temporal_ew),
        ("reinforcement30_only_v1", temporal_r30),
        ("combined65_reweight_new2_old1_v1", temporal_rw),
    ]:
        for s in STEPS:
            d = temporal[s]
            for bucket_name, b in d["buckets"].items():
                temp_rows.append(
                    {
                        "experiment": label, "checkpoint_step": s, "bucket": bucket_name,
                        "mae_shoulder_lift_deg": b["mae_per_joint"]["shoulder_lift"],
                        "mae_elbow_flex_deg": b["mae_per_joint"]["elbow_flex"],
                        "mae_key_joints_mean_deg": b["mae_key_joints_mean"],
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
            "rw_train_log_info": {k: v for k, v in rw_train_log.items()},
            "rw_offline_eval": rw_offline,
            "rw_in_sample_fit": rw_insample,
            "rw_sweeps": rw_sweeps,
            "rw_temporal": {s: temporal_rw[s] for s in STEPS},
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    for fname in ["checkpoint_metrics.csv", "first_action_diagnostics.csv", "temporal_chunk_error.csv", "summary.json"]:
        print(f"wrote {OUT_DIR / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
