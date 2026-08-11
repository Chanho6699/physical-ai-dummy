#!/usr/bin/env python3
"""Assemble reports/reinforcement30_only_v1/ - the 4-way comparison (V2 baseline / combined65
baseline / combined65 early-weight / reinforcement30-only) for the data-composition ablation.
Read-only aggregation of already-completed run artifacts - no training/eval performed here.
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
OUT_DIR = PROJECT_ROOT / "reports" / "reinforcement30_only_v1"
REPORTS = PROJECT_ROOT / "reports"


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v2_offline = load(REPORTS / "grid35_v2_midpoint_eval" / "summary.json")
    c65_offline = load(REPORTS / "pick_drop_combined65_offline_eval" / "summary.json")
    ew_offline = load(REPORTS / "pick_drop_combined65_early_weight_offline_eval" / "summary.json")
    r30_offline = load(REPORTS / "reinforcement30_only_v1_offline_eval" / "summary.json")
    r30_in_sample = load(REPORTS / "reinforcement30_only_v1_offline_eval" / "in_sample_fit.json")

    v2_sweep = load(REPORTS / "grid35_v2_T01_seed_sweep" / "seed_sweep.json")
    c65_sweeps = {s: load(REPORTS / "pick_drop_combined65_fresh_training" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    ew_sweeps = {s: load(REPORTS / "pick_drop_combined65_early_weight_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    r30_sweeps = {s: load(OUT_DIR / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}

    c65_train_log = parse_training_log(Path("/tmp/pick_drop_combined65_train.log"))
    ew_train_log = parse_training_log(Path("/tmp/pick_drop_combined65_early_weight_train.log"))
    r30_train_log = parse_training_log(Path("/tmp/reinforcement30_only_train.log"))

    temporal = {}
    for s in STEPS:
        temporal[("combined65_baseline", s)] = load(REPORTS / "pick_drop_combined65_early_weight_v1" / "temporal_chunk_error" / f"baseline_{s}" / "temporal_chunk_error.json")
        temporal[("combined65_early_weight", s)] = load(REPORTS / "pick_drop_combined65_early_weight_v1" / "temporal_chunk_error" / f"early_weight_{s}" / "temporal_chunk_error.json")
        temporal[("reinforcement30_only", s)] = load(OUT_DIR / "temporal_chunk_error" / f"reinforcement30_only_{s}" / "temporal_chunk_error.json")

    # ---- checkpoint_metrics.csv ----
    v2_off_by_step = {r["checkpoint_step"]: r for r in v2_offline["rows"]}
    c65_off_by_step = {r["checkpoint_step"]: r for r in c65_offline["rows"]}
    ew_off_by_step = {r["checkpoint_step"]: r for r in ew_offline["rows"]}
    r30_off_by_step = {r["checkpoint_step"]: r for r in r30_offline["rows"]}
    r30_insample_by_step = {r["checkpoint_step"]: r for r in r30_in_sample["rows"]}
    c65_tr_by_step = {r["checkpoint_step"]: r for r in c65_train_log["checkpoint_rows"]}
    ew_tr_by_step = {r["checkpoint_step"]: r for r in ew_train_log["checkpoint_rows"]}
    r30_tr_by_step = {r["checkpoint_step"]: r for r in r30_train_log["checkpoint_rows"]}

    ckpt_rows = []
    for s in STEPS:
        for exp, tr, off, insample in [
            ("v2_baseline", None, v2_off_by_step.get(s), None),
            ("combined65_baseline", c65_tr_by_step.get(s), c65_off_by_step.get(s), None),
            ("combined65_early_weight_v1", ew_tr_by_step.get(s), ew_off_by_step.get(s), None),
            ("reinforcement30_only_v1", r30_tr_by_step.get(s), r30_off_by_step.get(s), r30_insample_by_step.get(s)),
        ]:
            heldout_mae = off["action_mae"] if off else None
            insample_mae = insample["in_sample_action_mae"] if insample else None
            ckpt_rows.append(
                {
                    "experiment": exp,
                    "checkpoint_step": s,
                    "train_loss": tr["train_loss"] if tr else None,
                    "wall_time_since_start_min": tr["wall_time_since_start_min"] if tr else None,
                    "heldout_offline_action_mae": heldout_mae,
                    "in_sample_train_action_mae": insample_mae,
                    "train_heldout_gap": (heldout_mae - insample_mae) if (heldout_mae is not None and insample_mae is not None) else None,
                    "heldout_shoulder_lift_mae": off["action_mae_per_joint"]["shoulder_lift"] if off else None,
                    "heldout_elbow_flex_mae": off["action_mae_per_joint"]["elbow_flex"] if off else None,
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
                    "clamp_rate": pj["clamp_rate"],
                    "n_seeds": summ["n_seeds"],
                    "clamp_free_seed_count": summ["clamp_free_seed_count"],
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

    with open(OUT_DIR / "first_action_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fa_rows[0].keys()))
        w.writeheader()
        w.writerows(fa_rows)

    # ---- temporal_chunk_error.csv ----
    temp_rows = []
    for (label, s), d in temporal.items():
        for bucket_name, b in d["buckets"].items():
            temp_rows.append(
                {
                    "experiment": label,
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
    temp_rows.sort(key=lambda r: (r["experiment"], r["checkpoint_step"], r["bucket"]))
    with open(OUT_DIR / "temporal_chunk_error.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(temp_rows[0].keys()))
        w.writeheader()
        w.writerows(temp_rows)

    # ---- old35_vs_reinforcement30_distributional_stats.csv (from existing analysis, reused) ----
    cov = load(REPORTS / "grid35_v2_vs_start_coverage_v3" / "summary.json")
    dist_rows = []
    for label, key in [("old35_v2_clean", "v2_clean"), ("reinforcement30_v3", "v3_start_coverage")]:
        d = cov[key]
        div = d["start_pose_diversity"]
        fc = d["frame_coverage"]
        safety = d["immediate_action_safety_summary"]
        chunk = d["chunk_future_motion_summary"]
        timing = d["timing_summary"]
        row = {
            "dataset": label,
            "start_pose_pairwise_l2_median_deg": div["start_frame0_pairwise_l2_deg"]["median"],
            "static_low_motion_fraction": fc["static_segment_data_driven"]["fraction_of_total"],
            "static_length_median_frames": fc["static_length_distribution_frames"]["median"],
            "state_first_movement_median_frame": timing["state_first_movement"]["median_frame"],
            "action_first_movement_median_frame": timing["action_first_movement"]["median_frame"],
            "shoulder_lift_immediate_delta_mean_abs_deg": safety["shoulder_lift"]["mean_abs_delta_deg"],
            "elbow_flex_immediate_delta_mean_abs_deg": safety["elbow_flex"]["mean_abs_delta_deg"],
            "shoulder_lift_immediate_would_clamp_frac": safety["shoulder_lift"]["frac_would_clamp"],
            "elbow_flex_immediate_would_clamp_frac": safety["elbow_flex"]["frac_would_clamp"],
            "shoulder_lift_chunk_mean_abs_deg": chunk["shoulder_lift"]["chunk_mean_abs_delta_over_all_k"]["mean"],
            "elbow_flex_chunk_mean_abs_deg": chunk["elbow_flex"]["chunk_mean_abs_delta_over_all_k"]["mean"],
            "shoulder_lift_chunk_max_abs_deg": chunk["shoulder_lift"]["chunk_max_abs_delta_over_all_k"]["mean"],
            "elbow_flex_chunk_max_abs_deg": chunk["elbow_flex"]["chunk_max_abs_delta_over_all_k"]["mean"],
            "shoulder_lift_frac_static_frames_chunkmean_exceeds_would_clamp": chunk["shoulder_lift"]["fraction_static_frames_where_chunk_mean_exceeds_would_clamp"],
            "elbow_flex_frac_static_frames_chunkmean_exceeds_would_clamp": chunk["elbow_flex"]["fraction_static_frames_where_chunk_mean_exceeds_would_clamp"],
        }
        for j in JOINTS:
            row[f"{j}_start_std_deg"] = div["per_joint_start_frame0_stats"][j]["std"]
        dist_rows.append(row)
    with open(OUT_DIR / "old35_vs_reinforcement30_distributional_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
        w.writeheader()
        w.writerows(dist_rows)

    # ---- summary.json ----
    summary = {
        "checkpoint_metrics": ckpt_rows,
        "first_action_diagnostics": fa_rows,
        "temporal_chunk_error": temp_rows,
        "old35_vs_reinforcement30_distributional_stats": dist_rows,
        "raw": {
            "r30_train_log_info": {k: v for k, v in r30_train_log.items()},
            "r30_offline_eval": r30_offline,
            "r30_in_sample_fit": r30_in_sample,
            "r30_sweeps": r30_sweeps,
            "r30_temporal": {f"reinforcement30_only_{s}": temporal[("reinforcement30_only", s)] for s in STEPS},
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    for fname in ["checkpoint_metrics.csv", "first_action_diagnostics.csv", "temporal_chunk_error.csv",
                  "old35_vs_reinforcement30_distributional_stats.csv", "summary.json"]:
        print(f"wrote {OUT_DIR / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
