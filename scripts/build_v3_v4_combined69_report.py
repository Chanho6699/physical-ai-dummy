#!/usr/bin/env python3
"""Assemble reports/pick_drop_v3_v4_combined69_uniform_v1/ - dataset-composition ablation:
replace V2 with V4 in the "best" V2+V3 pool (V3 30ep + V4 39ep = 69ep, uniform sampling, no
loss tricks) and compare against the full A-F lineage. Read-only aggregation of already-completed
run artifacts - no training/eval performed here.
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
OUT_DIR = PROJECT_ROOT / "reports" / "pick_drop_v3_v4_combined69_uniform_v1"
REPORTS = PROJECT_ROOT / "reports"
NEW_TRAIN_LOG = Path(
    "/tmp/claude-1000/-home-rlack-Projects-physical-ai-dummy/4e197d46-f2bc-4543-b4b8-06756698c808/"
    "scratchpad/train_v3_v4_combined69_uniform.log"
)


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- offline heldout eval (historical test10, all experiments) + V4 heldout6 (D, F only) ----
    a_offline = load(REPORTS / "pick_drop_combined65_offline_eval" / "summary.json")
    b_offline = load(REPORTS / "combined65_reweight_new2_old1_v1_offline_eval" / "summary.json")
    b_insample = load(REPORTS / "combined65_reweight_new2_old1_v1_offline_eval" / "in_sample_fit.json")
    c_offline = load(REPORTS / "combined65_reweight_new3_old1_v1_offline_eval" / "summary.json")
    c_insample = load(REPORTS / "combined65_reweight_new3_old1_v1_offline_eval" / "in_sample_fit.json")
    e_offline = load(REPORTS / "combined65_reweight2_early_action_v1_offline_eval" / "summary.json")
    e_insample = load(REPORTS / "combined65_reweight2_early_action_v1_offline_eval" / "in_sample_fit.json")
    f_hist_offline = load(OUT_DIR.parent / "pick_drop_v3_v4_combined69_uniform_v1_historical_offline_eval" / "summary.json")
    f_hist_insample = load(OUT_DIR.parent / "pick_drop_v3_v4_combined69_uniform_v1_historical_offline_eval" / "in_sample_fit.json")
    f_v4h6_offline = load(OUT_DIR.parent / "pick_drop_v3_v4_combined69_uniform_v1_v4heldout6_offline_eval" / "summary.json")

    # D (V4 standalone) already-computed heldout numbers, reused verbatim
    d_hist_csv = list(csv.DictReader(open(REPORTS / "pick_drop_v4_fresh_training" / "historical_test10_metrics.csv", encoding="utf-8")))
    d_v4h6_csv = [r for r in csv.DictReader(open(REPORTS / "pick_drop_v4_fresh_training" / "v4_heldout6_metrics.csv", encoding="utf-8")) if r["row_type"] == "checkpoint_level"]

    # ---- first-action seed sweeps ----
    a_sweeps = {s: load(REPORTS / "pick_drop_combined65_fresh_training" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    b_sweeps = {s: load(REPORTS / "combined65_reweight_new2_old1_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    c_sweeps = {s: load(REPORTS / "combined65_reweight_new3_old1_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    d_sweeps = {s: load(REPORTS / "pick_drop_v4_fresh_training" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    e_sweeps = {s: load(REPORTS / "combined65_reweight2_early_action_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    f_sweeps = {s: load(OUT_DIR / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}

    # ---- training logs / train_loss+wall_time ----
    prior_ckpt_csv = REPORTS / "combined65_reweight_new3_old1_v1" / "checkpoint_metrics.csv"
    prior_rows: dict[tuple[str, int], dict[str, str]] = {}
    with open(prior_ckpt_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prior_rows[(row["experiment"], int(row["checkpoint_step"]))] = row

    def prior_tr(exp: str) -> dict[int, dict[str, Any]]:
        out = {}
        for s in STEPS:
            r = prior_rows.get((exp, s))
            if r and r["train_loss"]:
                out[s] = {"train_loss": float(r["train_loss"]), "wall_time_since_start_min": float(r["wall_time_min"])}
        return out

    a_tr = prior_tr("combined65_baseline")
    b_tr = prior_tr("combined65_reweight_new2_old1_v1")
    c_tr = prior_tr("combined65_reweight_new3_old1_v1")
    d_tr = {int(r["checkpoint_step"]): {"train_loss": float(r["train_loss"]), "wall_time_since_start_min": float(r["wall_time_since_start_min"])} for r in d_hist_csv}
    e_train_log_csv = list(csv.DictReader(open(REPORTS / "combined65_reweight2_early_action_v1" / "checkpoint_metrics.csv", encoding="utf-8")))
    e_tr = {int(r["checkpoint_step"]): {"train_loss": float(r["train_loss"]), "wall_time_since_start_min": float(r["wall_time_min"])} for r in e_train_log_csv if r["experiment"] == "combined65_reweight2_1_early_action"}
    f_train_log = parse_training_log(NEW_TRAIN_LOG)
    f_tr = {r["checkpoint_step"]: r for r in f_train_log["checkpoint_rows"]}

    # ---- temporal chunk error (A-F) ----
    temporal_a = {s: load(REPORTS / "pick_drop_combined65_early_weight_v1" / "temporal_chunk_error" / f"baseline_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_b = {s: load(REPORTS / "combined65_reweight_new2_old1_v1" / "temporal_chunk_error" / f"reweight_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_c = {s: load(REPORTS / "combined65_reweight_new3_old1_v1" / "temporal_chunk_error" / f"reweight_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_d = {s: load(REPORTS / "pick_drop_v4_fresh_training" / "temporal_chunk_error" / f"v4_fresh_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_e = {s: load(REPORTS / "combined65_reweight2_early_action_v1" / "temporal_chunk_error" / f"reweight2_early_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_f = {s: load(OUT_DIR / "temporal_chunk_error" / f"v3_v4_uniform_{s}" / "temporal_chunk_error.json") for s in STEPS}

    # =========================================================================
    # checkpoint_metrics.csv
    # =========================================================================
    def by_step(offline):
        return {r["checkpoint_step"]: r for r in offline["rows"]}

    a_off, b_off, c_off, e_off = (by_step(x) for x in (a_offline, b_offline, c_offline, e_offline))
    f_hist_off = by_step(f_hist_offline)
    f_v4h6_off = by_step(f_v4h6_offline)
    b_ins_by = {r["checkpoint_step"]: r for r in b_insample["rows"]}
    c_ins_by = {r["checkpoint_step"]: r for r in c_insample["rows"]}
    e_ins_by = {r["checkpoint_step"]: r for r in e_insample["rows"]}
    f_ins_by = {r["checkpoint_step"]: r for r in f_hist_insample["rows"]}
    d_hist_by = {int(r["checkpoint_step"]): r for r in d_hist_csv}
    d_v4h6_by = {int(r["checkpoint_step"]): r for r in d_v4h6_csv}

    ckpt_rows = []
    for s in STEPS:
        # A/B/C/E: historical heldout only (no V4 heldout6 relevant - A/B/C/E never trained with V4 data)
        for exp, tr, off, insample in [
            ("A_v2_v3_uniform", a_tr.get(s), a_off.get(s), None),
            ("B_v2_v3_reweight2_1", b_tr.get(s), b_off.get(s), b_ins_by.get(s)),
            ("C_v2_v3_reweight3_1", c_tr.get(s), c_off.get(s), c_ins_by.get(s)),
            ("E_v2_v3_reweight2_1_early", e_tr.get(s), e_off.get(s), e_ins_by.get(s)),
        ]:
            heldout = off["action_mae"] if off else None
            insamp = insample["in_sample_action_mae"] if insample else None
            ckpt_rows.append(
                {
                    "experiment": exp, "checkpoint_step": s,
                    "train_loss": tr["train_loss"] if tr else None,
                    "wall_time_min": tr["wall_time_since_start_min"] if tr else None,
                    "historical_test10_mae": heldout, "v4_heldout6_mae": None,
                    "in_sample_mae": insamp,
                    "historical_gap": (heldout - insamp) if (heldout is not None and insamp is not None) else None,
                }
            )
        # D: V4 standalone - historical test10 + its own v4 heldout6, no in-sample computed here (reuse existing report's own numbers, gap n/a)
        d_h = d_hist_by.get(s)
        d_v = d_v4h6_by.get(s)
        ckpt_rows.append(
            {
                "experiment": "D_v4_standalone", "checkpoint_step": s,
                "train_loss": d_tr[s]["train_loss"] if s in d_tr else None,
                "wall_time_min": d_tr[s]["wall_time_since_start_min"] if s in d_tr else None,
                "historical_test10_mae": float(d_h["action_mae"]) if d_h else None,
                "v4_heldout6_mae": float(d_v["action_mae"]) if d_v else None,
                "in_sample_mae": None, "historical_gap": None,
            }
        )
        # F: this experiment - both heldout sets + in-sample (measured against its own combined69 train set)
        f_h = f_hist_off.get(s)
        f_v = f_v4h6_off.get(s)
        f_ins = f_ins_by.get(s)
        f_heldout = f_h["action_mae"] if f_h else None
        f_insamp = f_ins["in_sample_action_mae"] if f_ins else None
        ckpt_rows.append(
            {
                "experiment": "F_v3_v4_uniform_THIS", "checkpoint_step": s,
                "train_loss": f_tr[s]["train_loss"] if s in f_tr else None,
                "wall_time_min": f_tr[s]["wall_time_since_start_min"] if s in f_tr else None,
                "historical_test10_mae": f_heldout,
                "v4_heldout6_mae": f_v["action_mae"] if f_v else None,
                "in_sample_mae": f_insamp,
                "historical_gap": (f_heldout - f_insamp) if (f_heldout is not None and f_insamp is not None) else None,
            }
        )
    with open(OUT_DIR / "checkpoint_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ckpt_rows[0].keys()))
        w.writeheader()
        w.writerows(ckpt_rows)

    # =========================================================================
    # first_action_diagnostics.csv
    # =========================================================================
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

    for label, sweeps in [
        ("A_v2_v3_uniform", a_sweeps), ("B_v2_v3_reweight2_1", b_sweeps), ("C_v2_v3_reweight3_1", c_sweeps),
        ("D_v4_standalone", d_sweeps), ("E_v2_v3_reweight2_1_early", e_sweeps), ("F_v3_v4_uniform_THIS", f_sweeps),
    ]:
        for s in STEPS:
            add_sweep_rows(label, sweeps[s], s)

    with open(OUT_DIR / "first_action_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fa_rows[0].keys()))
        w.writeheader()
        w.writerows(fa_rows)

    # =========================================================================
    # temporal_chunk_error.csv (A-F)
    # =========================================================================
    temp_rows = []
    for label, temporal in [
        ("A_v2_v3_uniform", temporal_a), ("B_v2_v3_reweight2_1", temporal_b), ("C_v2_v3_reweight3_1", temporal_c),
        ("D_v4_standalone", temporal_d), ("E_v2_v3_reweight2_1_early", temporal_e), ("F_v3_v4_uniform_THIS", temporal_f),
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
                        "nearest_demo_episode": d["nearest_demo_match"]["episode"],
                        "nearest_demo_frame": d["nearest_demo_match"]["frame"],
                    }
                )
    with open(OUT_DIR / "temporal_chunk_error.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(temp_rows[0].keys()))
        w.writeheader()
        w.writerows(temp_rows)

    # =========================================================================
    # comparison_vs_reweight2.csv (F vs B, checkpoint-by-checkpoint, key metrics)
    # =========================================================================
    def sweep_metric(sweep, joint):
        return sweep["summary"]["per_joint"][joint]

    cmp_rows = []
    for s in STEPS:
        b_h = b_off.get(s)
        f_h = f_hist_off.get(s)
        b_ins = b_ins_by.get(s)
        f_ins = f_ins_by.get(s)
        b_sw = b_sweeps[s]["summary"]
        f_sw = f_sweeps[s]["summary"]
        b_sl = sweep_metric(b_sweeps[s], "shoulder_lift")
        f_sl = sweep_metric(f_sweeps[s], "shoulder_lift")
        b_ef = sweep_metric(b_sweeps[s], "elbow_flex")
        f_ef = sweep_metric(f_sweeps[s], "elbow_flex")
        b_t = temporal_b[s]["buckets"]
        f_t = temporal_f[s]["buckets"]

        b_heldout = b_h["action_mae"] if b_h else None
        f_heldout = f_h["action_mae"] if f_h else None
        b_gap = (b_heldout - b_ins["in_sample_action_mae"]) if (b_h and b_ins) else None
        f_gap = (f_heldout - f_ins["in_sample_action_mae"]) if (f_h and f_ins) else None
        b_clampfree = b_sw["clamp_free_seed_count"] / b_sw["n_seeds"]
        f_clampfree = f_sw["clamp_free_seed_count"] / f_sw["n_seeds"]

        cmp_rows.append(
            {
                "checkpoint_step": s,
                "B_reweight2_1_historical_mae": b_heldout, "F_v3v4_historical_mae": f_heldout,
                "delta_historical_mae": (f_heldout - b_heldout) if (b_heldout is not None and f_heldout is not None) else None,
                "B_gap": b_gap, "F_gap": f_gap,
                "delta_gap": (f_gap - b_gap) if (b_gap is not None and f_gap is not None) else None,
                "B_clamp_free_rate": b_clampfree, "F_clamp_free_rate": f_clampfree,
                "B_shoulder_lift_clamp_rate": b_sl["clamp_rate"], "F_shoulder_lift_clamp_rate": f_sl["clamp_rate"],
                "B_elbow_flex_clamp_rate": b_ef["clamp_rate"], "F_elbow_flex_clamp_rate": f_ef["clamp_rate"],
                "B_l2_vs_gt": b_sw["l2_error_vs_gt"]["mean"], "F_l2_vs_gt": f_sw["l2_error_vs_gt"]["mean"],
                "B_step0_key_joint_mae": b_t["step0"]["mae_key_joints_mean"], "F_step0_key_joint_mae": f_t["step0"]["mae_key_joints_mean"],
                "B_step1_2_key_joint_mae": b_t["step1_2"]["mae_key_joints_mean"], "F_step1_2_key_joint_mae": f_t["step1_2"]["mae_key_joints_mean"],
                "B_step3_plus_key_joint_mae": b_t["step3_plus"]["mae_key_joints_mean"], "F_step3_plus_key_joint_mae": f_t["step3_plus"]["mae_key_joints_mean"],
            }
        )
    with open(OUT_DIR / "comparison_vs_reweight2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader()
        w.writerows(cmp_rows)

    # =========================================================================
    # summary.json
    # =========================================================================
    summary = {
        "checkpoint_metrics": ckpt_rows,
        "first_action_diagnostics": fa_rows,
        "temporal_chunk_error": temp_rows,
        "comparison_vs_reweight2": cmp_rows,
        "raw": {
            "f_train_log_info": {k: v for k, v in f_train_log.items() if k != "all_steps"},
            "f_historical_offline_eval": f_hist_offline,
            "f_v4heldout6_offline_eval": f_v4h6_offline,
            "f_in_sample_fit": f_hist_insample,
            "f_sweeps_summary": {s: f_sweeps[s]["summary"] for s in STEPS},
            "f_temporal": {s: temporal_f[s] for s in STEPS},
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    for fname in ["checkpoint_metrics.csv", "first_action_diagnostics.csv", "temporal_chunk_error.csv", "comparison_vs_reweight2.csv", "summary.json"]:
        print(f"wrote {OUT_DIR / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
