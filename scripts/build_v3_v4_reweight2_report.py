#!/usr/bin/env python3
"""Assemble reports/pick_drop_v3_v4_reweight2_v1/ - V3:V4=2:1 sampling ablation on top of the
V3+V4 uniform dataset-composition experiment. Read-only aggregation of already-completed run
artifacts - no training/eval performed here.
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
OUT_DIR = PROJECT_ROOT / "reports" / "pick_drop_v3_v4_reweight2_v1"
REPORTS = PROJECT_ROOT / "reports"
NEW_TRAIN_LOG = Path(
    "/tmp/claude-1000/-home-rlack-Projects-physical-ai-dummy/4e197d46-f2bc-4543-b4b8-06756698c808/"
    "scratchpad/train_v3_v4_reweight2.log"
)


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- offline heldout eval (historical test10, all) + V4 heldout6 (D, F, G) ----
    a_offline = load(REPORTS / "pick_drop_combined65_offline_eval" / "summary.json")
    b_offline = load(REPORTS / "combined65_reweight_new2_old1_v1_offline_eval" / "summary.json")
    b_insample = load(REPORTS / "combined65_reweight_new2_old1_v1_offline_eval" / "in_sample_fit.json")
    c_offline = load(REPORTS / "combined65_reweight_new3_old1_v1_offline_eval" / "summary.json")
    c_insample = load(REPORTS / "combined65_reweight_new3_old1_v1_offline_eval" / "in_sample_fit.json")
    e_offline = load(REPORTS / "combined65_reweight2_early_action_v1_offline_eval" / "summary.json")
    e_insample = load(REPORTS / "combined65_reweight2_early_action_v1_offline_eval" / "in_sample_fit.json")
    f_hist_offline = load(REPORTS / "pick_drop_v3_v4_combined69_uniform_v1_historical_offline_eval" / "summary.json")
    f_hist_insample = load(REPORTS / "pick_drop_v3_v4_combined69_uniform_v1_historical_offline_eval" / "in_sample_fit.json")
    f_v4h6_offline = load(REPORTS / "pick_drop_v3_v4_combined69_uniform_v1_v4heldout6_offline_eval" / "summary.json")
    g_hist_offline = load(REPORTS / "pick_drop_v3_v4_reweight2_v1_historical_offline_eval" / "summary.json")
    g_hist_insample = load(REPORTS / "pick_drop_v3_v4_reweight2_v1_historical_offline_eval" / "in_sample_fit.json")
    g_v4h6_offline = load(REPORTS / "pick_drop_v3_v4_reweight2_v1_v4heldout6_offline_eval" / "summary.json")

    d_hist_csv = list(csv.DictReader(open(REPORTS / "pick_drop_v4_fresh_training" / "historical_test10_metrics.csv", encoding="utf-8")))
    d_v4h6_csv = [r for r in csv.DictReader(open(REPORTS / "pick_drop_v4_fresh_training" / "v4_heldout6_metrics.csv", encoding="utf-8")) if r["row_type"] == "checkpoint_level"]

    # ---- first-action seed sweeps ----
    a_sweeps = {s: load(REPORTS / "pick_drop_combined65_fresh_training" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    b_sweeps = {s: load(REPORTS / "combined65_reweight_new2_old1_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    c_sweeps = {s: load(REPORTS / "combined65_reweight_new3_old1_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    d_sweeps = {s: load(REPORTS / "pick_drop_v4_fresh_training" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    e_sweeps = {s: load(REPORTS / "combined65_reweight2_early_action_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    f_sweeps = {s: load(REPORTS / "pick_drop_v3_v4_combined69_uniform_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    g_sweeps = {s: load(OUT_DIR / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}

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
    e_csv = list(csv.DictReader(open(REPORTS / "combined65_reweight2_early_action_v1" / "checkpoint_metrics.csv", encoding="utf-8")))
    e_tr = {int(r["checkpoint_step"]): {"train_loss": float(r["train_loss"]), "wall_time_since_start_min": float(r["wall_time_min"])} for r in e_csv if r["experiment"] == "combined65_reweight2_1_early_action"}
    f_csv = list(csv.DictReader(open(REPORTS / "pick_drop_v3_v4_combined69_uniform_v1" / "checkpoint_metrics.csv", encoding="utf-8")))
    f_tr = {int(r["checkpoint_step"]): {"train_loss": float(r["train_loss"]), "wall_time_since_start_min": float(r["wall_time_min"])} for r in f_csv if r["experiment"] == "F_v3_v4_uniform_THIS"}
    g_train_log = parse_training_log(NEW_TRAIN_LOG)
    g_tr = {r["checkpoint_step"]: r for r in g_train_log["checkpoint_rows"]}

    # ---- temporal chunk error (A-G) ----
    temporal_a = {s: load(REPORTS / "pick_drop_combined65_early_weight_v1" / "temporal_chunk_error" / f"baseline_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_b = {s: load(REPORTS / "combined65_reweight_new2_old1_v1" / "temporal_chunk_error" / f"reweight_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_c = {s: load(REPORTS / "combined65_reweight_new3_old1_v1" / "temporal_chunk_error" / f"reweight_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_d = {s: load(REPORTS / "pick_drop_v4_fresh_training" / "temporal_chunk_error" / f"v4_fresh_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_e = {s: load(REPORTS / "combined65_reweight2_early_action_v1" / "temporal_chunk_error" / f"reweight2_early_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_f = {s: load(REPORTS / "pick_drop_v3_v4_combined69_uniform_v1" / "temporal_chunk_error" / f"v3_v4_uniform_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_g = {s: load(OUT_DIR / "temporal_chunk_error" / f"v3_v4_reweight2_{s}" / "temporal_chunk_error.json") for s in STEPS}

    # =========================================================================
    # checkpoint_metrics.csv
    # =========================================================================
    def by_step(offline):
        return {r["checkpoint_step"]: r for r in offline["rows"]}

    a_off, b_off, c_off, e_off = (by_step(x) for x in (a_offline, b_offline, c_offline, e_offline))
    f_hist_off = by_step(f_hist_offline)
    f_v4h6_off = by_step(f_v4h6_offline)
    g_hist_off = by_step(g_hist_offline)
    g_v4h6_off = by_step(g_v4h6_offline)
    b_ins_by = {r["checkpoint_step"]: r for r in b_insample["rows"]}
    c_ins_by = {r["checkpoint_step"]: r for r in c_insample["rows"]}
    e_ins_by = {r["checkpoint_step"]: r for r in e_insample["rows"]}
    f_ins_by = {r["checkpoint_step"]: r for r in f_hist_insample["rows"]}
    g_ins_by = {r["checkpoint_step"]: r for r in g_hist_insample["rows"]}
    d_hist_by = {int(r["checkpoint_step"]): r for r in d_hist_csv}
    d_v4h6_by = {int(r["checkpoint_step"]): r for r in d_v4h6_csv}

    ckpt_rows = []
    for s in STEPS:
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
        d_h = d_hist_by.get(s); d_v = d_v4h6_by.get(s)
        ckpt_rows.append({
            "experiment": "D_v4_standalone", "checkpoint_step": s,
            "train_loss": d_tr[s]["train_loss"] if s in d_tr else None,
            "wall_time_min": d_tr[s]["wall_time_since_start_min"] if s in d_tr else None,
            "historical_test10_mae": float(d_h["action_mae"]) if d_h else None,
            "v4_heldout6_mae": float(d_v["action_mae"]) if d_v else None,
            "in_sample_mae": None, "historical_gap": None,
        })
        f_h = f_hist_off.get(s); f_v = f_v4h6_off.get(s); f_ins = f_ins_by.get(s)
        f_heldout = f_h["action_mae"] if f_h else None
        f_insamp = f_ins["in_sample_action_mae"] if f_ins else None
        ckpt_rows.append({
            "experiment": "F_v3_v4_uniform", "checkpoint_step": s,
            "train_loss": f_tr[s]["train_loss"] if s in f_tr else None,
            "wall_time_min": f_tr[s]["wall_time_since_start_min"] if s in f_tr else None,
            "historical_test10_mae": f_heldout,
            "v4_heldout6_mae": f_v["action_mae"] if f_v else None,
            "in_sample_mae": f_insamp,
            "historical_gap": (f_heldout - f_insamp) if (f_heldout is not None and f_insamp is not None) else None,
        })
        g_h = g_hist_off.get(s); g_v = g_v4h6_off.get(s); g_ins = g_ins_by.get(s)
        g_heldout = g_h["action_mae"] if g_h else None
        g_insamp = g_ins["in_sample_action_mae"] if g_ins else None
        ckpt_rows.append({
            "experiment": "G_v3_v4_reweight2_1_THIS", "checkpoint_step": s,
            "train_loss": g_tr[s]["train_loss"] if s in g_tr else None,
            "wall_time_min": g_tr[s]["wall_time_since_start_min"] if s in g_tr else None,
            "historical_test10_mae": g_heldout,
            "v4_heldout6_mae": g_v["action_mae"] if g_v else None,
            "in_sample_mae": g_insamp,
            "historical_gap": (g_heldout - g_insamp) if (g_heldout is not None and g_insamp is not None) else None,
        })
    with open(OUT_DIR / "checkpoint_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ckpt_rows[0].keys()))
        w.writeheader(); w.writerows(ckpt_rows)

    # =========================================================================
    # first_action_diagnostics.csv
    # =========================================================================
    fa_rows = []

    def add_sweep_rows(exp_label, sweep, step):
        summ = sweep["summary"]
        for j in JOINTS:
            pj = summ["per_joint"][j]
            fa_rows.append({
                "experiment": exp_label, "checkpoint_step": step, "joint": j,
                "delta_mean_deg": pj["mean"], "delta_std_deg": pj["std"],
                "would_clamp_threshold_deg": pj["threshold_deg"], "clamp_rate": pj["clamp_rate"],
                "n_seeds": summ["n_seeds"], "clamp_free_seed_count": summ["clamp_free_seed_count"],
                "l2_error_vs_gt_mean": summ["l2_error_vs_gt"]["mean"],
            })

    for label, sweeps in [
        ("A_v2_v3_uniform", a_sweeps), ("B_v2_v3_reweight2_1", b_sweeps), ("C_v2_v3_reweight3_1", c_sweeps),
        ("D_v4_standalone", d_sweeps), ("E_v2_v3_reweight2_1_early", e_sweeps), ("F_v3_v4_uniform", f_sweeps),
        ("G_v3_v4_reweight2_1_THIS", g_sweeps),
    ]:
        for s in STEPS:
            add_sweep_rows(label, sweeps[s], s)

    with open(OUT_DIR / "first_action_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fa_rows[0].keys()))
        w.writeheader(); w.writerows(fa_rows)

    # =========================================================================
    # temporal_chunk_error.csv (A-G)
    # =========================================================================
    temp_rows = []
    for label, temporal in [
        ("A_v2_v3_uniform", temporal_a), ("B_v2_v3_reweight2_1", temporal_b), ("C_v2_v3_reweight3_1", temporal_c),
        ("D_v4_standalone", temporal_d), ("E_v2_v3_reweight2_1_early", temporal_e), ("F_v3_v4_uniform", temporal_f),
        ("G_v3_v4_reweight2_1_THIS", temporal_g),
    ]:
        for s in STEPS:
            d = temporal[s]
            for bucket_name, b in d["buckets"].items():
                temp_rows.append({
                    "experiment": label, "checkpoint_step": s, "bucket": bucket_name,
                    "mae_shoulder_lift_deg": b["mae_per_joint"]["shoulder_lift"],
                    "mae_elbow_flex_deg": b["mae_per_joint"]["elbow_flex"],
                    "mae_key_joints_mean_deg": b["mae_key_joints_mean"],
                    "nearest_demo_episode": d["nearest_demo_match"]["episode"],
                    "nearest_demo_frame": d["nearest_demo_match"]["frame"],
                })
    with open(OUT_DIR / "temporal_chunk_error.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(temp_rows[0].keys()))
        w.writeheader(); w.writerows(temp_rows)

    # =========================================================================
    # comparison_vs_reweight2.csv (G vs A [accuracy baseline] and G vs F [safety baseline])
    # =========================================================================
    def sweep_metric(sweep, joint):
        return sweep["summary"]["per_joint"][joint]

    cmp_rows = []
    for s in STEPS:
        a_h = a_off.get(s); f_h = f_hist_off.get(s); g_h = g_hist_off.get(s)
        f_ins = f_ins_by.get(s); g_ins = g_ins_by.get(s)
        a_sw = a_sweeps[s]["summary"]; f_sw = f_sweeps[s]["summary"]; g_sw = g_sweeps[s]["summary"]
        a_sl = sweep_metric(a_sweeps[s], "shoulder_lift"); f_sl = sweep_metric(f_sweeps[s], "shoulder_lift"); g_sl = sweep_metric(g_sweeps[s], "shoulder_lift")
        a_ef = sweep_metric(a_sweeps[s], "elbow_flex"); f_ef = sweep_metric(f_sweeps[s], "elbow_flex"); g_ef = sweep_metric(g_sweeps[s], "elbow_flex")
        f_t = temporal_f[s]["buckets"]; g_t = temporal_g[s]["buckets"]

        a_heldout = a_h["action_mae"] if a_h else None
        f_heldout = f_h["action_mae"] if f_h else None
        g_heldout = g_h["action_mae"] if g_h else None
        f_gap = (f_heldout - f_ins["in_sample_action_mae"]) if (f_h and f_ins) else None
        g_gap = (g_heldout - g_ins["in_sample_action_mae"]) if (g_h and g_ins) else None
        f_clampfree = f_sw["clamp_free_seed_count"] / f_sw["n_seeds"]
        g_clampfree = g_sw["clamp_free_seed_count"] / g_sw["n_seeds"]

        cmp_rows.append({
            "checkpoint_step": s,
            "A_v2v3_reweight2_1_historical_mae": a_heldout,
            "F_v3v4_uniform_historical_mae": f_heldout,
            "G_v3v4_reweight2_1_historical_mae": g_heldout,
            "F_gap": f_gap, "G_gap": g_gap,
            "F_clamp_free_rate": f_clampfree, "G_clamp_free_rate": g_clampfree,
            "F_shoulder_lift_clamp_rate": f_sl["clamp_rate"], "G_shoulder_lift_clamp_rate": g_sl["clamp_rate"],
            "F_elbow_flex_clamp_rate": f_ef["clamp_rate"], "G_elbow_flex_clamp_rate": g_ef["clamp_rate"],
            "F_l2_vs_gt": f_sw["l2_error_vs_gt"]["mean"], "G_l2_vs_gt": g_sw["l2_error_vs_gt"]["mean"],
            "F_step3_plus_key_joint_mae": f_t["step3_plus"]["mae_key_joints_mean"],
            "G_step3_plus_key_joint_mae": g_t["step3_plus"]["mae_key_joints_mean"],
        })
    with open(OUT_DIR / "comparison_vs_reweight2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader(); w.writerows(cmp_rows)

    # =========================================================================
    # summary.json
    # =========================================================================
    summary = {
        "checkpoint_metrics": ckpt_rows,
        "first_action_diagnostics": fa_rows,
        "temporal_chunk_error": temp_rows,
        "comparison_vs_reweight2": cmp_rows,
        "raw": {
            "g_train_log_info": {k: v for k, v in g_train_log.items() if k != "all_steps"},
            "g_historical_offline_eval": g_hist_offline,
            "g_v4heldout6_offline_eval": g_v4h6_offline,
            "g_in_sample_fit": g_hist_insample,
            "g_sweeps_summary": {s: g_sweeps[s]["summary"] for s in STEPS},
            "g_temporal": {s: temporal_g[s] for s in STEPS},
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    for fname in ["checkpoint_metrics.csv", "first_action_diagnostics.csv", "temporal_chunk_error.csv", "comparison_vs_reweight2.csv", "summary.json"]:
        print(f"wrote {OUT_DIR / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
