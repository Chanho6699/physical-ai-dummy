#!/usr/bin/env python3
"""Assemble reports/combined65_reweight2_early_action_v1/ - the fresh 10k ablation that adds
early_action_loss_weights=[3.0,2.0,2.0] on top of the current best strategy (combined65 reweight
2:1 old35:reinforcement30). Single-variable change vs `combined65_reweight_new2_old1_v1`: loss
weighting only. Sampling ratio, dataset, architecture, optimizer/scheduler, chunk horizon, seed,
Safety Gate thresholds all unchanged. Read-only aggregation of already-completed run artifacts -
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
OUT_DIR = PROJECT_ROOT / "reports" / "combined65_reweight2_early_action_v1"
REPORTS = PROJECT_ROOT / "reports"
NEW_TRAIN_LOG = Path(
    "/tmp/claude-1000/-home-rlack-Projects-physical-ai-dummy/4e197d46-f2bc-4543-b4b8-06756698c808/"
    "scratchpad/train_reweight2_early.log"
)


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- offline heldout eval + in-sample fit ----
    c65_offline = load(REPORTS / "pick_drop_combined65_offline_eval" / "summary.json")
    ew_offline = load(REPORTS / "pick_drop_combined65_early_weight_offline_eval" / "summary.json")
    rw2_offline = load(REPORTS / "combined65_reweight_new2_old1_v1_offline_eval" / "summary.json")
    rw2_insample = load(REPORTS / "combined65_reweight_new2_old1_v1_offline_eval" / "in_sample_fit.json")
    rw3_offline = load(REPORTS / "combined65_reweight_new3_old1_v1_offline_eval" / "summary.json")
    rw3_insample = load(REPORTS / "combined65_reweight_new3_old1_v1_offline_eval" / "in_sample_fit.json")
    new_offline = load(REPORTS / "combined65_reweight2_early_action_v1_offline_eval" / "summary.json")
    new_insample = load(REPORTS / "combined65_reweight2_early_action_v1_offline_eval" / "in_sample_fit.json")

    # ---- first-action seed sweeps ----
    c65_sweeps = {s: load(REPORTS / "pick_drop_combined65_fresh_training" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    ew_sweeps = {s: load(REPORTS / "pick_drop_combined65_early_weight_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    rw2_sweeps = {s: load(REPORTS / "combined65_reweight_new2_old1_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    rw3_sweeps = {s: load(REPORTS / "combined65_reweight_new3_old1_v1" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    v4_sweeps = {s: load(REPORTS / "pick_drop_v4_fresh_training" / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}
    new_sweeps = {s: load(OUT_DIR / f"first_action_seed_sweep_{s}" / "seed_sweep.json") for s in STEPS}

    # ---- training logs ----
    # The original /tmp/*.log files from prior experiments (combined65 baseline, early-weight-only,
    # reweight2:1, reweight3:1) no longer exist on disk (session /tmp is not persistent) - their
    # train_loss/wall_time_min were already captured verbatim in the prior
    # combined65_reweight_new3_old1_v1/checkpoint_metrics.csv, which is reused here read-only
    # instead of re-deriving them. Only this new experiment's log is freshly parsed.
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

    c65_tr = prior_tr("combined65_baseline")
    ew_tr = prior_tr("combined65_early_weight_v1")
    rw2_tr = prior_tr("combined65_reweight_new2_old1_v1")
    rw3_tr = prior_tr("combined65_reweight_new3_old1_v1")
    new_train_log = parse_training_log(NEW_TRAIN_LOG)

    # ---- temporal chunk error (A: combined65 uniform, B: reweight2:1, C: reweight3:1, D: V4 fresh, E: this experiment) ----
    temporal_a = {s: load(REPORTS / "pick_drop_combined65_early_weight_v1" / "temporal_chunk_error" / f"baseline_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_b = {s: load(REPORTS / "combined65_reweight_new2_old1_v1" / "temporal_chunk_error" / f"reweight_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_c = {s: load(REPORTS / "combined65_reweight_new3_old1_v1" / "temporal_chunk_error" / f"reweight_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_d = {s: load(REPORTS / "pick_drop_v4_fresh_training" / "temporal_chunk_error" / f"v4_fresh_{s}" / "temporal_chunk_error.json") for s in STEPS}
    temporal_e = {s: load(OUT_DIR / "temporal_chunk_error" / f"reweight2_early_{s}" / "temporal_chunk_error.json") for s in STEPS}

    # =========================================================================
    # checkpoint_metrics.csv
    # =========================================================================
    def by_step(offline):
        return {r["checkpoint_step"]: r for r in offline["rows"]}

    c65_off, ew_off, rw2_off, rw3_off, new_off = (by_step(x) for x in (c65_offline, ew_offline, rw2_offline, rw3_offline, new_offline))
    rw2_ins_by = {r["checkpoint_step"]: r for r in rw2_insample["rows"]}
    rw3_ins_by = {r["checkpoint_step"]: r for r in rw3_insample["rows"]}
    new_ins_by = {r["checkpoint_step"]: r for r in new_insample["rows"]}
    new_tr = {r["checkpoint_step"]: r for r in new_train_log["checkpoint_rows"]}

    ckpt_rows = []
    for s in STEPS:
        for exp, tr, off, insample in [
            ("combined65_baseline_uniform", c65_tr.get(s), c65_off.get(s), None),
            ("combined65_early_weight_only", ew_tr.get(s), ew_off.get(s), None),
            ("combined65_reweight2_1", rw2_tr.get(s), rw2_off.get(s), rw2_ins_by.get(s)),
            ("combined65_reweight3_1", rw3_tr.get(s), rw3_off.get(s), rw3_ins_by.get(s)),
            ("combined65_reweight2_1_early_action", new_tr.get(s), new_off.get(s), new_ins_by.get(s)),
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
        ("combined65_baseline_uniform", c65_sweeps),
        ("combined65_early_weight_only", ew_sweeps),
        ("combined65_reweight2_1", rw2_sweeps),
        ("combined65_reweight3_1", rw3_sweeps),
        ("v4_fresh", v4_sweeps),
        ("combined65_reweight2_1_early_action", new_sweeps),
    ]:
        for s in STEPS:
            add_sweep_rows(label, sweeps[s], s)

    with open(OUT_DIR / "first_action_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fa_rows[0].keys()))
        w.writeheader()
        w.writerows(fa_rows)

    # =========================================================================
    # temporal_chunk_error.csv  (comparison A-E)
    # =========================================================================
    temp_rows = []
    for label, temporal in [
        ("A_combined65_uniform", temporal_a),
        ("B_reweight2_1", temporal_b),
        ("C_reweight3_1", temporal_c),
        ("D_v4_fresh", temporal_d),
        ("E_reweight2_1_early_action", temporal_e),
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
    # comparison_vs_reweight2.csv  (E vs B, checkpoint-by-checkpoint, key metrics only)
    # =========================================================================
    def sweep_metric(sweep, joint):
        return sweep["summary"]["per_joint"][joint]

    cmp_rows = []
    for s in STEPS:
        b_off = rw2_off.get(s)
        e_off = new_off.get(s)
        b_ins = rw2_ins_by.get(s)
        e_ins = new_ins_by.get(s)
        b_sw = rw2_sweeps[s]["summary"]
        e_sw = new_sweeps[s]["summary"]
        b_sl = sweep_metric(rw2_sweeps[s], "shoulder_lift")
        e_sl = sweep_metric(new_sweeps[s], "shoulder_lift")
        b_ef = sweep_metric(rw2_sweeps[s], "elbow_flex")
        e_ef = sweep_metric(new_sweeps[s], "elbow_flex")
        b_t = temporal_b[s]["buckets"]
        e_t = temporal_e[s]["buckets"]

        b_heldout = b_off["action_mae"] if b_off else None
        e_heldout = e_off["action_mae"] if e_off else None
        b_gap = (b_heldout - b_ins["in_sample_action_mae"]) if (b_off and b_ins) else None
        e_gap = (e_heldout - e_ins["in_sample_action_mae"]) if (e_off and e_ins) else None
        b_clampfree = b_sw["clamp_free_seed_count"] / b_sw["n_seeds"]
        e_clampfree = e_sw["clamp_free_seed_count"] / e_sw["n_seeds"]

        cmp_rows.append(
            {
                "checkpoint_step": s,
                "baseline_reweight2_1_heldout_mae": b_heldout,
                "new_reweight2_1_early_heldout_mae": e_heldout,
                "delta_heldout_mae": (e_heldout - b_heldout) if (b_heldout is not None and e_heldout is not None) else None,
                "baseline_train_heldout_gap": b_gap,
                "new_train_heldout_gap": e_gap,
                "delta_gap": (e_gap - b_gap) if (b_gap is not None and e_gap is not None) else None,
                "baseline_clamp_free_rate": b_clampfree,
                "new_clamp_free_rate": e_clampfree,
                "baseline_shoulder_lift_clamp_rate": b_sl["clamp_rate"],
                "new_shoulder_lift_clamp_rate": e_sl["clamp_rate"],
                "baseline_elbow_flex_clamp_rate": b_ef["clamp_rate"],
                "new_elbow_flex_clamp_rate": e_ef["clamp_rate"],
                "baseline_l2_vs_gt": b_sw["l2_error_vs_gt"]["mean"],
                "new_l2_vs_gt": e_sw["l2_error_vs_gt"]["mean"],
                "baseline_step0_key_joint_mae": b_t["step0"]["mae_key_joints_mean"],
                "new_step0_key_joint_mae": e_t["step0"]["mae_key_joints_mean"],
                "baseline_step1_2_key_joint_mae": b_t["step1_2"]["mae_key_joints_mean"],
                "new_step1_2_key_joint_mae": e_t["step1_2"]["mae_key_joints_mean"],
                "baseline_step3_plus_key_joint_mae": b_t["step3_plus"]["mae_key_joints_mean"],
                "new_step3_plus_key_joint_mae": e_t["step3_plus"]["mae_key_joints_mean"],
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
            "new_train_log_info": {k: v for k, v in new_train_log.items() if k != "all_steps"},
            "new_offline_eval": new_offline,
            "new_in_sample_fit": new_insample,
            "new_sweeps_summary": {s: new_sweeps[s]["summary"] for s in STEPS},
            "new_temporal": {s: temporal_e[s] for s in STEPS},
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    for fname in ["checkpoint_metrics.csv", "first_action_diagnostics.csv", "temporal_chunk_error.csv", "comparison_vs_reweight2.csv", "summary.json"]:
        print(f"wrote {OUT_DIR / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
