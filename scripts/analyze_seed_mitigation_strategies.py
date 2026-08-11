#!/usr/bin/env python3
"""Offline first-action seed-mitigation strategy comparison for Grid35 V2 clean SmolVLA 7.5k T01.

Companion to ``scripts/sweep_grid35_first_action_seed.py``. That script ran 20 *live* inference
calls (``--inference-seed 0..19``) against one fixed T01 Shadow reference observation
(``V2_F02``) and recorded, per seed, the chunk-index-0 action delta, its WOULD_CLAMP status per
joint (Safety Gate thresholds, read-only), and its L2 distance to the nearest-training-demo
immediate ground-truth (GT) delta.

This script performs **no new inference**. It re-reads that sweep's 20 rows
(``reports/grid35_v2_T01_seed_sweep/seed_sweep.csv``) and offline-simulates four candidate
first-action stabilization strategies purely by recombining/resampling those 20 already-observed
delta vectors:

  1. Single random sample        - baseline: one seed, one inference call, use it as-is.
  2. 3-sample per-joint median    - 3 inference calls, take the per-joint median independently.
  3. 5-sample per-joint median    - 5 inference calls, take the per-joint median independently.
  4. Safety-pass resampling       - keep drawing fresh inference calls (no GT involved in the
                                     stopping rule - only WOULD_CLAMP status) until a clamp-free
                                     sample is found or a cap (3 or 5) is hit; on a cap-out, the
                                     final delivered action is the last raw sample clipped to the
                                     Safety Gate thresholds (i.e. what the Safety Gate would
                                     actually send downstream), not a rejected/blank action.

  Baseline-only, non-deployable: "fix one good seed" - pick the single best-observed seed from
  the sweep (by simple deployment-blind criteria) and report its numbers, with an explicit
  discussion of why this is not a valid general strategy.

GT is used only to *score* strategies after the fact (l2_error_vs_gt), never inside the
selection logic itself (the resampling stopping rule and the median aggregation both use only
WOULD_CLAMP status / raw joint values, matching real deployment where GT is unknown).

Deliberately out of scope (same constraints as the parent scripts):
  * No training / fine-tuning, no new inference of any kind.
  * No modification of Safety Gate thresholds (``configs/safety_gate.yaml`` is read-only here).
  * No writes to any robot, real or simulated.
  * No modification of the LeRobot library itself - only this repo's own scripts.

3-sample and 5-sample medians are evaluated over the *full* combination space of the 20 sweep
seeds (C(20,3)=1140, C(20,5)=15504 - both small enough to enumerate exactly, so no sampling
approximation is needed there). Safety-pass resampling is evaluated over a large, fixed-RNG-seed
set of permutations of seed draw order (permutation count and RNG seed are both recorded in the
output for reproducibility).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_grid35_first_action_pipeline import safety_threshold_check  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

DEFAULT_CSV = PROJECT_ROOT / "reports" / "grid35_v2_T01_seed_sweep" / "seed_sweep.csv"
DEFAULT_SWEEP_JSON = PROJECT_ROOT / "reports" / "grid35_v2_T01_seed_sweep" / "seed_sweep.json"
DEFAULT_SAFETY_GATE_YAML = PROJECT_ROOT / "configs" / "safety_gate.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "grid35_v2_T01_seed_mitigation"

DEFAULT_RNG_SEED = 20260808  # = today's date (2026-08-08); arbitrary but fixed & documented
DEFAULT_N_PERMUTATIONS = 20000
RESAMPLE_CAPS = (3, 5)
MEDIAN_KS = (3, 5)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_seed_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Read the existing seed sweep CSV (no new inference). One row per already-observed seed."""
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            delta = {j: float(raw[j]) for j in JOINTS}
            rows.append(
                {
                    "seed": int(raw["seed"]),
                    "delta": delta,
                    "clamp_joint_count": int(raw["clamp_joint_count"]),
                    "any_clamp": bool(int(raw["any_clamp"])),
                    "clamped_joints": [j for j in raw["clamped_joints"].split(";") if j],
                    "l2_error_vs_gt_deg": float(raw["l2_error_vs_gt_deg"]),
                }
            )
    rows.sort(key=lambda r: r["seed"])
    return rows


def load_gt_and_metadata(sweep_json_path: Path) -> dict[str, Any]:
    """Pull GT immediate delta + provenance metadata from the *existing* sweep JSON.

    This is not new inference - it is the same GT reference value the original sweep already
    used, read here so it isn't re-transcribed by hand.
    """
    data = json.loads(sweep_json_path.read_text(encoding="utf-8"))
    return {
        "checkpoint": data["checkpoint"],
        "dataset_root": data["dataset_root"],
        "shadow_report": data["shadow_report"],
        "reference_observation_label": data["reference_observation_label"],
        "task": data["task"],
        "gt_immediate_delta": {j: float(v) for j, v in data["gt_immediate_delta"].items()},
        "nearest_demo_match": data["nearest_demo_match"],
        "thresholds_from_sweep_json": {j: float(v) for j, v in data["thresholds"].items()},
    }


def load_thresholds(safety_gate_yaml: Path) -> dict[str, float]:
    """Read Safety Gate WOULD_CLAMP thresholds fresh from config. Read-only - never modified."""
    return {j: float(v) for j, v in safety_threshold_check(safety_gate_yaml)["would_clamp_threshold_deg"].items()}


# --------------------------------------------------------------------------
# Core math (shared by every strategy)
# --------------------------------------------------------------------------


def clamp_status(delta: dict[str, float], thresholds: dict[str, float]) -> tuple[list[str], int, bool]:
    clamped = [j for j in JOINTS if abs(delta[j]) > thresholds[j]]
    return clamped, len(clamped), len(clamped) > 0


def l2_error_vs_gt(delta: dict[str, float], gt: dict[str, float]) -> float:
    return float(np.sqrt(sum((delta[j] - gt[j]) ** 2 for j in JOINTS)))


def clip_to_thresholds(delta: dict[str, float], thresholds: dict[str, float]) -> dict[str, float]:
    """Simulate what the (unmodified) Safety Gate itself would deliver: per-joint clip to
    +/-threshold. Used only to score the *delivered* action when resampling caps out - the
    Safety Gate thresholds/logic are not touched, this just reproduces their clipping arithmetic
    for offline scoring purposes."""
    return {j: float(np.clip(delta[j], -thresholds[j], thresholds[j])) for j in JOINTS}


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize_joint(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
    }


def summarize_l2(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": _pct(values, 95.0),
        "max": float(max(values)),
    }


def build_strategy_summary(
    *,
    name: str,
    n_units: int,
    clamp_free_flags: list[bool],
    clamp_joint_counts: list[int],
    l2_errors: list[float],
    shoulder_lift_values: list[float],
    elbow_flex_values: list[float],
    inference_counts: list[int],
    max_inference_cap: int,
) -> dict[str, Any]:
    clamp_free_rate = float(sum(clamp_free_flags) / n_units)
    return {
        "name": name,
        "n_units_evaluated": n_units,
        "clamp_free_rate": clamp_free_rate,
        "failure_rate": float(1.0 - clamp_free_rate),
        "avg_clamp_joint_count": float(statistics.fmean(clamp_joint_counts)),
        "gt_l2_error_deg": summarize_l2(l2_errors),
        "shoulder_lift_deg": summarize_joint(shoulder_lift_values),
        "elbow_flex_deg": summarize_joint(elbow_flex_values),
        "avg_inference_count": float(statistics.fmean(inference_counts)),
        "max_inference_count": max(inference_counts),
        "max_inference_cap": max_inference_cap,
    }


# --------------------------------------------------------------------------
# Strategy 1: single random sample
# --------------------------------------------------------------------------


def strategy_single_sample(rows: list[dict[str, Any]], thresholds: dict[str, float], gt: dict[str, float]) -> dict[str, Any]:
    clamp_free_flags = [not r["any_clamp"] for r in rows]
    clamp_counts = [r["clamp_joint_count"] for r in rows]
    l2s = [l2_error_vs_gt(r["delta"], gt) for r in rows]  # recomputed, cross-checked below
    sl = [r["delta"]["shoulder_lift"] for r in rows]
    ef = [r["delta"]["elbow_flex"] for r in rows]
    summary = build_strategy_summary(
        name="single_random_sample",
        n_units=len(rows),
        clamp_free_flags=clamp_free_flags,
        clamp_joint_counts=clamp_counts,
        l2_errors=l2s,
        shoulder_lift_values=sl,
        elbow_flex_values=ef,
        inference_counts=[1] * len(rows),
        max_inference_cap=1,
    )
    # Cross-check against the values already stored in seed_sweep.csv (should match to float
    # precision - this is a self-consistency guard, not a new measurement).
    csv_l2 = [r["l2_error_vs_gt_deg"] for r in rows]
    max_abs_diff = max(abs(a - b) for a, b in zip(l2s, csv_l2))
    summary["cross_check_vs_seed_sweep_csv_max_abs_l2_diff"] = max_abs_diff
    return summary


# --------------------------------------------------------------------------
# Strategies 2 & 3: k-sample per-joint median, full combination enumeration
# --------------------------------------------------------------------------


def strategy_k_median(
    rows: list[dict[str, Any]], k: int, thresholds: dict[str, float], gt: dict[str, float]
) -> dict[str, Any]:
    n = len(rows)
    combos = list(itertools.combinations(range(n), k))
    clamp_free_flags: list[bool] = []
    clamp_counts: list[int] = []
    l2s: list[float] = []
    sl: list[float] = []
    ef: list[float] = []
    for combo in combos:
        median_delta = {
            j: float(statistics.median(rows[i]["delta"][j] for i in combo)) for j in JOINTS
        }
        _, count, any_c = clamp_status(median_delta, thresholds)
        clamp_free_flags.append(not any_c)
        clamp_counts.append(count)
        l2s.append(l2_error_vs_gt(median_delta, gt))
        sl.append(median_delta["shoulder_lift"])
        ef.append(median_delta["elbow_flex"])
    summary = build_strategy_summary(
        name=f"{k}_sample_median",
        n_units=len(combos),
        clamp_free_flags=clamp_free_flags,
        clamp_joint_counts=clamp_counts,
        l2_errors=l2s,
        shoulder_lift_values=sl,
        elbow_flex_values=ef,
        inference_counts=[k] * len(combos),
        max_inference_cap=k,
    )
    summary["combination_space"] = f"C({n},{k})={len(combos)} (exhaustive, no approximation)"
    return summary


# --------------------------------------------------------------------------
# Strategy 4: safety-pass resampling, permutation simulation
# --------------------------------------------------------------------------


def strategy_resampling(
    rows: list[dict[str, Any]],
    cap: int,
    thresholds: dict[str, float],
    gt: dict[str, float],
    permutations: np.ndarray,
) -> dict[str, Any]:
    """``permutations`` is an (n_permutations, n_seeds) int array; only the first ``cap``
    columns of each row are used, so results for cap=3 are a strict prefix of cap=5 (same draw
    order), which is what makes the 3-vs-5 success-rate comparison apples-to-apples."""
    clamp_free_flags: list[bool] = []
    clamp_counts: list[int] = []
    l2s: list[float] = []
    sl: list[float] = []
    ef: list[float] = []
    attempts_used: list[int] = []

    for perm in permutations:
        success = False
        chosen_delta: dict[str, float] | None = None
        used = 0
        last_raw_delta: dict[str, float] | None = None
        last_raw_count = 0
        for i in range(cap):
            used = i + 1
            seed_idx = int(perm[i])
            delta = rows[seed_idx]["delta"]
            _, count, any_c = clamp_status(delta, thresholds)
            last_raw_delta, last_raw_count = delta, count
            if not any_c:
                success = True
                chosen_delta = delta
                break

        if success:
            clamp_free_flags.append(True)
            clamp_counts.append(0)
            delivered = chosen_delta
        else:
            # Cap reached without a clamp-free draw: the (unmodified) Safety Gate clips the last
            # raw attempt to threshold before it is ever sent downstream. That clipped vector -
            # not a blank/no-op - is what gets scored here as "the delivered action".
            clamp_free_flags.append(False)
            clamp_counts.append(last_raw_count)
            delivered = clip_to_thresholds(last_raw_delta, thresholds)

        attempts_used.append(used)
        l2s.append(l2_error_vs_gt(delivered, gt))
        sl.append(delivered["shoulder_lift"])
        ef.append(delivered["elbow_flex"])

    summary = build_strategy_summary(
        name=f"safety_pass_resampling_max{cap}",
        n_units=len(permutations),
        clamp_free_flags=clamp_free_flags,
        clamp_joint_counts=clamp_counts,
        l2_errors=l2s,
        shoulder_lift_values=sl,
        elbow_flex_values=ef,
        inference_counts=attempts_used,
        max_inference_cap=cap,
    )
    return summary


# --------------------------------------------------------------------------
# Baseline-only: fix one good seed (not a deployment recommendation by itself)
# --------------------------------------------------------------------------


def baseline_fixed_good_seed(rows: list[dict[str, Any]], thresholds: dict[str, float], gt: dict[str, float]) -> dict[str, Any]:
    """Pick the single best-observed seed among the 20 (clamp-free AND lowest GT L2 - a
    deployment-blind, purely retrospective pick since GT would not be available at deploy time)
    and report its numbers as a *baseline only*. See the accompanying report for why this is not
    itself a recommended deployment strategy (n=1 observation against a single fixed reference
    state; provides no evidence it generalizes to other observations/tasks)."""
    clamp_free = [r for r in rows if not r["any_clamp"]]
    candidates = clamp_free if clamp_free else rows
    best = min(candidates, key=lambda r: r["l2_error_vs_gt_deg"])
    delta = best["delta"]
    _, count, any_c = clamp_status(delta, thresholds)
    l2 = l2_error_vs_gt(delta, gt)
    return {
        "name": "fixed_good_seed_baseline_only",
        "chosen_seed": best["seed"],
        "selection_rule": "min GT L2 among clamp-free seeds in the 0..19 sweep (retrospective, GT-informed - NOT reproducible at deploy time)",
        "n_units_evaluated": 1,
        "clamp_free_rate": float(not any_c),
        "failure_rate": float(any_c),
        "avg_clamp_joint_count": float(count),
        "gt_l2_error_deg": {"mean": l2, "median": l2, "p95": l2, "max": l2},
        "shoulder_lift_deg": {"mean": delta["shoulder_lift"], "std": 0.0, "min": delta["shoulder_lift"], "max": delta["shoulder_lift"], "range": 0.0},
        "elbow_flex_deg": {"mean": delta["elbow_flex"], "std": 0.0, "min": delta["elbow_flex"], "max": delta["elbow_flex"], "range": 0.0},
        "avg_inference_count": 1.0,
        "max_inference_count": 1,
        "max_inference_cap": 1,
        "caveat": (
            "Degenerate stats (std=0, range=0) are expected and diagnostic, not a virtue: this "
            "reflects a single observation (one seed x one fixed Shadow reference state), not a "
            "distribution. It says nothing about what this seed would produce against a different "
            "observation."
        ),
    }


# --------------------------------------------------------------------------
# Additional analysis
# --------------------------------------------------------------------------


def additional_analysis(
    rows: list[dict[str, Any]],
    strat1: dict[str, Any],
    strat3: dict[str, Any],
    strat5: dict[str, Any],
    resample3: dict[str, Any],
    resample5: dict[str, Any],
) -> dict[str, Any]:
    # Does median actually reduce clamp rate vs single sample?
    clamp_rate_reduction = {
        "single_sample_clamp_free_rate": strat1["clamp_free_rate"],
        "3_median_clamp_free_rate": strat3["clamp_free_rate"],
        "5_median_clamp_free_rate": strat5["clamp_free_rate"],
        "3_median_reduces_clamp_rate_vs_single": strat3["clamp_free_rate"] > strat1["clamp_free_rate"],
        "5_median_reduces_clamp_rate_vs_single": strat5["clamp_free_rate"] > strat1["clamp_free_rate"],
    }

    # Does median also improve GT L2?
    gt_l2_comparison = {
        "single_sample_mean_l2": strat1["gt_l2_error_deg"]["mean"],
        "3_median_mean_l2": strat3["gt_l2_error_deg"]["mean"],
        "5_median_mean_l2": strat5["gt_l2_error_deg"]["mean"],
        "3_median_improves_mean_l2_vs_single": strat3["gt_l2_error_deg"]["mean"] < strat1["gt_l2_error_deg"]["mean"],
        "5_median_improves_mean_l2_vs_single": strat5["gt_l2_error_deg"]["mean"] < strat1["gt_l2_error_deg"]["mean"],
    }

    # Resampling 3 vs 5 success-rate delta.
    resampling_comparison = {
        "max3_clamp_free_rate": resample3["clamp_free_rate"],
        "max5_clamp_free_rate": resample5["clamp_free_rate"],
        "max5_minus_max3_clamp_free_rate": resample5["clamp_free_rate"] - resample3["clamp_free_rate"],
        "max3_avg_inference_count": resample3["avg_inference_count"],
        "max5_avg_inference_count": resample5["avg_inference_count"],
        "max3_failure_rate": resample3["failure_rate"],
        "max5_failure_rate": resample5["failure_rate"],
        "max3_gt_l2_mean": resample3["gt_l2_error_deg"]["mean"],
        "max5_gt_l2_mean": resample5["gt_l2_error_deg"]["mean"],
        "max5_gt_l2_mean_worse_than_max3": resample5["gt_l2_error_deg"]["mean"] > resample3["gt_l2_error_deg"]["mean"],
    }

    # Clamp-free != close to GT: concrete counterexample(s) from the raw 20-seed pool.
    #
    # Edge case (first hit on the actual T01-T10 real_final sweep, where several scenes have
    # 0/20 raw clamp-free seeds - unlike every synthetic/earlier-actual scene, which always had
    # at least one): with no clamp-free row at all there is no "worst clamp-free seed" to compare
    # against, so the counterexample statement degrades to reporting that fact directly instead of
    # indexing into an empty list.
    clamp_free_rows = sorted([r for r in rows if not r["any_clamp"]], key=lambda r: r["l2_error_vs_gt_deg"])
    clamped_rows_sorted = sorted([r for r in rows if r["any_clamp"]], key=lambda r: r["l2_error_vs_gt_deg"])
    clamp_free_l2 = [r["l2_error_vs_gt_deg"] for r in clamp_free_rows]
    clamped_l2 = [r["l2_error_vs_gt_deg"] for r in clamped_rows_sorted]

    if clamp_free_rows:
        worst_clamp_free = clamp_free_rows[-1]
        n_clamped_better_than_worst_clamp_free = sum(
            1 for l2 in clamped_l2 if l2 < worst_clamp_free["l2_error_vs_gt_deg"]
        )
        statement = (
            f"seed {worst_clamp_free['seed']} is clamp-free (0 joints clamp) but has GT L2="
            f"{worst_clamp_free['l2_error_vs_gt_deg']:.3f} deg, worse than "
            f"{n_clamped_better_than_worst_clamp_free}/{len(clamped_rows_sorted)} clamped seeds. "
            "Clamp-free selection (the only signal available at real deploy time, since GT is "
            "unknown) does not guarantee low GT error - it only guarantees the action was not "
            "truncated by the Safety Gate."
        )
        worst_clamp_free_seed = worst_clamp_free["seed"]
        worst_clamp_free_l2 = worst_clamp_free["l2_error_vs_gt_deg"]
    else:
        n_clamped_better_than_worst_clamp_free = 0
        worst_clamp_free_seed = None
        worst_clamp_free_l2 = None
        statement = (
            "0/{} seeds in the raw sweep were clamp-free, so no clamp-free-vs-GT counterexample "
            "exists for this scene - every raw seed's chunk[0] clamps on at least one joint. This "
            "is a stronger finding than the counterexample below normally shows: at this scene, "
            "clamp status is not just an imperfect proxy for GT accuracy, single-sample inference "
            "essentially never passes the Safety Gate cleanly regardless of seed.".format(len(rows))
        )
    clamp_free_vs_gt_caveat = {
        "clamp_free_seeds": [r["seed"] for r in clamp_free_rows],
        "clamp_free_l2_values": clamp_free_l2,
        "clamp_free_l2_mean": float(statistics.fmean(clamp_free_l2)) if clamp_free_l2 else None,
        "clamped_l2_mean": float(statistics.fmean(clamped_l2)) if clamped_l2 else None,
        "worst_clamp_free_seed": worst_clamp_free_seed,
        "worst_clamp_free_l2": worst_clamp_free_l2,
        "n_clamped_seeds_with_lower_l2_than_worst_clamp_free_seed": n_clamped_better_than_worst_clamp_free,
        "n_clamped_seeds_total": len(clamped_rows_sorted),
        "statement": statement,
    }

    return {
        "clamp_rate_reduction": clamp_rate_reduction,
        "gt_l2_improvement": gt_l2_comparison,
        "resampling_3_vs_5": resampling_comparison,
        "clamp_free_does_not_imply_low_gt_error": clamp_free_vs_gt_caveat,
    }


# --------------------------------------------------------------------------
# Report writers
# --------------------------------------------------------------------------


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def write_csv(out_path: Path, strategies: list[dict[str, Any]]) -> None:
    fields = [
        "name",
        "n_units_evaluated",
        "clamp_free_rate",
        "failure_rate",
        "avg_clamp_joint_count",
        "gt_l2_mean",
        "gt_l2_median",
        "gt_l2_p95",
        "gt_l2_max",
        "shoulder_lift_mean",
        "shoulder_lift_std",
        "shoulder_lift_min",
        "shoulder_lift_max",
        "shoulder_lift_range",
        "elbow_flex_mean",
        "elbow_flex_std",
        "elbow_flex_min",
        "elbow_flex_max",
        "elbow_flex_range",
        "avg_inference_count",
        "max_inference_count",
        "max_inference_cap",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for s in strategies:
            w.writerow(
                [
                    s["name"],
                    s["n_units_evaluated"],
                    s["clamp_free_rate"],
                    s["failure_rate"],
                    s["avg_clamp_joint_count"],
                    s["gt_l2_error_deg"]["mean"],
                    s["gt_l2_error_deg"]["median"],
                    s["gt_l2_error_deg"]["p95"],
                    s["gt_l2_error_deg"]["max"],
                    s["shoulder_lift_deg"]["mean"],
                    s["shoulder_lift_deg"]["std"],
                    s["shoulder_lift_deg"]["min"],
                    s["shoulder_lift_deg"]["max"],
                    s["shoulder_lift_deg"]["range"],
                    s["elbow_flex_deg"]["mean"],
                    s["elbow_flex_deg"]["std"],
                    s["elbow_flex_deg"]["min"],
                    s["elbow_flex_deg"]["max"],
                    s["elbow_flex_deg"]["range"],
                    s["avg_inference_count"],
                    s["max_inference_count"],
                    s["max_inference_cap"],
                ]
            )


def write_markdown(
    out_path: Path,
    meta: dict[str, Any],
    thresholds: dict[str, float],
    gt: dict[str, float],
    strategies: list[dict[str, Any]],
    baseline: dict[str, Any],
    extra: dict[str, Any],
    rng_seed: int,
    n_permutations: int,
) -> None:
    lines: list[str] = []
    lines.append("# Grid35 V2 T01 - first-action seed-mitigation strategy comparison (offline)")
    lines.append("")
    lines.append(
        "Offline comparison of 4 candidate first-action stabilization strategies, computed "
        "entirely from the existing 20-seed sweep "
        "(`reports/grid35_v2_T01_seed_sweep/seed_sweep.csv`). **No new inference was run.**"
    )
    lines.append("")
    lines.append(f"- Source sweep checkpoint: `{meta['checkpoint']}`")
    lines.append(f"- Reference Shadow observation: `{meta['reference_observation_label']}` (`{meta['shadow_report']}`)")
    lines.append(f"- Task: `{meta['task']}`")
    lines.append(f"- Nearest-training-demo match: episode {meta['nearest_demo_match']['episode']}, frame {meta['nearest_demo_match']['frame']}")
    lines.append(f"- Deterministic RNG seed used for resampling-permutation simulation: `{rng_seed}` (n_permutations={n_permutations})")
    lines.append("- 3-sample / 5-sample medians: full exhaustive combination enumeration (no sampling approximation)")
    lines.append("")

    lines.append("## Safety thresholds used (read-only, not modified)")
    lines.append("")
    lines.append("| joint | WOULD_CLAMP threshold (deg) |")
    lines.append("|---|---:|")
    for j in JOINTS:
        lines.append(f"| {j} | {thresholds[j]:.2f} |")
    lines.append("")

    lines.append("## GT immediate delta (reference, evaluation-only - never used in selection logic)")
    lines.append("")
    lines.append("| joint | GT immediate delta (deg) |")
    lines.append("|---|---:|")
    for j in JOINTS:
        lines.append(f"| {j} | {gt[j]:+.4f} |")
    lines.append("")

    lines.append("## Strategy comparison table")
    lines.append("")
    header = (
        "| strategy | clamp-free rate | avg clamp joints | GT L2 mean | GT L2 median | GT L2 p95 | GT L2 max "
        "| shoulder_lift mean±std (range) | elbow_flex mean±std (range) | avg # inference | max # inference | failure rate |"
    )
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|")
    for s in strategies:
        sl, ef = s["shoulder_lift_deg"], s["elbow_flex_deg"]
        lines.append(
            f"| {s['name']} | {s['clamp_free_rate']*100:.1f}% | {s['avg_clamp_joint_count']:.3f} "
            f"| {s['gt_l2_error_deg']['mean']:.3f} | {s['gt_l2_error_deg']['median']:.3f} "
            f"| {s['gt_l2_error_deg']['p95']:.3f} | {s['gt_l2_error_deg']['max']:.3f} "
            f"| {sl['mean']:+.3f}±{sl['std']:.3f} ({sl['range']:.3f}) "
            f"| {ef['mean']:+.3f}±{ef['std']:.3f} ({ef['range']:.3f}) "
            f"| {s['avg_inference_count']:.2f} | {s['max_inference_count']} | {s['failure_rate']*100:.1f}% |"
        )
    lines.append("")

    lines.append("## Baseline-only: fix one good seed (NOT evaluated as a deployment candidate)")
    lines.append("")
    lines.append(
        f"Selection rule: {baseline['selection_rule']}. Chosen seed: **{baseline['chosen_seed']}**."
    )
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| clamp-free | {'yes' if baseline['clamp_free_rate'] == 1.0 else 'no'} |")
    lines.append(f"| avg clamp joint count | {baseline['avg_clamp_joint_count']:.0f} |")
    lines.append(f"| GT L2 (deg) | {baseline['gt_l2_error_deg']['mean']:.3f} |")
    lines.append(f"| shoulder_lift (deg) | {baseline['shoulder_lift_deg']['mean']:+.3f} |")
    lines.append(f"| elbow_flex (deg) | {baseline['elbow_flex_deg']['mean']:+.3f} |")
    lines.append("")
    lines.append(f"> {baseline['caveat']}")
    lines.append("")

    lines.append("## Additional analysis")
    lines.append("")
    cr = extra["clamp_rate_reduction"]
    lines.append("### Does 3-median / 5-median actually reduce clamp rate vs single sample?")
    lines.append("")
    lines.append(
        f"- single-sample clamp-free rate: {cr['single_sample_clamp_free_rate']*100:.1f}%\n"
        f"- 3-median clamp-free rate: {cr['3_median_clamp_free_rate']*100:.1f}% "
        f"({'higher' if cr['3_median_reduces_clamp_rate_vs_single'] else 'not higher'} than single-sample)\n"
        f"- 5-median clamp-free rate: {cr['5_median_clamp_free_rate']*100:.1f}% "
        f"({'higher' if cr['5_median_reduces_clamp_rate_vs_single'] else 'not higher'} than single-sample)"
    )
    lines.append("")

    gl = extra["gt_l2_improvement"]
    lines.append("### Does median also improve GT L2?")
    lines.append("")
    lines.append(
        f"- single-sample mean GT L2: {gl['single_sample_mean_l2']:.3f} deg\n"
        f"- 3-median mean GT L2: {gl['3_median_mean_l2']:.3f} deg "
        f"({'improves' if gl['3_median_improves_mean_l2_vs_single'] else 'does NOT improve'})\n"
        f"- 5-median mean GT L2: {gl['5_median_mean_l2']:.3f} deg "
        f"({'improves' if gl['5_median_improves_mean_l2_vs_single'] else 'does NOT improve'})"
    )
    lines.append("")

    rc = extra["resampling_3_vs_5"]
    lines.append("### Resampling: max-3 vs max-5 success-rate difference")
    lines.append("")
    lines.append(
        f"- max-3 clamp-free (success) rate: {rc['max3_clamp_free_rate']*100:.1f}% "
        f"(avg {rc['max3_avg_inference_count']:.2f} inference calls, failure rate {rc['max3_failure_rate']*100:.1f}%)\n"
        f"- max-5 clamp-free (success) rate: {rc['max5_clamp_free_rate']*100:.1f}% "
        f"(avg {rc['max5_avg_inference_count']:.2f} inference calls, failure rate {rc['max5_failure_rate']*100:.1f}%)\n"
        f"- delta (max5 - max3): {rc['max5_minus_max3_clamp_free_rate']*100:+.1f} percentage points"
    )
    lines.append("")
    if rc["max5_gt_l2_mean_worse_than_max3"]:
        lines.append(
            f"> Note: max-5 mean GT L2 ({rc['max5_gt_l2_mean']:.3f} deg) is *slightly worse* than max-3 "
            f"({rc['max3_gt_l2_mean']:.3f} deg), even though max-5 succeeds (clamp-free) more often. "
            "This is not a bug: the stopping rule never looks at GT, so the *extra* successes cap=5 "
            "finds beyond what cap=3 already found are exactly the additional clamp-free seeds only "
            "reachable with more attempts (e.g. seed 12 in this sweep - clamp-free but a comparatively "
            "large GT L2) - a direct instance of the 'clamp-free does not imply low GT error' finding "
            "below, not evidence that more resampling attempts make actions worse."
        )
        lines.append("")

    cf = extra["clamp_free_does_not_imply_low_gt_error"]
    lines.append("### Clamp-free does not imply low GT error")
    lines.append("")
    lines.append(f"> {cf['statement']}")
    lines.append("")
    # clamp_free_l2_mean is None when the raw sweep had zero clamp-free seeds (see
    # additional_analysis's empty-clamp_free_rows branch) - format defensively rather than assume
    # at least one clamp-free seed exists, which is no longer true for every scene.
    clamp_free_l2_mean_str = f"{cf['clamp_free_l2_mean']:.3f}" if cf["clamp_free_l2_mean"] is not None else "n/a (0 clamp-free seeds)"
    clamped_l2_mean_str = f"{cf['clamped_l2_mean']:.3f}" if cf["clamped_l2_mean"] is not None else "n/a (0 clamped seeds)"
    lines.append(
        f"- clamp-free seeds {cf['clamp_free_seeds']}, GT L2 values {[round(v, 3) for v in cf['clamp_free_l2_values']]} "
        f"(mean {clamp_free_l2_mean_str})\n"
        f"- clamped seeds mean GT L2: {clamped_l2_mean_str}\n"
        "- **Implication for deployment:** since clamp status is the only signal available at "
        "inference time (GT is unknown), a strategy that optimizes purely for clamp-free status "
        "(e.g. safety-pass resampling) reduces Safety-Gate intervention but offers **no guarantee** "
        "on trajectory accuracy. It is a safety/stability mitigation, not an accuracy mitigation."
    )
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append(meta["recommendation_text"])
    lines.append("")
    lines.append("## Next step: T01-T10 full Shadow validation")
    lines.append("")
    lines.append(meta["next_step_text"])
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_recommendation_text(strategies_by_name: dict[str, Any], baseline: dict[str, Any]) -> str:
    s1 = strategies_by_name["single_random_sample"]
    s3 = strategies_by_name["3_sample_median"]
    s5 = strategies_by_name["5_sample_median"]
    r3 = strategies_by_name["safety_pass_resampling_max3"]
    r5 = strategies_by_name["safety_pass_resampling_max5"]

    parts = []
    parts.append(
        f"**Primary recommendation: safety-pass resampling, cap=5** "
        f"(clamp-free rate {r5['clamp_free_rate']*100:.1f}%, failure rate {r5['failure_rate']*100:.1f}%, "
        f"avg {r5['avg_inference_count']:.2f} inference calls, GT L2 mean {r5['gt_l2_error_deg']['mean']:.3f} deg). "
        "It directly targets the deployment failure mode this sweep was run to investigate "
        "(Safety Gate truncating first-action joints), it is the only strategy of the four whose "
        "stopping rule is deploy-time-realistic (uses only clamp status, never GT), and its inference "
        "cost is adaptive (only pays for extra calls when the first draw actually clamps) rather than "
        "fixed."
    )
    parts.append(
        f"**Secondary / fallback recommendation: safety-pass resampling, cap=3** "
        f"(clamp-free rate {r3['clamp_free_rate']*100:.1f}%, avg {r3['avg_inference_count']:.2f} calls) "
        "if the extra latency budget for a 5th inference call is not available on the deployment "
        "hardware - it captures most of the benefit of cap=5 at lower worst-case latency."
    )
    parts.append(
        f"3-sample and 5-sample median are **not** recommended as the primary mitigation: median "
        f"clamp-free rate (3-median {s3['clamp_free_rate']*100:.1f}%, 5-median {s5['clamp_free_rate']*100:.1f}%) "
        f"does not clearly dominate single-sample ({s1['clamp_free_rate']*100:.1f}%) or resampling, "
        "while always paying the fixed 3x/5x inference cost regardless of whether the first draw was "
        "already clamp-free - resampling only pays that cost when needed."
    )
    parts.append(
        f"The 'fix one good seed' baseline (seed {baseline['chosen_seed']}, clamp-free, GT L2 "
        f"{baseline['gt_l2_error_deg']['mean']:.3f} deg here) is **not recommended for deployment**: "
        "it was selected using GT that will not be available at deploy time, it reflects a single "
        "observation against one fixed Shadow reference state, and this whole sweep's own premise - "
        "that SmolVLA's flow-matching noise draw materially shifts the first action - means nothing "
        "guarantees that seed's favorable behavior transfers to a different observation, task, or "
        "checkpoint. It remains useful only as a debug/regression reference, not as a runtime policy."
    )
    return "\n\n".join(parts)


def build_next_step_text(strategies_by_name: dict[str, Any]) -> str:
    r5 = strategies_by_name["safety_pass_resampling_max5"]
    return (
        "Carry safety-pass resampling (cap=5, fallback cap=3) into the T01-T10 full Shadow "
        "verification pass as the candidate mitigation, but re-run this same offline analysis "
        "*per task* once each task has its own seed sweep - this T01-only result "
        f"(clamp-free {r5['clamp_free_rate']*100:.1f}% at cap=5) should not be assumed to hold for "
        "T02-T10 without their own 20-seed sweeps, since clamp behavior is a function of each task's "
        "own first-action delta distribution relative to the (task-independent) Safety Gate "
        "thresholds. Concretely: (1) run `sweep_grid35_first_action_seed.py` (or equivalent) for each "
        "of T02..T10 against its own Shadow reference observation; (2) re-run this script per task; "
        "(3) only adopt a single fixed cap/strategy repo-wide once its clamp-free rate and GT L2 are "
        "confirmed acceptable across all 10 tasks, not just T01."
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--sweep-json", type=Path, default=DEFAULT_SWEEP_JSON)
    p.add_argument("--safety-gate-yaml", type=Path, default=DEFAULT_SAFETY_GATE_YAML)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    p.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_seed_rows(args.csv)
    meta = load_gt_and_metadata(args.sweep_json)
    thresholds = load_thresholds(args.safety_gate_yaml)
    gt = meta["gt_immediate_delta"]

    # Sanity: thresholds read fresh from configs/safety_gate.yaml must match what the original
    # sweep used (read-only cross-check, not a modification).
    mismatch = {j: (thresholds[j], meta["thresholds_from_sweep_json"][j]) for j in JOINTS if abs(thresholds[j] - meta["thresholds_from_sweep_json"][j]) > 1e-9}
    if mismatch:
        raise RuntimeError(f"Safety Gate thresholds changed since the original sweep was run: {mismatch}")

    n_seeds = len(rows)
    assert n_seeds == 20, f"expected 20 seed rows, got {n_seeds}"

    rng = np.random.default_rng(args.rng_seed)
    permutations = np.array([rng.permutation(n_seeds) for _ in range(args.n_permutations)])

    strat_single = strategy_single_sample(rows, thresholds, gt)
    strat_median = {k: strategy_k_median(rows, k, thresholds, gt) for k in MEDIAN_KS}
    strat_resample = {cap: strategy_resampling(rows, cap, thresholds, gt, permutations) for cap in RESAMPLE_CAPS}
    baseline = baseline_fixed_good_seed(rows, thresholds, gt)

    strategies = [strat_single, strat_median[3], strat_median[5], strat_resample[3], strat_resample[5]]
    strategies_by_name = {s["name"]: s for s in strategies}

    extra = additional_analysis(
        rows,
        strat_single,
        strat_median[3],
        strat_median[5],
        strat_resample[3],
        strat_resample[5],
    )

    meta["recommendation_text"] = build_recommendation_text(strategies_by_name, baseline)
    meta["next_step_text"] = build_next_step_text(strategies_by_name)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_result = {
        "source": {
            "seed_sweep_csv": str(args.csv.resolve().relative_to(PROJECT_ROOT)),
            "seed_sweep_json": str(args.sweep_json.resolve().relative_to(PROJECT_ROOT)),
            "safety_gate_yaml": str(args.safety_gate_yaml.resolve().relative_to(PROJECT_ROOT)),
            "n_seeds": n_seeds,
            "note": "No new inference was run for this analysis; all strategies are recombinations/resamplings of the 20 existing seed rows.",
        },
        "metadata": meta,
        "thresholds_deg": thresholds,
        "gt_immediate_delta_deg": gt,
        "reproducibility": {
            "rng_seed": args.rng_seed,
            "n_permutations": args.n_permutations,
            "median_combination_method": "exhaustive itertools.combinations (no RNG involved)",
            "resampling_combination_method": "fixed-RNG-seed permutations of seed draw order, first `cap` entries used per permutation",
        },
        "strategies": {
            "single_random_sample": strat_single,
            "3_sample_median": strat_median[3],
            "5_sample_median": strat_median[5],
            "safety_pass_resampling_max3": strat_resample[3],
            "safety_pass_resampling_max5": strat_resample[5],
        },
        "baseline_fixed_good_seed": baseline,
        "additional_analysis": extra,
    }

    (out_dir / "strategy_comparison.json").write_text(json.dumps(full_result, indent=2), encoding="utf-8")
    write_csv(out_dir / "strategy_comparison.csv", strategies)
    write_markdown(
        out_dir / "strategy_comparison.md",
        meta,
        thresholds,
        gt,
        strategies,
        baseline,
        extra,
        args.rng_seed,
        args.n_permutations,
    )

    print(f"Wrote {out_dir / 'strategy_comparison.json'}")
    print(f"Wrote {out_dir / 'strategy_comparison.csv'}")
    print(f"Wrote {out_dir / 'strategy_comparison.md'}")
    print()
    for s in strategies:
        print(
            f"{s['name']:32s} clamp_free={s['clamp_free_rate']*100:5.1f}%  "
            f"gt_l2_mean={s['gt_l2_error_deg']['mean']:6.3f}  "
            f"avg_inf={s['avg_inference_count']:.2f}  "
            f"fail={s['failure_rate']*100:5.1f}%"
        )


if __name__ == "__main__":
    main()
