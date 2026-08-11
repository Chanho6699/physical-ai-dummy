#!/usr/bin/env python3
"""Assemble reports/pick_drop_combined65_fresh_training/ from the raw run artifacts.

Reads (never modifies):
  - the ``lerobot-train`` stdout/stderr log for the Combined65 fresh-training run (checkpoint
    train loss/LR/grad-norm/GPU-mem/wall-clock, parsed straight out of ``ot_train.py``'s own
    periodic ``step:N ... loss:... lr:... mem_gb:...`` log lines - no re-derivation).
  - ``scripts/evaluate_smolvla_midpoint.py``'s summary.json for the Combined65 checkpoints
    (offline held-out action MAE / per-joint MAE / Safety-Gate WOULD_CLAMP counts).
  - one ``scripts/sweep_grid35_first_action_seed.py`` seed_sweep.json per Combined65 checkpoint
    (first-action shoulder_lift/elbow_flex delta distribution over seeds 0..19, same reference
    Shadow observation as the V2 7.5k baseline).
  - the V2 7.5k baseline's own seed_sweep.json and the V2 offline-eval summary.json, for direct
    side-by-side comparison.

Writes reports/pick_drop_combined65_fresh_training/{summary.md,summary.json,
checkpoint_metrics.csv,first_action_diagnostics.csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_STEPS = [2500, 5000, 7500, 10000]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
KEY_JOINTS = ["shoulder_lift", "elbow_flex"]

STEP_LINE_RE = re.compile(
    r"INFO (?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) \S+:\d+ "
    r"step:(?P<step>[\d.]+K?) smpl:(?P<smpl>[\d.]+K?) ep:(?P<ep>\d+) epch:(?P<epch>[\d.]+) "
    r"loss:(?P<loss>[\d.eE+-]+) grdn:(?P<grdn>[\d.eE+-]+) lr:(?P<lr>[\d.eE+-]+) "
    r"updt_s:(?P<updt_s>[\d.]+) data_s:(?P<data_s>[\d.]+) smp/s:(?P<smp_s>\d+) "
    r"mem_gb:(?P<mem_gb>[\d.]+)"
)
CHECKPOINT_SAVED_RE = re.compile(r"INFO (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \S+:\d+ Checkpoint policy after step (\d+)")
OOM_PATTERNS = re.compile(r"CUDA out of memory|OutOfMemoryError|Killed|Traceback", re.IGNORECASE)


def _parse_k_suffixed(s: str) -> float:
    """tqdm/ot_train.py rounds step/smpl to the nearest thousand once >=1000 (e.g. 'step:2K' can
    mean any step in [2000,2999]) - only usable for an approximate display, never for pinpointing
    an exact checkpoint step. Exact step->metrics matching instead goes through the timestamp
    shared with the unambiguous 'Checkpoint policy after step N' log line (see below)."""
    if s.endswith("K"):
        return float(s[:-1]) * 1000
    return float(s)


def parse_training_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    all_steps = []
    for m in STEP_LINE_RE.finditer(text):
        ts = f"{m.group('date')} {m.group('time')}"
        all_steps.append(
            {
                "ts": ts,
                "date": m.group("date"),
                "time": m.group("time"),
                "step_approx": _parse_k_suffixed(m.group("step")),
                "loss": float(m.group("loss")),
                "grad_norm": float(m.group("grdn")),
                "lr": float(m.group("lr")),
                "mem_gb": float(m.group("mem_gb")),
            }
        )
    checkpoint_saves = [
        {"date": d, "time": t, "step": int(s), "ts": f"{d} {t}"} for d, t, s in CHECKPOINT_SAVED_RE.findall(text)
    ]
    errors = OOM_PATTERNS.findall(text)
    end_of_training_matches = re.findall(
        r"INFO (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \S+:\d+ End of training", text
    )
    end_of_training_at = f"{end_of_training_matches[0][0]} {end_of_training_matches[0][1]}" if end_of_training_matches else None

    # Exact-step metrics come only from the periodic step-log line sharing the checkpoint save
    # event's own timestamp (both are logged within the same training-loop iteration - verified
    # against this run's own log: e.g. step 2500's 'step:2K ...' line and 'Checkpoint policy after
    # step 2500' share the identical HH:MM:SS timestamp).
    by_ts: dict[str, dict] = {}
    for s in all_steps:
        by_ts[s["ts"]] = s  # last one wins if duplicate timestamps occur

    checkpoint_rows = []
    for target in CHECKPOINT_STEPS:
        save_evt = next((c for c in checkpoint_saves if c["step"] == target), None)
        row = by_ts.get(save_evt["ts"]) if save_evt else None
        checkpoint_rows.append(
            {
                "checkpoint_step": target,
                "logged_at": row["ts"] if row else None,
                "train_loss": row["loss"] if row else None,
                "grad_norm": row["grad_norm"] if row else None,
                "lr": row["lr"] if row else None,
                "gpu_mem_gb": row["mem_gb"] if row else None,
                "checkpoint_saved": save_evt is not None,
                "checkpoint_saved_at": save_evt["ts"] if save_evt else None,
            }
        )

    # Wall time: first step-log timestamp -> each checkpoint's save timestamp -> end-of-training.
    first_ts = all_steps[0]["ts"] if all_steps else None

    def _to_epoch(ts: str) -> float:
        import datetime

        return datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()

    if first_ts:
        t0 = _to_epoch(first_ts)
        for row in checkpoint_rows:
            if row["checkpoint_saved_at"]:
                row["wall_time_since_start_min"] = round((_to_epoch(row["checkpoint_saved_at"]) - t0) / 60.0, 2)
            else:
                row["wall_time_since_start_min"] = None

    total_wall_time_min = None
    if first_ts and end_of_training_at:
        total_wall_time_min = round((_to_epoch(end_of_training_at) - _to_epoch(first_ts)) / 60.0, 2)

    return {
        "log_path": str(log_path),
        "n_step_log_lines": len(all_steps),
        "n_checkpoint_saves_logged": len(checkpoint_saves),
        "checkpoint_save_events": checkpoint_saves,
        "end_of_training_seen": end_of_training_at is not None,
        "end_of_training_at": end_of_training_at,
        "training_started_at": first_ts,
        "total_wall_time_min": total_wall_time_min,
        "error_or_oom_matches": errors,
        "last_logged_step_approx": all_steps[-1] if all_steps else None,
        "checkpoint_rows": checkpoint_rows,
    }


def load_json(p: Path) -> dict[str, Any] | None:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_checkpoint_metrics_csv(
    out_path: Path,
    train_log_info: dict[str, Any],
    checkpoint_dirs: dict[int, Path],
    offline_eval_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    offline_by_step = {}
    if offline_eval_summary:
        for row in offline_eval_summary["rows"]:
            offline_by_step[row["checkpoint_step"]] = row

    rows = []
    for cr in train_log_info["checkpoint_rows"]:
        step = cr["checkpoint_step"]
        off = offline_by_step.get(step, {})
        rows.append(
            {
                "checkpoint_step": step,
                "checkpoint_path": str(checkpoint_dirs.get(step, "")),
                "train_loss": cr["train_loss"],
                "lr": cr["lr"],
                "grad_norm": cr["grad_norm"],
                "gpu_mem_gb": cr["gpu_mem_gb"],
                "logged_at": cr["logged_at"],
                "checkpoint_saved": cr["checkpoint_saved"],
                "offline_action_mae": off.get("action_mae"),
                "offline_pred_state_delta_median": off.get("pred_state_delta_median"),
                "offline_pred_state_delta_p95": off.get("pred_state_delta_p95"),
                "offline_would_pass": off.get("would_pass"),
                "offline_would_clamp": off.get("would_clamp"),
                "offline_would_reject": off.get("would_reject"),
                "offline_frames_evaluated": off.get("num_frames_evaluated"),
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def build_first_action_diagnostics_csv(
    out_path: Path,
    combined_sweeps: dict[int, dict[str, Any]],
    v2_baseline_sweep: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = []

    def rows_from_sweep(label: str, sweep: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        summ = sweep["summary"]
        for j in JOINTS:
            pj = summ["per_joint"][j]
            out.append(
                {
                    "checkpoint": label,
                    "joint": j,
                    "delta_mean_deg": pj["mean"],
                    "delta_std_deg": pj["std"],
                    "delta_min_deg": pj["min"],
                    "delta_max_deg": pj["max"],
                    "would_clamp_threshold_deg": pj["threshold_deg"],
                    "clamp_count": pj["clamp_count"],
                    "n_seeds": summ["n_seeds"],
                    "clamp_rate": pj["clamp_rate"],
                    "clamp_free_seed_count": summ["clamp_free_seed_count"],
                    "l2_error_vs_gt_mean": summ["l2_error_vs_gt"]["mean"],
                    "l2_error_vs_gt_std": summ["l2_error_vs_gt"]["std"],
                }
            )
        return out

    if v2_baseline_sweep is not None:
        rows.extend(rows_from_sweep("V2_7500_baseline", v2_baseline_sweep))
    for step in CHECKPOINT_STEPS:
        if step in combined_sweeps:
            rows.extend(rows_from_sweep(f"combined65_{step}", combined_sweeps[step]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="outputs/.../checkpoints")
    parser.add_argument("--offline-eval-summary", type=Path, required=True)
    parser.add_argument("--first-action-sweep-dir", type=Path, required=True, help="dir containing first_action_seed_sweep_<step>/seed_sweep.json")
    parser.add_argument("--v2-baseline-sweep", type=Path, default=PROJECT_ROOT / "reports" / "grid35_v2_T01_seed_sweep" / "seed_sweep.json")
    parser.add_argument("--v2-offline-eval-summary", type=Path, default=PROJECT_ROOT / "reports" / "grid35_v2_midpoint_eval" / "summary.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "pick_drop_combined65_fresh_training")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[report] parsing training log")
    train_log_info = parse_training_log(args.train_log.resolve())

    checkpoint_dirs = {s: args.checkpoint_root / f"{s:06d}" / "pretrained_model" for s in CHECKPOINT_STEPS}

    offline_eval_summary = load_json(args.offline_eval_summary.resolve())
    v2_offline_eval_summary = load_json(args.v2_offline_eval_summary.resolve())

    combined_sweeps = {}
    for step in CHECKPOINT_STEPS:
        p = args.first_action_sweep_dir / f"first_action_seed_sweep_{step}" / "seed_sweep.json"
        d = load_json(p)
        if d is not None:
            combined_sweeps[step] = d
    v2_baseline_sweep = load_json(args.v2_baseline_sweep.resolve())

    print("[report] writing checkpoint_metrics.csv")
    checkpoint_metrics_rows = build_checkpoint_metrics_csv(
        out_dir / "checkpoint_metrics.csv", train_log_info, checkpoint_dirs, offline_eval_summary
    )

    print("[report] writing first_action_diagnostics.csv")
    first_action_rows = build_first_action_diagnostics_csv(
        out_dir / "first_action_diagnostics.csv", combined_sweeps, v2_baseline_sweep
    )

    summary_json = {
        "train_log_info": {k: v for k, v in train_log_info.items() if k != "all_logged_steps"},
        "checkpoint_metrics": checkpoint_metrics_rows,
        "offline_eval_summary": offline_eval_summary,
        "v2_offline_eval_summary": v2_offline_eval_summary,
        "first_action_diagnostics": first_action_rows,
        "combined_sweeps_raw": combined_sweeps,
        "v2_baseline_sweep_raw": v2_baseline_sweep,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2, default=float)
    print(f"[report] wrote {out_dir / 'summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
