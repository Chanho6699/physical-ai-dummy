#!/usr/bin/env python3
"""In-sample (training-set) action MAE, reusing evaluate_smolvla_midpoint.py's own inference
pipeline verbatim, for measuring the train/held-out generalization gap of a small-dataset run
(reinforcement30_only_v1, 30 episodes - overfitting risk explicitly called out for this
experiment).

Why a separate script instead of `--eval-dataset X --train-dataset X`: evaluate_smolvla_midpoint.py
deliberately hard-stops when eval_root == train_root (its whole purpose is held-out evaluation, and
that guard exists to prevent accidentally measuring "held-out" performance on training data). Here
we *intentionally* want the training-set fit number, clearly labeled as such (never mixed with or
mistaken for the held-out numbers reported elsewhere) - so this script imports the exact same
`run_checkpoint`/`build_frame_plans`/`compute_gt_demo_delta_stats` functions directly, skipping
only that one CLI-level guard. Every actual computation is identical to the held-out evaluator.

Read-only: no dataset/checkpoint writes, no real-robot writes, no Safety Gate threshold changes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_smolvla_midpoint import (  # noqa: E402
    build_frame_plans,
    compute_gt_demo_delta_stats,
    infer_step_from_path,
    round_floats,
    run_checkpoint,
    try_build_safety_gate,
)

logger = logging.getLogger("evaluate_training_set_fit")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--train-dataset", required=True, help="the dataset the checkpoints were trained on - evaluated in-sample here")
    parser.add_argument("--train-repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    train_root = Path(args.train_dataset).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(repo_id=args.train_repo_id, root=str(train_root))
    episodes_meta = [dataset.meta.episodes[i] for i in range(dataset.num_episodes)]
    frame_plans = build_frame_plans(episodes_meta, max_frames_per_episode=None)
    total_frames = sum(len(p.dataset_indices) for p in frame_plans)
    logger.info(f"IN-SAMPLE (training-set) eval on {train_root}: {len(frame_plans)} episodes, {total_frames} frames")

    gt_demo_delta = compute_gt_demo_delta_stats(dataset, frame_plans)
    safety_gate, safety_reason = try_build_safety_gate()

    rows = []
    for ckpt_str in args.checkpoints:
        ckpt_dir = Path(ckpt_str)
        step = infer_step_from_path(ckpt_dir)
        logger.info(f"=== in-sample eval checkpoint step={step} ===")
        report = run_checkpoint(
            ckpt_dir, dataset, frame_plans, task=args.task, seed=args.seed, device=args.device,
            policy_type="smolvla", safety_gate=safety_gate, gt_demo_delta=gt_demo_delta,
        )
        row = {
            "checkpoint_step": step,
            "in_sample_action_mae": report["metrics"]["action_mae_overall"],
            "in_sample_action_mae_per_joint": report["metrics"]["action_mae_per_joint"],
            "num_frames_evaluated": report["num_frames_evaluated"],
        }
        logger.info(f"checkpoint step={step}: in_sample_action_mae={row['in_sample_action_mae']:.4f}")
        rows.append(row)

    payload = {"train_dataset": str(train_root), "task": args.task, "seed": args.seed, "rows": round_floats(rows)}
    out_path = output_dir / "in_sample_fit.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
