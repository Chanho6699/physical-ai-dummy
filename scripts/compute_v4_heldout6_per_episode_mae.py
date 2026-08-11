#!/usr/bin/env python3
"""Per-episode MAE breakdown for the V4 heldout6 offline eval (task requirement: report
episode-level MAE, never averaged together with the historical test10 numbers).

Reads (never writes to): reports/pick_drop_v4_heldout6_offline_eval/checkpoint_*.json (the
per-frame ``records`` list ``evaluate_smolvla_midpoint.py`` already saved - no re-inference).
Writes: reports/pick_drop_v4_fresh_training/v4_heldout6_per_episode_mae.json (consumed by the
final report-builder script).
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "reports" / "pick_drop_v4_heldout6_offline_eval"
OUT_PATH = PROJECT_ROOT / "reports" / "pick_drop_v4_fresh_training" / "v4_heldout6_per_episode_mae.json"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STEPS = [2500, 5000, 7500, 10000]


def main() -> int:
    out: dict[int, dict[int, dict]] = {}
    for step in STEPS:
        report = json.loads((IN_DIR / f"checkpoint_{step:06d}.json").read_text(encoding="utf-8"))
        per_episode: dict[int, list] = {}
        for rec in report["records"]:
            per_episode.setdefault(rec["episode_index"], []).append(rec)

        step_out = {}
        for ep, recs in sorted(per_episode.items()):
            all_abs_err = []
            per_joint_abs_err = {j: [] for j in JOINTS}
            for rec in recs:
                for j in JOINTS:
                    err = abs(rec["pred_action"][j] - rec["gt_action"][j])
                    all_abs_err.append(err)
                    per_joint_abs_err[j].append(err)
            step_out[ep] = {
                "n_frames": len(recs),
                "action_mae_overall": sum(all_abs_err) / len(all_abs_err),
                "action_mae_per_joint": {j: sum(v) / len(v) for j, v in per_joint_abs_err.items()},
            }
        out[step] = step_out
        print(f"step {step}: " + ", ".join(f"ep{ep}={v['action_mae_overall']:.4f}" for ep, v in step_out.items()))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
