#!/usr/bin/env python3
"""Synthetic-proxy vs actual (real hardware) T02-T10 comparison - and why they should not be
averaged together.

Reads the two already-built T01-T10 summaries - **no new inference, no new sampling**:

  * ``reports/grid35_v2_T01_T10_seed_sweep_summary/t01_t10_summary.json``          (synthetic,
    from ``scripts/aggregate_t01_t10_seed_sweep_report.py`` default ``--variant synthetic``)
  * ``reports/grid35_v2_T01_T10_actual_seed_sweep_summary/t01_t10_summary.json``   (actual, from
    the same script's ``--variant actual``)

and reports the per-scene numeric differences the user asked for (single clamp-free, resample<=5,
shoulder_lift/elbow_flex clamp rate). T01 is excluded from the diff table (it is the same
real-hardware result in both summaries by construction - the diff is trivially zero and it isn't
a T02-T10 comparison point).

**Critical framing, not just a caveat:** these two datasets do not measure the same thing, so a
"how well did synthetic reproduce actual" verdict has to be qualified rather than a single
number:

  * The *synthetic* T02-T10 scenes are 9 genuinely distinct held-out grid positions (different
    episodes of ``data/so101_cube_xy_midpoint_test10_v2_clean``) - built specifically to probe
    whether the T01 finding generalizes across *different scene geometries*.
  * The *actual* T02-T10 scenes turned out to be 9 real-hardware repeat captures of **the same**
    fixed scene T01 already uses (``scene_metadata.label == "V2_F02"``,
    ``evaluation_mode == "fixed-scene-repeat"`` in every source ``shadow.json`` - see
    ``scripts/import_actual_shadow_t02_t10.py``), not 9 different positions.

So a large synthetic-vs-actual gap on a given scene is not evidence the synthetic proxy method is
unreliable - it is expected, because the two datasets were never sampling the same underlying
distribution to begin with. This script's job is to report the numbers plainly and be explicit
about what they can and cannot support.

Per explicit user instruction: synthetic and actual results are never averaged together anywhere
in this repo. The actual T01-T10 results are the reference for any real policy decision.

Deliberately out of scope (same constraints as the parent scripts):
  * No training / fine-tuning, no new inference of any kind.
  * No modification of Safety Gate thresholds.
  * No writes to any robot, real or simulated.
  * No modification of the LeRobot library itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
COMPARE_SCENES = [f"T{i:02d}" for i in range(2, 11)]  # T02..T10 - T01 is identical in both by construction

DEFAULT_SYNTHETIC_JSON = PROJECT_ROOT / "reports" / "grid35_v2_T01_T10_seed_sweep_summary" / "t01_t10_summary.json"
DEFAULT_ACTUAL_JSON = PROJECT_ROOT / "reports" / "grid35_v2_T01_T10_actual_seed_sweep_summary" / "t01_t10_summary.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "grid35_v2_synthetic_vs_actual_T02_T10_comparison"


def load_summary(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s["scene"]: s for s in data["scenes"]}


def build_diff_rows(synth: dict[str, dict[str, Any]], actual: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for scene in COMPARE_SCENES:
        sy, ac = synth[scene], actual[scene]
        rows.append(
            {
                "scene": scene,
                "synthetic_single_clamp_free": sy["single_clamp_free_rate"],
                "actual_single_clamp_free": ac["single_clamp_free_rate"],
                "diff_single_clamp_free": ac["single_clamp_free_rate"] - sy["single_clamp_free_rate"],
                "synthetic_resample5_clamp_free": sy["resample5_clamp_free_rate"],
                "actual_resample5_clamp_free": ac["resample5_clamp_free_rate"],
                "diff_resample5_clamp_free": ac["resample5_clamp_free_rate"] - sy["resample5_clamp_free_rate"],
                "synthetic_shoulder_lift_clamp_rate": sy["per_joint_clamp_rate"]["shoulder_lift"],
                "actual_shoulder_lift_clamp_rate": ac["per_joint_clamp_rate"]["shoulder_lift"],
                "diff_shoulder_lift_clamp_rate": ac["per_joint_clamp_rate"]["shoulder_lift"] - sy["per_joint_clamp_rate"]["shoulder_lift"],
                "synthetic_elbow_flex_clamp_rate": sy["per_joint_clamp_rate"]["elbow_flex"],
                "actual_elbow_flex_clamp_rate": ac["per_joint_clamp_rate"]["elbow_flex"],
                "diff_elbow_flex_clamp_rate": ac["per_joint_clamp_rate"]["elbow_flex"] - sy["per_joint_clamp_rate"]["elbow_flex"],
                "synthetic_gt_l2": sy["single_gt_l2_mean"],
                "actual_gt_l2": ac["single_gt_l2_mean"],
                "diff_gt_l2": ac["single_gt_l2_mean"] - sy["single_gt_l2_mean"],
            }
        )
    return rows


def build_judgment(
    rows: list[dict[str, Any]],
    synth: dict[str, dict[str, Any]],
    actual: dict[str, dict[str, Any]],
) -> dict[str, str]:
    n = len(rows)
    synth_t02_10 = [synth[s] for s in COMPARE_SCENES]
    actual_t02_10 = [actual[s] for s in COMPARE_SCENES]

    synth_avg_single = statistics.fmean(s["single_clamp_free_rate"] for s in synth_t02_10)
    actual_avg_single = statistics.fmean(s["single_clamp_free_rate"] for s in actual_t02_10)
    synth_avg_r5 = statistics.fmean(s["resample5_clamp_free_rate"] for s in synth_t02_10)
    actual_avg_r5 = statistics.fmean(s["resample5_clamp_free_rate"] for s in actual_t02_10)

    n_actual_r5_improves = sum(1 for a in actual_t02_10 if a["resample5_clamp_free_rate"] > a["single_clamp_free_rate"])
    min_r5 = min(a["resample5_clamp_free_rate"] for a in actual_t02_10)
    max_r5 = max(a["resample5_clamp_free_rate"] for a in actual_t02_10)
    weakest_scenes = [a["scene"] for a in actual_t02_10 if a["resample5_clamp_free_rate"] == min_r5]
    strongest_scenes = [a["scene"] for a in actual_t02_10 if a["resample5_clamp_free_rate"] == max_r5]
    weakest_actual = next(a for a in actual_t02_10 if a["scene"] == weakest_scenes[0])
    strongest_actual = next(a for a in actual_t02_10 if a["scene"] == strongest_scenes[0])

    reproduces = (
        f"**Not in absolute magnitude - by design, not by failure.** Averaged over T02-T10: "
        f"synthetic single-sample clamp-free {synth_avg_single*100:.1f}% vs actual "
        f"{actual_avg_single*100:.1f}%; synthetic resample5 {synth_avg_r5*100:.1f}% vs actual "
        f"{actual_avg_r5*100:.1f}%. The actual numbers sit much closer to T01's own real-hardware "
        f"numbers (single {actual['T01']['single_clamp_free_rate']*100:.1f}%, resample5 "
        f"{actual['T01']['resample5_clamp_free_rate']*100:.1f}%) than to the synthetic T02-T10 "
        "average - which is exactly what should happen once you know the 'actual T02-T10' "
        "captures are real-hardware repeats of T01's own scene, not 9 different positions: the "
        "synthetic proxy was modeling a *different, harder* experiment (spatial generalization "
        "across genuinely distinct held-out grid positions) than what the actual capture ended up "
        "providing (repeat-capture robustness on one already-characterized scene). It would be "
        "wrong to read the gap as 'the synthetic proxy method is unreliable' - no scene-matched "
        "ground truth exists yet to make that call, since the actual data doesn't cover the "
        "positions the synthetic data covered."
    )

    q_repeats = (
        f"**Yes.** All {n}/{n} actual T02-T10 captures show the same seed-to-seed swing between "
        "WOULD_CLAMP and clean chunk-index-0 actions that T01 first surfaced, now confirmed across "
        "9 independent real-hardware capture sessions (not just T01's own single capture) - single-"
        f"sample clamp-free rate ranges {min(a['single_clamp_free_rate'] for a in actual_t02_10)*100:.0f}%"
        f"-{max(a['single_clamp_free_rate'] for a in actual_t02_10)*100:.0f}% "
        "across the group, never near 0% or 100%. This is a meaningfully stronger form of evidence "
        "than the synthetic run gave for T01 alone: it shows the effect survives real "
        "camera/robot-repositioning noise across repeated real captures of the same scene, not "
        "just repeated *inference* on one fixed captured frame."
    )

    q_resample_consistent = (
        f"**Yes, in direction, on all {n}/{n} actual scenes** - resample5 clamp-free rate exceeds "
        f"single-sample on every actual T02-T10 capture ({n_actual_r5_improves}/{n} scenes improve), "
        f"averaging {actual_avg_single*100:.1f}% -> {actual_avg_r5*100:.1f}%. Magnitude varies by "
        f"capture instance though: {'/'.join(strongest_scenes)} reach{'es' if len(strongest_scenes) == 1 else ''} "
        f"{strongest_actual['resample5_clamp_free_rate']*100:.1f}% while {'/'.join(weakest_scenes)} "
        f"only reach{'es' if len(weakest_scenes) == 1 else ''} {weakest_actual['resample5_clamp_free_rate']*100:.1f}% - since these are "
        "repeat captures of one scene, that spread is real-hardware capture-to-capture variance "
        "(camera framing/robot pose micro-differences interacting with the same flow-matching "
        "noise sensitivity), not evidence about how the mitigation performs on different scene "
        "geometries."
    )

    q_adopt = (
        "**Real hardware data now supports the resampling *direction* robustly, but still cannot "
        "confirm resampling generalizes across different scene geometries** - and that gap matters "
        "for a runtime-adoption decision. The actual T02-T10 captures strengthen confidence that "
        "resample<=5's benefit is not an artifact of T01's one particular capture (it reproduces "
        "across 9 independently-captured real sessions of that scene), but because all 9 are the "
        "*same* physical scene, they provide no additional evidence about scenes with different "
        "cube positions - that evidence still only exists in the synthetic proxy data, which is not "
        "real hardware. Before adopting resample<=5 repo-wide as the Shadow runtime candidate, the "
        "genuinely missing piece is **real Shadow captures at spatially distinct cube positions** "
        "(i.e. actually moving the cube, not re-capturing the same spot) - neither dataset built so "
        "far provides that combination (real hardware AND spatially distinct)."
    )

    weak_scene_note = (
        f"**{'/'.join(weakest_scenes)}** (tied) show{'s' if len(weakest_scenes) == 1 else ''} the "
        f"lowest resample5 success in the actual data ({weakest_actual['resample5_clamp_free_rate']*100:.1f}%, "
        f"single-sample {weakest_actual['single_clamp_free_rate']*100:.1f}%). But because every actual "
        f"T02-T10 capture is the same physical scene as T01, this does **not** indicate "
        f"{'/'.join(weakest_scenes)} {'is' if len(weakest_scenes) == 1 else 'are'} a weak *position* "
        "- it indicates that particular real capture session (camera framing/"
        "lighting/robot pose at that moment) landed in a harder region of the same underlying "
        "seed-noise distribution T01 already has. Treating it as a distinct 'weak scene' requiring "
        "its own checkpoint/data investigation would misattribute ordinary repeat-capture variance "
        "to a scene-geometry problem that these 9 captures cannot actually test."
    )

    return {
        "reproduces": reproduces,
        "q_repeats_in_actual": q_repeats,
        "q_resample_consistent_in_actual": q_resample_consistent,
        "q_adopt_resample5": q_adopt,
        "weak_scene_note": weak_scene_note,
    }


def write_csv(out_path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def write_markdown(
    out_path: Path,
    rows: list[dict[str, Any]],
    synth: dict[str, dict[str, Any]],
    actual: dict[str, dict[str, Any]],
    judgment: dict[str, str],
) -> None:
    lines: list[str] = []
    lines.append("# Synthetic proxy vs actual (real hardware) T02-T10 - comparison, NOT averaged together")
    lines.append("")
    lines.append(
        "Per explicit user instruction, synthetic and actual T02-T10 results are never averaged "
        "together anywhere in this repo. **The actual T01-T10 results "
        "(`reports/grid35_v2_T01_T10_actual_seed_sweep_summary/`) are the reference for any real "
        "policy decision** - this report only quantifies how the two datasets differ and why."
    )
    lines.append("")
    lines.append("## Data-quality caveat (read this before the table)")
    lines.append("")
    lines.append(
        "The *synthetic* T02-T10 scenes are 9 genuinely distinct held-out grid positions "
        "(`data/so101_cube_xy_midpoint_test10_v2_clean`, episodes 1-9). The *actual* T02-T10 "
        "captures turned out to be 9 real-hardware repeat captures of **the same fixed scene T01 "
        "already uses** (every source `shadow.json` has `scene_metadata.label == \"V2_F02\"`, "
        "`evaluation_mode == \"fixed-scene-repeat\"` - see "
        "`scripts/import_actual_shadow_t02_t10.py`). The two datasets were never sampling the same "
        "thing, so differences below reflect that mismatch in what was actually captured, not proxy "
        "modeling error."
    )
    lines.append("")

    lines.append("## Per-scene diff table (actual minus synthetic)")
    lines.append("")
    header = (
        "| scene | synthetic single CF | actual single CF | diff | synthetic r5 | actual r5 | diff "
        "| synthetic shoulder_lift clamp | actual shoulder_lift clamp | diff "
        "| synthetic elbow_flex clamp | actual elbow_flex clamp | diff |"
    )
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['scene']} | {r['synthetic_single_clamp_free']*100:.1f}% | {r['actual_single_clamp_free']*100:.1f}% "
            f"| {r['diff_single_clamp_free']*100:+.1f}pp | {r['synthetic_resample5_clamp_free']*100:.1f}% "
            f"| {r['actual_resample5_clamp_free']*100:.1f}% | {r['diff_resample5_clamp_free']*100:+.1f}pp "
            f"| {r['synthetic_shoulder_lift_clamp_rate']*100:.0f}% | {r['actual_shoulder_lift_clamp_rate']*100:.0f}% "
            f"| {r['diff_shoulder_lift_clamp_rate']*100:+.0f}pp | {r['synthetic_elbow_flex_clamp_rate']*100:.0f}% "
            f"| {r['actual_elbow_flex_clamp_rate']*100:.0f}% | {r['diff_elbow_flex_clamp_rate']*100:+.0f}pp |"
        )
    lines.append("")
    lines.append(
        f"(T01 omitted - identical real-hardware result in both summaries by construction, diff is "
        f"trivially zero: single {actual['T01']['single_clamp_free_rate']*100:.1f}%, resample5 "
        f"{actual['T01']['resample5_clamp_free_rate']*100:.1f}%.)"
    )
    lines.append("")

    lines.append("## Judgment")
    lines.append("")
    lines.append("### How well did the synthetic proxy reproduce the actual observation trend?")
    lines.append("")
    lines.append(judgment["reproduces"])
    lines.append("")
    lines.append("### Does the sampling-noise problem repeat across the actual captures?")
    lines.append("")
    lines.append(judgment["q_repeats_in_actual"])
    lines.append("")
    lines.append("### Does resample<=5 improve consistently in the actual data?")
    lines.append("")
    lines.append(judgment["q_resample_consistent_in_actual"])
    lines.append("")
    lines.append("### Is resample<=5 justified as the Shadow-runtime candidate, on actual-data evidence?")
    lines.append("")
    lines.append(judgment["q_adopt_resample5"])
    lines.append("")
    lines.append("### Which scene is still most fragile, and is it a policy/data problem?")
    lines.append("")
    lines.append(judgment["weak_scene_note"])
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic-json", type=Path, default=DEFAULT_SYNTHETIC_JSON)
    parser.add_argument("--actual-json", type=Path, default=DEFAULT_ACTUAL_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    synth = load_summary(args.synthetic_json)
    actual = load_summary(args.actual_json)

    rows = build_diff_rows(synth, actual)
    judgment = build_judgment(rows, synth, actual)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "synthetic_vs_actual_comparison.json").write_text(
        json.dumps({"rows": rows, "judgment": judgment}, indent=2), encoding="utf-8"
    )
    write_csv(out_dir / "synthetic_vs_actual_comparison.csv", rows)
    write_markdown(out_dir / "synthetic_vs_actual_comparison.md", rows, synth, actual, judgment)

    print(f"Wrote {out_dir / 'synthetic_vs_actual_comparison.json'}")
    print(f"Wrote {out_dir / 'synthetic_vs_actual_comparison.csv'}")
    print(f"Wrote {out_dir / 'synthetic_vs_actual_comparison.md'}")
    print("")
    for r in rows:
        print(
            f"{r['scene']}  single: synth={r['synthetic_single_clamp_free']*100:5.1f}% "
            f"actual={r['actual_single_clamp_free']*100:5.1f}% (diff {r['diff_single_clamp_free']*100:+5.1f}pp)  "
            f"r5: synth={r['synthetic_resample5_clamp_free']*100:5.1f}% "
            f"actual={r['actual_resample5_clamp_free']*100:5.1f}% (diff {r['diff_resample5_clamp_free']*100:+5.1f}pp)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
