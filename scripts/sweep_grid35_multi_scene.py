#!/usr/bin/env python3
"""T02-T10 first-action inference-seed sweep, batched over one checkpoint load.

Thin driver around ``scripts/sweep_grid35_first_action_seed.py``: reuses every one of that
script's functions (``load_shadow_observation``, ``load_images``, ``load_policy_bundle``,
``run_seed_sweep_with_bundle``, ``annotate_clamp``, ``annotate_gt_l2``, ``summarize``,
``write_report``) unchanged, and does **not** duplicate any of that inference/scoring logic here.
The only thing this script adds is looping over multiple scenes against a *single* loaded policy
bundle (checkpoint loaded once, not once per scene) while still re-seeding
(``torch.manual_seed``) and re-running ``predict_action_chunk`` independently per seed per scene -
exactly as the single-scene script does - so seed-to-seed reproducibility within a scene is
unaffected by batching across scenes.

Default scenes T02-T10 read their reference observation from the *synthetic* offline-proxy Shadow
reports built by ``scripts/build_synthetic_midpoint_shadow_reports.py`` (see that script's
docstring for why: no live SO-101 hardware is attached in this analysis session, so no new real
Shadow capture could be made for T02-T10). T01 already has a real hardware Shadow capture and is
handled by the single-scene script directly - this driver's default scene list starts at T02.

Deliberately out of scope (same constraints as the parent scripts):
  * No training / fine-tuning.
  * No modification of Safety Gate thresholds (``configs/safety_gate.yaml`` is read-only here).
  * No writes to any robot, real or simulated.
  * No modification of the LeRobot library itself - only this repo's own scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_grid35_first_action_pipeline import (  # noqa: E402
    nearest_frame_trajectory_analysis,
    safety_threshold_check,
)
from scripts.sweep_grid35_first_action_seed import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_SAFETY_GATE_YAML,
    DEFAULT_TASK,
    annotate_clamp,
    annotate_gt_l2,
    load_images,
    load_policy_bundle,
    load_shadow_observation,
    run_seed_sweep_with_bundle,
    summarize,
    write_report,
)

DEFAULT_SCENES = [f"T{i:02d}" for i in range(2, 11)]  # T02..T10


DEFAULT_SHADOW_REPORT_TEMPLATE = "reports/grid35_v2_shadow_{scene}/shadow_synthetic_{scene}.json"
DEFAULT_OUT_DIR_TEMPLATE = "reports/grid35_v2_{scene}_seed_sweep"

# Actual (real SO-101 hardware) T02-T10 scenes, imported by
# scripts/import_actual_shadow_t02_t10.py. See that script's docstring for the caveat that all 9
# are real-hardware repeats of T01's own fixed V2_F02 scene, not spatially distinct positions.
ACTUAL_SHADOW_REPORT_TEMPLATE = "reports/grid35_v2_shadow_{scene}_actual/shadow_patched.json"
ACTUAL_OUT_DIR_TEMPLATE = "reports/grid35_v2_{scene}_actual_seed_sweep"


def resolve_shadow_report_path(scene: str, template: str) -> Path:
    return PROJECT_ROOT / template.format(scene=scene)


def resolve_out_dir(scene: str, template: str) -> Path:
    return PROJECT_ROOT / template.format(scene=scene)


def run_one_scene(
    *,
    scene: str,
    shadow_report: Path,
    out_dir: Path,
    bundle: dict[str, Any],
    checkpoint: Path,
    dataset_root: Path,
    thresholds: dict[str, float],
    task: str,
    seeds: list[int],
) -> dict[str, Any]:
    print(f"[multi-scene] === {scene} ===")
    print(f"[multi-scene] loading reference Shadow observation from {shadow_report}")
    obs = load_shadow_observation(shadow_report)
    images = load_images(obs["camera_frame_paths"])
    if not images:
        print(f"[multi-scene] WARNING ({scene}): no camera frames found; running with no image inputs.")

    print(f"[multi-scene] computing nearest-training-demo immediate GT delta from {dataset_root}")
    traj = nearest_frame_trajectory_analysis(dataset_root, obs["state"], obs["raw_action_post_postprocessor"])
    gt_immediate_delta = traj["best_match"]["gt_immediate_delta_action_minus_state"]

    print(f"[multi-scene] running {len(seeds)} seeded inference calls for {scene} (seeds {seeds[0]}..{seeds[-1]})")
    rows = run_seed_sweep_with_bundle(bundle, obs["state"], images, task, seeds)

    annotate_clamp(rows, thresholds)
    annotate_gt_l2(rows, gt_immediate_delta)
    summary = summarize(rows, thresholds)

    payload = {
        "scene": scene,
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "shadow_report": str(shadow_report),
        "reference_observation_label": obs["label"],
        "reference_observation_state": obs["state"],
        "task": task,
        "seeds": seeds,
        "thresholds": thresholds,
        "gt_immediate_delta": gt_immediate_delta,
        "nearest_demo_match": {
            "episode": traj["best_match"]["episode"],
            "frame": traj["best_match"]["frame"],
            "l2_dist_deg": traj["best_match"]["l2_dist_deg"],
        },
        "rows": rows,
        "summary": summary,
    }

    write_report(out_dir, payload)
    print(
        f"[multi-scene] {scene}: {summary['clamp_free_seed_count']}/{summary['n_seeds']} seeds clamp-free, "
        f"L2 vs GT mean={summary['l2_error_vs_gt']['mean']:.2f}deg"
    )
    print("")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--safety-gate-yaml", type=Path, default=DEFAULT_SAFETY_GATE_YAML)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES, help="Scene labels, e.g. T02 T03 ... T10")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=19, help="Inclusive.")
    parser.add_argument(
        "--variant",
        choices=["synthetic", "actual"],
        default="synthetic",
        help="'synthetic' (default, backward-compatible): held-out midpoint_test10 offline-proxy "
        "scenes. 'actual': real SO-101 hardware Shadow captures imported by "
        "scripts/import_actual_shadow_t02_t10.py. Sets the shadow-report/out-dir templates below "
        "unless they are explicitly overridden.",
    )
    parser.add_argument(
        "--shadow-report-template",
        default=None,
        help="Path template with a {scene} placeholder. Defaults depend on --variant.",
    )
    parser.add_argument(
        "--out-dir-template",
        default=None,
        help="Path template with a {scene} placeholder. Defaults depend on --variant.",
    )
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_end + 1))
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()

    if args.variant == "actual":
        shadow_report_template = args.shadow_report_template or ACTUAL_SHADOW_REPORT_TEMPLATE
        out_dir_template = args.out_dir_template or ACTUAL_OUT_DIR_TEMPLATE
    else:
        shadow_report_template = args.shadow_report_template or DEFAULT_SHADOW_REPORT_TEMPLATE
        out_dir_template = args.out_dir_template or DEFAULT_OUT_DIR_TEMPLATE
    print(f"[multi-scene] variant={args.variant}  shadow_report_template={shadow_report_template!r}  "
          f"out_dir_template={out_dir_template!r}")

    print(f"[multi-scene] loading safety thresholds from {args.safety_gate_yaml}")
    thresholds = safety_threshold_check(args.safety_gate_yaml.resolve())["would_clamp_threshold_deg"]

    print(f"[multi-scene] loading checkpoint ONCE: {checkpoint}")
    bundle = load_policy_bundle(checkpoint)

    results: dict[str, dict[str, Any]] = {}
    for scene in args.scenes:
        shadow_report = resolve_shadow_report_path(scene, shadow_report_template)
        out_dir = resolve_out_dir(scene, out_dir_template)
        results[scene] = run_one_scene(
            scene=scene,
            shadow_report=shadow_report,
            out_dir=out_dir,
            bundle=bundle,
            checkpoint=checkpoint,
            dataset_root=dataset_root,
            thresholds=thresholds,
            task=args.task,
            seeds=seeds,
        )

    print("[multi-scene] summary across scenes:")
    for scene, payload in results.items():
        s = payload["summary"]
        print(
            f"  {scene}: clamp-free {s['clamp_free_seed_count']}/{s['n_seeds']}  "
            f"L2 mean={s['l2_error_vs_gt']['mean']:.2f}deg"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
