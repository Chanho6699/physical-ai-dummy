#!/usr/bin/env python3
"""Import the FINAL actual (real-hardware) T01-T10 Shadow captures into the repo's standard
``shadow_patched.json`` shape, from ``reports/real_midpoint_shadow_T01_T10/``.

Source layout (already inside this repo - not an external Downloads folder like
``scripts/import_actual_shadow_t02_t10.py``'s source): ``T0N/{shadow.json,workspace.jpg,
wrist.jpg}`` for N in 1..10, produced by a real SO-101 Shadow Mode run
(``scripts/run_shadow_mode.py``: real read-only follower state + real workspace/wrist camera
frames; ``real_robot_write_enabled: false`` in every source JSON). Unlike the extraction target
path named in the request, the actual checked-in data sits one directory deeper (a
double-nested ``real_midpoint_shadow_T01_T10/real_midpoint_shadow_T01_T10/T0N/`` - an artifact of
however the capture bundle was unpacked); ``_locate_source_root`` below transparently handles
either layout so callers can keep using the shallow path.

This is explicitly **not** the same data as the earlier ``import_actual_shadow_t02_t10.py``
import: that source had every one of its 9 ``shadow.json`` files stamped
``scene_metadata.label == "V2_F02"`` / ``evaluation_mode == "fixed-scene-repeat"`` - i.e. 9
real-hardware repeats of T01's *own* fixed scene, not spatially distinct positions. This new
source instead has ``evaluation_mode == "midpoint-shadow"`` and a distinct
``scene_metadata.label``/``heldout_episode_index`` (0..9) per scene, matching the same
episode-index convention ``scripts/build_synthetic_midpoint_shadow_reports.py`` uses for the
*synthetic* T02-T10 proxies - i.e. this is the real-hardware counterpart of that held-out
``midpoint_test10`` mapping, not a repeat of one scene. That said, the raw follower joint-state
readout is itself nearly constant across all 10 scenes here (see ``state_diff_vs_t01_deg`` in the
written provenance - this looks like an intentional fixed-"midpoint"-observation-pose capture
protocol, where only the workspace/wrist camera content changes per held-out cube placement, not
a duplicate-capture bug); this script computes and records a camera-frame RMSE-vs-T01 per scene
(``image_rmse_vs_t01``) alongside the state diff so downstream readers can judge the evidence for
themselves rather than trusting a one-line claim. Every output's ``repo_import_provenance``
carries this verbatim.

Only ``observation.camera_frame_paths`` is rewritten (source paths point at the capture machine's
own filesystem, e.g. ``/home/sunglee/Projects/...``, which does not exist here) - every other
field of the source ``shadow.json`` is carried through unmodified.

Deliberately out of scope:
  * No modification of the source ``reports/real_midpoint_shadow_T01_T10/`` folder (read-only
    source; this script only writes new ``reports/grid35_v2_shadow_T0N_real_final/`` dirs).
  * No training / fine-tuning, no modification of Safety Gate thresholds.
  * No writes to any robot, real or simulated.
  * No modification of the LeRobot library itself.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_REQUESTED_ROOT = PROJECT_ROOT / "reports" / "real_midpoint_shadow_T01_T10"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "reports"
DEFAULT_SCENES = [f"T{i:02d}" for i in range(1, 11)]  # T01..T10 (all 10 - T01 included, unlike
# import_actual_shadow_t02_t10.py, since this source has its own fresh T01 capture rather than
# reusing the original grid35_v2_shadow_T01 report).


def _locate_source_root(requested_root: Path, scenes: list[str]) -> Path:
    """Return the directory that directly contains ``T0N/`` scene folders.

    Handles the double-nested-on-disk layout (``<requested_root>/real_midpoint_shadow_T01_T10/``)
    transparently; falls back to ``requested_root`` itself if that already contains the scene
    folders directly.
    """
    if all((requested_root / scene / "shadow.json").is_file() for scene in scenes):
        return requested_root
    nested = requested_root / requested_root.name
    if all((nested / scene / "shadow.json").is_file() for scene in scenes):
        return nested
    raise FileNotFoundError(
        f"could not find {scenes[0]}/shadow.json under {requested_root} or {nested} - "
        "unexpected source layout."
    )


def _image_rmse(path_a: Path, path_b: Path) -> float:
    from PIL import Image

    a = np.array(Image.open(path_a).convert("RGB"), dtype=np.float64)
    b = np.array(Image.open(path_b).convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    return float(np.sqrt(((a - b) ** 2).mean()))


def import_scene(
    source_root: Path,
    out_root: Path,
    scene: str,
    *,
    t01_state: dict[str, float],
    t01_workspace: Path,
    t01_wrist: Path,
) -> Path:
    src_dir = source_root / scene
    src_json = src_dir / "shadow.json"
    src_workspace = src_dir / "workspace.jpg"
    src_wrist = src_dir / "wrist.jpg"
    for p in (src_json, src_workspace, src_wrist):
        if not p.is_file():
            raise FileNotFoundError(f"expected source file missing: {p}")

    out_dir = out_root / f"grid35_v2_shadow_{scene}_real_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) verbatim audit copies (source untouched; these are our own new files)
    raw_json_path = out_dir / "shadow_raw.json"
    workspace_path = out_dir / "workspace.jpg"
    wrist_path = out_dir / "wrist.jpg"
    shutil.copyfile(src_json, raw_json_path)
    shutil.copyfile(src_workspace, workspace_path)
    shutil.copyfile(src_wrist, wrist_path)

    raw = json.loads(raw_json_path.read_text(encoding="utf-8"))
    scene_label = raw.get("scene_metadata", {}).get("label")
    eval_mode = raw.get("evaluation_mode")
    heldout_idx = raw.get("scene_metadata", {}).get("heldout_episode_index")
    state = raw["observation"]["state"]

    state_diff_vs_t01_deg = {j: float(state[j] - t01_state[j]) for j in state}
    image_rmse_vs_t01 = {
        "workspace": _image_rmse(workspace_path, t01_workspace),
        "wrist": _image_rmse(wrist_path, t01_wrist),
    }

    patched = dict(raw)
    patched["observation"] = dict(raw["observation"])
    patched["observation"]["camera_frame_paths"] = {
        "workspace_path": str(workspace_path.resolve()),
        "wrist_path": str(wrist_path.resolve()),
    }
    patched["repo_import_provenance"] = {
        "statement": (
            "Real SO-101 hardware Shadow capture (real read-only follower state + real "
            "workspace/wrist camera), imported from reports/real_midpoint_shadow_T01_T10/ - the "
            "FINAL actual multi-position T01-T10 capture session (distinct from the earlier "
            "grid35_v2_shadow_T0N_actual/ import, which was a same-scene repeat; see "
            "scripts/import_actual_shadow_t02_t10.py). Only observation.camera_frame_paths was "
            "rewritten here (source paths pointed at the capture machine's own filesystem and do "
            "not exist in this repo/session) - every other field is verbatim from the source "
            "shadow.json (see shadow_raw.json in this same directory for the untouched original)."
        ),
        "source_file": str(src_json),
        "source_scene_folder_label": scene,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "imported_by": "scripts/import_actual_shadow_t01_t10_final.py",
        "data_quality_note": (
            f"Source shadow.json scene_metadata.label = {scene_label!r}, heldout_episode_index = "
            f"{heldout_idx!r}, evaluation_mode = {eval_mode!r} - a DISTINCT label/index per scene "
            "(0..9, matching the same T0N<->episode-index convention "
            "scripts/build_synthetic_midpoint_shadow_reports.py uses for the synthetic proxies), "
            "NOT the earlier import's 'fixed-scene-repeat'/'V2_F02'-for-all pattern. However, the "
            "raw follower joint-state readout (observation.state) is itself nearly constant across "
            f"all 10 scenes: this scene's state differs from T01's by {state_diff_vs_t01_deg} deg "
            "per joint. That pattern is consistent with an intentional fixed-'midpoint'-observation"
            "-pose capture protocol (the arm is held at one repeatable reading pose; only the cube "
            "placement in front of the camera differs per held-out scene) rather than a duplicate "
            "capture - the workspace/wrist camera frames differ meaningfully scene-to-scene "
            f"(RMSE vs T01, 0-255 scale: workspace={image_rmse_vs_t01['workspace']:.2f}, "
            f"wrist={image_rmse_vs_t01['wrist']:.2f}), which a literal duplicate capture would not "
            "show. This script does not independently verify actual cube ground-truth position; it "
            "records both signals (state diff + image RMSE) so downstream readers can judge the "
            "evidence themselves rather than trusting a one-line claim."
        ),
        "state_diff_vs_t01_deg": state_diff_vs_t01_deg,
        "image_rmse_vs_t01_0_255_scale": image_rmse_vs_t01,
    }

    patched_json_path = out_dir / "shadow_patched.json"
    patched_json_path.write_text(json.dumps(patched, indent=2, ensure_ascii=False), encoding="utf-8")
    return patched_json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_REQUESTED_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    args = parser.parse_args()

    if not args.source_root.is_dir():
        print(f"[import] ERROR: source root not found: {args.source_root}")
        return 2

    source_root = _locate_source_root(args.source_root.resolve(), args.scenes)
    print(f"[import] resolved source root: {source_root}")

    t01_dir = source_root / "T01"
    t01_state = json.loads((t01_dir / "shadow.json").read_text(encoding="utf-8"))["observation"]["state"]
    t01_workspace = t01_dir / "workspace.jpg"
    t01_wrist = t01_dir / "wrist.jpg"

    written = {}
    for scene in args.scenes:
        print(f"[import] {scene}: importing from {source_root / scene}")
        patched_path = import_scene(
            source_root,
            args.out_root.resolve(),
            scene,
            t01_state=t01_state,
            t01_workspace=t01_workspace,
            t01_wrist=t01_wrist,
        )
        written[scene] = str(patched_path)
        print(f"[import]   wrote {patched_path}")

    print("")
    print(f"[import] imported {len(written)} real_final scenes:")
    for label, path in written.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
