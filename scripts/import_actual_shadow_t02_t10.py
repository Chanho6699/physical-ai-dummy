#!/usr/bin/env python3
"""Import the real (hardware-captured) T02-T10 Shadow reports into the repo.

Source: a Windows Downloads folder (``real_shadow_T02_T10/T0N/{shadow.json,workspace.jpg,
wrist.jpg}``) produced by an actual SO-101 Shadow Mode run (real follower state, real
workspace/wrist camera frames) on a different machine (``run_shadow_mode.py`` there, HTTP VLA
backend - see ``vla.model_id``/``communication.server_url`` in each ``shadow.json``).

This script only **copies and non-destructively patches** that data into the repo - it never
writes to the source folder:

  * ``shadow_raw.json`` + ``workspace.jpg`` + ``wrist.jpg`` - byte-for-byte copies of the source
    files (audit trail, untouched original content).
  * ``shadow_patched.json`` - a *separate* copy of the same JSON with only
    ``observation.camera_frame_paths.{workspace_path,wrist_path}`` rewritten to point at the
    repo-local jpg copies above (the source JSON's paths point at the capture machine's own
    filesystem, e.g. ``/home/sunglee/Projects/...``, which does not exist here). A
    ``repo_import_provenance`` key is added (new top-level key - does not shadow/overwrite
    anything ``load_shadow_observation()`` reads) documenting the patch. This patched copy is
    what ``scripts/sweep_grid35_multi_scene.py`` actually consumes for inference.

Important data-quality note surfaced here (see each output dir's provenance block and the
T01-T10 actual summary report for the full discussion): every one of the 9 source
``shadow.json`` files has ``scene_metadata.label == "V2_F02"`` and
``evaluation_mode == "fixed-scene-repeat"`` - i.e. **all 9 are repeat real-hardware captures of
the same fixed scene T01 already uses**, not 9 spatially distinct held-out positions. This
script preserves that fact verbatim (it does not relabel or reinterpret the source metadata) and
flags it in every generated ``shadow_patched.json``.

Deliberately out of scope:
  * No modification of the source Downloads folder (read-only source).
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE_ROOT = Path("/mnt/c/Users/rlack/Downloads/real_shadow_T02_T10/real_shadow_T02_T10")
DEFAULT_OUT_ROOT = PROJECT_ROOT / "reports"
DEFAULT_SCENES = [f"T{i:02d}" for i in range(2, 11)]  # T02..T10


def import_scene(source_root: Path, out_root: Path, scene: str) -> Path:
    src_dir = source_root / scene
    src_json = src_dir / "shadow.json"
    src_workspace = src_dir / "workspace.jpg"
    src_wrist = src_dir / "wrist.jpg"
    for p in (src_json, src_workspace, src_wrist):
        if not p.is_file():
            raise FileNotFoundError(f"expected source file missing: {p}")

    out_dir = out_root / f"grid35_v2_shadow_{scene}_actual"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) verbatim audit copies (source untouched; these are our own new files)
    raw_json_path = out_dir / "shadow_raw.json"
    workspace_path = out_dir / "workspace.jpg"
    wrist_path = out_dir / "wrist.jpg"
    shutil.copyfile(src_json, raw_json_path)
    shutil.copyfile(src_workspace, workspace_path)
    shutil.copyfile(src_wrist, wrist_path)

    # 2) patched copy: rewrite camera_frame_paths only, add provenance, everything else verbatim
    raw = json.loads(raw_json_path.read_text(encoding="utf-8"))
    scene_label = raw.get("scene_metadata", {}).get("label")
    eval_mode = raw.get("evaluation_mode")

    patched = dict(raw)
    patched["observation"] = dict(raw["observation"])
    patched["observation"]["camera_frame_paths"] = {
        "workspace_path": str(workspace_path.resolve()),
        "wrist_path": str(wrist_path.resolve()),
    }
    patched["repo_import_provenance"] = {
        "statement": (
            "Real SO-101 hardware Shadow capture (actual follower state + actual workspace/wrist "
            "camera), imported from a Windows Downloads folder produced on a different machine. "
            "Only observation.camera_frame_paths was rewritten here (source paths pointed at the "
            "capture machine's own filesystem and do not exist in this repo/session) - every other "
            "field is verbatim from the source shadow.json (see shadow_raw.json in this same "
            "directory for the untouched original)."
        ),
        "source_file": str(src_json),
        "source_scene_folder_label": scene,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "imported_by": "scripts/import_actual_shadow_t02_t10.py",
        "data_quality_note": (
            f"Source shadow.json scene_metadata.label = {scene_label!r}, evaluation_mode = "
            f"{eval_mode!r}. This matches T01's own fixed scene (V2_F02, fixed-scene-repeat), NOT "
            "a spatially distinct held-out position - i.e. despite the T02..T10 folder naming, "
            "this capture is a real-hardware repeat of T01's scene, not a different scene "
            "geometry. See the actual T01-T10 summary report for the full discussion and its "
            "effect on the synthetic-vs-actual comparison."
        ),
    }

    patched_json_path = out_dir / "shadow_patched.json"
    patched_json_path.write_text(json.dumps(patched, indent=2, ensure_ascii=False), encoding="utf-8")
    return patched_json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    args = parser.parse_args()

    if not args.source_root.is_dir():
        print(f"[import] ERROR: source root not found: {args.source_root}")
        return 2

    written = {}
    for scene in args.scenes:
        print(f"[import] {scene}: importing from {args.source_root / scene}")
        patched_path = import_scene(args.source_root.resolve(), args.out_root.resolve(), scene)
        written[scene] = str(patched_path)
        print(f"[import]   wrote {patched_path}")

    print("")
    print(f"[import] imported {len(written)} actual scenes:")
    for label, path in written.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
