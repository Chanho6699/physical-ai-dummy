"""Shadow Mode 단일 실행의 JSON report (섹션 18) - ``reports/shadow_mode/shadow_<timestamp>.json``."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "shadow_mode"

RESULT_SHADOW_PASS = "SHADOW_PASS"
RESULT_SHADOW_PASS_WITH_CLAMP = "SHADOW_PASS_WITH_CLAMP"
RESULT_SHADOW_FAIL = "SHADOW_FAIL"


def make_session_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def resolve_report_path(reports_dir: Path | None, session_id: str) -> Path:
    directory = reports_dir or DEFAULT_REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"shadow_{session_id}.json"


def resolve_camera_frame_paths(reports_dir: Path | None, session_id: str) -> dict[str, Path]:
    directory = (reports_dir or DEFAULT_REPORTS_DIR) / "images" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return {"workspace": directory / "workspace.jpg", "wrist": directory / "wrist.jpg"}


def save_camera_frames(images: dict[str, Any], *, reports_dir: Path | None, session_id: str) -> dict[str, str]:
    """VLA에 실제로 보낸 workspace/wrist 프레임(HWC uint8 RGB)을 JPEG로 저장한다.

    과거 Shadow 실험에는 raw camera input이 보존되지 않아 "같은 입력으로 fixed-seed 재평가"가
    불가능했다 - 이 함수가 그 gap을 메운다. ``images``는
    ``runtime.laptop.observation_builder.build_observation``이 반환한
    ``BuiltObservation.images``(``CAMERA_WORKSPACE_KEY``/``CAMERA_WRIST_KEY`` -> numpy 배열)
    형식을 그대로 받는다.

    저장에 실패해도(PIL 미설치, 이미지 없음 등) 예외를 던지지 않고 빈/부분 dict를 반환한다 -
    카메라 저장 실패가 Shadow 실행 자체를 막으면 안 되기 때문이다(리포트에는 그만큼 경로가
    비어 있는 채로 남는다).
    """
    from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY

    key_to_label = {CAMERA_WORKSPACE_KEY: "workspace", CAMERA_WRIST_KEY: "wrist"}
    saved: dict[str, str] = {}
    if not images:
        return saved
    try:
        from PIL import Image
    except ImportError:
        return saved

    paths = resolve_camera_frame_paths(reports_dir, session_id)
    for wire_key, label in key_to_label.items():
        arr = images.get(wire_key)
        if arr is None:
            continue
        try:
            Image.fromarray(arr, mode="RGB").save(paths[label], format="JPEG", quality=95)
        except Exception:  # noqa: BLE001 - 이미지 하나 저장 실패가 리포트 전체를 막지 않는다
            continue
        saved[f"{label}_path"] = str(paths[label])
    return saved


def build_report(
    *,
    task: str,
    backend: str,
    observation: dict[str, Any],
    communication: dict[str, Any],
    vla: dict[str, Any],
    adapter: dict[str, Any],
    safety: dict[str, Any],
    mujoco: dict[str, Any],
    validation: dict[str, Any],
    real_follower_write_count: int,
    result: str,
    result_reasons: list[str],
    checkpoint: dict[str, Any] | None = None,
    evaluation_mode: str | None = None,
    scene_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert real_follower_write_count == 0, "REAL_SO101_WRITE 불변식 위반 - 리포트를 만들지 않고 즉시 중단해야 합니다."
    return {
        "mode": "SHADOW",
        "backend": backend,
        "real_robot_write_enabled": False,
        "generated_at": time.time(),
        "task": task,
        # checkpoint 절대경로/step/학습 dataset provenance - 과거 20ep/old 10k 체크포인트와
        # 섞어 해석하지 않기 위한 필수 기록 (None이면 어떤 backend도 checkpoint 메타데이터를
        # 주지 않은 것 - 예: 구성이 잘못됐거나 순수 통신 계약 테스트).
        "checkpoint": checkpoint,
        # "fixed-scene-repeat" | "midpoint-shadow" | None(미지정) - 요구사항 7번.
        "evaluation_mode": evaluation_mode,
        # T01~T10 라벨 등 사람이 CLI로 지정한 값만 담는다 - 물체의 실제 XY를 추정/기록하지
        # 않는다 (요구사항: "자동으로 물체 XY를 알고 있다고 가정하지 말 것").
        "scene_metadata": scene_metadata,
        "observation": observation,
        "communication": communication,
        "vla": vla,
        "adapter": adapter,
        "safety": safety,
        "mujoco": {**mujoco, "validation_result": validation},
        "hardware": {"real_follower_write_count": real_follower_write_count},
        "result": result,
        "result_reasons": result_reasons,
    }


def write_report(report: dict[str, Any], *, reports_dir: Path | None = None, session_id: str | None = None) -> Path:
    sid = session_id or make_session_id()
    path = resolve_report_path(reports_dir, sid)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
