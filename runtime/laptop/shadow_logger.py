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
) -> dict[str, Any]:
    assert real_follower_write_count == 0, "REAL_SO101_WRITE 불변식 위반 - 리포트를 만들지 않고 즉시 중단해야 합니다."
    return {
        "mode": "SHADOW",
        "backend": backend,
        "real_robot_write_enabled": False,
        "generated_at": time.time(),
        "task": task,
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
