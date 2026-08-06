"""원격 MuJoCo 진단 세션의 CSV/JSON 리포트 작성.

이 모듈은 순수 직렬화만 담당한다 - 안전 판정(safety_checks.py)이나 이상 패턴 감지
(diagnostic_analysis.py) 로직은 여기 두지 않는다. API 토큰 등 민감한 값은 이 모듈이
받는 인자에 애초에 포함되지 않아야 하며(호출자 책임), 혹시라도 summary dict에
``api_token``류 키가 섞여 있으면 저장 직전에 방어적으로 제거한다.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CSV_FIELDNAMES: tuple[str, ...] = (
    "local_timestamp",
    "remote_timestamp",
    "sequence",
    "network_latency_ms",
    "state_age_ms",
    "joint_name",
    "leader_position_deg",
    "follower_position_deg",
    "difference_deg",
    "leader_raw_tick",
    "follower_raw_tick",
    "mujoco_target_deg",
    "mujoco_qpos_deg",
    "mujoco_limit_margin_deg",
    "safety_status",
    "event_code",
    "blocked_reason",
)

# 리포트에 절대 포함하면 안 되는 민감한 키 (방어적 필터링용).
_FORBIDDEN_SUMMARY_KEYS = ("api_token", "token", "authorization")


@dataclass(frozen=True)
class SessionPaths:
    csv_path: Path | None
    json_path: Path


def resolve_session_paths(
    *,
    reports_dir: Path,
    session_id: str,
    explicit_report_path: Path | None,
    write_csv: bool,
) -> SessionPaths:
    """CSV/JSON 저장 경로를 결정한다.

    ``explicit_report_path``를 주면 그 경로를 JSON 기본 경로로 쓰고, CSV는 같은 이름에
    확장자만 ``.csv``로 바꿔 나란히 저장한다 (파일을 덮어쓰지 않도록 이미 존재하면 아무
    것도 하지 않는다 - 호출자가 이미 unique한 session_id를 넘긴다고 가정).
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    if explicit_report_path is not None:
        json_path = explicit_report_path
        csv_path = explicit_report_path.with_suffix(".csv") if write_csv else None
    else:
        json_path = reports_dir / f"session_{session_id}.json"
        csv_path = reports_dir / f"session_{session_id}.csv" if write_csv else None
    if json_path.parent != reports_dir:
        json_path.parent.mkdir(parents=True, exist_ok=True)
    return SessionPaths(csv_path=csv_path, json_path=json_path)


def make_session_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def write_csv_report(path: Path, rows: list[dict]) -> None:
    """CSV_FIELDNAMES 순서로 rows를 저장한다. 없는 필드는 빈 칸으로 채운다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})


def _strip_forbidden_keys(summary: dict) -> dict:
    cleaned = dict(summary)
    for key in list(cleaned.keys()):
        if any(bad in key.lower() for bad in _FORBIDDEN_SUMMARY_KEYS):
            cleaned.pop(key)
    return cleaned


def write_json_report(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_summary = _strip_forbidden_keys(summary)
    path.write_text(json.dumps(safe_summary, indent=2, ensure_ascii=False), encoding="utf-8")


def build_json_summary(
    *,
    server_url: str,
    duration_sec: float,
    requested_rate_hz: float,
    actual_sample_rate_hz: float,
    sample_count: int,
    latency_mean_ms: float,
    latency_max_ms: float,
    stale_count: int,
    timeout_count: int,
    joint_names: list[str],
    max_abs_difference: dict[str, float],
    mean_abs_difference: dict[str, float],
    persistent_difference_events: int,
    follower_saturation_events: int,
    sign_mismatch_events: int,
    offset_suspected: list[str],
    mujoco_blocked_events: int,
    network_pause_events: int,
    warnings: list[str],
    final_result: str,
) -> dict:
    """CSV/디스플레이 로직과 무관하게, 최종 JSON에 들어갈 키를 명시적으로 고정한다.

    (요구사항 12번 목록과 1:1로 대응 - 새 필드를 추가하고 싶으면 여기 시그니처에 먼저
    추가할 것. api_token은 애초에 인자로 받지 않는다.)
    """
    return {
        "server_url": server_url,
        "duration_sec": duration_sec,
        "requested_rate_hz": requested_rate_hz,
        "actual_sample_rate_hz": actual_sample_rate_hz,
        "sample_count": sample_count,
        "latency_mean_ms": latency_mean_ms,
        "latency_max_ms": latency_max_ms,
        "stale_count": stale_count,
        "timeout_count": timeout_count,
        "joint_names": joint_names,
        "max_abs_difference": max_abs_difference,
        "mean_abs_difference": mean_abs_difference,
        "persistent_difference_events": persistent_difference_events,
        "follower_saturation_events": follower_saturation_events,
        "sign_mismatch_events": sign_mismatch_events,
        "offset_suspected": offset_suspected,
        "mujoco_blocked_events": mujoco_blocked_events,
        "network_pause_events": network_pause_events,
        "warnings": warnings,
        "final_result": final_result,
    }
