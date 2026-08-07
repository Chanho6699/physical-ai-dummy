"""Instrumented Teleop 다중-run 통합(offline) 분석기.

**완전 offline 분석이다.** 이 모듈은 이미 디스크에 저장된 CSV/JSON 결과 파일만 읽는다 -
``lerobot``도, ``hardware/safety/*``의 serial 접근 클래스(``FeetechMotorsBus`` 등)도, 실제
로봇 객체도 이 파일 어디에서도 import/생성/호출하지 않는다. 유일하게 재사용하는 것은
``hardware/diagnostics/instrumented_teleop.py``의 **순수 계산 함수/데이터클래스**
(``TeleopCycleSample``/``TeleopRunResult``/``compute_deadband_summary``/
``classify_causal_response``/``_percentile`` 등)뿐이다 - 이 함수들은 그 자체로 하드웨어에
접근하지 않는 순수 계산이므로(``instrumented_teleop.py``는 lerobot이 설치되어 있지 않은
환경에서도 import된다 - 실제로 이 모듈을 시스템 python으로 확인했다), 이 모듈을 import해도
안전하다.

## 재사용 원칙

새 causal deadband 로직을 다시 만들지 않는다 - CSV의 각 행을 실제
``run_instrumented_teleop_loop()``가 만들었던 것과 동일한 ``TeleopCycleSample``로
복원(reconstruct)한 뒤, 단일-run 분석에 쓰였던 ``compute_deadband_summary()``를 run마다
그대로 호출하고, 그 결과(버킷별 response/no_response/opposite_motion 카운트)를 6개 run에
걸쳐 합산한다. percentile 계산도 ``instrumented_teleop.py``의 ``_percentile``(private이지만
동일 저장소 내부이므로 재사용 - 로직 중복을 피하는 것이 이름 규칙보다 우선한다)을 그대로
쓴다.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hardware.diagnostics.instrumented_teleop import (
    DEFAULT_DEADBAND_LOOKAHEAD_MS,
    DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS,
    INSUFFICIENT_DATA,
    INSUFFICIENT_FOR_DEADBAND_ESTIMATE,
    JOINT_ORDER,
    TARGET_JOINT,
    NO_RESPONSE,
    OPPOSITE_MOTION,
    RESPONSE,
    TeleopCycleSample,
    TeleopRunResult,
    _percentile,  # noqa: PLC2701 - 의도적 재사용, 위 모듈 docstring 참고
    compute_deadband_summary,
)

__all__ = [
    "RUN_FILENAME_RE",
    "QUALITY_OK",
    "QUALITY_WARNING",
    "NO_RESPONSE_REGION",
    "TRANSITION_REGION",
    "HIGH_RESPONSE_REGION",
    "DEFAULT_MIN_CSV_BYTES",
    "DEFAULT_STALE_ACTUAL_HZ_MIN",
    "DEFAULT_STALE_ACTUAL_HZ_MAX",
    "DEFAULT_MIN_DT_S",
    "DEFAULT_MAX_DT_S",
    "RunBundle",
    "discover_run_files",
    "select_latest_runs",
    "row_to_sample",
    "load_run_samples",
    "load_run_report",
    "load_run_bundle",
    "assess_run_quality",
    "compute_joint_value_series",
    "compute_frame_deltas",
    "compute_velocities",
    "percentile_summary",
    "compute_joint_aggregate",
    "compute_latency_aggregate",
    "compute_deadband_aggregate",
    "classify_deadband_region",
    "compute_run_to_run_stability",
    "compute_candidates",
    "build_aggregate_report",
    "render_markdown_report",
]

RUN_FILENAME_RE = re.compile(r"^instrumented_wrist_roll_(\d{8}_\d{6})\.csv$")

QUALITY_OK = "OK"
QUALITY_WARNING = "QUALITY_WARNING"

NO_RESPONSE_REGION = "NO_RESPONSE_REGION"
TRANSITION_REGION = "TRANSITION_REGION"
HIGH_RESPONSE_REGION = "HIGH_RESPONSE_REGION"

# 근거: dry-run/빈 파일은 헤더 한 줄(수십 바이트) 정도라 실제 20초 실행(수백 KB)과 확연히
# 구분된다 - 1KB를 문턱값으로 삼는다(실측 튜닝값 아님, 명백한 이상치만 거른다).
DEFAULT_MIN_CSV_BYTES = 1024

# 근거: 정상 SO-101 teleop 루프는 지금까지 관측된 범위가 대략 30~120Hz다(59Hz 정상 실행,
# 89.67Hz는 timing 버그가 있었던 비정상 실행으로 이미 확인됨) - run-level 품질 점검에서
# "정상 범위 밖"을 감지하기 위한 느슨한 참고 범위일 뿐, 확정 스펙이 아니다.
DEFAULT_STALE_ACTUAL_HZ_MIN = 30.0
DEFAULT_STALE_ACTUAL_HZ_MAX = 120.0

# frame-to-frame velocity 계산에서 dt가 이 범위를 벗어나면 quality flag 처리하고 그 샘플은
# velocity 계산에서 제외한다(연속된 두 timestamp가 사실상 같거나, 계측 사이에 비정상적으로
# 긴 공백이 있었다는 뜻이라 순간 속도로 나누면 왜곡된다) - 실측 튜닝값이 아니라 보수적
# 이상치 필터다.
DEFAULT_MIN_DT_S = 1e-4
DEFAULT_MAX_DT_S = 0.5


# ---------------------------------------------------------------------------
# run 파일 탐색/선택
# ---------------------------------------------------------------------------


def discover_run_files(directory: Path) -> list[tuple[str, Path, Path | None]]:
    """``instrumented_wrist_roll_<timestamp>.csv`` 패턴의 run을 timestamp 오름차순으로 찾는다.

    대응하는 ``..._report.json``이 없으면 세 번째 값이 ``None``이다. 하드웨어/네트워크
    접근 없음 - 순수 파일시스템 조회.
    """
    results: list[tuple[str, Path, Path | None]] = []
    for csv_path in sorted(directory.glob("instrumented_wrist_roll_*.csv")):
        m = RUN_FILENAME_RE.match(csv_path.name)
        if not m:
            continue
        ts = m.group(1)
        json_path = directory / f"instrumented_wrist_roll_{ts}_report.json"
        results.append((ts, csv_path, json_path if json_path.is_file() else None))
    results.sort(key=lambda t: t[0])
    return results


def select_latest_runs(
    directory: Path,
    *,
    count: int = 6,
    require_json: bool = True,
    min_csv_bytes: int = DEFAULT_MIN_CSV_BYTES,
) -> list[tuple[str, Path, Path | None]]:
    """가장 최근 ``count``개의 정상적인 run을 고른다.

    dry-run/빈 파일(``min_csv_bytes`` 미만)과 JSON 리포트가 없는 run은 제외한다(테스트
    fixture나 미완료 실행을 걸러내기 위함). 그 뒤 timestamp 기준 최신 것부터 ``count``개를
    반환한다.
    """
    all_runs = discover_run_files(directory)
    filtered: list[tuple[str, Path, Path | None]] = []
    for ts, csv_path, json_path in all_runs:
        try:
            if csv_path.stat().st_size < min_csv_bytes:
                continue
        except OSError:
            continue
        if require_json and json_path is None:
            continue
        filtered.append((ts, csv_path, json_path))
    return filtered[-count:] if count is not None else filtered


# ---------------------------------------------------------------------------
# CSV 행 -> TeleopCycleSample 복원 (offline, 하드웨어 접근 없음)
# ---------------------------------------------------------------------------


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    parsed = _parse_float(value)
    return int(parsed) if parsed is not None else None


def _parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


_REQUIRED_ROW_FIELDS = (
    "loop_index",
    "elapsed_sec",
    "loop_hz",
    "leader_wrist_roll_deg",
    "leader_wrist_roll_delta_from_start_deg",
    "command_wrist_roll_deg",
)


def row_to_sample(row: dict[str, str]) -> TeleopCycleSample | None:
    """CSV 한 행을 ``TeleopCycleSample``로 복원한다. 필수 필드가 없거나 숫자로 변환할 수
    없으면(malformed row) ``None``을 반환한다 - 호출부가 건너뛴다.
    """
    try:
        loop_index = int(float(row["loop_index"]))
        elapsed_sec = float(row["elapsed_sec"])
        loop_hz = float(row["loop_hz"])
        leader_wrist_roll_deg = float(row["leader_wrist_roll_deg"])
        leader_wrist_roll_delta_from_start_deg = float(row["leader_wrist_roll_delta_from_start_deg"])
        command_wrist_roll_deg = float(row["command_wrist_roll_deg"])
    except (KeyError, ValueError, TypeError):
        return None

    leader_command_all_joints: dict[str, float] = {}
    follower_sent_all_joints: dict[str, float] = {}
    follower_observation_all_joints: dict[str, float] = {}
    for joint in JOINT_ORDER:
        v = _parse_float(row.get(f"leader_command_{joint}"))
        if v is not None:
            leader_command_all_joints[f"{joint}.pos"] = v
        v = _parse_float(row.get(f"follower_sent_{joint}"))
        if v is not None:
            follower_sent_all_joints[f"{joint}.pos"] = v
        v = _parse_float(row.get(f"follower_observation_{joint}"))
        if v is not None:
            follower_observation_all_joints[f"{joint}.pos"] = v

    warning_types_raw = (row.get("warning_types") or "").strip()
    warning_types = tuple(w for w in warning_types_raw.split(";") if w)

    return TeleopCycleSample(
        loop_index=loop_index,
        timestamp_iso=row.get("timestamp", ""),
        elapsed_sec=elapsed_sec,
        loop_hz=loop_hz,
        leader_wrist_roll_deg=leader_wrist_roll_deg,
        leader_wrist_roll_delta_from_start_deg=leader_wrist_roll_delta_from_start_deg,
        command_wrist_roll_deg=command_wrist_roll_deg,
        follower_goal_raw=_parse_int(row.get("follower_goal_raw")),
        follower_goal_deg=_parse_float(row.get("follower_goal_deg")),
        follower_present_raw=_parse_int(row.get("follower_present_raw")),
        follower_present_deg=_parse_float(row.get("follower_present_deg")),
        goal_present_error_raw=_parse_int(row.get("goal_present_error_raw")),
        goal_present_error_deg=_parse_float(row.get("goal_present_error_deg")),
        follower_present_delta_from_prev_raw=_parse_int(row.get("follower_present_delta_from_prev_raw")),
        follower_present_delta_from_prev_deg=_parse_float(row.get("follower_present_delta_from_prev_deg")),
        follower_present_delta_from_start_deg=_parse_float(row.get("follower_present_delta_from_start_deg")),
        follower_torque_enable=_parse_int(row.get("follower_torque_enable")),
        follower_acceleration=_parse_int(row.get("follower_acceleration")),
        follower_acceleration_multiplier=_parse_int(row.get("follower_acceleration_multiplier")),
        follower_moving=_parse_int(row.get("follower_moving")),
        follower_status=_parse_int(row.get("follower_status")),
        send_action_executed=_parse_bool(row.get("send_action_executed")),
        leader_command_all_joints=leader_command_all_joints,
        follower_sent_all_joints=follower_sent_all_joints,
        follower_observation_all_joints=follower_observation_all_joints,
        register_read_error=(row.get("register_read_error") or None),
        warning_types=warning_types,
    )


def load_run_samples(csv_path: Path) -> tuple[list[TeleopCycleSample], int]:
    """CSV를 읽어 ``(samples, malformed_row_count)``를 반환한다. 원본 CSV는 읽기만 한다."""
    samples: list[TeleopCycleSample] = []
    malformed = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sample = row_to_sample(row)
            if sample is None:
                malformed += 1
                continue
            samples.append(sample)
    return samples, malformed


def load_run_report(json_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# RunBundle: run 하나(=CSV+JSON)를 메모리에 올린 결과
# ---------------------------------------------------------------------------


@dataclass
class RunBundle:
    timestamp: str
    csv_path: Path
    json_path: Path | None
    report: dict[str, Any] | None
    samples: list[TeleopCycleSample]
    malformed_row_count: int
    quality: dict[str, Any] = field(default_factory=dict)


def load_run_bundle(timestamp: str, csv_path: Path, json_path: Path | None) -> RunBundle:
    samples, malformed = load_run_samples(csv_path)
    report = load_run_report(json_path) if json_path is not None else None
    bundle = RunBundle(
        timestamp=timestamp,
        csv_path=csv_path,
        json_path=json_path,
        report=report,
        samples=samples,
        malformed_row_count=malformed,
    )
    bundle.quality = assess_run_quality(bundle)
    return bundle


def assess_run_quality(
    bundle: RunBundle,
    *,
    min_hz: float = DEFAULT_STALE_ACTUAL_HZ_MIN,
    max_hz: float = DEFAULT_STALE_ACTUAL_HZ_MAX,
) -> dict[str, Any]:
    """섹션 3: run-level 품질 점검. 이상 run을 자동 제외하지 않고 ``QUALITY_WARNING``만 남긴다."""
    reasons: list[str] = []
    report = bundle.report or {}
    analysis = report.get("analysis", {}) if report else {}

    stopped_reason = report.get("stopped_reason")
    sample_count = analysis.get("sample_count", len(bundle.samples))
    actual_loop_hz = analysis.get("actual_loop_hz")
    register_read_error_count = analysis.get("register_read_error_count", 0)
    status_ever_nonzero = analysis.get("status_ever_nonzero", False)

    if bundle.report is None:
        reasons.append("JSON report를 읽지 못했습니다.")
    if stopped_reason is not None and stopped_reason not in ("DURATION_ELAPSED", "KEYBOARD_INTERRUPT"):
        reasons.append(f"stopped_reason={stopped_reason} (정상 종료가 아닙니다).")
    if sample_count is not None and sample_count < 100:
        reasons.append(f"sample_count가 매우 적습니다 ({sample_count}).")
    if actual_loop_hz is not None and not (min_hz <= actual_loop_hz <= max_hz):
        reasons.append(f"actual_loop_hz={actual_loop_hz:.2f}가 참고 범위 [{min_hz:g}, {max_hz:g}]를 벗어납니다.")
    if register_read_error_count:
        reasons.append(f"register_read_error_count={register_read_error_count} (계측 read 실패가 있었습니다).")
    if status_ever_nonzero:
        reasons.append("status_ever_nonzero=True (Status 레지스터 이상이 관측됐습니다).")
    if bundle.malformed_row_count:
        reasons.append(f"CSV에 malformed row {bundle.malformed_row_count}개가 있었습니다(분석에서 제외됨).")

    return {
        "verdict": QUALITY_WARNING if reasons else QUALITY_OK,
        "reasons": reasons,
        "duration_sec": analysis.get("elapsed_sec"),
        "sample_count": sample_count,
        "actual_loop_hz": actual_loop_hz,
        "stopped_reason": stopped_reason,
        "register_read_error_count": register_read_error_count,
        "status_ever_nonzero": status_ever_nonzero,
        "command_movement_range_deg": analysis.get("command_movement_range_deg"),
        "follower_movement_range_deg": analysis.get("follower_movement_range_deg"),
    }


# ---------------------------------------------------------------------------
# joint 시계열 -> range / frame delta / velocity / tracking error
# ---------------------------------------------------------------------------


def compute_joint_value_series(
    samples: list[TeleopCycleSample], joint: str, *, field_name: str = "follower_observation_all_joints"
) -> tuple[list[float], list[float]]:
    """``samples``에서 ``joint``의 값 시계열과 대응 ``elapsed_sec`` 시계열을 뽑는다.

    ``field_name``은 ``leader_command_all_joints``/``follower_sent_all_joints``/
    ``follower_observation_all_joints`` 중 하나 - 전부 ``TeleopCycleSample``에 이미 있는
    필드라 새로 계산하지 않는다.
    """
    values: list[float] = []
    timestamps: list[float] = []
    key = f"{joint}.pos"
    for s in samples:
        d = getattr(s, field_name)
        v = d.get(key)
        if v is not None:
            values.append(v)
            timestamps.append(s.elapsed_sec)
    return values, timestamps


def compute_frame_deltas(values: list[float]) -> list[float]:
    """연속된 샘플 간 절대 위치 변화량 - timestamp 없이도 계산 가능한 순수 위치 델타다."""
    return [abs(values[i] - values[i - 1]) for i in range(1, len(values))]


def compute_velocities(
    values: list[float],
    timestamps: list[float],
    *,
    min_dt_s: float = DEFAULT_MIN_DT_S,
    max_dt_s: float = DEFAULT_MAX_DT_S,
) -> tuple[list[float], int]:
    """timestamp 기반 순간 속도(deg/s 등, 단위는 ``values``와 동일)를 계산한다.

    ``dt``가 ``[min_dt_s, max_dt_s]`` 밖이면(사실상 동시 timestamp, 또는 비정상적으로 긴
    공백) 그 구간은 속도 계산에서 제외하고 flag 카운트만 올린다 - 순간 속도로 나누면 값이
    왜곡되기 때문이다.
    """
    velocities: list[float] = []
    flagged = 0
    for i in range(1, len(values)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt < min_dt_s or dt > max_dt_s:
            flagged += 1
            continue
        velocities.append(abs(values[i] - values[i - 1]) / dt)
    return velocities, flagged


def percentile_summary(values: list[float]) -> dict[str, float] | None:
    """``instrumented_teleop._percentile``을 재사용해 mean/p50/p90/p95/p99/max를 만든다."""
    if not values:
        return None
    sorted_values = sorted(values)
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(sorted_values, 50),
        "p90": _percentile(sorted_values, 90),
        "p95": _percentile(sorted_values, 95),
        "p99": _percentile(sorted_values, 99),
        "max": sorted_values[-1],
        "min": sorted_values[0],
        "n": len(values),
    }


def compute_joint_aggregate(
    bundles: list[RunBundle],
    joint: str,
    *,
    min_dt_s: float = DEFAULT_MIN_DT_S,
    max_dt_s: float = DEFAULT_MAX_DT_S,
) -> dict[str, Any]:
    """섹션 4: joint 하나에 대한 range/frame-delta/velocity/tracking-error 통합 결과.

    range/frame-delta/velocity는 ``follower_observation_{joint}``(실제 관측된 follower
    상태) 시계열로 계산한다. tracking error는 다음을 쓴다:
    - ``wrist_roll``: register 수준 ``goal_present_error_deg``(가장 신뢰도 높은 신호,
      단일-run 분석과 동일한 정의).
    - 그 외 5개 joint: ``follower_sent_{joint}``(이번 cycle에 실제로 전송한 command) 대비
      ``follower_observation_{joint}``의 차이 - 이 5개 joint는 개별 Goal/Present 레지스터를
      읽지 않으므로(계측 대상이 wrist_roll 하나였다) 사용 가능한 유일한 tracking 신호다.
    """
    per_run: list[dict[str, Any]] = []
    pooled_values: list[float] = []
    pooled_frame_deltas: list[float] = []
    pooled_velocities: list[float] = []
    pooled_dt_flagged = 0
    pooled_tracking_errors: list[float] = []

    for bundle in bundles:
        values, timestamps = compute_joint_value_series(bundle.samples, joint)
        frame_deltas = compute_frame_deltas(values)
        velocities, dt_flagged = compute_velocities(values, timestamps, min_dt_s=min_dt_s, max_dt_s=max_dt_s)

        if joint == TARGET_JOINT:
            tracking_errors = [abs(s.goal_present_error_deg) for s in bundle.samples if s.goal_present_error_deg is not None]
            tracking_error_source = "goal_vs_present_register"
        else:
            sent, _ts_sent = compute_joint_value_series(bundle.samples, joint, field_name="follower_sent_all_joints")
            obs, _ts_obs = compute_joint_value_series(bundle.samples, joint, field_name="follower_observation_all_joints")
            n = min(len(sent), len(obs))
            tracking_errors = [abs(sent[i] - obs[i]) for i in range(n)]
            tracking_error_source = "command_vs_observation"

        pooled_values.extend(values)
        pooled_frame_deltas.extend(frame_deltas)
        pooled_velocities.extend(velocities)
        pooled_dt_flagged += dt_flagged
        pooled_tracking_errors.extend(tracking_errors)

        per_run.append(
            {
                "timestamp": bundle.timestamp,
                "range_min": min(values) if values else None,
                "range_max": max(values) if values else None,
                "frame_delta": percentile_summary(frame_deltas),
                "velocity": percentile_summary(velocities),
                "tracking_error": percentile_summary(tracking_errors),
                "dt_quality_flagged_count": dt_flagged,
            }
        )

    sorted_pooled = sorted(pooled_values)
    return {
        "joint": joint,
        "tracking_error_source": "goal_vs_present_register" if joint == TARGET_JOINT else "command_vs_observation",
        "per_run": per_run,
        "aggregate": {
            "range": {
                "min": sorted_pooled[0] if sorted_pooled else None,
                "max": sorted_pooled[-1] if sorted_pooled else None,
                "p01": _percentile(sorted_pooled, 1) if sorted_pooled else None,
                "p99": _percentile(sorted_pooled, 99) if sorted_pooled else None,
                "run_min_max_spread": (
                    (max(r["range_max"] for r in per_run if r["range_max"] is not None) - min(r["range_min"] for r in per_run if r["range_min"] is not None))
                    if any(r["range_max"] is not None for r in per_run)
                    else None
                ),
            },
            "frame_delta": percentile_summary(pooled_frame_deltas),
            "velocity": percentile_summary(pooled_velocities),
            "velocity_dt_quality_flagged_count": pooled_dt_flagged,
            "tracking_error": percentile_summary(pooled_tracking_errors),
        },
    }


# ---------------------------------------------------------------------------
# latency 통합 (섹션 5) - 각 run의 사전 계산된 lag 추정값을 재사용/집계만 한다
# ---------------------------------------------------------------------------


def compute_latency_aggregate(bundles: list[RunBundle]) -> dict[str, Any]:
    """각 run이 이미 계산해 둔(``compute_run_analysis``가 생성한)
    ``command_to_actual_lag_frames``/``..._ms_timestamp_based``를 모아 집계한다 - lag
    추정 알고리즘 자체를 다시 실행하지 않는다(그 알고리즘은 단일 episode 안에서만 의미가
    있고, 여러 run을 이어붙여 재추정하면 episode 경계에서 허위 상관이 생길 수 있다).
    """
    per_run: list[dict[str, Any]] = []
    ms_values: list[float] = []
    frame_values: list[int] = []

    for bundle in bundles:
        analysis = (bundle.report or {}).get("analysis", {}) if bundle.report else {}
        lag_estimate = analysis.get("command_to_actual_lag_estimate")
        lag_frames = analysis.get("command_to_actual_lag_frames")
        lag_ms_ts = analysis.get("command_to_actual_lag_ms_timestamp_based")
        lag_ms_frame = analysis.get("command_to_actual_lag_ms_frame_based")

        available = lag_estimate != INSUFFICIENT_DATA and lag_ms_ts is not None
        per_run.append(
            {
                "timestamp": bundle.timestamp,
                "available": available,
                "lag_frames": lag_frames,
                "lag_ms_timestamp_based": lag_ms_ts,
                "lag_ms_frame_based": lag_ms_frame,
            }
        )
        if available:
            ms_values.append(lag_ms_ts)
            if lag_frames is not None:
                frame_values.append(lag_frames)

    if not ms_values:
        return {"verdict": INSUFFICIENT_DATA, "per_run": per_run}

    summary = percentile_summary(ms_values)
    std = statistics.pstdev(ms_values) if len(ms_values) > 1 else 0.0
    return {
        "verdict": "AVAILABLE",
        "n_runs_with_valid_lag": len(ms_values),
        "n_runs_total": len(bundles),
        "lag_ms_median": statistics.median(ms_values),
        "lag_ms_mean": summary["mean"],
        "lag_ms_min": summary["min"],
        "lag_ms_max": summary["max"],
        "lag_ms_p90": summary["p90"] if len(ms_values) >= 3 else None,
        "lag_ms_p95": summary["p95"] if len(ms_values) >= 3 else None,
        "lag_ms_std": std,
        "lag_frames_values": frame_values,
        "per_run": per_run,
    }


# ---------------------------------------------------------------------------
# wrist_roll causal deadband 통합 (섹션 6~7) - compute_deadband_summary 그대로 재사용
# ---------------------------------------------------------------------------


def classify_deadband_region(response_fraction: float | None) -> str | None:
    """섹션 7: 분석 편의용 3분류. **hardware safety threshold로 확정하는 것이 아니다.**

    기준(요구사항 예시 그대로 채택): response_fraction < 10% -> NO_RESPONSE_REGION,
    10~80% -> TRANSITION_REGION, > 80% -> HIGH_RESPONSE_REGION.
    """
    if response_fraction is None:
        return None
    if response_fraction < 0.10:
        return NO_RESPONSE_REGION
    if response_fraction <= 0.80:
        return TRANSITION_REGION
    return HIGH_RESPONSE_REGION


def compute_deadband_aggregate(
    bundles: list[RunBundle],
    *,
    lookahead_ms: float = DEFAULT_DEADBAND_LOOKAHEAD_MS,
    noise_threshold_ticks: int = DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS,
    min_samples_per_bucket: int = 3,
) -> dict[str, Any]:
    """``compute_deadband_summary()``(단일-run causal 판정 로직, 재구현 없음)를 run마다
    그대로 호출한 뒤 버킷별로 합산한다. run별 response_fraction도 함께 보고해서, 특정
    bucket이 한 run에서만 우연히 반응했는지 여러 run에서 반복되는지 구분할 수 있게 한다.
    """
    per_run_summaries: list[dict[str, Any]] = []
    pooled: dict[Any, dict[str, int]] = {}

    for bundle in bundles:
        report = bundle.report or {}
        follower_start_present_raw = report.get("follower_start_present_raw")
        follower_start_present_deg = report.get("follower_start_present_deg")
        result = TeleopRunResult(
            samples=bundle.samples,
            stopped_reason=(report.get("stopped_reason") or "UNKNOWN"),
            follower_start_present_raw=follower_start_present_raw,
            follower_start_present_deg=follower_start_present_deg,
        )
        run_summary = compute_deadband_summary(
            result, lookahead_ms=lookahead_ms, noise_threshold_ticks=noise_threshold_ticks, min_samples_per_bucket=min_samples_per_bucket
        )
        per_run_summaries.append({"timestamp": bundle.timestamp, **run_summary})

        for row in run_summary.get("buckets", []):
            key = row["abs_goal_present_error_ticks"]
            agg = pooled.setdefault(key, {"sample_count": 0, "response_count": 0, "no_response_count": 0, "opposite_motion_count": 0})
            agg["sample_count"] += row["sample_count"]
            agg["response_count"] += row["response_count"]
            agg["no_response_count"] += row["no_response_count"]
            agg["opposite_motion_count"] += row["opposite_motion_count"]

    def _sort_key(k: Any) -> int:
        return 6 if k == "6+" else int(k)

    aggregate_buckets = []
    for key in sorted(pooled, key=_sort_key):
        agg = pooled[key]
        response_fraction = (agg["response_count"] / agg["sample_count"]) if agg["sample_count"] else None
        per_run_fractions = []
        for run in per_run_summaries:
            row = next((r for r in run.get("buckets", []) if r["abs_goal_present_error_ticks"] == key), None)
            per_run_fractions.append(
                {
                    "timestamp": run["timestamp"],
                    "response_fraction": row["response_fraction"] if row else None,
                    "sample_count": row["sample_count"] if row else 0,
                }
            )
        aggregate_buckets.append(
            {
                "abs_goal_present_error_ticks": key,
                "sample_count": agg["sample_count"],
                "response_count": agg["response_count"],
                "no_response_count": agg["no_response_count"],
                "opposite_motion_count": agg["opposite_motion_count"],
                "response_fraction": response_fraction,
                "region_candidate": classify_deadband_region(response_fraction),
                "per_run_response_fraction": per_run_fractions,
                "runs_with_any_response": sum(1 for r in per_run_fractions if (r["response_fraction"] or 0) > 0),
            }
        )

    if not aggregate_buckets:
        return {"verdict": INSUFFICIENT_FOR_DEADBAND_ESTIMATE, "per_run": per_run_summaries, "buckets": []}

    return {
        "verdict": "DEADBAND_AGGREGATE_AVAILABLE",
        "lookahead_ms": lookahead_ms,
        "noise_threshold_ticks": noise_threshold_ticks,
        "buckets": aggregate_buckets,
        "per_run": per_run_summaries,
    }


# ---------------------------------------------------------------------------
# run-to-run 안정성 (섹션 9)
# ---------------------------------------------------------------------------


def _mean_std_cv(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "cv": None, "n": 0}
    mean = sum(values) / len(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    cv = (std / mean) if mean not in (0, None) else None
    return {"mean": mean, "std": std, "cv": cv, "n": len(values)}


def compute_run_to_run_stability(bundles: list[RunBundle], joint_aggregates: dict[str, dict], latency_aggregate: dict) -> dict[str, Any]:
    """주요 scalar metric들의 run 간 mean/std/CV. CV(변동계수)가 클수록 run마다 들쭉날쭉하다."""
    actual_hz_values = [b.quality.get("actual_loop_hz") for b in bundles if b.quality.get("actual_loop_hz") is not None]

    per_joint: dict[str, Any] = {}
    for joint, agg in joint_aggregates.items():
        mae_values = [r["tracking_error"]["mean"] for r in agg["per_run"] if r["tracking_error"] is not None]
        mean_velocity_values = [r["velocity"]["mean"] for r in agg["per_run"] if r["velocity"] is not None]
        per_joint[joint] = {
            "tracking_error_mean_across_runs": _mean_std_cv(mae_values),
            "mean_velocity_across_runs": _mean_std_cv(mean_velocity_values),
        }

    latency_stability = None
    if latency_aggregate.get("verdict") == "AVAILABLE":
        ms_values = [r["lag_ms_timestamp_based"] for r in latency_aggregate["per_run"] if r["available"]]
        latency_stability = _mean_std_cv(ms_values)

    return {
        "actual_loop_hz": _mean_std_cv(actual_hz_values),
        "latency_ms": latency_stability,
        "per_joint": per_joint,
    }


# ---------------------------------------------------------------------------
# VLA candidate 값 (섹션 8) - 전부 CANDIDATE_ONLY
# ---------------------------------------------------------------------------


def compute_candidates(joint_aggregates: dict[str, dict], deadband_aggregate: dict, latency_aggregate: dict) -> dict[str, Any]:
    candidates: dict[str, Any] = {"label": "CANDIDATE_ONLY", "joints": {}, "wrist_roll_deadband": None, "latency": None}

    for joint, agg in joint_aggregates.items():
        a = agg["aggregate"]
        candidates["joints"][joint] = {
            "label": "CANDIDATE_ONLY",
            "historical_safe_range_candidate_deg_or_pct": (
                [a["range"]["p01"], a["range"]["p99"]] if a["range"]["p01"] is not None else None
            ),
            "typical_frame_delta_candidate": a["frame_delta"]["p50"] if a["frame_delta"] else None,
            "p95_frame_delta_candidate": a["frame_delta"]["p95"] if a["frame_delta"] else None,
            "p99_frame_delta_candidate": a["frame_delta"]["p99"] if a["frame_delta"] else None,
            "typical_velocity_candidate": a["velocity"]["p50"] if a["velocity"] else None,
            "p95_velocity_candidate": a["velocity"]["p95"] if a["velocity"] else None,
            "tracking_error_p95_candidate": a["tracking_error"]["p95"] if a["tracking_error"] else None,
            "tracking_error_p99_candidate": a["tracking_error"]["p99"] if a["tracking_error"] else None,
        }

    if deadband_aggregate.get("verdict") == "DEADBAND_AGGREGATE_AVAILABLE":
        transition_or_higher = [
            b["abs_goal_present_error_ticks"]
            for b in deadband_aggregate["buckets"]
            if b["region_candidate"] in (TRANSITION_REGION, HIGH_RESPONSE_REGION)
        ]
        candidates["wrist_roll_deadband"] = {
            "label": "CANDIDATE_ONLY",
            "no_response_region_ticks": [
                b["abs_goal_present_error_ticks"] for b in deadband_aggregate["buckets"] if b["region_candidate"] == NO_RESPONSE_REGION
            ],
            "transition_or_higher_region_ticks": transition_or_higher,
            "note": "이 구간 분류는 분석 편의용이며 hardware safety threshold로 확정된 것이 아닙니다.",
        }

    if latency_aggregate.get("verdict") == "AVAILABLE":
        candidates["latency"] = {
            "label": "CANDIDATE_ONLY",
            "range_ms": [latency_aggregate["lag_ms_min"], latency_aggregate["lag_ms_max"]],
            "median_ms": latency_aggregate["lag_ms_median"],
        }
    else:
        candidates["latency"] = INSUFFICIENT_DATA

    return candidates


# ---------------------------------------------------------------------------
# 최상위 조립
# ---------------------------------------------------------------------------


def build_aggregate_report(
    bundles: list[RunBundle],
    *,
    lookahead_ms: float = DEFAULT_DEADBAND_LOOKAHEAD_MS,
    noise_threshold_ticks: int = DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS,
) -> dict[str, Any]:
    joint_aggregates = {joint: compute_joint_aggregate(bundles, joint) for joint in JOINT_ORDER}
    latency_aggregate = compute_latency_aggregate(bundles)
    deadband_aggregate = compute_deadband_aggregate(bundles, lookahead_ms=lookahead_ms, noise_threshold_ticks=noise_threshold_ticks)
    stability = compute_run_to_run_stability(bundles, joint_aggregates, latency_aggregate)
    candidates = compute_candidates(joint_aggregates, deadband_aggregate, latency_aggregate)

    return {
        "run_count": len(bundles),
        "runs": [
            {
                "timestamp": b.timestamp,
                "csv_path": str(b.csv_path),
                "json_path": str(b.json_path) if b.json_path else None,
                "quality": b.quality,
                "malformed_row_count": b.malformed_row_count,
            }
            for b in bundles
        ],
        "joint_aggregates": joint_aggregates,
        "latency_aggregate": latency_aggregate,
        "deadband_aggregate": deadband_aggregate,
        "run_to_run_stability": stability,
        "candidates": candidates,
        "direct_register_write_count": 0,
        "hardware_execution_count": 0,
    }


# ---------------------------------------------------------------------------
# markdown 렌더링 (순수 문자열 생성)
# ---------------------------------------------------------------------------


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Instrumented Teleop 통합 분석 (offline, 실물 write 0회)")
    lines.append("")
    lines.append(f"- 분석에 사용한 run 수: {report['run_count']}")
    lines.append("")
    lines.append("## Run-level 품질 요약")
    lines.append("")
    lines.append("| timestamp | quality | sample_count | actual_hz | stopped_reason | reasons |")
    lines.append("|---|---|---|---|---|---|")
    for r in report["runs"]:
        q = r["quality"]
        reasons = "; ".join(q.get("reasons", [])) or "-"
        lines.append(
            f"| {r['timestamp']} | {q['verdict']} | {q.get('sample_count')} | "
            f"{_fmt(q.get('actual_loop_hz'), '.2f')} | {q.get('stopped_reason')} | {reasons} |"
        )
    lines.append("")

    lines.append("## Joint별 range / frame delta / velocity / tracking error (pooled aggregate)")
    lines.append("")
    lines.append("| joint | range min | range max | frame_delta p95 | velocity p95 | tracking_error p95 | tracking_error p99 |")
    lines.append("|---|---|---|---|---|---|---|")
    for joint, agg in report["joint_aggregates"].items():
        a = agg["aggregate"]
        lines.append(
            f"| {joint} | {_fmt(a['range']['min'])} | {_fmt(a['range']['max'])} | "
            f"{_fmt(a['frame_delta']['p95'] if a['frame_delta'] else None)} | "
            f"{_fmt(a['velocity']['p95'] if a['velocity'] else None)} | "
            f"{_fmt(a['tracking_error']['p95'] if a['tracking_error'] else None)} | "
            f"{_fmt(a['tracking_error']['p99'] if a['tracking_error'] else None)} |"
        )
    lines.append("")

    lines.append("## Latency (command -> actual)")
    lines.append("")
    if report["latency_aggregate"].get("verdict") == "AVAILABLE":
        la = report["latency_aggregate"]
        lines.append(
            f"- median={_fmt(la['lag_ms_median'], '.2f')}ms mean={_fmt(la['lag_ms_mean'], '.2f')}ms "
            f"min={_fmt(la['lag_ms_min'], '.2f')}ms max={_fmt(la['lag_ms_max'], '.2f')}ms std={_fmt(la['lag_ms_std'], '.2f')}ms "
            f"(신뢰 가능 run {la['n_runs_with_valid_lag']}/{la['n_runs_total']})"
        )
    else:
        lines.append(f"- {INSUFFICIENT_DATA}")
    lines.append("")

    lines.append("## wrist_roll causal deadband (aggregate, run별 response_fraction)")
    lines.append("")
    db = report["deadband_aggregate"]
    if db.get("verdict") == "DEADBAND_AGGREGATE_AVAILABLE":
        for bucket in db["buckets"]:
            lines.append(
                f"### {bucket['abs_goal_present_error_ticks']} tick(s) - region_candidate={bucket['region_candidate']}"
            )
            for run_frac in bucket["per_run_response_fraction"]:
                pct = f"{run_frac['response_fraction'] * 100:.1f}%" if run_frac["response_fraction"] is not None else "?"
                lines.append(f"- {run_frac['timestamp']}: {pct} (n={run_frac['sample_count']})")
            agg_pct = f"{bucket['response_fraction'] * 100:.1f}%" if bucket["response_fraction"] is not None else "?"
            lines.append(f"- **aggregate: {agg_pct}** (n={bucket['sample_count']}, {bucket['runs_with_any_response']}개 run에서 반응 관측)")
            lines.append("")
    else:
        lines.append(f"- {db.get('verdict')}")
    lines.append("")

    lines.append("## VLA candidate 값 (전부 CANDIDATE_ONLY - 확정 아님)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["candidates"], indent=2, ensure_ascii=False, default=str))
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append(f"direct_register_write_count={report['direct_register_write_count']}, hardware_execution_count={report['hardware_execution_count']}")

    return "\n".join(lines)
