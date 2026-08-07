"""6-run instrumented teleop aggregate -> "VLA/follower 제어용 candidate control profile v1".

**완전 offline, 순수 계산 모듈이다.** ``lerobot``, ``hardware/safety/*``의 serial 접근
클래스, 실제 로봇 객체를 이 파일 어디에서도 import/생성/호출하지 않는다. 입력은 이미
디스크에 저장된 ``hardware/diagnostics/instrumented_teleop_aggregate.py``의 산출물
(``aggregate_6runs_*.json``)뿐이고, 그 dict를 읽어서 새 dict를 만들어 반환할 뿐이다.

이 모듈이 만드는 것은 어디까지나 **CANDIDATE_ONLY** 후보값이다. 여기서 계산한 어떤 값도
실제 서보 write, safety threshold, rate limiter, follower-safe mapper 설정에 자동으로
연결되지 않는다 - 이 모듈은 dict를 반환할 뿐 파일 쓰기조차 하지 않는다(파일 쓰기는
``scripts/generate_control_profile_candidate.py``가 담당).

## 왜 이 모듈이 필요한가

``instrumented_teleop_aggregate.py``의 ``compute_candidates()``가 이미 최소한의
candidate 블록(joint별 range/frame_delta/velocity/tracking_error percentile, wrist_roll
deadband 2분류, latency range)을 만들어 aggregate JSON에 포함시킨다. 이 모듈은 그 위에서:

  1. joint별 confidence(HIGH/MEDIUM/LOW/INSUFFICIENT_DATA)를 aggregate의
     ``run_to_run_stability`` 블록(coefficient of variation)에서 **유도**한다 - 새로
     추측하지 않는다.
  2. wrist_roll deadband를 "0~5 tick 무반응, 6+ tick TRANSITION, HIGH_RESPONSE는
     아직 미확립"으로 명시하고 tick당 각도 환산표를 붙인다.
  3. latency에 "이것은 local instrumented teleop 계측값이지 end-to-end VLA latency가
     아니다"라는 scope 라벨을 강제로 붙인다.
  4. gripper 단위를 percent_0_100으로 명시하고 degree 후보값과 절대 섞이지 않게 한다.
  5. provenance(``status``/``source``/``run_count``/``apply_automatically``)를 top-level에
     강제한다.

## Confidence 산출 근거

``run_to_run_stability.per_joint.<joint>.{tracking_error_mean_across_runs,
mean_velocity_across_runs}.cv`` (6-run에 걸친 coefficient of variation, 이미 aggregate가
계산해 둔 값)를 그대로 재사용한다. cv가 낮을수록 6개 run에서 그 joint의 움직임 특성이
반복적으로 비슷했다는 뜻이므로, 그 percentile 기반 후보값을 더 신뢰할 수 있다고 본다.
이 임계값(0.3 / 0.6)은 이번 작업에서 새로 정한 분류 편의값이며, 실측으로 검증된 통계적
경계가 아니다 - 그래서 이 상수들에도 "튜닝값 아님"이라는 주석을 남긴다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from hardware.diagnostics.instrumented_teleop import JOINT_ORDER

__all__ = [
    "SCRIPT_VERSION",
    "STATUS_CANDIDATE_ONLY",
    "SOURCE_LABEL",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MEDIUM",
    "CONFIDENCE_LOW",
    "CONFIDENCE_INSUFFICIENT_DATA",
    "CONFIDENCE_LEGEND",
    "DEGREE_JOINTS",
    "GRIPPER_JOINT",
    "DEG_PER_WRIST_ROLL_TICK",
    "LATENCY_SCOPE_LOCAL_INSTRUMENTED_TELEOP",
    "ControlProfileCandidateError",
    "confidence_from_cv",
    "joint_confidence",
    "tick_to_degree_table",
    "build_joint_candidate",
    "build_wrist_roll_deadband_candidate",
    "build_timing_candidate",
    "build_control_profile_candidate",
    "compare_with_existing_rate_limits",
    "render_comparison_markdown",
]

SCRIPT_VERSION = "1.0.0"

STATUS_CANDIDATE_ONLY = "CANDIDATE_ONLY"
SOURCE_LABEL = "instrumented_teleop_6runs"

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"
CONFIDENCE_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

CONFIDENCE_LEGEND: dict[str, str] = {
    CONFIDENCE_HIGH: "6개 run에서 반복적으로 안정적으로 관측됨 (run-to-run cv < 0.3, 또는 6-run 전체에서 반복 확인된 정성적 패턴).",
    CONFIDENCE_MEDIUM: "데이터는 있으나 run 간 변동(cv)이 상당하거나, 일부 run에서만 유효한 표본이 관측됨.",
    CONFIDENCE_LOW: "일부 run에서만 관찰되었거나 run 간 변동(cv)이 매우 큼.",
    CONFIDENCE_INSUFFICIENT_DATA: "candidate를 계산할 근거 표본이 부족하거나 없음.",
}

# 이 저장소에서 arm 5개 joint는 degree, gripper는 percent_0_100 semantics를 쓴다
# (근거: hardware/state_server/readonly_so101_reader.py, configs/follower_safe_mapper.yaml
# 상단 주석, configs/generated/teleop_safe_ranges_candidate.json의 unit_by_joint_group).
GRIPPER_JOINT = "gripper"
DEGREE_JOINTS: tuple[str, ...] = tuple(j for j in JOINT_ORDER if j != GRIPPER_JOINT)

# wrist_roll raw tick -> degree 환산. STS3215 resolution=4096 -> max_res=4095
# (LeRobot MotorNormMode.DEGREES 공식: degree = (raw - mid) * 360 / max_res, 이 저장소
# 안에서는 hardware/diagnostics/instrumented_teleop.py의 wrist_roll 각도 계산과
# configs/follower_safe_mapper.yaml의 motor_resolution=4096 주석에서 동일하게 확인된다).
# tick *차이*(delta)를 각도로 바꿀 때는 mid가 상쇄되므로 이 상수 하나로 충분하다.
DEG_PER_WRIST_ROLL_TICK = 360.0 / 4095.0

LATENCY_SCOPE_LOCAL_INSTRUMENTED_TELEOP = "local_instrumented_teleop_command_to_actual"

# confidence_from_cv()의 분류 경계값 - 편의상 정한 값이며 실측 검증된 통계 경계가 아니다.
_CV_HIGH_MAX = 0.3
_CV_MEDIUM_MAX = 0.6


class ControlProfileCandidateError(RuntimeError):
    """입력 aggregate dict가 필요한 schema를 갖추지 못했을 때."""


REQUIRED_AGGREGATE_KEYS = ("joint_aggregates", "latency_aggregate", "deadband_aggregate", "run_count")


def _validate_aggregate_schema(aggregate: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_AGGREGATE_KEYS if k not in aggregate]
    if missing:
        raise ControlProfileCandidateError(
            f"aggregate JSON에 필요한 필드가 없습니다: {missing} "
            "(hardware/diagnostics/instrumented_teleop_aggregate.py의 build_aggregate_report() 산출물이 맞는지 확인하세요)."
        )


def confidence_from_cv(cv: float | None) -> str:
    """coefficient of variation -> confidence 등급. 근거는 모듈 docstring 참고."""
    if cv is None:
        return CONFIDENCE_INSUFFICIENT_DATA
    if cv < _CV_HIGH_MAX:
        return CONFIDENCE_HIGH
    if cv < _CV_MEDIUM_MAX:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def joint_confidence(run_to_run_stability: dict[str, Any], joint: str) -> dict[str, Any]:
    """``run_to_run_stability.per_joint.<joint>``의 tracking_error/velocity cv를 평균내어
    joint 하나의 candidate 전체(range/frame_delta/velocity/tracking_error)에 공통으로
    쓸 confidence를 만든다. 세부 항목마다 별도 cv가 없으므로, "이 joint의 움직임 특성이
    6-run에서 얼마나 반복적이었는가"라는 하나의 신호를 공유해서 쓰는 것이 과장을 피하는
    보수적인 선택이다.
    """
    per_joint = (run_to_run_stability or {}).get("per_joint", {})
    entry = per_joint.get(joint)
    if not entry:
        return {"confidence": CONFIDENCE_INSUFFICIENT_DATA, "tracking_error_cv": None, "velocity_cv": None}

    tracking_cv = (entry.get("tracking_error_mean_across_runs") or {}).get("cv")
    velocity_cv = (entry.get("mean_velocity_across_runs") or {}).get("cv")
    cvs = [c for c in (tracking_cv, velocity_cv) if c is not None]
    combined_cv = sum(cvs) / len(cvs) if cvs else None
    return {
        "confidence": confidence_from_cv(combined_cv),
        "tracking_error_cv": tracking_cv,
        "velocity_cv": velocity_cv,
    }


def tick_to_degree_table(max_tick: int = 6) -> dict[str, float]:
    """0..max_tick tick 각각의 degree 환산표 (문자열 키 - JSON 직렬화 안전)."""
    return {str(t): round(t * DEG_PER_WRIST_ROLL_TICK, 6) for t in range(max_tick + 1)}


def build_joint_candidate(
    joint: str,
    joint_agg_entry: dict[str, Any],
    run_to_run_stability: dict[str, Any],
    *,
    unit: str,
) -> dict[str, Any]:
    """단일 joint의 candidate 블록. aggregate에 실제 존재하는 percentile만 옮겨 쓴다 -
    존재하지 않는 통계는 만들어내지 않고 ``None``으로 남긴다.
    """
    agg = (joint_agg_entry or {}).get("aggregate") or {}
    rng = agg.get("range") or {}
    fd = agg.get("frame_delta") or {}
    vel = agg.get("velocity") or {}
    te = agg.get("tracking_error") or {}

    conf = joint_confidence(run_to_run_stability, joint)

    return {
        "label": STATUS_CANDIDATE_ONLY,
        "unit": unit,
        "confidence": conf["confidence"],
        "confidence_basis": {
            "tracking_error_cv_across_runs": conf["tracking_error_cv"],
            "velocity_cv_across_runs": conf["velocity_cv"],
        },
        "historical_operating_range": {
            "note": (
                "이 값은 6-run teleop 동안 실제로 관측된 위치 범위 후보입니다. "
                "calibration mechanical limit(hard_limit)이 아닙니다."
            ),
            "observed_min": rng.get("min"),
            "observed_max": rng.get("max"),
            "p01": rng.get("p01"),
            "p99": rng.get("p99"),
            "candidate_historical_inner_range": (
                [rng["p01"], rng["p99"]] if rng.get("p01") is not None and rng.get("p99") is not None else None
            ),
            "run_min_max_spread": rng.get("run_min_max_spread"),
        },
        "frame_delta": {
            "p50": fd.get("p50"),
            "p95": fd.get("p95"),
            "p99": fd.get("p99"),
            "max": fd.get("max"),
            "candidate_frame_delta_soft_limit": fd.get("p99"),
            "candidate_basis": "p99 값을 그대로 사용 (multiplier=1.0x, 추가 배율 적용 없음).",
        },
        "velocity": {
            "observed_velocity_profile": {
                "p50": vel.get("p50"),
                "p95": vel.get("p95"),
                "p99": vel.get("p99"),
                "max": vel.get("max"),
            },
            "candidate_soft_velocity_limit": vel.get("p99"),
            "candidate_basis": "p99 값을 그대로 사용 (multiplier=1.0x, 추가 배율 적용 없음).",
        },
        "tracking_error": {
            "mae": te.get("mean"),
            "p95": te.get("p95"),
            "p99": te.get("p99"),
            "max": te.get("max"),
            "tracking_warning_candidate": te.get("p95"),
            "tracking_severe_candidate": te.get("p99"),
            "candidate_basis": "warning ~= p95, severe ~= p99 (execution monitor 설계용 제안일 뿐, 확정 safety threshold 아님).",
        },
    }


def build_wrist_roll_deadband_candidate(deadband_aggregate: dict[str, Any]) -> dict[str, Any]:
    """섹션 5 요구사항: 0~5 tick NO_RESPONSE, 6+ TRANSITION, HIGH_RESPONSE는 NOT_ESTABLISHED.

    기존 ``deadband_aggregate.buckets[*].region_candidate``(이미
    ``instrumented_teleop_aggregate.classify_deadband_region()``이 계산해 둔 값)를 그대로
    읽어서 재분류 없이 재사용한다.
    """
    if (deadband_aggregate or {}).get("verdict") != "DEADBAND_AGGREGATE_AVAILABLE":
        return {
            "label": STATUS_CANDIDATE_ONLY,
            "joint": "wrist_roll",
            "verdict": "INSUFFICIENT_DATA",
            "no_response_region_ticks": None,
            "transition_region_start_ticks": None,
            "high_response_region": "NOT_ESTABLISHED",
        }

    buckets = deadband_aggregate.get("buckets") or []
    no_response_ticks = sorted(
        int(b["abs_goal_present_error_ticks"])
        for b in buckets
        if b.get("region_candidate") == "NO_RESPONSE_REGION" and b["abs_goal_present_error_ticks"] != "6+"
    )
    transition_buckets = [b for b in buckets if b.get("region_candidate") == "TRANSITION_REGION"]
    high_response_buckets = [b for b in buckets if b.get("region_candidate") == "HIGH_RESPONSE_REGION"]

    six_plus = next((b for b in buckets if b.get("abs_goal_present_error_ticks") == "6+"), None)

    no_response_evidence = {
        str(b["abs_goal_present_error_ticks"]): {
            "aggregate_response_fraction": b.get("response_fraction"),
            "sample_count": b.get("sample_count"),
            "runs_with_any_response": b.get("runs_with_any_response"),
        }
        for b in buckets
        if b.get("region_candidate") == "NO_RESPONSE_REGION" and b["abs_goal_present_error_ticks"] != "6+"
    }

    # confidence: 0~5 tick 구간은 aggregate response_fraction이 전부 1% 미만이고 6-run에
    # 걸쳐 반복 확인되므로 HIGH. 6+ 구간은 response_fraction(70.6%)이 TRANSITION 판정
    # 경계(<=80%) 안에 있고, 6개 run 중 2개는 이 구간에서 표본 자체가 없어(모션 없음)
    # "response_fraction=0"으로 잡혀 있어 실제로는 4/6 run에서만 판단 가능하다 - 그래서
    # MEDIUM으로 낮춘다 (과장 금지).
    transition_confidence = CONFIDENCE_INSUFFICIENT_DATA
    transition_response_fraction = None
    transition_runs_with_response = None
    if six_plus is not None:
        transition_response_fraction = six_plus.get("response_fraction")
        transition_runs_with_response = six_plus.get("runs_with_any_response")
        transition_confidence = CONFIDENCE_MEDIUM

    return {
        "label": STATUS_CANDIDATE_ONLY,
        "joint": "wrist_roll",
        "verdict": "DEADBAND_ESTIMATE_AVAILABLE",
        "unit": "abs_goal_present_error_ticks",
        "no_response_region_ticks": (
            [no_response_ticks[0], no_response_ticks[-1]] if no_response_ticks else None
        ),
        "no_response_region_confidence": CONFIDENCE_HIGH if no_response_ticks else CONFIDENCE_INSUFFICIENT_DATA,
        "no_response_region_evidence": no_response_evidence,
        "transition_region_start_ticks": 6 if (transition_buckets or six_plus is not None) else None,
        "transition_region_aggregate_response_fraction": transition_response_fraction,
        "transition_region_runs_with_response": transition_runs_with_response,
        "transition_region_confidence": transition_confidence,
        "high_response_region": "NOT_ESTABLISHED",
        "high_response_region_rationale": (
            "6+ tick 버킷의 aggregate response_fraction은 약 70.6%로, 이 저장소가 쓰는 HIGH_RESPONSE_REGION "
            "판정 기준(>80%)에 못 미친다. 게다가 6개 run 중 2개(20260807_093136, 20260807_093204)는 이 구간에서 "
            "실제 움직임 표본 자체가 없어 판단에서 사실상 제외된다. '6 ticks = guaranteed motion'이라고 단정할 수 없다."
            if high_response_buckets == []
            else "일부 run에서 HIGH_RESPONSE_REGION 버킷이 관측되었으나 6-run aggregate로는 아직 일관되게 확립되지 않았다."
        ),
        "tick_to_degree_equivalent": tick_to_degree_table(6),
        "note": (
            "이 구간 분류는 분석 편의용이며 hardware safety threshold로 확정된 것이 아닙니다. "
            "6 ticks가 '항상 반응하는 지점'을 보장하지 않습니다."
        ),
    }


def build_timing_candidate(latency_aggregate: dict[str, Any], run_to_run_stability: dict[str, Any]) -> dict[str, Any]:
    hz_stats = (run_to_run_stability or {}).get("actual_loop_hz") or {}
    lat = latency_aggregate or {}

    hz_confidence = confidence_from_cv(hz_stats.get("cv"))

    n_valid = lat.get("n_runs_with_valid_lag")
    n_total = lat.get("n_runs_total")
    latency_confidence = confidence_from_cv(lat.get("lag_ms_std") and (lat["lag_ms_std"] / lat["lag_ms_mean"]) if lat.get("lag_ms_mean") else None)
    # run 검증 가능 비율이 낮으면(6개 중 4개) cv가 낮아도 한 단계 낮춰 과신을 막는다.
    if n_valid is not None and n_total and n_valid < n_total and latency_confidence == CONFIDENCE_HIGH:
        latency_confidence = CONFIDENCE_MEDIUM

    return {
        "label": STATUS_CANDIDATE_ONLY,
        "nominal_control_hz": hz_stats.get("mean"),
        "control_hz_confidence": hz_confidence,
        "control_hz_std": hz_stats.get("std"),
        "observed_latency_ms": {
            "median": lat.get("lag_ms_median"),
            "mean": lat.get("lag_ms_mean"),
            "min": lat.get("lag_ms_min"),
            "max": lat.get("lag_ms_max"),
            "std": lat.get("lag_ms_std"),
            "valid_runs": f"{n_valid}/{n_total}" if n_valid is not None and n_total is not None else None,
        },
        "latency_confidence": latency_confidence,
        "latency_scope": LATENCY_SCOPE_LOCAL_INSTRUMENTED_TELEOP,
        "latency_scope_note": (
            "이 latency는 leader->follower local instrumented teleop 계측에서 얻은 "
            "'command to actual' 값입니다. 아직 Desktop VLA -> network -> Laptop -> robot의 "
            "end-to-end latency가 아닙니다."
        ),
    }


def build_control_profile_candidate(
    aggregate: dict[str, Any],
    *,
    source_aggregate_path: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """메인 진입점. aggregate JSON(dict)을 받아 candidate control profile v1(dict)을 만든다.

    이 함수는 파일을 읽거나 쓰지 않는다 - 순수 변환이다.
    """
    _validate_aggregate_schema(aggregate)

    run_to_run_stability = aggregate.get("run_to_run_stability") or {}
    joint_aggregates = aggregate.get("joint_aggregates") or {}

    joints_out: dict[str, Any] = {}
    for joint in JOINT_ORDER:
        entry = joint_aggregates.get(joint)
        unit = "percent_0_100" if joint == GRIPPER_JOINT else "degree"
        if entry is None:
            joints_out[joint] = {
                "label": STATUS_CANDIDATE_ONLY,
                "unit": unit,
                "confidence": CONFIDENCE_INSUFFICIENT_DATA,
                "note": "이 6-run aggregate에 이 joint 데이터가 없습니다.",
            }
            continue
        joints_out[joint] = build_joint_candidate(joint, entry, run_to_run_stability, unit=unit)

    return {
        "status": STATUS_CANDIDATE_ONLY,
        "source": SOURCE_LABEL,
        "run_count": aggregate.get("run_count"),
        "apply_automatically": False,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "source_aggregate_path": source_aggregate_path,
        "source_aggregate_generated_at": aggregate.get("generated_at"),
        "generator": {
            "module": "hardware.diagnostics.control_profile_candidate",
            "version": SCRIPT_VERSION,
        },
        "confidence_legend": dict(CONFIDENCE_LEGEND),
        "gripper_unit_note": (
            "gripper는 arm 5개 joint(degree)와 달리 percent_0_100 semantics를 씁니다 "
            "(hardware/state_server/readonly_so101_reader.py, configs/follower_safe_mapper.yaml 확인). "
            "gripper의 range/frame_delta/velocity candidate 값을 degree로 해석하지 마세요."
        ),
        "usage_restrictions": [
            "이 candidate 값은 실제 로봇 제어 코드에 자동 적용되지 않습니다 (apply_automatically=false).",
            "이 파일 생성 과정에서 leader/follower connect, teleop 실행, servo write는 발생하지 않았습니다.",
            "safety threshold로 확정된 값이 아니며, production config가 아닙니다.",
        ],
        "joints": joints_out,
        "wrist_roll_deadband_analysis": build_wrist_roll_deadband_candidate(aggregate.get("deadband_aggregate") or {}),
        "timing": build_timing_candidate(aggregate.get("latency_aggregate") or {}, run_to_run_stability),
    }


# ---------------------------------------------------------------------------
# 섹션 9/10: 기존 follower-safe mapper rate_limit_deg_per_sec와의 read-only 비교
# ---------------------------------------------------------------------------

VERDICT_MORE_CONSERVATIVE = "CURRENT_LIMIT_MORE_CONSERVATIVE_THAN_TELEOP"
VERDICT_LOOSER = "CURRENT_LIMIT_LOOSER_THAN_OBSERVED_P95"
VERDICT_EQUAL = "CURRENT_LIMIT_EQUALS_OBSERVED_P95"
VERDICT_NO_EXISTING_LIMIT = "NO_EXISTING_LIMIT"
VERDICT_NO_OBSERVED_DATA = "NO_OBSERVED_DATA"


def compare_with_existing_rate_limits(
    candidate_profile: dict[str, Any],
    existing_rate_limits: dict[str, float] | None,
) -> dict[str, Any]:
    """``configs/follower_safe_mapper.yaml``의 ``rate_limit_deg_per_sec``(읽기 전용으로만
    전달받는다 - 이 함수는 그 파일을 직접 열지 않는다)와 이번 6-run observed velocity p95를
    비교한다. **이 비교는 어떤 파일도 수정하지 않는다.**
    """
    existing_rate_limits = existing_rate_limits or {}
    rows: dict[str, Any] = {}
    for joint in JOINT_ORDER:
        joint_candidate = (candidate_profile.get("joints") or {}).get(joint) or {}
        observed_p95 = ((joint_candidate.get("velocity") or {}).get("observed_velocity_profile") or {}).get("p95")
        candidate_soft_limit = (joint_candidate.get("velocity") or {}).get("candidate_soft_velocity_limit")
        existing_limit = existing_rate_limits.get(joint)

        if existing_limit is None:
            verdict = VERDICT_NO_EXISTING_LIMIT
        elif observed_p95 is None:
            verdict = VERDICT_NO_OBSERVED_DATA
        elif observed_p95 > existing_limit:
            verdict = VERDICT_MORE_CONSERVATIVE
        elif observed_p95 < existing_limit:
            verdict = VERDICT_LOOSER
        else:
            verdict = VERDICT_EQUAL

        rows[joint] = {
            "existing_rate_limit": existing_limit,
            "existing_rate_limit_unit": (
                "percent/s (config key 이름은 deg_per_sec이지만 gripper는 percent_0_100 semantics - "
                "configs/follower_safe_mapper.yaml 상단 주석 참고)"
                if joint == GRIPPER_JOINT
                else "deg/s"
            ),
            "observed_velocity_p95": observed_p95,
            "candidate_soft_velocity_limit_p99": candidate_soft_limit,
            "verdict": verdict,
        }

    return {"label": STATUS_CANDIDATE_ONLY, "joints": rows}


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "| joint | 기존 rate_limit_deg_per_sec | 6-run observed velocity p95 | candidate soft limit (p99) | verdict |",
        "|---|---|---|---|---|",
    ]
    for joint, row in comparison.get("joints", {}).items():
        existing = row["existing_rate_limit"]
        existing_str = f"{existing:.2f}" if isinstance(existing, (int, float)) else "N/A"
        observed = row["observed_velocity_p95"]
        observed_str = f"{observed:.2f}" if isinstance(observed, (int, float)) else "N/A"
        cand = row["candidate_soft_velocity_limit_p99"]
        cand_str = f"{cand:.2f}" if isinstance(cand, (int, float)) else "N/A"
        lines.append(f"| {joint} | {existing_str} | {observed_str} | {cand_str} | {row['verdict']} |")
    return "\n".join(lines)
