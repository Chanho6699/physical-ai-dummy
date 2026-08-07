"""Realistic SO-101 Control Layer (v1) - MuJoCo actuator에 명령을 전달하기 직전 계층.

::

    raw command (desired action, degree/percent - profile과 동일 단위)
        -> RealisticControlLayer.process()
        -> processed command (같은 단위)
        -> (호출자) 기존 action_mapping.map_positions_dict()로 rad 변환
        -> 기존 MuJoCo executor (mj_step)

이 모듈은 ``mujoco``를 import하지 않는다 - 순수 파이썬 로직만 담아서 하드웨어/시뮬레이터
없이도 단위테스트할 수 있게 한다 (섹션 5 요구사항: "pure logic과 MuJoCo API 호출을 분리").
실물 팔로워 write 코드도 전혀 없다.

v1에서 반영하는 4가지 특성 (섹션 6, 그 이상은 다루지 않음):
    A. rate/frame-delta characteristic  - profile의 candidate_frame_delta_soft_limit을
       "realism_limit"으로만 사용 (하드 safety limit이 아님, 새 multiplier 없음).
    B. wrist_roll deadband              - no-response candidate 폭 이내 변화는 직전 명령 유지.
       REALISM_APPROXIMATION으로만 표시하고, 6+ tick을 "guaranteed response"라 부르지 않음.
    C. command latency                  - profile median latency만큼 지연 후 반영 (on/off 가능).
    D. historical operating range       - 벗어나도 clip하지 않고 진단만 남김.

tracking error는 이 레이어가 만들지 않는다 (섹션 7) - :func:`compute_tracking_error`는
processed_command 대 caller가 물리 스텝 이후 관측한 simulated_actual_position을 비교하는
순수 진단 함수일 뿐이다.
"""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

from simulation.realism.so101_control_profile import ControlProfile

WRIST_ROLL_JOINT = "wrist_roll"

DEADBAND_REASON = "REALISM_APPROXIMATION"
OUTSIDE_RANGE_REASON = "OUTSIDE_HISTORICAL_OPERATING_RANGE"


class RealisticControlError(RuntimeError):
    """레이어 구성/사용 오류 (예: profile에 없는 관절로 process() 호출)."""


@dataclass(frozen=True)
class RealisticControlConfig:
    """v1 특성 각각을 개별적으로 켜고 끌 수 있는 스위치.

    모두 기본값 True(전부 적용)이지만, ``RealisticControlLayer``를 생성할지 여부 자체가
    ``control_mode``(baseline/realistic) 선택에 달려 있으므로 - 즉 baseline 모드는 이 레이어를
    아예 만들지 않는다. 이 config는 "realistic 모드 안에서" 개별 특성을 더 좁게 켜고 끄기
    위한 것이다 (예: latency만 끄고 나머지는 비교하고 싶을 때, 테스트에서 결정론적으로
    비교하고 싶을 때).
    """

    enable_latency: bool = True
    enable_deadband: bool = True
    enable_rate_limit: bool = True
    enable_historical_range_diagnostic: bool = True

    # None이면 profile.timing.latency_median_ms를 그대로 쓴다. 테스트/비교 목적으로만
    # override한다 - production 경로에서는 항상 None(profile 값 그대로)이어야 한다.
    latency_ms_override: float | None = None

    def __post_init__(self) -> None:
        if self.latency_ms_override is not None and self.latency_ms_override < 0:
            raise RealisticControlError(f"latency_ms_override는 0 이상이어야 합니다: {self.latency_ms_override}")


def config_all_disabled() -> RealisticControlConfig:
    """4가지 특성을 모두 끈 config - baseline과 완전히 동일한 passthrough가 되어야 한다
    (섹션 10 "Baseline" 테스트가 레이어 자체의 identity를 확인하는 데 사용)."""
    return RealisticControlConfig(
        enable_latency=False,
        enable_deadband=False,
        enable_rate_limit=False,
        enable_historical_range_diagnostic=False,
    )


@dataclass(frozen=True)
class JointDiagnostic:
    """섹션 11에서 요구하는 필드 그대로. ``tracking_error``는 이 레이어가 채우지 않고
    ``compute_tracking_error()``로 별도 계산해 caller가 합쳐 기록한다."""

    joint: str
    raw_action: float
    processed_action: float
    simulated_actual_state: float | None
    command_delta: float | None  # processed_action - 직전 processed_action
    deadband_applied: bool
    latency_applied_ms: float
    rate_limited: bool
    outside_historical_range: bool
    tracking_error: float | None = None


@dataclass(frozen=True)
class ProcessedCommandResult:
    processed_action: dict[str, float]
    diagnostics: dict[str, JointDiagnostic]


def compute_tracking_error(processed_action: dict[str, float], simulated_actual: dict[str, float]) -> dict[str, float]:
    """processed_command 대 실제 시뮬레이션 위치의 순수 diff (섹션 7 - noise 주입 없음).

    양쪽에 다 있는 관절만 계산한다. 이 함수는 상태를 갖지 않으며 MuJoCo를 호출하지 않는다.
    """
    return {
        joint: processed_action[joint] - simulated_actual[joint]
        for joint in processed_action
        if joint in simulated_actual
    }


class RealisticControlLayer:
    """관절별 latency queue / 직전 processed 값을 들고 있는 상태 기반 pure-logic 레이어."""

    def __init__(self, profile: ControlProfile, config: RealisticControlConfig | None = None) -> None:
        self.profile = profile
        self.config = config or RealisticControlConfig()
        self._pending: dict[str, Deque[tuple[float, float]]] = {}
        self._delayed_current: dict[str, float] = {}
        self._last_output: dict[str, float] = {}

    # -- 설정/리셋 ---------------------------------------------------------

    def reset(self, initial_state: dict[str, float] | None = None) -> None:
        """상태를 비운다. ``initial_state``를 주면 그 값으로 last_output/delayed_current를
        미리 채워 첫 프레임부터 rate-limit/deadband 기준점을 갖게 한다 (선택 사항)."""
        self._pending = {}
        self._delayed_current = {}
        self._last_output = {}
        if initial_state:
            for joint, value in initial_state.items():
                self._delayed_current[joint] = float(value)
                self._last_output[joint] = float(value)

    def _latency_s(self) -> float:
        if self.config.latency_ms_override is not None:
            return self.config.latency_ms_override / 1000.0
        median_ms = self.profile.timing.latency_median_ms
        return (median_ms or 0.0) / 1000.0

    # -- 단계별 처리 --------------------------------------------------------

    def _apply_latency(self, joint: str, desired_value: float, now: float) -> tuple[float, float]:
        """command latency queue. 반환: (지연 후 현재 적용해야 할 값, 이번 프레임 적용된 지연 ms)."""
        first_sample = joint not in self._delayed_current
        if first_sample:
            # 이 joint에 대해 한 번도 값을 받은 적이 없다 - "연결이 막 시작된" 시점으로 보고
            # 지연 없이 즉시 채택한다 (그래야 이후 delta부터 delay를 측정할 기준점이 생긴다).
            self._delayed_current[joint] = desired_value
            self._pending[joint] = deque()
            return desired_value, 0.0

        if not self.config.enable_latency:
            self._delayed_current[joint] = desired_value
            return desired_value, 0.0

        delay_s = self._latency_s()
        queue = self._pending.setdefault(joint, deque())
        queue.append((now, desired_value))
        # ready(=충분히 시간이 지난) 항목 중 가장 최근 것만 "현재 값"으로 채택하고, 그보다
        # 오래된 대기 항목은 버린다(더 최근 값이 이미 그 시점을 지났으므로 의미가 없다).
        while queue and (now - queue[0][0]) >= delay_s:
            self._delayed_current[joint] = queue.popleft()[1]
        return self._delayed_current[joint], delay_s * 1000.0

    def _apply_deadband(self, joint: str, delayed_value: float, simulated_actual: float | None) -> tuple[float, bool]:
        """wrist_roll에만 적용되는 v1 approximation (섹션 6-B). 다른 관절은 그대로 통과."""
        if joint != WRIST_ROLL_JOINT or not self.config.enable_deadband:
            return delayed_value, False

        last_output = self._last_output.get(joint)
        reference = simulated_actual if simulated_actual is not None else last_output
        if reference is None:
            return delayed_value, False

        threshold_deg = self.profile.wrist_roll_deadband.no_response_upper_deg
        if abs(delayed_value - reference) <= threshold_deg:
            # no-response candidate 폭 이내 - 직전 명령을 유지한다 (clamp/reject 아님).
            # last_output이 아직 없으면(첫 프레임) 유지할 "이전 값"이 없으므로 그대로 통과.
            return (last_output if last_output is not None else delayed_value), True
        return delayed_value, False

    def _apply_rate_limit(self, joint: str, value: float) -> tuple[float, bool]:
        """profile의 candidate_frame_delta_soft_limit을 realism_limit으로만 사용한다
        (하드 safety limit 아님 - 섹션 6-A). 새 multiplier 없이 profile 값 그대로 clip."""
        last_output = self._last_output.get(joint)
        limit = self.profile.joints[joint].frame_delta_soft_limit
        if not self.config.enable_rate_limit or limit is None or last_output is None:
            return value, False
        delta = value - last_output
        if abs(delta) > limit:
            return last_output + math.copysign(limit, delta), True
        return value, False

    def _outside_historical_range(self, joint: str, value: float) -> bool:
        if not self.config.enable_historical_range_diagnostic:
            return False
        rng = self.profile.joints[joint].historical_range
        if rng is None:
            return False
        lo, hi = rng
        return value < lo or value > hi

    # -- 메인 진입점 ---------------------------------------------------------

    def process(
        self,
        desired_action: dict[str, float],
        *,
        now: float,
        simulated_actual: dict[str, float] | None = None,
    ) -> ProcessedCommandResult:
        """한 control step 분의 desired_action(raw command)을 처리한다.

        Args:
            desired_action: {joint: value} - profile과 동일 단위(arm=degree, gripper=percent).
            now: 단조 증가 시계 (latency queue 계산 기준). ``time.monotonic()`` 등을 넘긴다.
            simulated_actual: (선택) 이번 프레임의 시뮬레이션 실제 위치 - wrist_roll deadband
                기준값으로 쓰인다. 없으면 직전 processed 값을 기준으로 쓴다.

        Raises:
            RealisticControlError: profile에 정의되지 않은 관절이 desired_action에 있는 경우.
        """
        unknown = [j for j in desired_action if j not in self.profile.joints]
        if unknown:
            raise RealisticControlError(f"control profile에 없는 관절입니다: {unknown}")

        processed: dict[str, float] = {}
        diagnostics: dict[str, JointDiagnostic] = {}

        for joint, raw_value in desired_action.items():
            actual = simulated_actual.get(joint) if simulated_actual else None

            delayed_value, latency_ms = self._apply_latency(joint, raw_value, now)
            deadband_value, deadband_applied = self._apply_deadband(joint, delayed_value, actual)
            rate_value, rate_limited = self._apply_rate_limit(joint, deadband_value)
            outside_range = self._outside_historical_range(joint, rate_value)

            previous_output = self._last_output.get(joint)
            command_delta = (rate_value - previous_output) if previous_output is not None else None

            self._last_output[joint] = rate_value
            processed[joint] = rate_value
            diagnostics[joint] = JointDiagnostic(
                joint=joint,
                raw_action=raw_value,
                processed_action=rate_value,
                simulated_actual_state=actual,
                command_delta=command_delta,
                deadband_applied=deadband_applied,
                latency_applied_ms=latency_ms,
                rate_limited=rate_limited,
                outside_historical_range=outside_range,
                tracking_error=None,
            )

        return ProcessedCommandResult(processed_action=processed, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# 진단 로깅 (CSV/JSON) - follower_safe_mapper.FollowerSafeRecorder와 같은 패턴 재사용.
# ---------------------------------------------------------------------------

CSV_FIELDNAMES: tuple[str, ...] = (
    "step",
    "joint",
    "raw_action",
    "processed_action",
    "simulated_actual_state",
    "command_delta",
    "deadband_applied",
    "latency_applied_ms",
    "rate_limited",
    "outside_historical_range",
    "tracking_error",
)


@dataclass
class RealisticControlRecorder:
    """process() 결과를 프레임마다 누적해 JSON/CSV로 남긴다 (섹션 11 - console spam 방지용).

    스텝 수가 매우 커질 수 있으므로(합성 trajectory 벤치마크 등) 이 레코더는 stdout에는
    아무것도 출력하지 않는다 - 파일로만 남긴다.
    """

    _rows: list[dict] = field(default_factory=list)

    def record(self, step: int, diagnostics: dict[str, JointDiagnostic]) -> None:
        for joint, d in diagnostics.items():
            self._rows.append(
                {
                    "step": step,
                    "joint": joint,
                    "raw_action": d.raw_action,
                    "processed_action": d.processed_action,
                    "simulated_actual_state": d.simulated_actual_state,
                    "command_delta": d.command_delta,
                    "deadband_applied": d.deadband_applied,
                    "latency_applied_ms": d.latency_applied_ms,
                    "rate_limited": d.rate_limited,
                    "outside_historical_range": d.outside_historical_range,
                    "tracking_error": d.tracking_error,
                }
            )

    @property
    def rows(self) -> list[dict]:
        return list(self._rows)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": self._rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)
