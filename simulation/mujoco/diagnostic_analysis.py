"""리더-팔로워 실시간 비교 스트림에서 이상 패턴을 감지하는 온라인 분석기.

``simulation/mujoco/joint_range_diagnostics.py``는 저장된 데이터셋 전체를 한 번에
훑는 "배치" 분석이지만, 이 모듈은 원격 진단 루프(remote_diagnostic.py)가 매 샘플마다
호출하는 "온라인/스트리밍" 분석이다. 판정 로직을 순수 함수/클래스로 분리해 실제 MuJoCo나
HTTP 없이 단위 테스트할 수 있게 했다.

감지하는 이벤트 (요구사항 9번 참고):

- ``persistent_difference``: 리더-팔로워 차이가 같은 방향으로 일정 시간 이상 지속.
- ``follower_saturation_suspected``: 리더는 계속 움직이는데 팔로워는 거의 정지 + 차이가 커짐.
- ``sign_mismatch_suspected``: 리더 변화량과 팔로워(또는 MuJoCo) 변화량의 부호가 반복적으로 반대.
- ``offset_suspected``: 리더가 여러 자세를 거치는 동안 리더-팔로워 차이가 거의 일정하게 유지.
- ``leader_out_of_mujoco_range`` / ``follower_inside_mujoco_range``: 리더만 MuJoCo 관절
  range를 벗어나고 팔로워는 range 안쪽인 경우 (wrist_flex 문제를 명확히 잡아내기 위한 이벤트).

각 이벤트는 "새로 감지된 시점"에 한 번만 발생한다(edge-triggered). 조건이 계속 유지되는
동안 매 샘플마다 반복 출력되지 않도록 하기 위함이며, 이는 콘솔/리포트가 같은 문제로
도배되는 것을 막기 위한 의도적인 설계다. 단, ``leader_out_of_mujoco_range``는 CSV의
``event_code`` 열에 프레임 단위로 남아야 하므로 매 샘플 판정 결과를 별도 필드
(``DiagnosticSample.leader_out_of_range``)로 항상 제공한다.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field

Level = str  # "WARN" (이 모듈은 진단/관찰용이라 BLOCKED를 만들지 않는다 - 실제 차단은 safety_checks.py 담당)


@dataclass(frozen=True)
class DiagnosticConfig:
    """configs/remote_mujoco_diagnostic.yaml의 ``diagnostic`` 섹션과 대응된다.

    persistent_difference_deg 이상, persistent_duration_sec 이상 지속되면
    persistent_difference 이벤트를 낸다. 나머지 필드는 문서(docs/remote_mujoco_diagnostic.md)에
    설명된 대로 초기값이 실측이 아닌 추정치다.
    """

    difference_warn_deg: float = 3.0
    persistent_difference_deg: float = 5.0
    persistent_duration_sec: float = 1.0
    follower_stationary_delta_deg: float = 0.3
    leader_motion_delta_deg: float = 1.0
    saturation_duration_sec: float = 1.0
    # 아래는 yaml 기본 스펙에는 없지만 offset/sign 판정에 필요해 추가한 값들
    # (문서에 근거/추정치임을 명시).
    offset_window_sec: float = 3.0
    offset_pose_variation_deg: float = 2.0
    offset_stability_deg: float = 0.5
    offset_min_samples: int = 8
    sign_mismatch_window_size: int = 5
    sign_mismatch_min_count: int = 3
    sign_mismatch_min_delta_deg: float = 0.3


@dataclass(frozen=True)
class DiagnosticEvent:
    code: str
    level: Level
    joint: str
    message: str
    timestamp: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "level": self.level,
            "joint": self.joint,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass(frozen=True)
class DiagnosticSample:
    """분석에 사용한 한 샘플의 정규화된 결과 (CSV/리포트가 그대로 참고할 수 있게 노출)."""

    joint: str
    timestamp: float
    leader_deg: float
    follower_deg: float
    difference_deg: float
    leader_out_of_range: bool
    follower_out_of_range: bool


@dataclass
class _JointHistory:
    times: deque = field(default_factory=lambda: deque(maxlen=4096))
    leader: deque = field(default_factory=lambda: deque(maxlen=4096))
    follower: deque = field(default_factory=lambda: deque(maxlen=4096))
    difference: deque = field(default_factory=lambda: deque(maxlen=4096))

    # persistent_difference 상태
    persistent_run_start: float | None = None
    persistent_run_sign: int = 0
    persistent_active: bool = False

    # follower_saturation 상태
    saturation_active: bool = False

    # sign_mismatch 상태
    sign_mismatch_recent: deque = field(default_factory=lambda: deque(maxlen=16))
    sign_mismatch_active: bool = False

    # offset 상태 (세션당 1회만 보고)
    offset_reported: bool = False


class DiagnosticAnalyzer:
    """관절별 리더/팔로워 히스토리를 들고, 샘플이 들어올 때마다 이상 패턴을 판정한다."""

    def __init__(self, config: DiagnosticConfig | None = None) -> None:
        self._config = config or DiagnosticConfig()
        self._joints: dict[str, _JointHistory] = {}

    def _history(self, joint: str) -> _JointHistory:
        if joint not in self._joints:
            self._joints[joint] = _JointHistory()
        return self._joints[joint]

    def update(
        self,
        joint: str,
        timestamp: float,
        leader_deg: float,
        follower_deg: float,
        *,
        joint_range_deg: tuple[float, float] | None = None,
    ) -> tuple[DiagnosticSample, list[DiagnosticEvent]]:
        """새 샘플 하나를 반영하고 (정규화된 샘플, 새로 감지된 이벤트 목록)을 반환한다."""
        history = self._history(joint)
        difference = leader_deg - follower_deg

        history.times.append(timestamp)
        history.leader.append(leader_deg)
        history.follower.append(follower_deg)
        history.difference.append(difference)

        leader_out = False
        follower_out = False
        if joint_range_deg is not None:
            lo, hi = joint_range_deg
            leader_out = leader_deg < lo or leader_deg > hi
            follower_out = follower_deg < lo or follower_deg > hi

        sample = DiagnosticSample(
            joint=joint,
            timestamp=timestamp,
            leader_deg=leader_deg,
            follower_deg=follower_deg,
            difference_deg=difference,
            leader_out_of_range=leader_out,
            follower_out_of_range=follower_out,
        )

        events: list[DiagnosticEvent] = []
        events.extend(self._check_range_exceed(joint, history, sample))
        events.extend(self._check_persistent_difference(joint, history, timestamp, difference))
        events.extend(self._check_follower_saturation(joint, history, timestamp))
        events.extend(self._check_sign_mismatch(joint, history, timestamp))
        events.extend(self._check_offset_suspected(joint, history, timestamp))
        return sample, events

    # -- range exceed ----------------------------------------------------

    def _check_range_exceed(self, joint: str, history: _JointHistory, sample: DiagnosticSample) -> list[DiagnosticEvent]:
        if sample.leader_out_of_range and not sample.follower_out_of_range:
            return [
                DiagnosticEvent(
                    code="leader_out_of_mujoco_range",
                    level="WARN",
                    joint=joint,
                    message=(
                        f"{joint}: 리더 값({sample.leader_deg:.2f}deg)만 MuJoCo 관절 range를 벗어났고 "
                        f"팔로워 값({sample.follower_deg:.2f}deg)은 range 안쪽입니다."
                    ),
                    timestamp=sample.timestamp,
                    details={
                        "leader_deg": sample.leader_deg,
                        "follower_deg": sample.follower_deg,
                        "sub_event": "follower_inside_mujoco_range",
                    },
                )
            ]
        return []

    # -- persistent difference -------------------------------------------

    def _check_persistent_difference(
        self, joint: str, history: _JointHistory, timestamp: float, difference: float
    ) -> list[DiagnosticEvent]:
        cfg = self._config
        sign = 1 if difference > 0 else (-1 if difference < 0 else 0)
        over_threshold = abs(difference) > cfg.persistent_difference_deg and sign != 0

        if not over_threshold:
            history.persistent_run_start = None
            history.persistent_run_sign = 0
            history.persistent_active = False
            return []

        if history.persistent_run_start is None or sign != history.persistent_run_sign:
            history.persistent_run_start = timestamp
            history.persistent_run_sign = sign
            history.persistent_active = False
            return []

        duration = timestamp - history.persistent_run_start
        if duration >= cfg.persistent_duration_sec and not history.persistent_active:
            history.persistent_active = True
            direction = "리더가 더 큼" if sign > 0 else "팔로워가 더 큼"
            return [
                DiagnosticEvent(
                    code="persistent_difference",
                    level="WARN",
                    joint=joint,
                    message=(
                        f"{joint}: 리더-팔로워 차이({difference:+.2f}deg)가 {duration:.1f}초간 "
                        f"같은 방향({direction})으로 지속되고 있습니다."
                    ),
                    timestamp=timestamp,
                    details={
                        "difference_deg": difference,
                        "duration_sec": duration,
                        "possible_causes": ["캘리브레이션 range 차이", "팔로워 포화(saturation)"],
                    },
                )
            ]
        return []

    # -- follower saturation ----------------------------------------------

    def _check_follower_saturation(self, joint: str, history: _JointHistory, timestamp: float) -> list[DiagnosticEvent]:
        cfg = self._config
        window_start = timestamp - cfg.saturation_duration_sec
        idx = None
        for i in range(len(history.times) - 1, -1, -1):
            if history.times[i] <= window_start:
                idx = i
                break
        if idx is None:
            # 윈도우 길이만큼 데이터가 아직 쌓이지 않음 -> 단일 샘플로 판정하지 않는다.
            history.saturation_active = False
            return []

        leader_delta = history.leader[-1] - history.leader[idx]
        follower_delta = history.follower[-1] - history.follower[idx]
        diff_delta = history.difference[-1] - history.difference[idx]

        leader_moving = abs(leader_delta) >= cfg.leader_motion_delta_deg
        follower_stationary = abs(follower_delta) <= cfg.follower_stationary_delta_deg
        diff_growing = abs(diff_delta) >= cfg.follower_stationary_delta_deg  # 차이가 실제로 벌어지는 중인지

        if leader_moving and follower_stationary and diff_growing:
            if not history.saturation_active:
                history.saturation_active = True
                return [
                    DiagnosticEvent(
                        code="follower_saturation_suspected",
                        level="WARN",
                        joint=joint,
                        message=(
                            f"{joint}: 최근 {cfg.saturation_duration_sec:.1f}초 동안 리더는 "
                            f"{leader_delta:+.2f}deg 움직였지만 팔로워는 {follower_delta:+.2f}deg만 "
                            "움직여 포화(saturation)가 의심됩니다."
                        ),
                        timestamp=timestamp,
                        details={
                            "leader_delta_deg": leader_delta,
                            "follower_delta_deg": follower_delta,
                            "difference_delta_deg": diff_delta,
                        },
                    )
                ]
            return []

        history.saturation_active = False
        return []

    # -- sign mismatch ------------------------------------------------------

    def _check_sign_mismatch(self, joint: str, history: _JointHistory, timestamp: float) -> list[DiagnosticEvent]:
        cfg = self._config
        if len(history.leader) < 2:
            return []

        leader_delta = history.leader[-1] - history.leader[-2]
        follower_delta = history.follower[-1] - history.follower[-2]

        if abs(leader_delta) < cfg.sign_mismatch_min_delta_deg or abs(follower_delta) < cfg.sign_mismatch_min_delta_deg:
            mismatch_now = False
        else:
            mismatch_now = (leader_delta > 0) != (follower_delta > 0)

        history.sign_mismatch_recent.append(mismatch_now)
        recent_count = sum(1 for v in history.sign_mismatch_recent if v)

        if (
            len(history.sign_mismatch_recent) >= cfg.sign_mismatch_window_size
            and recent_count >= cfg.sign_mismatch_min_count
        ):
            if not history.sign_mismatch_active:
                history.sign_mismatch_active = True
                return [
                    DiagnosticEvent(
                        code="sign_mismatch_suspected",
                        level="WARN",
                        joint=joint,
                        message=(
                            f"{joint}: 최근 {len(history.sign_mismatch_recent)}개 구간 중 {recent_count}개에서 "
                            "리더와 팔로워의 변화 방향이 반대였습니다 (부호/방향 불일치 의심)."
                        ),
                        timestamp=timestamp,
                        details={"mismatch_count": recent_count, "window_size": len(history.sign_mismatch_recent)},
                    )
                ]
            return []

        if recent_count == 0:
            history.sign_mismatch_active = False
        return []

    # -- offset suspected -----------------------------------------------------

    def _check_offset_suspected(self, joint: str, history: _JointHistory, timestamp: float) -> list[DiagnosticEvent]:
        cfg = self._config
        if history.offset_reported:
            return []

        window_start = timestamp - cfg.offset_window_sec
        window_leader: list[float] = []
        window_diff: list[float] = []
        for t, leader_deg, diff in zip(history.times, history.leader, history.difference):
            if t >= window_start:
                window_leader.append(leader_deg)
                window_diff.append(diff)

        if len(window_leader) < cfg.offset_min_samples:
            return []

        leader_variation = max(window_leader) - min(window_leader)
        if leader_variation < cfg.offset_pose_variation_deg:
            # 리더가 거의 한 자세에 머물러 있으면 "여러 자세에서" 조건을 만족하지 못한다.
            return []

        diff_stability = statistics.pstdev(window_diff)
        if diff_stability <= cfg.offset_stability_deg:
            history.offset_reported = True
            mean_diff = statistics.fmean(window_diff)
            return [
                DiagnosticEvent(
                    code="offset_suspected",
                    level="WARN",
                    joint=joint,
                    message=(
                        f"{joint}: 리더가 {leader_variation:.2f}deg 범위로 여러 자세를 거치는 동안 "
                        f"리더-팔로워 차이가 평균 {mean_diff:+.2f}deg(표준편차 {diff_stability:.2f}deg)로 "
                        "거의 일정하게 유지되어 고정 offset(캘리브레이션 영점 차이)이 의심됩니다."
                    ),
                    timestamp=timestamp,
                    details={
                        "mean_difference_deg": mean_diff,
                        "difference_stddev_deg": diff_stability,
                        "leader_variation_deg": leader_variation,
                        "sample_count": len(window_leader),
                    },
                )
            ]
        return []


def summarize_events(events: list[DiagnosticEvent]) -> dict[str, int]:
    """이벤트 리스트를 code별 카운트로 요약한다 (JSON 리포트용)."""
    counts: dict[str, int] = {}
    for evt in events:
        counts[evt.code] = counts.get(evt.code, 0) + 1
    return counts
