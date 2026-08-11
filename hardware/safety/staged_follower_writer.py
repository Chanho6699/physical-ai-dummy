"""6관절 SO-101 follower로 SmolVLA action을 실제로 write하는 armed writer.

``hardware/safety/single_joint_writer.py``(wrist_roll 1개, 평생 1회)의 원칙을 그대로
따르되, 관절 6개 전체와 스테이지당 N회 write로 확장한다. 새 원칙을 만들지 않는다:

    - write budget은 write **시도 직전**에 소비된다 (``consume()``이 실패하면
      ``bus``/``follower`` 쪽 코드는 아예 실행되지 않는다) - 두 번째 이상 시도가 예산을
      넘으면 시리얼 패킷 자체가 나가지 않는다.
    - 이 클래스는 ``Goal_Position``을 쓰는 ``follower.send_action()`` 외에 다른 어떤
      write 경로도 갖지 않는다 - ``enable_torque``/``disable_torque``/``configure``/
      ``write_calibration``/``set_half_turn_homings``를 이 파일 어디에서도 호출하지 않는다
      (``SOFollower.connect()``가 내부적으로 ``configure()``를 호출하는 것은 LeRobot
      표준 동작이고, ``hardware/diagnostics/instrumented_teleop.py``에 이미 문서화된
      전례다 - 이 클래스가 추가로 만드는 write가 아니다).
    - retry 없음 (``send_action`` 실패 시 재시도하지 않는다).
    - ``disconnect()``는 ``SOFollowerConfig.disable_torque_on_disconnect`` 설정값을
      그대로 통과시킨다 - 이 클래스가 그 값을 오버라이드하지 않는다.

## 실제 write API (설치된 lerobot에서 확인)

``~/lerobot/src/lerobot/robots/so_follower/so_follower.py``::

    def get_observation(self) -> RobotObservation:
        obs_dict = self.bus.sync_read("Present_Position", ...)
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        ...

    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        if self.config.max_relative_target is not None:
            ...  # LeRobot 자체의 추가 clip (이 저장소가 만든 것 아님)
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}  # 실제로 보낸 값

``send_action``이 "실제로 보낸 action"을 반환한다는 점이 중요하다 - ``max_relative_target``이
설정돼 있으면 LeRobot이 자체적으로 값을 자를 수 있으므로, 이 writer는 그 반환값을 그대로
캡처해 보고서에 남긴다(무시하지 않는다).

이 writer는 이미지를 다루지 않는다 - 카메라는 ``runtime/laptop/camera_source.py``의
``RealCameraObservationSource``(이미 감사된 read-only 경로)가 별도로 담당하고, 이 writer가
감싸는 ``SOFollower``는 ``cameras={}``로 생성해 로봇 bus 하나만 이 클래스가 점유한다
(포트 충돌 방지 - 관측용 별도 커넥션을 안 만든다).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "StagedWriteBudget",
    "StagedWriteBudgetExceededError",
    "StagedFollowerArmedWriter",
    "JOINT_ORDER",
]

# runtime.common.vla_contract.JOINT_ORDER와 동일한 순서를 그대로 가져온다 (새로 정의하지
# 않음) - import 시점에 무거운 의존성(mujoco 등)을 끌고 오지 않기 위해 값만 복제해 두고,
# 테스트에서 두 값이 같음을 확인한다.
JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class StagedWriteBudgetExceededError(RuntimeError):
    """이번 stage의 write budget을 초과해서 write를 시도했을 때."""


class StagedWriteBudget:
    """write 횟수를 추적/강제하는 카운터. ``single_joint_writer.WriteBudget``과 동일하게
    reset 메서드가 없다 - 새 budget이 필요하면 새 writer/새 프로세스를 만들어야 한다."""

    def __init__(self, max_write_count: int) -> None:
        if max_write_count < 1:
            raise ValueError(f"max_write_count는 1 이상이어야 합니다: {max_write_count}")
        self._max = max_write_count
        self._count = 0

    @property
    def max_write_count(self) -> int:
        return self._max

    @property
    def write_count(self) -> int:
        return self._count

    def consume(self) -> None:
        if self._count >= self._max:
            raise StagedWriteBudgetExceededError(
                f"write budget 초과: 이미 {self._count}회 write했고 이번 stage 최대 {self._max}회까지만 허용됩니다."
            )
        self._count += 1


class _FollowerLike(Protocol):
    """``StagedFollowerArmedWriter``가 요구하는 ``SOFollower`` 최소 인터페이스
    (fake follower 테스트용 duck type)."""

    def connect(self) -> None: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]: ...
    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class WriteAttemptResult:
    """write 한 번의 전/후 상태 - budget 소진으로 write 자체를 못 한 경우에도 만들어진다
    (``executed=False``)."""

    executed: bool
    requested_action_deg: dict[str, float]
    sent_action_deg: dict[str, float] | None  # send_action()이 실제로 보고한 값 (LeRobot이 자체 clip했을 수 있음)
    write_count_before: int
    write_count_after: int
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "executed": self.executed,
            "requested_action_deg": dict(self.requested_action_deg),
            "sent_action_deg": dict(self.sent_action_deg) if self.sent_action_deg is not None else None,
            "write_count_before": self.write_count_before,
            "write_count_after": self.write_count_after,
            "error": self.error,
        }


class StagedFollowerArmedWriter:
    """6관절 SO-101 follower에 스테이지당 최대 N회 ``send_action`` write를 허용하는 래퍼.

    이 클래스가 아는 메서드는 ``connect``/``get_observation``/``send_action``/``disconnect``
    딱 넷뿐이다(``_FollowerLike`` 참고) - torque/calibration 관련 메서드는 아예 참조하지
    않는다.
    """

    def __init__(self, *, follower: _FollowerLike, max_write_count: int) -> None:
        self._follower = follower
        self._budget = StagedWriteBudget(max_write_count=max_write_count)
        self._connected = False

    @property
    def write_count(self) -> int:
        return self._budget.write_count

    @property
    def max_write_count(self) -> int:
        return self._budget.max_write_count

    def connect(self) -> None:
        """``SOFollower.connect()``를 그대로 호출한다 (LeRobot 표준 ``configure()`` 포함,
        모듈 docstring 참고). 이 메서드 자체는 write budget을 소비하지 않는다."""
        self._follower.connect()
        self._connected = True

    def read_state_deg(self) -> dict[str, float]:
        """``follower.get_observation()``에서 관절 위치만 골라 ``JOINT_ORDER`` dict로
        변환한다 (``.pos`` 접미사 제거). 카메라 키가 섞여 있어도 무시한다."""
        obs = self._follower.get_observation()
        state: dict[str, float] = {}
        for joint in JOINT_ORDER:
            key = f"{joint}.pos"
            if key not in obs:
                raise KeyError(f"get_observation() 결과에 '{key}'가 없습니다: keys={list(obs.keys())}")
            state[joint] = float(obs[key])
        return state

    def write_action_once(self, safe_action_deg: dict[str, float]) -> WriteAttemptResult:
        """정확히 한 번의 ``send_action`` 시도. 호출부(``staged_real_rollout.py``)는 이
        메서드를 Safety Gate가 ``ACCEPT``를 반환한 action에 대해서만 호출해야 한다 -
        이 클래스 자체는 ACCEPT/WOULD_CLAMP/REJECT를 모른다(그건 Safety Gate의 책임이고,
        이 writer는 "무엇을 쓸지 이미 결정된" action만 받는다 - 섹션 요구사항: write
        경로와 판정 경로를 분리)."""
        missing = [j for j in JOINT_ORDER if j not in safe_action_deg]
        if missing:
            raise ValueError(f"safe_action_deg에 관절이 없습니다: {missing}")

        write_count_before = self._budget.write_count
        try:
            self._budget.consume()  # 예외가 나면 아래 send_action은 절대 실행되지 않는다
        except StagedWriteBudgetExceededError as exc:
            return WriteAttemptResult(
                executed=False, requested_action_deg=dict(safe_action_deg), sent_action_deg=None,
                write_count_before=write_count_before, write_count_after=self._budget.write_count,
                error=str(exc),
            )

        action = {f"{j}.pos": safe_action_deg[j] for j in JOINT_ORDER}
        try:
            sent = self._follower.send_action(action)
        except Exception as exc:  # noqa: BLE001 - 통신 오류를 폭넓게 잡아 결과에 남긴다
            return WriteAttemptResult(
                executed=False, requested_action_deg=dict(safe_action_deg), sent_action_deg=None,
                write_count_before=write_count_before, write_count_after=self._budget.write_count,
                error=f"send_action 실패: {exc}",
            )

        sent_deg = {j: float(sent[f"{j}.pos"]) for j in JOINT_ORDER if f"{j}.pos" in sent}
        return WriteAttemptResult(
            executed=True, requested_action_deg=dict(safe_action_deg), sent_action_deg=sent_deg,
            write_count_before=write_count_before, write_count_after=self._budget.write_count,
        )

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._follower.disconnect()
        self._connected = False
