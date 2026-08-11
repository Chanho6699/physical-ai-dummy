"""Phase C-3B: continuous 60Hz follower action writer 추상화.

``hardware/safety/staged_follower_writer.py``(스테이지당 최대 N회, 이미 검증된
ACCEPT-only 호출 규율)의 원칙을 재사용한다 - **새 safety writer를 중복으로 만들지
않는다**(요구사항). 이 파일이 하는 일은 오직:

    1. 계속 도는 60Hz 제어 루프가 쓸 수 있는 얇은 ``FollowerActionWriter`` 인터페이스를
       정의하고(``write(action_deg) -> WriteResult`` 딱 하나),
    2. 오프라인 테스트용 구현(``FakeFollowerWriter``/``RecordingFollowerWriter``)을 주고,
    3. production 구현(``SO101FollowerActionWriter``)은 ``StagedFollowerArmedWriter``를
       **합성(compose)**해서 그 검증된 ``send_action`` 래핑 로직을 그대로 물려받는다 -
       "Goal_Position 외 다른 write 경로가 없다"는 그 클래스의 안전 논리를 재구현하지
       않는다.

# ACCEPT-only invariant는 이 클래스가 아니라 호출자(control loop)의 책임

``staged_follower_writer.py``와 동일한 관례를 그대로 따른다: 이 writer들은
ACCEPT/WOULD_CLAMP/REJECT를 전혀 모른다 - "무엇을 쓸지 이미 안전하다고 확정된" action만
받는다. ``RealTimeFollowerControlLoop``가 ``target_valid==True``(=Intent ACCEPT AND
Final Safety ACCEPT)인 tick에 대해서만 ``write()``를 호출해야 하는 invariant를 지킨다
(``tests/test_realtime_control_loop.py::test_writer_invariant_*`` 참고) - write 경로와
판정 경로를 분리한다는 원칙을 그대로 유지한다.

# 이번 phase에서 하지 않는 것

``SO101FollowerActionWriter``는 정의만 되고 **이 저장소 어디에서도 실제 follower
객체와 함께 인스턴스화되지 않는다** - ``connect()``/``write()``가 실제 포트에 닿는
경로가 없다. 테스트에서도 ``test_staged_follower_writer.py``와 동일한 fake follower
double만 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from hardware.safety.staged_follower_writer import StagedFollowerArmedWriter, _FollowerLike
from runtime.common.vla_contract import JOINT_ORDER

# StagedFollowerArmedWriter는 "스테이지당 최대 N회"를 강제하도록 설계됐다(유한 테스트
# 시나리오용). 연속 60Hz 제어는 그 "N"이 사실상 무제한이어야 하지만, "완전 무제한"은
# 안전 관점에서 나쁜 기본값이다(버그로 무한 루프가 나도 최후의 circuit breaker가 있어야
# 함). 그래서 "무제한처럼 보이는 값"이 아니라 명시적인 세션 상한을 상수로 둔다 - 60Hz
# 기준 약 46시간 연속 가동에 해당하는 값(10**7 tick)으로, 정상적인 단일 세션에서는
# 절대 닿지 않지만 진짜 폭주 버그가 있으면 결국 멈춘다.
DEFAULT_SESSION_WRITE_CAP = 10_000_000


@dataclass(frozen=True)
class WriteResult:
    """``FollowerActionWriter.write()`` 공통 반환 타입."""

    executed: bool
    requested_action_deg: dict[str, float]
    sent_action_deg: dict[str, float] | None
    write_index: int  # 이 writer 인스턴스 생애 동안 시도된(성공 여부 무관) write 순번, 0부터
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "executed": self.executed,
            "requested_action_deg": dict(self.requested_action_deg),
            "sent_action_deg": dict(self.sent_action_deg) if self.sent_action_deg is not None else None,
            "write_index": self.write_index,
            "error": self.error,
        }


class FollowerActionWriter(Protocol):
    """control loop이 요구하는 최소 계약."""

    def write(self, action_deg: dict[str, float]) -> WriteResult: ...


@dataclass
class FakeFollowerWriter:
    """오프라인 전용 - 아무 데도 쓰지 않고 성공을 시뮬레이션한다. ``fail_on_indices``로
    특정 순번의 write만 실패하게 주입할 수 있다(writer exception 테스트용, 섹션 13)."""

    fail_on_indices: frozenset[int] = field(default_factory=frozenset)
    calls: list[dict[str, float]] = field(default_factory=list, init=False)
    call_monotonic_times: list[float] = field(default_factory=list, init=False)
    _count: int = field(default=0, init=False)

    def write(self, action_deg: dict[str, float]) -> WriteResult:
        missing = [j for j in JOINT_ORDER if j not in action_deg]
        if missing:
            raise ValueError(f"action_deg에 관절이 없습니다: {missing}")
        idx = self._count
        self._count += 1
        self.calls.append(dict(action_deg))
        if idx in self.fail_on_indices:
            return WriteResult(
                executed=False, requested_action_deg=dict(action_deg), sent_action_deg=None,
                write_index=idx, error="FakeFollowerWriter: 주입된 실패 (시뮬레이션)",
            )
        return WriteResult(
            executed=True, requested_action_deg=dict(action_deg), sent_action_deg=dict(action_deg), write_index=idx,
        )

    @property
    def write_count(self) -> int:
        return self._count


class RecordingFollowerWriter(FakeFollowerWriter):
    """``FakeFollowerWriter``와 동작은 같지만, 매 호출의 monotonic timestamp도 함께
    기록해 trajectory continuity(섹션 15 - "연속적으로 target을 받는지", "long
    unintended pause 없음")를 테스트에서 직접 검증할 수 있게 한다."""

    def __init__(self, *, monotonic_fn=None, fail_on_indices: frozenset[int] = frozenset()) -> None:
        import time as _time

        super().__init__(fail_on_indices=fail_on_indices)
        self._monotonic_fn = monotonic_fn or _time.monotonic

    def write(self, action_deg: dict[str, float]) -> WriteResult:
        result = super().write(action_deg)
        self.call_monotonic_times.append(self._monotonic_fn())
        return result


class SO101FollowerActionWriter:
    """production 실물 writer - ``StagedFollowerArmedWriter``를 합성 재사용한다(중복
    safety writer를 새로 만들지 않는다는 요구사항). 연속 60Hz 제어이므로 유한
    "스테이지" 개념 대신 ``DEFAULT_SESSION_WRITE_CAP``을 circuit breaker로 쓴다.

    이번 phase에서는 이 클래스가 실제 follower 객체(``SOFollower``)와 함께
    인스턴스화되는 곳이 이 저장소 어디에도 없다 - ``connect()``/``write()``가 실제
    포트에 닿을 방법이 없다. 테스트도 ``test_staged_follower_writer.py``와 동일한
    fake follower double만 쓴다.
    """

    def __init__(self, *, follower: _FollowerLike, session_write_cap: int = DEFAULT_SESSION_WRITE_CAP) -> None:
        self._inner = StagedFollowerArmedWriter(follower=follower, max_write_count=session_write_cap)

    @property
    def write_count(self) -> int:
        return self._inner.write_count

    def connect(self) -> None:
        """``StagedFollowerArmedWriter.connect()``에 그대로 위임한다 - 이번 phase에서
        어떤 코드 경로도 이 메서드를 실제 follower로 호출하지 않는다."""
        self._inner.connect()

    def read_state_deg(self) -> dict[str, float]:
        return self._inner.read_state_deg()

    def write(self, action_deg: dict[str, float]) -> WriteResult:
        result = self._inner.write_action_once(action_deg)
        write_index = result.write_count_after - 1 if result.executed else result.write_count_after
        return WriteResult(
            executed=result.executed, requested_action_deg=result.requested_action_deg,
            sent_action_deg=result.sent_action_deg, write_index=write_index, error=result.error,
        )

    def disconnect(self) -> None:
        self._inner.disconnect()
