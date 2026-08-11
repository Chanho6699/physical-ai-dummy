"""Phase C-3B: 오프라인 전용 fake follower state source.

``runtime/laptop/follower_state_source.py``(실물 wrapper, 이번 phase에서 손대지 않음)의
``FollowerStateSnapshot``을 그대로 재사용한다 - 새 snapshot 타입을 만들지 않는다. 이
파일은 그 real 모듈을 import만 하고, 실물 reader/connect 관련 코드는 전혀 참조하지
않는다 - 구조적으로 하드웨어에 닿을 방법이 없다(실물 모듈처럼 write 메서드 자체가
없다는 안전 논리를 그대로 물려받음).

``RealTimeFollowerControlLoop``이 기대하는 최소 인터페이스(``read() ->
FollowerStateSnapshot``)를 duck-type으로 만족한다 - 실물 ``ReadOnlyRealFollowerStateSource``와
동일한 모양이라 향후 그 자리에 그대로 바꿔 끼울 수 있다(production path 목표).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Protocol

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.follower_state_source import FollowerStateSnapshot


class FollowerStateSourceProtocol(Protocol):
    """control loop이 요구하는 최소 계약 - 실물/fake 둘 다 이 모양이면 된다."""

    def read(self) -> FollowerStateSnapshot: ...


class FakeFollowerStateSourceError(RuntimeError):
    """``fail_next_read()``로 주입된 시뮬레이션 실패."""


class FakeFollowerStateSource:
    """오프라인 전용 - 테스트가 ``set_state()``로 언제든 현재 관절 상태를 바꿀 수 있는
    fake state source. 여러 thread에서 동시에 읽고/쓸 수 있어야 하므로(제어 루프
    thread가 읽는 동안 테스트 thread가 갱신) 내부 lock으로 보호한다."""

    def __init__(
        self,
        *,
        initial_state_deg: dict[str, float] | None = None,
        monotonic_fn: Callable[[], float] = time.monotonic,
        wall_fn: Callable[[], float] = time.time,
    ) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, float] = dict(initial_state_deg) if initial_state_deg is not None else {j: 0.0 for j in JOINT_ORDER}
        self._monotonic_fn = monotonic_fn
        self._wall_fn = wall_fn
        self._fail_next = False
        self.read_count = 0

    def set_state(self, state_deg: dict[str, float]) -> None:
        missing = [j for j in JOINT_ORDER if j not in state_deg]
        if missing:
            raise ValueError(f"state_deg에 관절이 없습니다: {missing}")
        with self._lock:
            self._state = dict(state_deg)

    def fail_next_read(self) -> None:
        """다음 ``read()`` 호출 한 번만 예외를 던지게 한다(state read exception 주입
        테스트용, 섹션 13) - 그 이후 호출은 다시 정상."""
        with self._lock:
            self._fail_next = True

    def read(self) -> FollowerStateSnapshot:
        with self._lock:
            self.read_count += 1
            if self._fail_next:
                self._fail_next = False
                raise FakeFollowerStateSourceError("FakeFollowerStateSource: 주입된 읽기 실패 (시뮬레이션)")
            positions = dict(self._state)
        return FollowerStateSnapshot(
            positions_deg=positions,
            read_at_monotonic=self._monotonic_fn(),
            read_at_wall=self._wall_fn(),
        )
