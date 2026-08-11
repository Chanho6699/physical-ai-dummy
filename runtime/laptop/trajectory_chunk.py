"""Phase C-1B: 하나의 ``/predict_chunk`` 응답을 담는 immutable, timestamp-aware trajectory
표현.

# 왜 필요한가

Phase C-1A(``runtime/desktop/vla_server.py``의 ``/predict_chunk``,
``runtime/laptop/vla_client.py``의 ``ChunkPredictResult``)는 "이 chunk가 어떤 관측 시점을
기준으로 만들어졌고, 지금(``now``) 기준으로 그 50-step 중 어디부터가 아직 미래인지"를
계산하는 헬퍼가 없었다 - ``ChunkPredictResult``는 순수 wire 응답 파싱 결과일 뿐이다. 이
모듈은 그 위에 "시간 계산" 레이어 하나만 추가한다 - 이 세션에서는 아직 그 계산 결과를
실제로 뭔가에 쓰지 않는다(interpolation/ensemble/write는 전부 금지 범위).

# monotonic vs wall clock (섹션 1 요구사항)

이 클래스의 모든 timing 판단(``current_index``/``first_future_index``/
``remaining_future_count``/``age_ms``/``is_expired``)은 **``time.monotonic()`` 기준
필드만** 사용한다 - ``server_received_at``/``server_responded_at``(wall clock, Desktop이
보고한 값)은 로깅/모니터링용으로만 보존하고 control timing 계산에는 절대 쓰지 않는다.
NTP 보정/시스템 시계 변경에 흔들리는 wall clock으로 "지금 이 action이 몇 번째 index인가"를
계산하면 안 되기 때문이다(``staged_real_rollout.py``/``FollowerStateSnapshot``이 이미
따르는 것과 같은 관례).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# IEEE754 부동소수점 표현 오차 보정용 - ``_floor_index()`` 참고. spacing(수십 ms) 대비
# 1e-9(나노초)는 무시 가능한 크기이므로 진짜 슬롯 경계 판단에는 영향을 주지 않는다.
_FLOAT_INDEX_EPSILON = 1e-9


@dataclass(frozen=True)
class TimestampedActionChunk:
    """하나의 fresh ``/predict_chunk`` 응답 - 항상 chunk 전체(``chunk_size``개)를 그대로
    보존한다(일부만 잘라 담지 않음 - 향후 temporal ensembling이 과거 chunk의 뒷부분도
    다시 참조할 수 있어야 하기 때문).

    ``actions``는 ``tuple[dict[str, float], ...]``(list가 아님)로 강제한다 - 이 객체가
    frozen이어도 내부 list는 여전히 mutate 가능하다는 흔한 함정을 피하기 위함이다.
    """

    sequence: int
    session_id: str
    # -- monotonic 기준 (control timing 판단용, 섹션 1) --------------------------------
    observation_time_monotonic: float  # 이 chunk의 근거가 된 관측(camera/state)을 캡처한 시각
    request_started_time_monotonic: float  # predict_chunk() HTTP/in-process 호출 시작 시각
    response_received_time_monotonic: float  # predict_chunk() 응답을 받은 시각
    # -- wall clock (로깅/모니터링 전용, control timing에 쓰지 않음) ---------------------
    server_received_at: float | None
    server_responded_at: float | None
    # -- 추론/궤적 메타데이터 --------------------------------------------------------
    inference_latency_ms: float | None
    chunk_index_spacing_s: float  # chunk[k]와 chunk[k+1] 사이 시간 간격(초) = 1/dataset_fps
    chunk_size: int
    actions: tuple[dict[str, float], ...]  # 길이 = chunk_size, 각 원소 = 6-joint 절대좌표(deg/percent)
    model_id: str | None
    backend: str | None

    def validate(self) -> tuple[bool, str | None]:
        """이 객체 자체의 내적 일관성만 검사한다(외부 상태/이전 chunk와의 비교는
        ``TrajectoryBuffer.publish()``의 책임 - 섹션 2). 신뢰할 수 없는 입력(네트워크에서
        온 값)에서 만들어졌을 수 있으므로 예외를 던지지 않고 (bool, reason) 규약을 쓴다
        (``runtime/common/vla_contract.py``의 ``validate_joint_dict``/``validate_action_chunk``와
        동일 관례)."""
        if self.chunk_size <= 0:
            return False, f"chunk_size가 0 이하입니다: {self.chunk_size}"
        if len(self.actions) != self.chunk_size:
            return False, f"actions 길이({len(self.actions)})가 chunk_size({self.chunk_size})와 다릅니다."
        if not math.isfinite(self.chunk_index_spacing_s) or self.chunk_index_spacing_s <= 0:
            return False, f"chunk_index_spacing_s가 유효한 양수가 아닙니다: {self.chunk_index_spacing_s}"
        if not math.isfinite(self.observation_time_monotonic):
            return False, f"observation_time_monotonic이 finite하지 않습니다: {self.observation_time_monotonic}"
        for k, action in enumerate(self.actions):
            for joint, value in action.items():
                if not math.isfinite(value):
                    return False, f"actions[{k}]['{joint}']가 finite하지 않습니다: {value}"
        return True, None

    # -- 시간 계산 헬퍼 (섹션 1/6) ------------------------------------------------------

    def nominal_target_time(self, k: int) -> float:
        """chunk의 k번째 action이 "원래" 실행돼야 할 monotonic 시각 -
        ``observation_time + k * chunk_index_spacing_s``."""
        return self.observation_time_monotonic + k * self.chunk_index_spacing_s

    @property
    def horizon_duration_s(self) -> float:
        """이 chunk가 커버하는 전체 시간 길이(초) = chunk_size * spacing (Candidate B는
        50/30 ≈ 1.667s)."""
        return self.chunk_size * self.chunk_index_spacing_s

    @property
    def horizon_end_time_monotonic(self) -> float:
        """이 chunk의 마지막 index 다음(=완전히 경과) 시각 - ``nominal_target_time(chunk_size)``."""
        return self.observation_time_monotonic + self.horizon_duration_s

    def current_index(self, now_monotonic: float) -> int:
        """``now``가 현재 속한 index 슬롯 - ``floor((now - observation_time) / spacing)``.

        Boundary(섹션 6): ``now < observation_time``이면 0으로 clamp한다(관측 이전 시각을
        묻는 것은 정상적인 control-loop 상황이 아니지만, 방어적으로 0을 반환한다 - 음수
        index를 반환하지 않음). 반환값은 ``chunk_size`` 이상일 수 있다(=이미 chunk 전체가
        지났음, 호출자가 ``is_expired()``/``remaining_future_count()``로 판단)."""
        if now_monotonic <= self.observation_time_monotonic:
            return 0
        return self._floor_index((now_monotonic - self.observation_time_monotonic) / self.chunk_index_spacing_s)

    def first_future_index(self, now_monotonic: float) -> int | None:
        """``now`` 시점 기준 아직 지나지 않은(``nominal_target_time(k) > now``) 가장 이른
        index. chunk 전체가 이미 지났으면 ``None``(섹션 6: "now beyond chunk end ->
        expired").

        ``now < observation_time``이면 index 0 자체가 이미 미래이므로 0을 반환한다
        (섹션 6 boundary 요구사항과 정확히 일치)."""
        if now_monotonic < self.observation_time_monotonic:
            idx = 0
        else:
            idx = self._floor_index((now_monotonic - self.observation_time_monotonic) / self.chunk_index_spacing_s) + 1
        if idx >= self.chunk_size:
            return None
        return idx

    @staticmethod
    def _floor_index(raw: float) -> int:
        """``math.floor(raw)``인데, IEEE754 부동소수점 표현 오차를 보정한다.

        ``chunk_index_spacing_s``가 실제로는 1/30(=0.0333..., 이진 소수로 정확히 표현
        불가능)이므로, 예를 들어 ``k * spacing / spacing``이 수학적으로는 정확히 ``k``여야
        하는데 부동소수점 연산에서는 ``k``보다 아주 조금 작은 값(예: 0.9999999999999998)이
        나올 수 있다 - 이걸 그냥 floor하면 슬롯 경계 정확히 그 순간에 index가 1 작게
        나오는 off-by-one이 생긴다(실제로 이 파일의 boundary 테스트에서 이 문제가
        재현됐다). ``spacing``(수십 ms) 대비 무시 가능한 수준의 상대 epsilon(1e-9, 즉
        나노초 단위)만 더해서 이 표현 오차만 흡수하고, 진짜 다른 슬롯에 속하는 경우는
        전혀 건드리지 않는다."""
        return int(math.floor(raw + _FLOAT_INDEX_EPSILON))

    def remaining_future_count(self, now_monotonic: float) -> int:
        """``first_future_index(now)``부터 chunk 끝까지 남은 "아직 쓸 수 있는" action 개수.
        0이면(또는 ``first_future_index``가 ``None``이면) unusable(섹션 7 기본 정책)."""
        idx = self.first_future_index(now_monotonic)
        if idx is None:
            return 0
        return self.chunk_size - idx

    def is_expired(self, now_monotonic: float, *, max_chunk_age_ms: float | None = None) -> bool:
        """기본 정책(섹션 7): ``remaining_future_count(now) <= 0``이면 무조건 expired.
        ``max_chunk_age_ms``를 주면(선택, 하드코딩된 기본값 없음 - 호출자가 명시해야 함)
        chunk의 관측 시각 기준 나이가 그 값을 넘어도 expired로 취급한다(예: 아직 미래
        index가 남아 있어도, observation 자체가 너무 오래됐다고 보수적으로 판단하고 싶은
        경우를 위한 추가 guard - 섹션 7 "추가 freshness guard 후보: observation age")."""
        if self.remaining_future_count(now_monotonic) <= 0:
            return True
        if max_chunk_age_ms is not None:
            if self.age_ms(now_monotonic) > max_chunk_age_ms:
                return True
        return False

    def age_ms(self, now_monotonic: float) -> float:
        """이 chunk 근거 관측이 캡처된 시각(``observation_time_monotonic``) 기준 경과
        시간(ms). ``response_received_time_monotonic``이 아니라 ``observation_time``을
        기준으로 삼는다 - "이 chunk의 예측 유효성"은 HTTP 응답이 도착한 시각이 아니라
        그 예측의 근거가 된 관측이 얼마나 오래됐는지에 달려 있기 때문이다(섹션 11:
        "response arrival 시점에는 보통 chunk 앞 10~12 action은 이미 과거"라는 관찰과
        일치하는 기준점)."""
        return (now_monotonic - self.observation_time_monotonic) * 1000.0

    def response_latency_ms(self) -> float:
        """참고용 - observation 캡처부터 응답 수신까지 걸린 전체 시간(추론+통신
        포함, ``inference_latency_ms`` 하나만으론 안 보이는 왕복 전체)."""
        return (self.response_received_time_monotonic - self.observation_time_monotonic) * 1000.0
