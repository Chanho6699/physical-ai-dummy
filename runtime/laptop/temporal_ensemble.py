"""Phase C-2: 여러 겹치는(overlapping) ``TimestampedActionChunk``를 절대 시간축으로 정렬해
하나의 안정적인 absolute joint target으로 합치는 timestamp-aware temporal ensembling.

# 전체 control flow에서 이 모듈의 위치 (섹션 9 요구사항 - 문서화만, 아래 단계 중 이
# 모듈이 실제로 구현/호출하는 건 "temporal ensemble" 한 칸뿐이다)

::

    multiple overlapping SmolVLA chunk predictions (TrajectoryBuffer.snapshot()/valid_chunks())
      -> time alignment + interpolation (이 모듈, _sample_action_at())
      -> temporal ensemble (이 모듈, TemporalEnsembler.compute_target())
      -> final absolute target (EnsembledTarget.action)
      -> [향후, 이번 세션 범위 밖] motion guard (velocity/acceleration/jerk limiting)
      -> [향후, 이번 세션 범위 밖] SafetyGate.evaluate()  <- 반드시 "최종 실행 target"에 적용
      -> [향후, 이번 세션 범위 밖] follower write

``EnsembledTarget``은 그 자체로 아무것도 실행하지 않는다 - Safety Gate를 전혀 모르고
(이 모듈이 import하지도 않음), follower에 아무것도 쓰지 않는다.

# chunk index 평균이 아니라 절대 시간 정렬 (섹션 2 핵심 요구사항)

여러 chunk를 합칠 때 "chunk[5]끼리 평균"처럼 index를 그대로 맞춰 쓰면 안 된다 - 각
chunk는 서로 다른 시점에 캡처된 관측에서 나왔으므로, index가 같아도 가리키는 절대
시각(``nominal_target_time``)이 다르다(Phase C-1B ``TimestampedActionChunk`` 참고). 이
모듈은 반드시 "이 chunk에서 절대시각 T에 해당하는 action이 뭔가"부터 계산한 뒤에 합친다.

# interpolation은 motor smoothing이 아니다 (섹션 2 명시적 구분)

여기서 하는 linear interpolation은 "30Hz(spacing=1/30s)로 discrete하게 샘플된 policy
trajectory에서, 정확히 T가 아닌 슬롯 사이의 값을 하나의 chunk 안에서 다시 샘플링하는 것"
(trajectory resampling)이다 - 이후 단계에서 할 "여러 절대좌표 target 사이를 매끄럽게
잇는" motor smoothing/interpolation(아직 미구현)과는 완전히 다른 목적/레이어다. 이
모듈이 만드는 건 여전히 "이 순간의 절대 목표"이지, 궤적 보간이 아니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from runtime.common.vla_contract import JOINT_ORDER, validate_joint_dict
from runtime.laptop.trajectory_chunk import TimestampedActionChunk

# 부동소수점 표현 오차 보정용 (runtime/laptop/trajectory_chunk.py의 _FLOAT_INDEX_EPSILON과
# 동일한 이유 - spacing=1/30 같은 값이 이진 소수로 정확히 표현되지 않아 슬롯 경계에서
# floor/ceil이 off-by-one을 낼 수 있다).
_FLOAT_EPSILON = 1e-9

# 향후 ensemble 가능성을 위해 TrajectoryBuffer가 보존하는 chunk가 여러 개여도, 실제로
# 한 target 계산에 섞을 chunk 수는 이 기본값으로 제한한다(요구사항: magic number로 박지
# 말고 configurable) - "3개"라는 숫자 자체는 TrajectoryBuffer.DEFAULT_MAX_CHUNKS(섹션 2,
# Phase C-1B)와 같은 근거(향후 ensemble 후보 보존용 보수적 기본값)를 그대로 재사용한다.
DEFAULT_MAX_CONTRIBUTORS = 3

# half-life 기본값의 근거: Phase B 조사에서 실측한 predict_chunk() steady-state 추론
# latency median = 338ms(이 sandbox RTX 3050, in-process) - "보통 이 정도 지나면 새 chunk가
# 도착한다"는 실제 cadence와 같은 크기로 half-life를 잡으면, 딱 한 inference 주기가 지난
# chunk는 가중치가 절반으로 줄고 그보다 오래된 chunk는 더 빠르게 무시된다는 의미가 직관과
# 맞아떨어진다(요구사항 제시 범위 0.3~0.5s 안에도 들어간다). 실제 배포 GPU의 실측 cadence가
# 다르면 이 값도 다시 맞춰야 한다 - 하드코딩된 "정답"이 아니라 이 sandbox 실측 기준 초기값.
DEFAULT_HALF_LIFE_S = 0.338

# lookahead 기본값 - "지금(now)"을 그대로 target_time으로 쓴다(0.0). network/control latency
# 보정용으로 향후 양수 값을 실험할 수 있게 configurable로만 노출한다(섹션 12) - 임의의 큰
# 값을 여기서 미리 정해두지 않는다.
DEFAULT_LOOKAHEAD_S = 0.0


@dataclass(frozen=True)
class EnsembledTarget:
    """``TemporalEnsembler.compute_target()``의 결과 - 하나의 절대시각 T에 대한 weighted
    ensemble target. 이 자체는 아직 "실행 가능한 명령"이 아니다(Safety Gate 이전 단계)."""

    target_time_monotonic: float
    action: dict[str, float]  # 6-joint 절대좌표(degree/percent) - weighted 평균 결과
    contributing_sequences: tuple[int, ...]  # 기여한 chunk들의 sequence (최신순)
    contributing_chunk_indices: tuple[tuple[int, int], ...]  # 각 기여 chunk의 (lower, upper) 샘플 index쌍
    weights: tuple[float, ...]  # contributing_sequences와 같은 순서, 합=1.0로 정규화됨
    newest_observation_age_ms: float  # target_time 기준 가장 "신선한" 기여 chunk의 관측 나이(ms)
    oldest_observation_age_ms: float  # target_time 기준 가장 "오래된" 기여 chunk의 관측 나이(ms)
    num_contributors: int
    phase_offset_s: float = 0.0


class TemporalEnsembler:
    """여러 ``TimestampedActionChunk``를 절대 시간 T 기준으로 정렬 + interpolation +
    exponential recency weighting해서 하나의 ``EnsembledTarget``을 만든다.

    Safety Gate/follower write/motion guard를 전혀 모른다(섹션 9 - 이 클래스는 "temporal
    ensemble" 한 단계만 담당).
    """

    def __init__(
        self,
        *,
        max_contributors: int = DEFAULT_MAX_CONTRIBUTORS,
        half_life_s: float = DEFAULT_HALF_LIFE_S,
        lookahead_s: float = DEFAULT_LOOKAHEAD_S,
        phase_continuity: bool = False,
        phase_fade_cadence_scale: float = 1.0,
    ) -> None:
        if max_contributors < 1:
            raise ValueError(f"max_contributors는 1 이상이어야 합니다: {max_contributors}")
        if half_life_s <= 0:
            raise ValueError(f"half_life_s는 양수여야 합니다: {half_life_s}")
        if lookahead_s < 0:
            raise ValueError(f"lookahead_s는 음수일 수 없습니다: {lookahead_s}")
        if phase_fade_cadence_scale <= 0:
            raise ValueError("phase_fade_cadence_scale must be positive")
        self._max_contributors = max_contributors
        self._half_life_s = half_life_s
        self._lambda = math.log(2.0) / half_life_s
        self._lookahead_s = lookahead_s
        self._phase_continuity = phase_continuity
        self._phase_fade_cadence_scale = phase_fade_cadence_scale

    @property
    def max_contributors(self) -> int:
        return self._max_contributors

    @property
    def half_life_s(self) -> float:
        return self._half_life_s

    @property
    def lookahead_s(self) -> float:
        return self._lookahead_s

    # -- 공개 API (섹션 13) -------------------------------------------------------------

    def compute_target(
        self,
        chunks: Sequence[TimestampedActionChunk],
        target_time_monotonic: float,
    ) -> EnsembledTarget | None:
        return self._compute_target_with_offsets(chunks, target_time_monotonic, {})

    def _compute_target_with_offsets(
        self,
        chunks: Sequence[TimestampedActionChunk],
        target_time_monotonic: float,
        offsets: dict[int, float],
    ) -> EnsembledTarget | None:
        candidates = [
            c for c in chunks
            if self._covers(c, target_time_monotonic + offsets.get(c.sequence, 0.0))
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda c: c.observation_time_monotonic, reverse=True)
        # Continuity mode retains one rolling boundary contributor. Its phase
        # envelope reaches zero before removal, avoiding a hard top-k swap.
        candidate_limit = self._max_contributors + 1 if self._phase_continuity else self._max_contributors
        candidates = candidates[:candidate_limit]

        samples: list[tuple[TimestampedActionChunk, dict[str, float], int, int, float]] = []
        for chunk in candidates:
            ok, _ = chunk.validate()
            if not ok:
                continue
            offset = offsets.get(chunk.sequence, 0.0)
            sampled, lower, upper = self._sample_action_at(chunk, target_time_monotonic + offset)
            valid_sample, _ = validate_joint_dict(sampled, context="ensemble contributor")
            if valid_sample is not None:
                samples.append((chunk, valid_sample, lower, upper, offset))
        if not samples:
            return None

        newest_obs_time = max(chunk.observation_time_monotonic for chunk, _, _, _, _ in samples)
        recency_weights = [
            math.exp(-self._lambda * (newest_obs_time - chunk.observation_time_monotonic))
            for chunk, _, _, _, _ in samples
        ]
        weights_raw = list(recency_weights)
        if self._phase_continuity and len(samples) > 1:
            observation_times = sorted({chunk.observation_time_monotonic for chunk, _, _, _, _ in samples})
            cadence_values = [
                b - a for a, b in zip(observation_times, observation_times[1:])
                if b - a > _FLOAT_EPSILON
            ]
            cadence_s = (
                sorted(cadence_values)[len(cadence_values) // 2]
                if cadence_values
                else min(chunk.chunk_index_spacing_s for chunk, _, _, _, _ in samples)
            )
            cadence_s *= self._phase_fade_cadence_scale
            newest_chunk = samples[0][0]
            admission_age_s = max(
                0.0, target_time_monotonic - newest_chunk.response_received_time_monotonic
            )
            admission_phase = min(1.0, admission_age_s / cadence_s)
            weights_raw[0] *= admission_phase
            if len(samples) == self._max_contributors + 1:
                weights_raw[-1] *= 1.0 - admission_phase
            else:
                oldest_chunk, _, _, _, oldest_offset = samples[-1]
                remaining_phase_s = max(
                    0.0,
                    oldest_chunk.nominal_target_time(oldest_chunk.chunk_size - 1)
                    - (target_time_monotonic + oldest_offset),
                )
                weights_raw[-1] *= min(1.0, remaining_phase_s / cadence_s)
        weight_sum = sum(weights_raw)
        if weight_sum <= _FLOAT_EPSILON:
            weights_raw = recency_weights
            weight_sum = sum(weights_raw)
        weights = [w / weight_sum for w in weights_raw]
        action = {
            joint: sum(w * sample[joint] for (_, sample, _, _, _), w in zip(samples, weights))
            for joint in JOINT_ORDER
        }
        valid_action, _ = validate_joint_dict(action, context="ensembled target")
        if valid_action is None:
            return None
        observation_ages_ms = [
            (target_time_monotonic - chunk.observation_time_monotonic) * 1000.0
            for chunk, _, _, _, _ in samples
        ]
        weighted_phase_offset = sum(w * offset for (_, _, _, _, offset), w in zip(samples, weights))
        return EnsembledTarget(
            target_time_monotonic=target_time_monotonic,
            action=valid_action,
            contributing_sequences=tuple(chunk.sequence for chunk, _, _, _, _ in samples),
            contributing_chunk_indices=tuple((lower, upper) for _, _, lower, upper, _ in samples),
            weights=tuple(weights),
            newest_observation_age_ms=min(observation_ages_ms),
            oldest_observation_age_ms=max(observation_ages_ms),
            num_contributors=len(samples),
            phase_offset_s=weighted_phase_offset,
        )

    def compute_target_for_now(
        self, chunks: Sequence[TimestampedActionChunk], now_monotonic: float
    ) -> EnsembledTarget | None:
        """``compute_target()``의 편의 wrapper - ``target_time = now + lookahead_s``를
        여기서 계산해준다(섹션 12). 향후 motor control loop가 매 tick ``now``만 넘기고
        lookahead 적용은 이 ensembler 설정에 맡기고 싶을 때 쓴다 - 이번 세션에서는 아직
        어떤 control loop도 이 메서드를 호출하지 않는다(구현 금지 범위)."""
        return self.compute_target(chunks, now_monotonic + self._lookahead_s)

    # -- 내부 헬퍼 --------------------------------------------------------------------

    @staticmethod
    def _covers(chunk: TimestampedActionChunk, target_time_monotonic: float) -> bool:
        """이 chunk가 절대시각 T를 실제 샘플된 범위 안에서 커버하는지 (섹션 3 조건:
        ``observation_time <= T <= chunk_end_time``). 여기서 ``chunk_end_time``은
        ``TimestampedActionChunk.horizon_end_time_monotonic``(마지막 index "다음", 아직
        실제로 샘플되지 않은 경계)이 아니라 **마지막으로 실제 샘플이 존재하는 시각**
        (``nominal_target_time(chunk_size-1)``)이다 - interpolation은 upper index가 실제로
        존재해야 하므로 이 경계가 맞다. (``TrajectoryBuffer.is_expired()``가 쓰는 "now
        기준 남은 미래 index" 개념과는 별개 - 이 메서드는 항상 T 하나만 기준으로 그 chunk
        자체의 실제 데이터 범위 안인지만 본다.)"""
        if chunk.chunk_size <= 0 or chunk.chunk_index_spacing_s <= 0:
            return False
        last_sample_time = chunk.nominal_target_time(chunk.chunk_size - 1)
        return (chunk.observation_time_monotonic - _FLOAT_EPSILON) <= target_time_monotonic <= (last_sample_time + _FLOAT_EPSILON)

    @staticmethod
    def _sample_action_at(
        chunk: TimestampedActionChunk, target_time_monotonic: float
    ) -> tuple[dict[str, float], int, int]:
        """chunk 내부에서 절대시각 T에 해당하는 action을 linear interpolation으로 샘플링한다
        (섹션 2). ``_covers()``가 True인 chunk에서만 호출된다는 전제 - 그래도 index는
        방어적으로 ``[0, chunk_size-1]``로 clamp한다."""
        raw = (target_time_monotonic - chunk.observation_time_monotonic) / chunk.chunk_index_spacing_s
        nearest = round(raw)
        if abs(raw - nearest) < _FLOAT_EPSILON:
            raw = float(nearest)  # 슬롯 경계 부동소수점 오차 보정 (trajectory_chunk.py와 동일 원칙)

        lower = max(0, min(int(math.floor(raw)), chunk.chunk_size - 1))
        upper = max(0, min(int(math.ceil(raw)), chunk.chunk_size - 1))

        if lower == upper:
            return dict(chunk.actions[lower]), lower, upper

        fraction = raw - lower
        action_lower = chunk.actions[lower]
        action_upper = chunk.actions[upper]
        sampled = {
            joint: action_lower[joint] + fraction * (action_upper[joint] - action_lower[joint])
            for joint in JOINT_ORDER
        }
        return sampled, lower, upper


# ---------------------------------------------------------------------------
# TODO (섹션 8/9 - 이번 세션에서 구현하지 않음, 향후 검토 항목만 명시)
# ---------------------------------------------------------------------------
#
# 1. Gripper 전용 전략 (섹션 8): gripper는 지금 다른 5개 arm 관절과 완전히 동일한 weighted
#    average를 적용받는다. gripper는 사실상 discrete(열림/닫힘) 성격이 강해서, 두 기여자가
#    "닫는 중"과 "연 상태"처럼 크게 다른 값을 보고할 때 weighted average가 "어중간하게 반쯤
#    닫힌" 값을 만들어낼 수 있다 - 실제 그리퍼 동작으로는 부자연스러울 수 있다. thresholding/
#    hysteresis(예: "많이 닫힌 쪽으로 편향" 또는 "가장 최근 값만 신뢰") 같은 별도 전략이
#    필요할 수 있으나, 이번 단계에서는 구현하지 않는다(요구사항).
#
# 2. Outlier contributor 오염 가능성 (섹션 9): 지금은 순수 weighted average라서, flow-matching
#    stochastic noise 때문에 유독 튄 contributor 하나가 (특히 가중치가 비슷한 다른 기여자와
#    섞일 때) 최종 target을 왜곡시킬 수 있다. 예: 3개 기여자 중 하나가 다른 둘과 5deg 이상
#    떨어진 값을 예측했다면, 그 하나가 (weight가 완전히 무시할 수준으로 작지 않은 이상)
#    평균을 그 방향으로 끌고 간다. 향후 옵션으로 검토할 만한 것:
#      - contributor 간 median/MAD 기반 outlier prefilter(가중치 계산 전에 극단값 제거)
#      - robust weighted median(단순 평균 대신)
#      - per-joint 편차가 threshold를 넘는 contributor는 아예 배제
#    이번 세션에서는 구현하지 않는다 - Safety Gate가 최종적으로 excessive-step을 걸러주는
#    안전망이 있다는 것과는 별개로, ensemble 자체의 품질 문제로 남겨둔다.
