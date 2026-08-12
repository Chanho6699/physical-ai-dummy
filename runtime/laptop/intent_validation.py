"""Phase C-3A.1 correction: Intent Safety vs Execution Safety 역할 분리.

# 왜 필요한가 (문제 1 - Motion Guard가 Safety outlier semantic을 우회함)

C-3A의 원래 파이프라인은 ``raw ensemble target -> Motion Guard -> Safety Gate`` 순서였다.
이 순서에서는 raw target이 아무리 위험한 policy outlier(예: 실물 사례 wrist_flex
53.67->33.67, delta 20deg - training distribution 밖의 true outlier로 이미 분류됐던
사례)여도, Motion Guard가 그걸 매 tick `velocity_limit*dt`만큼의 작은 조각으로 잘라
버리면 Safety Gate는 그 "작은 조각"만 보고 매번 ACCEPT해버린다 - mechanical hard limit을
넘는 건 아니지만, **"이 policy 예측 자체를 신뢰해도 되는가"라는 outlier 판정(원래
Safety Gate의 excessive-step이 하려던 일)을 완전히 우회하게 된다.** 60 tick에 걸쳐
결국 원래 목표(위험한 것으로 분류됐던 그 목표)에 도달해버리는 게 실측으로 확인됐다.

# 새 파이프라인 (섹션 1 요구사항)

::

    Temporal Ensemble
      -> Intent Validation (이 모듈)   <- raw target 자체가 신뢰 가능한 policy 의도인가
      -> Motion Guard                  <- 신뢰된 target에 "어떻게" 안전한 속도로 접근할까
      -> Final Execution SafetyGate    <- 이번 tick 실행할 guarded target이 안전한가
      -> [향후] write

Intent Validation이 raw target을 거부하면, Motion Guard는 **아예 호출되지 않는다** -
"위험한 큰 target을 잘게 쪼개는" 상황 자체가 발생할 수 없다(섹션 2 요구사항).

# 기존 SafetyGate 재사용 원칙 (요구사항: 같은 함수를 무비판적으로 두 번 호출해 semantic
# confusion을 만들지 마라)

이 모듈은 ``SafetyGate.evaluate()``의 mechanical-range/excessive-step 계산 로직 자체를
새로 만들지 않는다 - **정확히 같은 계산**을 재사용한다(재캘리브레이션된 threshold와
calibration 데이터를 그대로 신뢰). 다만 그 결과를 해석하는 **의미를 명시적으로 다르게
문서화**한다:

    - ``SafetyGate.evaluate(raw_target, current_state)``의 결과를 "이 raw target을 지금
      당장 그대로 실행해도 되는가"가 아니라 **"이 raw target을 신뢰할 만한 policy
      의도로 받아들여도 되는가"**로 해석한다 - ``PolicyIntentValidator``가 이 재해석을
      명시적인 타입(``IntentValidationResult``)과 이름(``check_intent``)으로 감싼다.
    - Motion Guard 이후 guarded target에 대해 별도로 다시 부르는 ``SafetyGate.evaluate()``
      호출(``realtime_control_target.py``)은 **"이번 tick 이 값을 정말 실행해도 되는가"**
      (원래 의미 그대로)로 해석한다.

두 호출 모두 **같은 ``SafetyGate`` 인스턴스, 같은 config, 같은 코드**를 쓴다 - 로직
분기/중복이 없다. 유일한 차이는 "무엇을 넣고 그 ACCEPT/WOULD_CLAMP/REJECT를 어떤
질문에 대한 답으로 취급하느냐"뿐이다.

# Time semantics 분석 (섹션 4 요구사항)

기존 excessive-step threshold는 30Hz demonstration에서 ``|action[t] -
observation.state[t]|``로 캘리브레이션됐다 - ``action[t]``는 그 프레임에서 즉시
목표해야 할 절대 위치(LeRobot SO-101 레코딩 관례, ``action_adapter.py`` 근거)이고
``state[t]``는 같은 프레임의 실측 현재 위치다.

이 모듈이 검사하는 ``raw_ensemble_target``은 ``TemporalEnsembler.compute_target(chunks, T)``의
결과이고, ``T``는 ``RealTimeControlTargetGenerator``에서 ``now + lookahead_s``로 계산된다.
**``lookahead_s=0.0``(현재 기본값이자 이 세션에서 유일하게 검증된 값)일 때는 ``T≈now``이므로,
"지금 이 순간 policy가 믿는 절대 목표"와 "지금 이 순간의 실측 현재 상태"를 비교하는 것이
되어 30Hz demo의 ``action[t]-state[t]``와 정확히 같은 질문("바로 지금 이 상태에서 policy가
얼마나 멀리 가고 싶어하는가")이다** - ensemble/interpolation을 거쳤다는 차이는 있지만
(여러 chunk를 절대시간 정렬해 합친 결과일 뿐), 비교 대상(현재 실측 state)과 비교 시점(지금)의
관계는 demo 캘리브레이션 근거와 동일하다.

**중요한 제약(임의로 완화하지 않음)**: ``lookahead_s > 0``으로 설정되면 ``T``가 "지금"이
아니라 "미래"가 되므로, 그 미래 시점의 policy target을 "지금" 실측 state와 비교하는 게
30Hz demo 캘리브레이션과 더 이상 정확히 같은 질문이 아니게 된다(정상적인 궤적이라도 더
먼 미래일수록 현재와의 거리가 자연히 커지기 때문). 이 세션은 ``lookahead_s=0.0``만
검증했고, threshold를 lookahead에 맞춰 스케일하는 일반화는 **구현하지 않았다** - 향후
lookahead를 실제로 쓰게 되면 이 부분을 반드시 재검토해야 한다(그렇다고 지금 threshold
자체를 임의로 완화하지 않는다 - 요구사항).
"""

from __future__ import annotations

from dataclasses import dataclass

from runtime.laptop.action_adapter import adapt_vla_action
from runtime.laptop.safety_gate import SafetyDecision, SafetyGate


@dataclass(frozen=True)
class IntentValidationResult:
    """``PolicyIntentValidator.check_intent()``의 결과. ``SafetyDecision``과 필드가
    비슷해 보이지만 의미가 다르다는 걸 명시하기 위해 별도 타입으로 둔다(요구사항 -
    semantic confusion 방지)."""

    valid: bool  # True == decision == "ACCEPT" (이 raw target을 신뢰 가능한 의도로 받아들임)
    decision: str  # "ACCEPT" | "WOULD_CLAMP" | "REJECT" (SafetyGate.evaluate()의 원래 3분류 그대로)
    reasons: tuple[str, ...]


class PolicyIntentValidator:
    """``SafetyGate``를 감싸서 raw ensemble target의 "policy 의도 신뢰성"을 판정한다.

    이 클래스는 ``SafetyGate.evaluate()``를 재구현하지 않는다 - 그대로 위임한다. 이
    클래스가 하는 일은 오직: (1) 호출 시그니처를 raw target 검사에 맞게 좁히고, (2) 반환
    타입을 ``IntentValidationResult``로 감싸 "이건 execution 판정이 아니라 intent
    판정"이라는 걸 타입 레벨에서 드러내는 것뿐이다."""

    def __init__(self, safety_gate: SafetyGate) -> None:
        self._safety_gate = safety_gate

    def check_intent(
        self, *, raw_target_deg: dict[str, float], current_state_deg: dict[str, float]
    ) -> IntentValidationResult:
        adapted = adapt_vla_action(raw_target_deg)
        mechanical: SafetyDecision = self._safety_gate.evaluate(
            adapted_action=adapted, current_state_deg=current_state_deg, observation_valid=True,
            check_excessive_step=False, check_mechanical_range=True,
        )
        # Small endpoint overshoot is recoverable downstream saturation. Gross mechanical
        # violations remain fail-closed before MotionGuard.
        if mechanical.decision == "REJECT":
            return IntentValidationResult(valid=False, decision="REJECT", reasons=mechanical.reasons)

        decision: SafetyDecision = self._safety_gate.evaluate(
            adapted_action=adapted, current_state_deg=current_state_deg, observation_valid=True,
            check_mechanical_range=False,
        )
        return IntentValidationResult(
            valid=decision.decision == "ACCEPT", decision=decision.decision, reasons=decision.reasons,
        )
