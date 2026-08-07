"""VLA raw action -> SO-101 command semantics.

# action semantics (근거 - 섹션 7)

``data/so101_cube_train_v6/meta/stats.json``에서 ``action``과 ``observation.state``의
min/max/mean이 관절별로 거의 동일한 범위를 갖는다 (예: shoulder_lift action
[-98.59, 48.92] vs observation.state [-98.77, 49.36]). LeRobot SO-101 레코딩 관례상
(``simulation/mujoco/action_mapping.py`` 모듈 docstring이 이미 동일하게 확인한 바)
데이터셋의 ``action``은 **그 프레임에서 팔로워가 즉시 향해야 할 절대 목표 관절 위치**다 -
델타(delta)도, 정규화([-1,1] 등)된 값도 아니다. 단위는 몸통 5관절=degree,
gripper=percent_0_100 (``observation.state``와 동일 - ``readonly_so101_reader.py``의
gripper RANGE_0_100 관례).

따라서 SmolVLA가 이 데이터셋으로 파인튜닝되어 있다면(``hardware/state_server`` 조사와
동일하게, 이 코드는 그 가정을 추측이 아니라 이 근거로 명시한다) 반환되는 action도 같은
단위의 절대 목표 위치다. 이 Action Adapter는 **단위 변환을 하지 않는다** - 그런 변환은
"새로운 mapping을 임의로 만드는" 것이며, 이미 근거로 확인된 사실(단위 동일)과 모순된다
(섹션 7 요구사항: 기존 Action Adapter가 있으면 재사용, 없으면 근거 기반으로만 만들 것).
이 저장소에는 사전에 구현된 Desktop VLA Action Adapter가 없었다 (최종 보고서 참고) -
이 모듈이 그 근거를 바탕으로 새로 만든 최소 adapter다.

이 Adapter가 하는 일은 다음 세 가지뿐이다:
    1. 6개 관절 dict 형태/타입 검증 (``JOINT_ORDER``와 정확히 일치).
    2. NaN/Inf 거부.
    3. gripper 값이 0~100 percent 범위의 "숫자"라는 최소 sanity(진짜 clamp/hold 판단은
       Safety Gate가 한다 - 이 모듈은 판단하지 않는다).

향후 backend(baseline MuJoCo/realistic MuJoCo/real SO-101) 전환 시에도 이 adapter의
출력(같은 단위의 6개 관절 dict)은 그대로 재사용된다 - backend별 변환을 새로 만들지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.common.vla_contract import GRIPPER_JOINT, JOINT_ORDER, validate_joint_dict


class ActionAdapterError(RuntimeError):
    """adapter 구성 오류 (입력 자체는 예외를 던지지 않고 AdaptedAction.valid로 표현한다)."""


@dataclass(frozen=True)
class AdaptedAction:
    valid: bool
    command_deg: dict[str, float]  # 몸통 5관절=degree, gripper=percent_0_100 (SO-101 command semantics)
    invalid_reason: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def adapt_vla_action(raw_action: object) -> AdaptedAction:
    """VLA ``/predict`` 응답의 ``action``을 SO-101 command dict로 변환한다.

    단위 변환 없음 (모듈 docstring 참고) - 검증만 하고 그대로 통과시킨다.
    """
    validated, reason = validate_joint_dict(raw_action, context="action")
    if validated is None:
        return AdaptedAction(valid=False, command_deg={}, invalid_reason=reason)

    warnings: list[str] = []
    gripper = validated[GRIPPER_JOINT]
    if gripper < -50.0 or gripper > 150.0:
        # percent_0_100이어야 할 값이 이 범위조차 벗어나면 단위 자체가 잘못됐을 가능성이
        # 높다는 진단 경고 - REJECT 여부는 Safety Gate가 결정한다 (이 adapter는 판단하지 않음).
        warnings.append(f"gripper 값이 percent_0_100 범위에서 크게 벗어났습니다: {gripper}")

    return AdaptedAction(valid=True, command_deg=validated, warnings=tuple(warnings))
