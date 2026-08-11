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

# gripper tiny-negative normalization (2026-08 실물 Stage 3 조사 근거)

실물 Stage 3(``docs/real_follower_staged_safety_test.md``)에서 10 step 중 9 step까지
전 관절 ACCEPT/write 성공 후, 마지막 step에서 gripper raw action이 ``-0.0253772736``으로
나와 Safety Gate가 ``MECHANICAL_LIMIT_CLAMPED``(``WOULD_CLAMP``)를 반환했고, 그 결과
gripper 외 5개 arm 관절(전혀 문제없던 값)까지 포함한 전체 action이 write되지 못하고
stage가 중단됐다.

이 세션에서 조사한 근거(요약 - 전체는 별도 조사 보고서 참고):
    - ``data/so101_cube_pick_drop_start_coverage_v3_clean``,
      ``data/so101_cube_pick_drop_generalization_v4``,
      ``data/so101_cube_pick_drop_v3_v4_combined69_v1`` 세 dataset의 gripper
      ``action``/``observation.state`` 원본에는 음수가 0건이다 (min은 ``action`` 기준
      약 0.71). 학습 라벨은 정확히 0에도 도달한 적이 없다.
    - Candidate B 체크포인트(``outputs/pick_drop_v3_v4_combined69/.../checkpoints/010000/
      pretrained_model``)의 ``config.json``은 ``normalization_mapping.ACTION == "MEAN_STD"``다
      (MIN_MAX가 아님) - unnormalize(``x = z*std + mean``)가 [0,100] 안에 구조적으로
      갇히지 않으므로, 0 근처에서 z-score 공간의 미세한 흔들림만으로도 tiny negative가
      나올 수 있다.
    - ``reports/mujoco_full_rollout_candidate_comparison_v1/trajectories/`` (Candidate B,
      raw prediction 22,229 step)를 집계하면 음수 예측의 median은 약 -0.29이고, 이번
      실물 사례(-0.025)급의 극소 음수는 음수 전체 중 소수(약 7%)에 불과하다 - 즉
      "-0.025 같은 극소 음수"와 "-0.3급 음수"는 같은 population이 아니다.

이 근거로, gripper 채널에 한해 **아주 작은** epsilon(``GRIPPER_TINY_NEGATIVE_EPSILON``)
이내의 음수만 0으로 정규화한다 - RANGE_0_100 관례상 0은 이미 "완전히 닫힘"이고 서보가
그보다 더 닫을 방법이 없으므로, epsilon 이내의 음수는 물리적으로 0과 구분 불가능한
numerical noise로 취급하는 것이 타당하다. epsilon보다 큰 음수(예: -0.3)는 정규화하지
않고 그대로 Safety Gate에 넘긴다 - Safety Gate의 ``MECHANICAL_LIMIT_CLAMPED`` 판정
능력을 우회하지 않기 위함이다 (모든 음수를 무조건 ``max(0, gripper)``로 clamp하는 방식은
Safety Gate를 사실상 무력화하므로 채택하지 않았다). 다른 5개 arm 관절에는 이 normalization을
전혀 적용하지 않는다 - arm 관절의 음수는 gripper와 달리 "이미 닫힘"과 동일한 물리적 의미를
갖지 않으므로 같은 근거가 성립하지 않는다. Safety Gate(``runtime/laptop/safety_gate.py``)의
mechanical limit/excessive step/gross reject 로직은 이 변경과 무관하게 그대로 유지된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.common.vla_contract import GRIPPER_JOINT, JOINT_ORDER, validate_joint_dict

# gripper raw action이 이 값 이내의 음수(즉 [-EPSILON, 0.0) 구간)면 0.0으로 정규화한다.
# -EPSILON보다 작은(더 음수인) 값은 절대 건드리지 않고 그대로 Safety Gate로 넘긴다.
#
# 근거(모듈 docstring "gripper tiny-negative normalization" 절 참고): 실측된 두 anchor
# (-0.025 = numerical noise로 보는 게 타당한 사례, -0.3 = Candidate B negative prediction의
# median에 해당하는, tolerance로 삼키면 안 되는 사례) 사이에는 약 10배 차이가 있고,
# 학습 gripper action min(~0.71) 대비로도 0.05는 ~7% 수준의 작은 마진이다. 조사에서 근거
# 있는 epsilon 후보 범위는 0.05~0.1이었으며, 이번 최초 구현은 그중 더 보수적인(더 적게
# 정규화하는) 하한값 0.05를 사용한다 - 필요시 실물 로그를 더 모은 뒤에만 조정한다.
GRIPPER_TINY_NEGATIVE_EPSILON = 0.05


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

    단위 변환 없음 (모듈 docstring 참고) - 검증만 하고 그대로 통과시킨다. 유일한 예외는
    gripper의 tiny-negative normalization(``GRIPPER_TINY_NEGATIVE_EPSILON`` 참고) -
    [-epsilon, 0.0) 구간의 gripper 값만 0.0으로 정규화하고, 그 외에는 값을 바꾸지 않는다.
    """
    validated, reason = validate_joint_dict(raw_action, context="action")
    if validated is None:
        return AdaptedAction(valid=False, command_deg={}, invalid_reason=reason)

    warnings: list[str] = []
    gripper = validated[GRIPPER_JOINT]

    # gripper tiny-negative normalization (모듈 docstring 근거) - 이 관절에만, 이 범위에만
    # 적용한다. arm 5관절은 절대 이 정규화를 거치지 않는다.
    if -GRIPPER_TINY_NEGATIVE_EPSILON <= gripper < 0.0:
        gripper = 0.0
        validated = dict(validated)
        validated[GRIPPER_JOINT] = gripper

    if gripper < -50.0 or gripper > 150.0:
        # percent_0_100이어야 할 값이 이 범위조차 벗어나면 단위 자체가 잘못됐을 가능성이
        # 높다는 진단 경고 - REJECT 여부는 Safety Gate가 결정한다 (이 adapter는 판단하지 않음).
        warnings.append(f"gripper 값이 percent_0_100 범위에서 크게 벗어났습니다: {gripper}")

    return AdaptedAction(valid=True, command_deg=validated, warnings=tuple(warnings))
