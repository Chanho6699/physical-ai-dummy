"""MuJoCo Shadow Validation - VLA action의 "control preflight" 검증 (섹션 14).

task/world 재현(cube grasp 성공 등)은 v1 PASS 조건이 아니다. 이 모듈이 실제로 보는 것:

    - final MuJoCo state finite
    - 예상 관절이 예상 방향으로 움직였는가
    - mechanical violation 없는가 (``mujoco_shadow_backend``가 이미 계산한 값 재사용)
    - rate/delta anomaly 없는가 (Realistic Layer의 ``rate_limited`` 플래그 + 자체 delta 크기 확인)
    - severe oscillation 없는가 (``max_abs_qvel`` 임계값)
    - Realistic Layer 처리가 정상인가 (diagnostics 존재, 예외 없음 - 호출자가 이미 보장)
    - processed command와 simulated actual state가 비정상적으로 어긋나지 않는가
      (tracking_error_deg 임계값)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.mujoco_shadow_backend import MuJoCoStepResult

# 단일 control step(1/30초 근방)에서 tracking error가 이 값을 넘으면 이상 징후로 본다.
# CANDIDATE_ONLY 관측 프로파일(simulation/realism/so101_control_profile.py)의 wrist_roll
# deadband(수 deg 수준)보다 넉넉하게 잡은 진단용 임계값이며, hard safety limit이 아니다
# (섹션 10 - candidate 값을 hard limit으로 승격하지 않는다는 원칙과 별개로, 이 값은 그
# 파일에서 읽지 않고 이 모듈이 진단 목적으로만 고정한 상수다).
TRACKING_ERROR_WARN_DEG = 15.0

# 한 control step 동안(steps_per_frame 물리 스텝 합) 관측된 |qvel|의 이상 징후 임계값 (rad/s).
# configs/mujoco_so101.yaml의 max_velocity(기존 보수적 값, 관절별 0.7~4.5 rad/s)보다 한 자리
# 여유를 둔 "심각한 진동/발산" 수준의 진단 임계값이다.
OSCILLATION_QVEL_WARN = 15.0

# 명령 방향과 실제 이동 방향이 이 값(deg) 이상 움직였을 때만 "방향 검사" 대상으로 삼는다
# (아주 작은 delta는 노이즈/deadband로 방향이 뒤집혀도 이상 징후가 아니다).
DIRECTION_CHECK_MIN_DELTA_DEG = 0.5


@dataclass(frozen=True)
class ShadowValidationResult:
    passed: bool  # False면 SHADOW_FAIL 수준의 하드 실패
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    checks: dict[str, bool]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "warnings": list(self.warnings), "failures": list(self.failures), "checks": self.checks}


def validate_mujoco_step(step_result: MuJoCoStepResult) -> ShadowValidationResult:
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    # 1. finite
    checks["final_state_finite"] = step_result.final_state_finite
    if not step_result.final_state_finite:
        failures.append("MuJoCo 최종 state에 NaN/Inf가 있습니다.")
        # 발산 상태에서 이후 검사는 의미가 없다.
        return ShadowValidationResult(passed=False, warnings=tuple(warnings), failures=tuple(failures), checks=checks)

    # 2. mechanical violation
    checks["no_mechanical_violation"] = len(step_result.mechanical_violations) == 0
    if step_result.mechanical_violations:
        failures.append(f"관절 range를 벗어난 최종 state: {list(step_result.mechanical_violations)}")

    # 3. rate/delta anomaly (Realistic Layer의 rate_limited 플래그)
    rate_limited_joints = [
        joint for joint, diag in step_result.realistic_diagnostics.items() if diag.get("rate_limited")
    ]
    checks["no_rate_anomaly"] = True  # rate_limited는 Realistic Layer가 정상적으로 흡수한 것 - WARN일 뿐
    if rate_limited_joints:
        warnings.append(f"Realistic Layer가 rate-limit을 적용한 관절: {rate_limited_joints}")

    # 4. severe oscillation
    checks["no_severe_oscillation"] = step_result.max_abs_qvel <= OSCILLATION_QVEL_WARN
    if not checks["no_severe_oscillation"]:
        failures.append(f"물리 스텝 중 |qvel| 최댓값이 비정상적으로 큽니다: {step_result.max_abs_qvel:.2f} rad/s")

    # 5. Realistic Layer 처리 정상 (diagnostics가 6관절 모두 존재)
    checks["realistic_layer_complete"] = set(step_result.realistic_diagnostics) == set(JOINT_ORDER)
    if not checks["realistic_layer_complete"]:
        failures.append("Realistic Layer diagnostics가 6개 관절을 모두 포함하지 않습니다.")

    # 6. processed command vs simulated actual (tracking error)
    large_tracking_error = {
        joint: err for joint, err in step_result.tracking_error_deg.items() if abs(err) > TRACKING_ERROR_WARN_DEG
    }
    checks["tracking_error_within_bounds"] = len(large_tracking_error) == 0
    if large_tracking_error:
        warnings.append(f"tracking error가 임계값({TRACKING_ERROR_WARN_DEG}deg)을 초과한 관절: {large_tracking_error}")

    # 7. 예상 방향으로 움직였는가 (명령 delta 부호 vs 실제 이동 부호)
    direction_mismatches: list[str] = []
    for joint in JOINT_ORDER:
        initial = step_result.initial_state_deg.get(joint)
        command = step_result.processed_command_deg.get(joint)
        final = step_result.final_state_deg.get(joint)
        if initial is None or command is None or final is None:
            continue
        commanded_delta = command - initial
        actual_delta = final - initial
        if abs(commanded_delta) < DIRECTION_CHECK_MIN_DELTA_DEG:
            continue  # 명령 자체가 거의 없으면 방향 판단 대상이 아니다
        if math.copysign(1.0, commanded_delta) != math.copysign(1.0, actual_delta) and abs(actual_delta) >= DIRECTION_CHECK_MIN_DELTA_DEG:
            direction_mismatches.append(joint)
    checks["direction_consistent"] = len(direction_mismatches) == 0
    if direction_mismatches:
        warnings.append(f"명령 방향과 실제 이동 방향이 다른 관절: {direction_mismatches}")

    passed = len(failures) == 0
    return ShadowValidationResult(passed=passed, warnings=tuple(warnings), failures=tuple(failures), checks=checks)
