"""Desktop SmolVLA FastAPI 서버 <-> Laptop VLA HTTP client 공용 계약.

# 조사 결과 (이번 세션에서 확인한 근거)

`~/Projects/physical-ai-dummy` 저장소 자체에는 `/health`/`/session/reset`/`/predict`/
`/action/ack`를 구현한 코드가 전혀 없었다 (``grep -rn`` 결과 0건 - 최종 보고서 참고).
`VLA`라는 단어가 등장하는 유일한 곳은 ``hardware/diagnostics/instrumented_teleop.py``의
"나중에 VLA deadband/threshold 설계 시 다시 봐야 한다"는 주석 하나뿐이었다.

인접 저장소 ``~/Projects/physical-ai-recycling-cell-with-sim``에는 ``/predict``/``/health``
계약을 쓰는 VLA 백엔드 설정(``configs/vla_backend_smolvla_config.json`` 등)이 있었지만,
그 계약의 ``action_schema``는 ``delta_ee_7dof``(엔드이펙터 delta, Franka Panda 임베디먼트
가정)이다. 이 저장소(physical-ai-dummy)의 실제 25-episode 데이터셋
(``data/so101_cube_train_v6/meta/info.json``)은 6-dim
``[shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos,
gripper.pos]`` **절대 관절 위치**(도/퍼센트, delta 아님 - 아래 "action semantics" 참고)를
쓴다. 두 프로젝트는 로봇 임베디먼트와 action space 자체가 다르므로, 그 계약을 그대로
가져오면 오히려 "새로운(그리고 틀린) mapping을 임의로 만드는" 것이 된다 - 그래서 여기서는
엔드포인트 이름(``/health``, ``/predict`` 등)과 JSON envelope 스타일만 참고하고,
``action_schema``는 이 저장소의 실측 데이터셋 근거로 새로 정의했다.

# action semantics (섹션 7 근거)

``data/so101_cube_train_v6/meta/stats.json``에서 ``action``과 ``observation.state``의
min/max/mean이 관절별로 거의 동일한 범위를 가진다 (예: shoulder_lift action
[-98.59, 48.92], observation.state [-98.77, 49.36]). LeRobot SO-101 레코딩 관례
(리더의 현재 위치를 그대로 그 프레임의 ``action``으로 기록 - `~/lerobot/src/lerobot`의
so_follower/so_leader 관례, `simulation/mujoco/action_mapping.py` 모듈 docstring도 동일하게
확인)상 이는 **절대 목표 관절 위치**이지 delta가 아니다. 따라서 SmolVLA가 이 데이터셋으로
파인튜닝됐다면 반환하는 action도 절대 목표 위치(같은 단위: 몸통 5관절=degree,
gripper=percent_0_100)로 취급한다 - 새로 정규화/변환을 발명하지 않는다.

# 카메라 매핑 (섹션 5 근거)

``configs/hardware.local.json``의 카메라 key(``workspace``/``wrist``)가 데이터셋 feature
이름(``observation.images.workspace``/``observation.images.wrist``, 위 info.json)과
접두사만 다르고 나머지 이름이 동일하다 - 이 파일은 그 접두사만 붙여서 그대로 wire key로
쓴다 (``camera_fixed`` 같은 새 이름을 만들지 않는다).
"""

from __future__ import annotations

import math

# 이 저장소 여러 모듈(so101_model.SO101_JOINT_NAMES, remote_state_client.JOINT_NAMES,
# readonly_so101_reader.JOINT_ORDER, so101_control_profile.JOINT_NAMES)이 각자 독립적으로
# 이 6개 관절 이름/순서를 정의해 두는 기존 관례를 그대로 따라 여기서도 새로 정의한다.
JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRIPPER_JOINT = "gripper"

# data/so101_cube_train_v6/meta/info.json의 observation.images.* feature 이름 그대로.
CAMERA_WORKSPACE_KEY = "observation.images.workspace"
CAMERA_WRIST_KEY = "observation.images.wrist"
CAMERA_KEYS: tuple[str, ...] = (CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY)

# info.json: observation.images.* shape = [480, 640, 3] (height, width, channels), HWC.
EXPECTED_IMAGE_HEIGHT = 480
EXPECTED_IMAGE_WIDTH = 640
EXPECTED_IMAGE_CHANNELS = 3

HEALTH_PATH = "/health"
SESSION_RESET_PATH = "/session/reset"
PREDICT_PATH = "/predict"
ACTION_ACK_PATH = "/action/ack"

SCHEMA_VERSION = "shadow_vla_contract_v1"
BACKEND_SMOLVLA = "smolvla"
BACKEND_FAKE = "fake"


def validate_joint_dict(raw: object, *, context: str) -> tuple[dict[str, float] | None, str | None]:
    """{joint: number} 6개가 ``JOINT_ORDER``와 정확히 일치하는지 검증한다 (state/action 공용).

    NaN/Inf/타입 오류/누락/여분 키를 모두 거부 사유로 구분해 반환한다. 예외를 던지지
    않는다 - 호출자(ObservationBuilder/ActionAdapter/SafetyGate)가 무효 상태를 안전하게
    판단할 수 있게 하기 위함이다 (``remote_state_client.py``의 defensive parsing 패턴 재사용).
    """
    if not isinstance(raw, dict):
        return None, f"{context}가 object(dict)가 아닙니다: {type(raw)!r}"

    missing = [j for j in JOINT_ORDER if j not in raw]
    if missing:
        return None, f"{context}에 관절이 누락되었습니다: {missing}"

    extra = [k for k in raw if k not in JOINT_ORDER]
    if extra:
        return None, f"{context}에 알 수 없는 관절 키가 있습니다: {extra}"

    result: dict[str, float] = {}
    for joint in JOINT_ORDER:
        value = raw[joint]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f"{context}['{joint}']가 숫자가 아닙니다: {value!r}"
        fvalue = float(value)
        if math.isnan(fvalue):
            return None, f"{context}['{joint}']가 NaN입니다."
        if math.isinf(fvalue):
            return None, f"{context}['{joint}']가 Inf입니다."
        result[joint] = fvalue
    return result, None
