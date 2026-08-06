"""데이터셋 action <-> MuJoCo joint/actuator 매핑.

단위 변환 근거 (추측이 아니라 소스 확인 결과):

1. 데이터셋 action feature 이름은 `shoulder_pan.pos`, `shoulder_lift.pos`, ... 형태이며
   (`data/so101_cube_xy_train_v1/meta/info.json`의 `features.action.names`), 이는 LeRobot의
   `SOFollower` 로봇 클래스가 각 모터의 `.pos` 값을 그대로 기록한 것이다.
   (~/lerobot/src/lerobot/robots/so_follower/so_follower.py)

2. `SOFollowerConfig.use_degrees` 기본값은 `True`이다.
   (~/lerobot/src/lerobot/robots/so_follower/config_so_follower.py:42)
   `use_degrees=True`이면 모터 정규화 모드가 `MotorNormMode.DEGREES`로 설정된다.
   (~/lerobot/src/lerobot/robots/so_follower/so_follower.py:50)

3. `MotorNormMode.DEGREES`의 변환식은 다음과 같다 (~/lerobot/src/lerobot/motors/motors_bus.py:874):
       normalized = (raw_ticks - calibration_mid) * 360 / (motor_resolution - 1)
   즉 각 모터의 "0"은 캘리브레이션 시 정해진 중앙값이고, 단위는 degree이다.
   => 데이터셋 action 값의 단위는 **degree**이며, MuJoCo 모델(so101.xml)의 관절 range/actuator
   ctrlrange는 **radian**이다. 따라서 scale = pi/180 (deg -> rad) 이 필요하다.

4. 위 사실은 실제 데이터로도 교차 확인했다: `data/so101_cube_xy_train_v1` episode 0~3의
   action 값 범위(예: shoulder_lift -98.59 ~ 55.25)가 MuJoCo 모델의 관절 range를 degree로
   환산한 값(shoulder_lift: -100.0 ~ 100.0)과 같은 자릿수/부호로 일치했다. (docs 참고)

5. **[확인되지 않음]** 데이터셋을 기록한 실물 로봇의 캘리브레이션 "0" 위치가 이 MuJoCo
   모델(robotstudio_so101, mujoco_menagerie)이 가정하는 "0" 위치(각 관절의 기구학적 중립 자세)와
   정확히 같은지는 검증하지 못했다. 방향 부호(sign)도 마찬가지로 실물에서 직접 검증한 적이
   없다. 이번 구현에서는 sign=+1 (부호 반전 없음)을 기본값으로 사용하며, 이는 "매핑 이름이
   같으면 부호도 같다"는 상식적 가정일 뿐 하드웨어로 확인된 값이 아니다.
   실제로 wrist_flex 관절은 데이터셋 전 에피소드에서 모델의 range(±95.0도)를 최대 약 2도
   초과하는 값이 반복 관측되었는데, 이는 이 가정과 정확한 range/캘리브레이션 차이를 보여주는
   사례이다 (docs/mujoco_action_replay.md 참고). 이 replay 도구가 이런 사례를 BLOCKED로
   잡아내는 것이 이번 1단계 안전 검증의 목적이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco

from simulation.mujoco.so101_model import SO101_JOINT_NAMES, get_actuator_id, get_joint_id

DEG2RAD = math.pi / 180.0


class ActionMappingError(RuntimeError):
    """action mapping 구성 또는 검증 실패."""


@dataclass(frozen=True)
class JointMapping:
    """dataset feature 1개 <-> MuJoCo joint/actuator 1개 매핑.

    scale/offset/sign은 다음 순서로 적용된다:
        mujoco_target = (dataset_value * scale + offset) * sign
    """

    dataset_feature_name: str  # 예: "shoulder_pan.pos" (info.json에 기록된 원래 이름)
    dataset_name: str  # 예: "shoulder_pan" (".pos" 제거, MuJoCo 이름과 비교용)
    dataset_index: int  # action 배열에서의 열 인덱스
    mujoco_joint_name: str
    mujoco_actuator_name: str
    unit: str  # 데이터셋 원본 값의 단위 (예: "deg")
    scale: float  # deg -> rad 등 단위 변환 계수
    offset: float
    sign: float
    basis: str  # 이 scale/offset/sign 값의 근거 (모듈 docstring 요약 참조)


def strip_pos_suffix(feature_name: str) -> str:
    return feature_name.removesuffix(".pos")


# 데이터셋 action feature 이름 -> (MuJoCo joint 이름, MuJoCo actuator 이름) 기본 매핑표.
# 이름이 서로 같다는 사실은 so101.xml의 joint/actuator 선언과 info.json의 feature 이름을
# 직접 대조해 확인했다 (simulation/mujoco/so101_model.py의 SO101_JOINT_NAMES 참고).
_DEFAULT_TARGETS: dict[str, tuple[str, str]] = {name: (name, name) for name in SO101_JOINT_NAMES}

_BASIS_TEXT = (
    "SOFollowerConfig.use_degrees=True 기본값 + MotorNormMode.DEGREES 변환식 "
    "(lerobot/motors/motors_bus.py) => 단위=degree. scale=pi/180으로 deg->rad 변환. "
    "offset=0 (모터 정규화 중앙값이 이미 0). sign=+1은 미검증 가정 "
    "(action_mapping.py 모듈 docstring 5번 항목 참고)."
)


def build_default_mapping(
    dataset_action_names: list[str],
) -> tuple[JointMapping, ...]:
    """데이터셋 info.json의 action feature 이름 순서를 기준으로 매핑을 만든다.

    Args:
        dataset_action_names: info.json의 features.action.names (예: ["shoulder_pan.pos", ...]).

    Raises:
        ActionMappingError: 알 수 없는 관절 이름이 있거나, 중복된 이름이 있는 경우.
    """
    stripped = [strip_pos_suffix(name) for name in dataset_action_names]

    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for name in stripped:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            duplicates.append(name)
    if duplicates:
        raise ActionMappingError(
            f"데이터셋 action feature 이름에 중복이 있습니다: {sorted(set(duplicates))}"
        )

    unknown = [name for name in stripped if name not in _DEFAULT_TARGETS]
    if unknown:
        raise ActionMappingError(
            f"알 수 없는 관절 이름입니다 (기본 매핑표에 없음): {unknown}. "
            f"지원되는 이름: {sorted(_DEFAULT_TARGETS)}"
        )

    mappings = []
    for index, (feature_name, name) in enumerate(zip(dataset_action_names, stripped)):
        joint_name, actuator_name = _DEFAULT_TARGETS[name]
        mappings.append(
            JointMapping(
                dataset_feature_name=feature_name,
                dataset_name=name,
                dataset_index=index,
                mujoco_joint_name=joint_name,
                mujoco_actuator_name=actuator_name,
                unit="deg",
                scale=DEG2RAD,
                offset=0.0,
                sign=1.0,
                basis=_BASIS_TEXT,
            )
        )
    return tuple(mappings)


def validate_mapping_against_model(
    mapping: tuple[JointMapping, ...], model: mujoco.MjModel
) -> None:
    """매핑에 등장하는 모든 joint/actuator가 실제 모델에 존재하는지 확인한다.

    Raises:
        ActionMappingError: joint 또는 actuator가 모델에 없는 경우.
    """
    for entry in mapping:
        try:
            get_joint_id(model, entry.mujoco_joint_name)
        except Exception as exc:
            raise ActionMappingError(
                f"'{entry.dataset_name}' -> MuJoCo joint '{entry.mujoco_joint_name}' 매핑 실패: {exc}"
            ) from exc
        try:
            get_actuator_id(model, entry.mujoco_actuator_name)
        except Exception as exc:
            raise ActionMappingError(
                f"'{entry.dataset_name}' -> MuJoCo actuator '{entry.mujoco_actuator_name}' 매핑 실패: {exc}"
            ) from exc


def map_action_value(value: float, entry: JointMapping) -> float:
    """단일 값을 dataset 단위에서 MuJoCo(radian) 단위로 변환한다."""
    return (value * entry.scale + entry.offset) * entry.sign


def map_action_row(action_row, mapping: tuple[JointMapping, ...]) -> dict[str, float]:
    """action 배열의 한 프레임(row)을 {actuator_name: target_rad} 딕셔너리로 변환한다."""
    return {
        entry.mujoco_actuator_name: map_action_value(float(action_row[entry.dataset_index]), entry)
        for entry in mapping
    }
