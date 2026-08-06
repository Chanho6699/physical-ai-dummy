"""SO-101 MuJoCo 모델 로딩.

모델 출처와 라이선스는 `simulation/mujoco/assets/ATTRIBUTION.md`에 기록되어 있다.
관절 이름, 관절 range, actuator ctrlrange는 이 모듈에서 새로 정의하지 않고
로딩된 MuJoCo 모델(XML)에서 직접 읽어온다. 값을 여기서 재정의하면 XML과 값이
어긋날 위험이 있기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_SCENE_PATH = ASSETS_DIR / "scene.xml"

# so101.xml / scene.xml에 정의된 실제 관절/actuator 이름 순서.
# LeRobot 데이터셋의 action feature 이름(접미사 ".pos" 제외)과 동일하다.
# 출처: simulation/mujoco/assets/robotstudio_so101/so101.xml (joint/actuator 선언 순서)
SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# scene.xml에서 실제 하드웨어 테이블을 대리하는 평면 geom 이름.
TABLE_GEOM_NAME = "table_surface"


class SO101ModelError(RuntimeError):
    """MuJoCo 모델 로딩 또는 조회 중 발생한 오류."""


@dataclass(frozen=True)
class JointLimits:
    """MuJoCo 모델에서 직접 읽은 관절/actuator 제한값 (라디안)."""

    joint_name: str
    joint_range: tuple[float, float]
    actuator_name: str | None
    actuator_ctrlrange: tuple[float, float] | None


def load_model(scene_path: str | Path | None = None) -> mujoco.MjModel:
    """MuJoCo 씬(XML)을 로딩한다.

    Args:
        scene_path: 씬 XML 경로. 지정하지 않으면 벤더링된 SO-101 기본 씬을 사용한다.

    Raises:
        SO101ModelError: 파일이 없거나 MuJoCo가 XML 파싱에 실패한 경우.
    """
    path = Path(scene_path) if scene_path is not None else DEFAULT_SCENE_PATH
    if not path.is_file():
        raise SO101ModelError(f"MuJoCo 씬 파일을 찾을 수 없습니다: {path}")
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # mujoco raises plain ValueError on XML errors
        raise SO101ModelError(f"MuJoCo 모델 로딩 실패 ({path}): {exc}") from exc


def make_data(model: mujoco.MjModel) -> mujoco.MjData:
    """모델에 대응하는 새 MjData(시뮬레이션 상태)를 만든다."""
    return mujoco.MjData(model)


def get_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"<unnamed_joint_{i}>"
        for i in range(model.njnt)
    ]


def get_actuator_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"<unnamed_actuator_{i}>"
        for i in range(model.nu)
    ]


def get_joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise SO101ModelError(f"MuJoCo 모델에 '{joint_name}' 관절이 없습니다.")
    return joint_id


def get_actuator_id(model: mujoco.MjModel, actuator_name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise SO101ModelError(f"MuJoCo 모델에 '{actuator_name}' actuator가 없습니다.")
    return actuator_id


def get_geom_id(model: mujoco.MjModel, geom_name: str) -> int | None:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    return geom_id if geom_id >= 0 else None


def get_joint_limits(model: mujoco.MjModel, joint_names: tuple[str, ...] = SO101_JOINT_NAMES) -> dict[str, JointLimits]:
    """joint_names에 대해 모델에서 직접 읽은 range/ctrlrange를 반환한다.

    관절과 actuator는 1:1로 이름이 같다고 가정하지 않고, actuator가 없으면
    actuator_name/actuator_ctrlrange는 None으로 표시한다.
    """
    limits: dict[str, JointLimits] = {}
    actuator_names = set(get_actuator_names(model))
    for name in joint_names:
        joint_id = get_joint_id(model, name)
        joint_range = (float(model.jnt_range[joint_id][0]), float(model.jnt_range[joint_id][1]))

        actuator_name = name if name in actuator_names else None
        ctrlrange = None
        if actuator_name is not None:
            actuator_id = get_actuator_id(model, actuator_name)
            ctrlrange = (
                float(model.actuator_ctrlrange[actuator_id][0]),
                float(model.actuator_ctrlrange[actuator_id][1]),
            )

        limits[name] = JointLimits(
            joint_name=name,
            joint_range=joint_range,
            actuator_name=actuator_name,
            actuator_ctrlrange=ctrlrange,
        )
    return limits
