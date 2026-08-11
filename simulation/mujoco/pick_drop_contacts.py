"""cube-vs-gripper contact 판정 (Track B/physics 전용) - Primary/Secondary rollout이 공유한다.

``simulation/mujoco/safety_checks.py``의 self-collision 판정과 같은 원칙을 따른다: 이름이 없는
mesh collision geom(``docs/mujoco_action_replay.md`` §11.4에 이미 기록된 한계 - 벤더링된 MJCF의
gripper collision mesh 다수가 이름 없음)은 쓰지 않고, ``scene_pick_drop.xml``에 실제로 이름이
붙어 있는 jaw geom(``fixed_jaw_*``/``moving_jaw_*``)만 대상으로 한다. 이름 없는 mesh geom까지
포함하면 오탐(false positive)이 늘어 physics 판정의 신뢰도가 더 떨어진다는 것이 이미 그 문서에서
확인된 사실이다.
"""

from __future__ import annotations

import mujoco

JAW_GEOM_PREFIXES = ("fixed_jaw_", "moving_jaw_")


def get_jaw_geom_ids(model: mujoco.MjModel) -> set[int]:
    ids: set[int] = set()
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and name.startswith(JAW_GEOM_PREFIXES):
            ids.add(i)
    return ids


def get_cube_geom_id(model: mujoco.MjModel) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    if geom_id < 0:
        raise ValueError("모델에 'cube_geom'이 없습니다 - scene_pick_drop.xml을 로딩했는지 확인하세요.")
    return geom_id


def cube_jaw_contact_active(model: mujoco.MjModel, data: mujoco.MjData, *, cube_geom_id: int, jaw_geom_ids: set[int]) -> bool:
    """이번 step(``mj_step`` 직후)에 cube_geom과 jaw geom 중 하나라도 접촉 중이면 True."""
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if (g1 == cube_geom_id and g2 in jaw_geom_ids) or (g2 == cube_geom_id and g1 in jaw_geom_ids):
            return True
    return False
