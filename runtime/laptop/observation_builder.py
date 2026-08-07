"""실제 카메라 프레임 + 실제 follower state -> VLA 요청용 observation.

학습 데이터셋 schema(``data/so101_cube_train_v6/meta/info.json``)와의 대응 근거:

    observation.state             <- follower_state_source.FollowerStateSnapshot.positions_deg
                                      (JOINT_ORDER 6개, 몸통 5관절=degree, gripper=percent_0_100 -
                                      readonly_so101_reader.py가 이미 학습 데이터와 동일한 LeRobot
                                      정규화 관례를 재현했음을 확인했다)
    observation.images.workspace  <- camera_source의 "workspace" 카메라 (RGB, HWC, uint8)
    observation.images.wrist      <- camera_source의 "wrist" 카메라 (RGB, HWC, uint8)

dtype/채널 순서(RGB, not BGR)/HWC는 info.json의 ``video.pix_fmt: yuv420p``(RGB 계열
color space, OpenCV BGR을 여기서 RGB로 변환 - ``camera_source.py`` 참고)와 shape
``[480, 640, 3]``에 맞춘 것이다. 정확한 스케일(0~255 uint8 vs 0~1 float) 변환은 이 모듈이
하지 않는다 - Desktop 서버의 policy backend가 체크포인트 preprocessor에서 처리한다
(섹션 4: "정규화 위치"는 policy 쪽).

schema가 확실하지 않으면(예: 해상도가 크게 다르거나 관절이 누락되면) inference로
넘어가지 않고 명확한 invalid 사유를 반환한다 (섹션 4 요구사항).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.common.vla_contract import (
    CAMERA_KEYS,
    EXPECTED_IMAGE_CHANNELS,
    JOINT_ORDER,
)
from runtime.laptop.camera_source import CameraFrame
from runtime.laptop.follower_state_source import FollowerStateSnapshot

# 완전히 다른 해상도(예: 카메라 오설정)는 명백한 schema 불일치로 보고 REJECT한다.
# 정확히 480x640이 아니어도(리사이즈로 흡수 가능한 범위) 통과시키되, 채널 수/HWC는 엄격히 검사한다.
MIN_REASONABLE_DIM = 32


@dataclass(frozen=True)
class BuiltObservation:
    state: dict[str, float]
    images: dict[str, "object"]  # wire_key -> HWC uint8 RGB numpy array
    schema_valid: bool
    fixed_camera_valid: bool
    wrist_camera_valid: bool
    state_valid: bool
    invalid_reasons: list[str] = field(default_factory=list)


def build_observation(
    *,
    state_snapshot: FollowerStateSnapshot,
    camera_frames: dict[str, CameraFrame],
) -> BuiltObservation:
    reasons: list[str] = []

    state_valid = True
    missing_joints = [j for j in JOINT_ORDER if j not in state_snapshot.positions_deg]
    if missing_joints:
        state_valid = False
        reasons.append(f"observation.state에 관절이 누락되었습니다: {missing_joints}")
    else:
        import math

        non_finite = [j for j in JOINT_ORDER if not math.isfinite(state_snapshot.positions_deg[j])]
        if non_finite:
            state_valid = False
            reasons.append(f"observation.state에 NaN/Inf 관절이 있습니다: {non_finite}")

    camera_valid: dict[str, bool] = {}
    images: dict[str, object] = {}
    for wire_key in CAMERA_KEYS:
        frame = camera_frames.get(wire_key)
        if frame is None:
            camera_valid[wire_key] = False
            reasons.append(f"{wire_key} 프레임이 없습니다.")
            continue
        arr = frame.image_rgb
        ok = True
        if getattr(arr, "ndim", None) != 3 or arr.shape[2] != EXPECTED_IMAGE_CHANNELS:
            ok = False
            reasons.append(f"{wire_key} 이미지 shape이 올바르지 않습니다: {getattr(arr, 'shape', None)}")
        elif arr.shape[0] < MIN_REASONABLE_DIM or arr.shape[1] < MIN_REASONABLE_DIM:
            ok = False
            reasons.append(f"{wire_key} 이미지 해상도가 비정상적으로 작습니다: {arr.shape}")
        elif str(arr.dtype) != "uint8":
            ok = False
            reasons.append(f"{wire_key} 이미지 dtype이 uint8이 아닙니다: {arr.dtype}")
        camera_valid[wire_key] = ok
        if ok:
            images[wire_key] = arr

    fixed_ok = camera_valid.get("observation.images.workspace", False)
    wrist_ok = camera_valid.get("observation.images.wrist", False)
    schema_valid = state_valid and fixed_ok and wrist_ok

    return BuiltObservation(
        state=dict(state_snapshot.positions_deg) if state_valid else {},
        images=images,
        schema_valid=schema_valid,
        fixed_camera_valid=fixed_ok,
        wrist_camera_valid=wrist_ok,
        state_valid=state_valid,
        invalid_reasons=reasons,
    )
