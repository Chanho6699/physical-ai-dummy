"""Phase C-1B: ``AsyncVLAChunkInferenceWorker``가 매 inference마다 필요로 하는 "fresh
observation 한 벌"을 얻기 위한 최소 read-only 인터페이스.

# 왜 기존 CameraSourceProtocol/StateSourceProtocol/build_observation을 재사용하지 않았는가

``runtime/laptop/staged_real_rollout.py``는 ``CameraSourceProtocol.capture_all()``과
``StateSourceProtocol.read()``를 따로 호출한 뒤 ``observation_builder.build_observation()``으로
합쳐서 Safety Gate용 스키마 검증까지 겸한다 - 그 파이프라인은 "Safety Gate에 넘길 수 있는
관측인가"까지 판단하는 무거운 계약이고, 이 모듈의 목적과 다르다. 이 worker는 Safety Gate를
전혀 모르고(이 세션 범위 밖) 그냥 "이미지+state+task를 한 번에, timestamp와 함께" 얻고 싶을
뿐이다 - 그래서 별도의 가벼운 단일 ``capture()`` 계약을 새로 정의했다(기존 걸 억지로
재사용하지 않는다 - 서로 다른 관심사를 섞지 않기 위함).

실제 하드웨어 구현(진짜 카메라+진짜 follower state)은 이 세션 범위 밖이다 - 여기서는
인터페이스와 테스트용 Fake만 만든다. 향후 실제 runtime에서는 이 Protocol을 만족하는
클래스가 ``runtime/laptop/camera_source.py``/``follower_state_source.py``를 내부적으로
호출해 만들어질 것이다(아직 만들지 않음).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ObservationSnapshot:
    """한 번의 fresh 관측 - image/state/task를 한 시점에 묶어서 캡처한 것."""

    images: dict[str, object]  # CAMERA_KEYS -> HWC uint8 RGB numpy 배열 (VLAHttpClient.predict_chunk와 동일 계약)
    state: dict[str, float]  # JOINT_ORDER 6개 - follower 현재 state
    task: str
    capture_monotonic_time: float  # 이 관측을 "캡처했다"고 볼 수 있는 monotonic 시각
    sequence: int  # 이 캡처를 요청한 worker의 sequence (그대로 echo)


class ObservationSnapshotProvider(Protocol):
    """``AsyncVLAChunkInferenceWorker``가 의존하는 유일한 관측 인터페이스. 실제 하드웨어를
    하드코딩하지 않기 위한 추상화(섹션 4 요구사항) - 구현체는 이 세션에서 Fake뿐이다."""

    def capture(self, *, sequence: int) -> ObservationSnapshot: ...
