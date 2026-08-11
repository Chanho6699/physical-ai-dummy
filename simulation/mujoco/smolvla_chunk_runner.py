"""SmolVLA checkpoint 로딩 + chunk 단위 추론 - Primary/Secondary rollout 트랙이 공유하는 유일한
추론 경로.

로딩 순서(``get_policy_class`` -> ``from_pretrained`` -> ``make_pre_post_processors`` ->
``preprocessor(batch)`` -> ``policy.predict_action_chunk(processed)`` -> ``postprocessor(...)``)는
새로 만든 것이 아니다 - ``runtime/desktop/vla_server.py::SmolVLAPolicyRunner._load``,
``scripts/evaluate_smolvla_midpoint.py::load_policy_bundle``,
``scripts/sweep_grid35_first_action_seed.py::load_policy_bundle``/``run_seed_sweep_with_bundle``에서
이미 세 번 독립적으로 검증된 것과 동일한 패턴을 그대로 따른다.

두 가지 호출 방식을 제공한다 (둘 다 같은 로딩된 policy/preprocessor/postprocessor를 공유):

- ``predict_chunk()``: 항상 새로 chunk를 샘플링한다 (내부 큐를 쓰지 않음). Primary track이
  chunk 경계마다 "실제 기록된 그 시점의 real observation"으로 새 chunk를 뽑아 그 전체를
  MuJoCo에서 물리적으로 재생할 때 쓴다.
- ``next_queued_action()``: SmolVLA의 실제 배포 동작(``policy.select_action()``이 내부에서 하는
  것과 동일하게 - ``runtime/desktop/vla_server.py``의 ``SmolVLAPolicyRunner.predict`` 참고)을
  그대로 재현한다 - 큐가 비어 있을 때만 새로 chunk를 샘플링하고, 아니면 큐에서 다음 action을
  꺼낸다. Secondary(synthetic closed-loop) track이 "매 step마다 predict를 부른다"는 실제
  배포 시맨틱을 그대로 재현하면서도, 이 모듈에서는 (내부 구현인 LeRobot ``select_action()``과
  달리) chunk 경계 시점을 밖에서 관측할 수 있게 ``chunk_boundary`` bool을 함께 반환한다 -
  진단/로깅 목적으로만 다르고 반환되는 action 자체의 의미는 동일하다.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from runtime.common.vla_contract import JOINT_ORDER


class ChunkRunnerError(RuntimeError):
    """checkpoint 로딩 실패 (preflight에서 즉시 드러나야 한다)."""


@dataclass
class ChunkPredictResult:
    chunk_deg: list[dict[str, float]]  # 길이 chunk_size, 각 원소는 JOINT_ORDER dict (deg/percent)
    chunk_size: int
    seed: int | None


def _images_to_batch(images: dict[str, np.ndarray], device, torch) -> dict[str, object]:
    """numpy HWC uint8(or float 0-1) RGB -> policy 입력 tensor. 카메라 키는 rename하지 않고
    원본 이름 그대로 넣는다 (``InProcessSmolVLAClient``와 동일 원칙 - 저장된 preprocessor가
    카메라 rename을 처리한다)."""
    out: dict[str, object] = {}
    for key, arr in images.items():
        img = torch.from_numpy(np.ascontiguousarray(arr))
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        else:
            img = img.float()
        img = img.permute(2, 0, 1).contiguous()
        out[key] = img.unsqueeze(0).to(device)
    return out


class SmolVLAChunkRunner:
    def __init__(self, checkpoint_dir: str, *, policy_type: str = "smolvla", device: str | None = None) -> None:
        self.checkpoint_dir = str(checkpoint_dir)
        self._queue: collections.deque[dict[str, float]] = collections.deque()
        self._chunk_count = 0
        try:
            import torch
            from lerobot.policies import get_policy_class, make_pre_post_processors

            self._torch = torch
            policy_cls = get_policy_class(policy_type)
            self.policy = policy_cls.from_pretrained(self.checkpoint_dir)
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.policy.to(self.device)
            self.policy.eval()
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config, pretrained_path=self.checkpoint_dir
            )
        except Exception as exc:  # noqa: BLE001 - checkpoint/의존성 문제를 한 종류의 예외로 통일
            raise ChunkRunnerError(f"SmolVLA checkpoint 로딩 실패 ({checkpoint_dir}): {exc}") from exc

        action_dim = self.policy.config.action_feature.shape[0]
        if action_dim != len(JOINT_ORDER):
            raise ChunkRunnerError(
                f"{checkpoint_dir}: action dim={action_dim}, 기대값={len(JOINT_ORDER)} - "
                "이 checkpoint는 6-dim SO-101 action schema와 다를 수 있습니다."
            )

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    def close(self) -> None:
        import gc

        del self.policy
        del self.preprocessor
        del self.postprocessor
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def reset(self, *, task: str | None = None) -> None:
        self.policy.reset()
        self._queue.clear()
        self._chunk_count = 0

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)

    def _predict_chunk_rows(
        self, *, state_deg: dict[str, float], images: dict[str, np.ndarray], task: str, seed: int | None
    ) -> np.ndarray:
        torch = self._torch
        state_arr = np.array([state_deg[j] for j in JOINT_ORDER], dtype=np.float32)
        batch: dict[str, object] = {"observation.state": torch.from_numpy(state_arr).unsqueeze(0).to(self.device)}
        batch.update(_images_to_batch(images, self.device, torch))
        batch["task"] = [task]

        with torch.inference_mode():
            processed = self.preprocessor(batch)
            self._set_seed(seed)
            raw_chunk = self.policy.predict_action_chunk(processed)  # (1, chunk_size, action_dim)
            postproc_chunk = self.postprocessor(raw_chunk)
        chunk = postproc_chunk.detach().to("cpu").numpy()[0]  # (chunk_size, action_dim)
        return chunk

    def predict_chunk(
        self, *, state_deg: dict[str, float], images: dict[str, np.ndarray], task: str, seed: int | None = None
    ) -> ChunkPredictResult:
        """항상 새 chunk를 샘플링한다 (내부 큐 무시). Primary track 전용."""
        chunk = self._predict_chunk_rows(state_deg=state_deg, images=images, task=task, seed=seed)
        self._chunk_count += 1
        rows = [{j: float(chunk[i, k]) for k, j in enumerate(JOINT_ORDER)} for i in range(chunk.shape[0])]
        return ChunkPredictResult(chunk_deg=rows, chunk_size=len(rows), seed=seed)

    def next_queued_action(
        self, *, state_deg: dict[str, float], images: dict[str, np.ndarray], task: str, seed: int | None = None
    ) -> tuple[dict[str, float], bool]:
        """실제 배포(``select_action`` 내부 큐)와 동일한 시맨틱 - 큐가 비었을 때만 재추론한다.
        Secondary(synthetic closed-loop) track 전용."""
        chunk_boundary = False
        if not self._queue:
            chunk = self._predict_chunk_rows(state_deg=state_deg, images=images, task=task, seed=seed)
            self._chunk_count += 1
            for i in range(chunk.shape[0]):
                self._queue.append({j: float(chunk[i, k]) for k, j in enumerate(JOINT_ORDER)})
            chunk_boundary = True
        return self._queue.popleft(), chunk_boundary
