#!/usr/bin/env python3
"""Grid35 SmolVLA held-out(offline) checkpoint evaluator - ``data/so101_cube_xy_midpoint_test10_v1`` 기준.

# 조사 결과 (이 스크립트가 재사용하는 기존 경로)

이번 grid35 실험(``data/so101_cube_xy_grid35_v1``로 학습한
``outputs/grid35/smolvla_grid35_fresh_v1``)은 20-episode pilot 실험(``data/so101_cube_train_v6``,
``outputs/pilot``)과 완전히 독립적이다 - 이 스크립트는 어떤 pilot 체크포인트/데이터셋도
건드리지 않고, ``--checkpoints``/``--eval-dataset``/``--train-dataset`` 인자로 명시된
grid35 전용 경로만 다룬다.

- **SmolVLA 추론 경로**: ``runtime/desktop/vla_server.py``의 ``SmolVLAPolicyRunner``가 이미
  ``lerobot.policies.get_policy_class("smolvla").from_pretrained(checkpoint)`` +
  ``lerobot.policies.make_pre_post_processors(policy.config, pretrained_path=checkpoint)`` +
  ``preprocessor(batch) -> policy.select_action(...) -> postprocessor(...)`` 순서로 추론한다.
  이 스크립트는 그 순서를 그대로 재사용한다(``load_policy_bundle``/``build_inference_batch``/
  ``run_checkpoint`` 참고) - 정규화/역정규화를 다시 구현하지 않는다.
- **카메라 rename**: ``runtime/common/vla_contract.py``의 ``CAMERA_KEYS``
  (``observation.images.workspace``/``observation.images.wrist``)를 그대로 배치에 넣는다.
  이번 grid35 체크포인트는 학습 시 ``--rename_map``으로
  ``workspace -> camera1``, ``wrist -> camera2``를 저장된 preprocessor 파이프라인
  (``policy_preprocessor.json``의 ``rename_observations_processor`` 단계)에 이미 구운 상태다 -
  ``vla_server.py``처럼 원본 데이터셋 키 이름으로만 배치를 채우면 저장된 preprocessor가 알아서
  rename한다. 이 스크립트가 별도로 ``observation.images.camera1``을 만들지 않는 이유가 이것이다
  (``validate_checkpoint_camera_mapping``이 체크포인트마다 이 가정을 실제로 검증한다 - 추측하지
  않는다).
- **action/state semantics**: ``runtime/laptop/action_adapter.py`` 모듈 docstring 근거대로
  action/state는 절대 목표 관절 위치(몸통 5관절=degree, gripper=percent_0_100)이고, 단위 변환을
  하지 않는다. ``LeRobotDataset``이 반환하는 ``action``/``observation.state`` 텐서도 같은 단위의
  raw 값이다(정규화는 policy의 preprocessor 내부에서만 일어난다).
- **Safety Gate 재사용**: ``runtime/laptop/safety_gate.py``의 ``SafetyGate``/``SafetyGateConfig``는
  순수 계산(파일 I/O만 하고 하드웨어/네트워크 write가 전혀 없음) - ``adapt_vla_action`` +
  ``SafetyGate.evaluate()``를 그대로 호출해 ACCEPT/WOULD_CLAMP/REJECT 판정을 "실제 write 없이"
  재사용한다. (2026-08 Grid35 Safety 감사 이후) ``SafetyGateConfig.from_repo_defaults()``는
  더 이상 mujoco 임포트 체인이 필요 없다 - excessive-step 임계값이 mujoco 전용 진단 도구
  설정(``configs/mujoco_so101.yaml``)이 아니라 Safety Gate 전용 ``configs/safety_gate.yaml``에서
  온다. 그래도 calibration/fallback 로딩 자체가 실패할 수는 있으므로(``SafetyGateConfigError``),
  그 경우 이 부분만 자동으로 비활성화하고 나머지 metric은 정상 계산한다(``try_build_safety_gate``
  참고) - 요구사항 8번 "신뢰성 있게 재사용 가능한 경우에만 적용"을 그대로 구현한 것이다.

# 이 스크립트가 하지 않는 것

- 실물 SO-101/MuJoCo에 어떤 write도 하지 않는다 (순수 오프라인 평가).
- 기존 pilot 체크포인트(1k/5k/10k/15k/20k)나 ``so101_cube_train_v6``를 로딩/평가하지 않는다.
- held-out ``midpoint_test10`` 데이터를 학습에 쓰지 않는다 (``--eval-dataset``과
  ``--train-dataset``이 같은 경로면 즉시 중단한다 - ``main()`` 참고).

사용 예::

    source ~/lerobot/.venv/bin/activate
    python scripts/evaluate_smolvla_midpoint.py --seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.common.checkpoint_provenance import (  # noqa: E402
    check_train_dataset_provenance as _shared_check_train_dataset_provenance,
    infer_checkpoint_step,
    validate_checkpoint_camera_mapping as _shared_validate_checkpoint_camera_mapping,
)
from runtime.common.vla_contract import (  # noqa: E402
    CAMERA_WORKSPACE_KEY,
    CAMERA_WRIST_KEY,
    JOINT_ORDER,
)
from runtime.laptop.action_adapter import adapt_vla_action  # noqa: E402

try:
    from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig, SafetyGateConfigError

    _SAFETY_GATE_IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 - mujoco 등 optional dependency 부재를 폭넓게 흡수
    SafetyGate = None  # type: ignore[assignment,misc]
    SafetyGateConfig = None  # type: ignore[assignment,misc]
    SafetyGateConfigError = Exception  # type: ignore[assignment,misc]
    _SAFETY_GATE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

logger = logging.getLogger("evaluate_smolvla_midpoint")

# ---------------------------------------------------------------------------
# 기본값 (요구사항의 경로/설정을 그대로 옮김)
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_DATASET = PROJECT_ROOT / "data" / "so101_cube_xy_grid35_v1"
DEFAULT_EVAL_DATASET = PROJECT_ROOT / "data" / "so101_cube_xy_midpoint_test10_v1"
DEFAULT_EVAL_REPO_ID = "local/so101_cube_xy_midpoint_test10_v1"

_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "grid35" / "smolvla_grid35_fresh_v1" / "checkpoints"
DEFAULT_CHECKPOINTS: tuple[Path, ...] = tuple(
    _CHECKPOINT_ROOT / step / "pretrained_model" for step in ("002500", "005000", "007500", "010000")
)

DEFAULT_TASK = "Pick up the cube and place it in the target area."
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "grid35_midpoint_eval"
DEFAULT_SEED = 42

# 학습 시 실제로 쓴 rename_map(문제 설명 "SmolVLA mapping" 근거) - 체크포인트마다 저장된
# policy_preprocessor.json이 정말 이 매핑을 담고 있는지 검증할 때 기준값으로 쓴다.
EXPECTED_CAMERA_RENAME_MAP: dict[str, str] = {
    CAMERA_WORKSPACE_KEY: "observation.images.camera1",
    CAMERA_WRIST_KEY: "observation.images.camera2",
}


# ---------------------------------------------------------------------------
# 순수 계산 유틸 (GPU/lerobot 없이도 단위 테스트 가능)
# ---------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """선형보간 percentile - numpy 없이 표준 라이브러리만으로 계산한다."""
    if not values:
        raise ValueError("빈 리스트의 percentile은 정의되지 않습니다.")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    k = (len(ordered) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def error_stats(values: list[float]) -> dict[str, float | int | None]:
    """median/p95/max - 요구사항 7번의 delta/gt-delta 통계 공용 계산."""
    if not values:
        return {"median": None, "p95": None, "max": None, "n": 0}
    return {
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": float(max(values)),
        "n": len(values),
    }


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def select_frame_local_indices(length: int, max_frames: int | None) -> list[int]:
    """에피소드 길이 ``length`` 안에서 평가할 로컬 frame index 목록을 고른다.

    ``max_frames`` 미지정(None) 또는 ``length`` 이상이면 전체 프레임을 그대로 쓴다(기본값 -
    요구사항 9번 "골고루 포함"을 가장 확실하게 만족). ``max_frames``를 준 경우에는 앞부분만
    자르지 않고(요구사항 9번이 명시적으로 금지) 에피소드 전체에 걸쳐 균등 간격으로 샘플링한다 -
    OOM 등으로 계산량을 줄여야 할 때만 쓰는 fallback(요구사항 "테스트" 항목)이다.
    """
    if length <= 0:
        return []
    if max_frames is None or max_frames >= length:
        return list(range(length))
    if max_frames <= 1:
        return [0]
    step = (length - 1) / (max_frames - 1)
    idxs = sorted({round(i * step) for i in range(max_frames)})
    return idxs


@dataclass(frozen=True)
class FramePlan:
    """한 episode에서 평가할 (전역 dataset row index) 목록 - 항상 시간순으로 증가한다."""

    episode_index: int
    dataset_indices: tuple[int, ...]


def build_frame_plans(episodes_meta: list[dict], *, max_frames_per_episode: int | None = None) -> list[FramePlan]:
    """``dataset.meta.episodes``(또는 그와 동일한 필드를 가진 dict 목록)에서 FramePlan을 만든다.

    모든 episode를 episode_index 순서로 포함한다(요구사항 9번 "test10 전체 episode가
    골고루" - episode를 건너뛰지 않는다).
    """
    plans: list[FramePlan] = []
    for ep in sorted(episodes_meta, key=lambda e: int(e["episode_index"])):
        length = int(ep["length"])
        base = int(ep["dataset_from_index"])
        local = select_frame_local_indices(length, max_frames_per_episode)
        plans.append(FramePlan(episode_index=int(ep["episode_index"]), dataset_indices=tuple(base + i for i in local)))
    return plans


# checkpoint step 추론/학습 dataset 출처 확인/카메라 rename 검증은 Shadow Mode
# (runtime/laptop/shadow_mode_runner.py)와 공유한다 - 두 곳이 각자 다른 방식으로 "이
# checkpoint가 정말 이 실험의 산출물인가"를 판단하면 결과가 어긋날 수 있기 때문이다
# (runtime/common/checkpoint_provenance.py 참고).
infer_step_from_path = infer_checkpoint_step


def round_floats(obj, ndigits: int = 6):
    """JSON 출력 크기를 줄이기 위해 float만 반올림한다(구조는 그대로 유지)."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def tensor_to_joint_dict(tensor) -> dict[str, float]:
    """(6,) 또는 (1,6) 텐서 -> ``JOINT_ORDER`` dict. 차원이 6이 아니면 명확히 실패한다.

    ``runtime/desktop/vla_server.py``의 ``SmolVLAPolicyRunner.predict``와 동일한 원칙 -
    schema가 확실하지 않으면 추측으로 채우지 않고 예외를 던진다.
    """
    flat = tensor.detach().to("cpu").reshape(-1)
    if flat.shape[0] != len(JOINT_ORDER):
        raise ValueError(f"action/state 텐서 차원이 {len(JOINT_ORDER)}이 아닙니다: shape={tuple(flat.shape)}")
    return {joint: float(flat[i]) for i, joint in enumerate(JOINT_ORDER)}


# ---------------------------------------------------------------------------
# Safety Gate 재사용 (요구사항 8번 - 신뢰성 있게 재사용 가능한 경우에만)
# ---------------------------------------------------------------------------


@dataclass
class SafetyTally:
    accept: int = 0
    would_clamp: int = 0
    reject: int = 0
    joint_clamp_count: dict[str, int] = field(default_factory=lambda: dict.fromkeys(JOINT_ORDER, 0))
    joint_reject_count: dict[str, int] = field(default_factory=lambda: dict.fromkeys(JOINT_ORDER, 0))

    def record(self, decision: str, per_joint: dict) -> None:
        if decision == "ACCEPT":
            self.accept += 1
        elif decision == "WOULD_CLAMP":
            self.would_clamp += 1
        elif decision == "REJECT":
            self.reject += 1
        else:  # pragma: no cover - SafetyGate가 이 값들만 반환함을 신뢰
            raise ValueError(f"알 수 없는 SafetyGate decision: {decision}")
        for joint, report in per_joint.items():
            if report.clamped:
                self.joint_clamp_count[joint] += 1
            if report.rejected:
                self.joint_reject_count[joint] += 1

    def to_dict(self) -> dict:
        return {
            "WOULD_PASS": self.accept,
            "WOULD_CLAMP": self.would_clamp,
            "WOULD_REJECT": self.reject,
            "joint_clamp_count": dict(self.joint_clamp_count),
            "joint_reject_count": dict(self.joint_reject_count),
        }


def try_build_safety_gate() -> tuple["SafetyGate | None", str | None]:
    """가능하면 ``SafetyGate``를 만들고, 아니면 (None, 이유)를 반환한다 - 절대 예외를 던지지 않는다.

    호출하는 쪽(``main``)이 이 반환값을 보고 metric 전체를 건너뛸지 판단한다(요구사항 8번:
    "억지로 끼워 맞추지 말 것").
    """
    if SafetyGate is None:
        return None, f"safety_gate 모듈을 임포트하지 못했습니다 (mujoco 등 의존성 부재 가능): {_SAFETY_GATE_IMPORT_ERROR}"
    try:
        config = SafetyGateConfig.from_repo_defaults()
    except SafetyGateConfigError as exc:
        return None, f"SafetyGateConfig.from_repo_defaults() 실패: {exc}"
    return SafetyGate(config), None


# ---------------------------------------------------------------------------
# checkpoint 학습 데이터 출처 검증 (요구사항: "기존 모델과 완전히 독립적")
# ---------------------------------------------------------------------------


def check_train_dataset_provenance(checkpoint_dir: Path, expected_train_dataset: Path) -> str | None:
    """``runtime.common.checkpoint_provenance``에 위임 - 문서는 그 모듈 참고."""
    return _shared_check_train_dataset_provenance(checkpoint_dir, expected_train_dataset)


def validate_checkpoint_camera_mapping(checkpoint_dir: Path) -> list[str]:
    """``runtime.common.checkpoint_provenance``에 위임 - 이 스크립트가 기대하는
    ``EXPECTED_CAMERA_RENAME_MAP``을 기준으로 검증한다."""
    return _shared_validate_checkpoint_camera_mapping(checkpoint_dir, EXPECTED_CAMERA_RENAME_MAP)


# ---------------------------------------------------------------------------
# GPU/lerobot 필요한 부분 (checkpoint 로딩/추론) - import를 함수 안에서 지연시켜
# 순수 계산 유틸/테스트는 lerobot 없이도 임포트 가능하게 한다.
# ---------------------------------------------------------------------------


@dataclass
class PolicyBundle:
    policy: object
    preprocessor: object
    postprocessor: object
    device: object


def load_policy_bundle(checkpoint_dir: Path, *, policy_type: str = "smolvla", device: str | None = None) -> PolicyBundle:
    """``runtime/desktop/vla_server.py``의 ``SmolVLAPolicyRunner._load``와 동일한 순서."""
    import torch
    from lerobot.policies import get_policy_class, make_pre_post_processors

    policy_cls = get_policy_class(policy_type)
    policy = policy_cls.from_pretrained(str(checkpoint_dir))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy.to(dev)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint_dir))
    return PolicyBundle(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor, device=dev)


def unload_policy_bundle(bundle: PolicyBundle) -> None:
    """다음 checkpoint를 로딩하기 전에 GPU 메모리를 확실히 반환한다 (요구사항: 동시 로딩 금지)."""
    import torch

    del bundle.policy
    del bundle.preprocessor
    del bundle.postprocessor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_inference_batch(sample: dict, device, task: str) -> dict:
    """``LeRobotDataset`` 샘플 -> policy 배치. 카메라 키는 rename하지 않고 원본 데이터셋 이름
    그대로 넣는다(모듈 docstring "카메라 rename" 절 참고 - 저장된 preprocessor가 처리한다)."""
    return {
        "observation.state": sample["observation.state"].unsqueeze(0).to(device),
        CAMERA_WORKSPACE_KEY: sample[CAMERA_WORKSPACE_KEY].unsqueeze(0).to(device),
        CAMERA_WRIST_KEY: sample[CAMERA_WRIST_KEY].unsqueeze(0).to(device),
        "task": [task],
    }


def compute_gt_demo_delta_stats(dataset, frame_plans: list[FramePlan]) -> dict:
    """ground-truth demonstration의 state -> action 1-step delta 통계 (요구사항 7번 마지막 항목).

    모델과 무관한 dataset 고유값이므로 checkpoint 루프 밖에서 한 번만 계산하고 모든 checkpoint
    report에 동일하게 끼워 넣는다. 영상 디코딩 없이 ``dataset.hf_dataset``(parquet 백엔드)만
    읽어 빠르다.
    """
    pooled: list[float] = []
    for plan in frame_plans:
        for row in plan.dataset_indices:
            raw = dataset.hf_dataset[row]
            state = tensor_to_joint_dict(raw["observation.state"])
            action = tensor_to_joint_dict(raw["action"])
            for joint in JOINT_ORDER:
                pooled.append(abs(action[joint] - state[joint]))
    return error_stats(pooled)


def run_checkpoint(
    checkpoint_dir: Path,
    dataset,
    frame_plans: list[FramePlan],
    *,
    task: str,
    seed: int,
    device: str | None,
    policy_type: str,
    safety_gate,
    gt_demo_delta: dict,
) -> dict:
    """checkpoint 하나를 로딩 -> 평가 -> 언로드까지 전부 수행하고 report dict를 반환한다."""
    import torch

    warnings: list[str] = list(validate_checkpoint_camera_mapping(checkpoint_dir))

    bundle = load_policy_bundle(checkpoint_dir, policy_type=policy_type, device=device)
    action_dim = bundle.policy.config.action_feature.shape[0]
    if action_dim != len(JOINT_ORDER):
        unload_policy_bundle(bundle)
        raise ValueError(
            f"{checkpoint_dir}: action dim={action_dim}, 기대값={len(JOINT_ORDER)} - "
            "이 체크포인트는 이번 6-dim grid35 실험과 다른 action schema로 학습됐을 수 있습니다."
        )

    records: list[dict] = []
    safety_tally = SafetyTally() if safety_gate is not None else None

    t_start = time.time()
    try:
        for plan in frame_plans:
            # 요구사항 6번: episode마다 동일 seed로 재현 가능하게 하되, checkpoint 간에는
            # "같은 episode = 같은 seed"라서 비교가 노이즈(seed 차이)가 아니라 가중치 차이만
            # 반영하게 한다.
            episode_seed = seed + plan.episode_index
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)
            bundle.policy.reset()

            for row in plan.dataset_indices:
                sample = dataset[row]
                batch = build_inference_batch(sample, bundle.device, task)
                with torch.inference_mode():
                    processed = bundle.preprocessor(batch)
                    raw_action = bundle.policy.select_action(processed)
                    action = bundle.postprocessor(raw_action)

                pred = tensor_to_joint_dict(action)
                gt = tensor_to_joint_dict(sample["action"])
                state = tensor_to_joint_dict(sample["observation.state"])

                record: dict = {
                    "episode_index": plan.episode_index,
                    "frame_index": int(sample["frame_index"]),
                    "dataset_row": row,
                    "pred_action": pred,
                    "gt_action": gt,
                    "state": state,
                }

                if safety_gate is not None:
                    adapted = adapt_vla_action(pred)
                    decision = safety_gate.evaluate(
                        adapted_action=adapted, current_state_deg=state, observation_valid=True
                    )
                    record["safety_decision"] = decision.decision
                    record["safety_per_joint"] = {
                        j: {"clamped": r.clamped, "rejected": r.rejected} for j, r in decision.per_joint.items()
                    }
                    safety_tally.record(decision.decision, decision.per_joint)

                records.append(record)
    finally:
        unload_policy_bundle(bundle)

    elapsed_sec = time.time() - t_start
    report = aggregate_report(
        checkpoint_dir=checkpoint_dir,
        records=records,
        safety_tally=safety_tally,
        seed=seed,
        elapsed_sec=elapsed_sec,
        warnings=warnings,
        gt_demo_delta=gt_demo_delta,
    )
    return report


def aggregate_report(
    *,
    checkpoint_dir: Path,
    records: list[dict],
    safety_tally: SafetyTally | None,
    seed: int,
    elapsed_sec: float,
    warnings: list[str],
    gt_demo_delta: dict,
) -> dict:
    per_joint_abs_err: dict[str, list[float]] = {j: [] for j in JOINT_ORDER}
    all_abs_err: list[float] = []
    pooled_pred_state_delta: list[float] = []
    per_episode_counts: dict[int, int] = {}

    for rec in records:
        per_episode_counts[rec["episode_index"]] = per_episode_counts.get(rec["episode_index"], 0) + 1
        for joint in JOINT_ORDER:
            err = abs(rec["pred_action"][joint] - rec["gt_action"][joint])
            per_joint_abs_err[joint].append(err)
            all_abs_err.append(err)
            pooled_pred_state_delta.append(abs(rec["pred_action"][joint] - rec["state"][joint]))

    metrics = {
        "action_mae_overall": mean_or_none(all_abs_err),
        "action_mae_per_joint": {j: mean_or_none(v) for j, v in per_joint_abs_err.items()},
        "pred_state_delta": error_stats(pooled_pred_state_delta),
        "gt_demo_state_delta": gt_demo_delta,
    }

    report = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_step": infer_step_from_path(checkpoint_dir),
        "seed": seed,
        "elapsed_sec": elapsed_sec,
        "num_frames_evaluated": len(records),
        "num_episodes_evaluated": len(per_episode_counts),
        "frames_per_episode": dict(sorted(per_episode_counts.items())),
        "warnings": warnings,
        "metrics": metrics,
        "safety": safety_tally.to_dict() if safety_tally is not None else None,
        "records": round_floats(records),
    }
    return report


# ---------------------------------------------------------------------------
# summary.json / summary.md
# ---------------------------------------------------------------------------


def build_summary(reports: list[dict], *, eval_dataset: str, train_dataset: str, task: str, seed: int, safety_gate_available: bool, safety_gate_unavailable_reason: str | None) -> dict:
    ordered = sorted(reports, key=lambda r: (r["checkpoint_step"] is None, r["checkpoint_step"]))
    rows = []
    for r in ordered:
        m = r["metrics"]
        s = r["safety"] or {}
        rows.append(
            {
                "checkpoint_step": r["checkpoint_step"],
                "checkpoint": r["checkpoint"],
                "num_frames_evaluated": r["num_frames_evaluated"],
                "num_episodes_evaluated": r["num_episodes_evaluated"],
                "action_mae": m["action_mae_overall"],
                "action_mae_per_joint": m["action_mae_per_joint"],
                "pred_state_delta_median": m["pred_state_delta"]["median"],
                "pred_state_delta_p95": m["pred_state_delta"]["p95"],
                "pred_state_delta_max": m["pred_state_delta"]["max"],
                "gt_demo_delta_median": m["gt_demo_state_delta"]["median"],
                "gt_demo_delta_p95": m["gt_demo_state_delta"]["p95"],
                "gt_demo_delta_max": m["gt_demo_state_delta"]["max"],
                "would_pass": s.get("WOULD_PASS"),
                "would_clamp": s.get("WOULD_CLAMP"),
                "would_reject": s.get("WOULD_REJECT"),
                "joint_clamp_count": s.get("joint_clamp_count"),
                "warnings": r["warnings"],
            }
        )
    return {
        "eval_dataset": eval_dataset,
        "train_dataset": train_dataset,
        "task": task,
        "seed": seed,
        "safety_gate_available": safety_gate_available,
        "safety_gate_unavailable_reason": safety_gate_unavailable_reason,
        "rows": round_floats(rows),
    }


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_summary_markdown(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Grid35 SmolVLA - Held-out Midpoint10 Offline Evaluation")
    lines.append("")
    lines.append(f"- eval dataset (held-out, 학습에 미사용): `{summary['eval_dataset']}`")
    lines.append(f"- train dataset (provenance 확인용, 이 평가에는 사용 안 함): `{summary['train_dataset']}`")
    lines.append(f"- task: {summary['task']}")
    lines.append(f"- seed: {summary['seed']}")
    if summary["safety_gate_available"]:
        lines.append("- Safety Gate: 재사용됨 (`runtime/laptop/safety_gate.py`, write 없음)")
    else:
        lines.append(f"- Safety Gate: 사용 불가 - {summary['safety_gate_unavailable_reason']}")
    lines.append("")
    lines.append(
        "> **중요**: 이 offline metric(action MAE/delta)은 실제 task success를 단정하지 않는다. "
        "train35(학습 데이터) 성능은 이번 checkpoint 선택 기준이 아니며, 아래는 전부 unseen "
        "midpoint10 기준이다. 이 표의 목적은 Shadow Mode에 올릴 후보 1~2개를 거르는 것이다."
    )
    lines.append("")

    lines.append("## 핵심 비교 표 (unseen midpoint10 기준)")
    lines.append("")
    lines.append(
        "| checkpoint | frames | episodes | action MAE | delta median | delta p95 | delta max | "
        "WOULD_PASS | WOULD_CLAMP | WOULD_REJECT |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in summary["rows"]:
        lines.append(
            f"| {row['checkpoint_step']} | {row['num_frames_evaluated']} | {row['num_episodes_evaluated']} | "
            f"{_fmt(row['action_mae'])} | {_fmt(row['pred_state_delta_median'])} | "
            f"{_fmt(row['pred_state_delta_p95'])} | {_fmt(row['pred_state_delta_max'])} | "
            f"{_fmt(row['would_pass'])} | {_fmt(row['would_clamp'])} | {_fmt(row['would_reject'])} |"
        )
    lines.append("")

    lines.append("## Ground-truth demonstration state->action delta (checkpoint 무관, 참고용 baseline)")
    lines.append("")
    if summary["rows"]:
        gt = summary["rows"][0]
        lines.append(
            f"- median={_fmt(gt['gt_demo_delta_median'])}, p95={_fmt(gt['gt_demo_delta_p95'])}, "
            f"max={_fmt(gt['gt_demo_delta_max'])}"
        )
    lines.append("")

    lines.append("## Joint별 action MAE")
    lines.append("")
    lines.append("| checkpoint | " + " | ".join(JOINT_ORDER) + " |")
    lines.append("|---|" + "---|" * len(JOINT_ORDER))
    for row in summary["rows"]:
        per_joint = row["action_mae_per_joint"]
        lines.append(f"| {row['checkpoint_step']} | " + " | ".join(_fmt(per_joint.get(j)) for j in JOINT_ORDER) + " |")
    lines.append("")

    if summary["safety_gate_available"]:
        lines.append("## Joint별 WOULD_CLAMP count")
        lines.append("")
        lines.append("| checkpoint | " + " | ".join(JOINT_ORDER) + " |")
        lines.append("|---|" + "---|" * len(JOINT_ORDER))
        for row in summary["rows"]:
            jc = row["joint_clamp_count"] or {}
            lines.append(f"| {row['checkpoint_step']} | " + " | ".join(_fmt(jc.get(j)) for j in JOINT_ORDER) + " |")
        lines.append("")

    any_warnings = any(row["warnings"] for row in summary["rows"])
    if any_warnings:
        lines.append("## Checkpoint별 경고")
        lines.append("")
        for row in summary["rows"]:
            if row["warnings"]:
                lines.append(f"- checkpoint {row['checkpoint_step']}:")
                for w in row["warnings"]:
                    lines.append(f"  - {w}")
        lines.append("")

    lines.append("## 한계 (읽는 사람이 반드시 알아야 함)")
    lines.append("")
    lines.append(
        "- teacher forcing 평가다: 매 프레임 policy에게 넣는 state/이미지는 데이터셋에 실제로 "
        "기록된 값이지, policy 자신이 이전 스텝에서 만든 action을 따라간 결과가 아니다. 즉 진짜 "
        "closed-loop rollout이 아니라 \"기록된 궤적을 그대로 재생하면서 각 시점에서 policy라면 "
        "무엇을 예측했을까\"를 보는 것이다."
    )
    lines.append(
        "- SmolVLA는 action chunk(＝chunk_size)를 한 번에 생성하고 재사용한다 - 실제로 매 프레임마다 "
        "새로 forward하지 않고, chunk가 소진될 때만(이번 설정 기준 매 50프레임마다) 새로 샘플링한다. "
        "이는 실제 배포(`runtime/desktop/vla_server.py`)와 동일한 동작이다."
    )
    lines.append(
        "- action MAE/delta는 offline metric이다 - 실제 task success(큐브를 집어서 target에 놓았는지)를 "
        "이 수치만으로 단정할 수 없다."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid35 SmolVLA held-out(midpoint10) offline checkpoint evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[str(p) for p in DEFAULT_CHECKPOINTS],
        help="평가할 pretrained_model 디렉터리 목록",
    )
    parser.add_argument("--eval-dataset", default=str(DEFAULT_EVAL_DATASET), help="held-out 평가 전용 dataset root")
    parser.add_argument(
        "--eval-repo-id", default=DEFAULT_EVAL_REPO_ID, help="LeRobotDataset repo_id (로컬 root와 함께 씀)"
    )
    parser.add_argument(
        "--train-dataset",
        default=str(DEFAULT_TRAIN_DATASET),
        help="checkpoint 학습에 실제 쓰인 dataset root (provenance 검증용 - 이 스크립트는 이 dataset을 로딩/평가하지 않는다)",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="episode별 재현성을 위한 base seed")
    parser.add_argument("--device", default=None, help="cuda/cpu (미지정 시 자동 감지)")
    parser.add_argument("--policy-type", default="smolvla")
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="episode당 평가할 최대 프레임 수 (기본: 전체). OOM 등으로 줄여야 할 때만 쓰며, "
        "지정해도 episode 전체에 균등 간격으로 샘플링한다(앞부분만 자르지 않음).",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-safety-gate", action="store_true", help="Safety Gate 재사용을 건너뛴다")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    eval_root = Path(args.eval_dataset).resolve()
    train_root = Path(args.train_dataset).resolve()
    if eval_root == train_root:
        logger.error("--eval-dataset과 --train-dataset이 동일합니다 (%s) - held-out 평가에 학습 데이터를 쓸 수 없습니다.", eval_root)
        return 2
    if not eval_root.is_dir():
        logger.error("--eval-dataset 경로가 존재하지 않습니다: %s", eval_root)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=args.eval_repo_id, root=str(eval_root))
    episodes_meta = [dataset.meta.episodes[i] for i in range(dataset.num_episodes)]
    frame_plans = build_frame_plans(episodes_meta, max_frames_per_episode=args.max_frames_per_episode)
    total_frames = sum(len(p.dataset_indices) for p in frame_plans)
    logger.info(
        "held-out eval dataset=%s: %d episodes, 평가 대상 프레임 %d개 (전체 프레임 %d개)",
        eval_root,
        len(frame_plans),
        total_frames,
        len(dataset),
    )

    gt_demo_delta = compute_gt_demo_delta_stats(dataset, frame_plans)
    logger.info("ground-truth demo state->action delta: %s", gt_demo_delta)

    safety_gate = None
    safety_reason = None
    if not args.no_safety_gate:
        safety_gate, safety_reason = try_build_safety_gate()
        if safety_gate is None:
            logger.warning("Safety Gate 재사용 불가 (metric에서 생략됨): %s", safety_reason)
        else:
            logger.info("Safety Gate 재사용 활성화됨 (runtime/laptop/safety_gate.py)")
    else:
        safety_reason = "--no-safety-gate로 비활성화됨"

    reports = []
    for ckpt_str in args.checkpoints:
        ckpt_dir = Path(ckpt_str)
        if not ckpt_dir.is_dir():
            logger.error("checkpoint 경로가 존재하지 않습니다: %s", ckpt_dir)
            return 2

        provenance_warning = check_train_dataset_provenance(ckpt_dir, train_root)
        step = infer_step_from_path(ckpt_dir)
        logger.info("=== checkpoint step=%s (%s) 평가 시작 ===", step, ckpt_dir)

        report = run_checkpoint(
            ckpt_dir,
            dataset,
            frame_plans,
            task=args.task,
            seed=args.seed,
            device=args.device,
            policy_type=args.policy_type,
            safety_gate=safety_gate,
            gt_demo_delta=gt_demo_delta,
        )
        if provenance_warning:
            report["warnings"].append(provenance_warning)
            logger.warning("[checkpoint step=%s] %s", step, provenance_warning)
        for w in report["warnings"]:
            if w != provenance_warning:
                logger.warning("[checkpoint step=%s] %s", step, w)

        report["eval_dataset"] = str(eval_root)
        report["train_dataset"] = str(train_root)
        report["task"] = args.task

        out_name = f"checkpoint_{step:06d}.json" if step is not None else f"checkpoint_{ckpt_dir.parent.name}.json"
        out_path = output_dir / out_name
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "checkpoint step=%s 저장: %s (action_mae=%.4f, frames=%d, elapsed=%.1fs)",
            step,
            out_path,
            report["metrics"]["action_mae_overall"],
            report["num_frames_evaluated"],
            report["elapsed_sec"],
        )
        reports.append(report)

    summary = build_summary(
        reports,
        eval_dataset=str(eval_root),
        train_dataset=str(train_root),
        task=args.task,
        seed=args.seed,
        safety_gate_available=safety_gate is not None,
        safety_gate_unavailable_reason=safety_reason,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    logger.info("summary 저장 완료: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
