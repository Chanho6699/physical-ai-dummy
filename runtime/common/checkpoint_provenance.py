"""SmolVLA checkpoint의 출처(어떤 dataset으로 학습됐는지/어떤 step인지/카메라 rename이
저장된 preprocessor에 있는지)를 파일 기반으로 검증하는 순수 계산 유틸.

``scripts/evaluate_smolvla_midpoint.py``(offline held-out evaluator)와
``runtime/laptop/shadow_mode_runner.py``/``scripts/run_shadow_mode.py``(Shadow Mode) 양쪽
모두 "이 checkpoint가 정말 내가 생각하는 실험의 산출물인가"를 같은 방식으로 검증해야 한다
(요구사항: "기존 20ep / old 10k checkpoint를 암묵적으로 기본값으로 사용하지 않는다" +
"checkpoint provenance 기록") - 그래서 이 로직을 한 곳으로 뽑아 두 호출부가 공유한다.

여기 있는 함수는 전부 로컬 파일(``pretrained_model`` 디렉터리 안의 ``train_config.json``/
``policy_preprocessor.json``)만 읽는다 - GPU/torch/lerobot import가 전혀 없어 가볍고,
lerobot 없는 환경에서도 임포트/테스트할 수 있다.
"""

from __future__ import annotations

import json
from pathlib import Path


def infer_checkpoint_step(checkpoint_dir: Path) -> int | None:
    """``checkpoints/007500/pretrained_model`` -> 7500. 숫자가 아니면(``last`` 등) None.

    ``pretrained_model``이 아닌 디렉터리(예: 이미 ``checkpoints/007500``까지만 준 경우)도
    받아들인다.
    """
    name = checkpoint_dir.name
    candidate = checkpoint_dir.parent.name if name == "pretrained_model" else name
    try:
        return int(candidate)
    except ValueError:
        return None


def read_checkpoint_train_dataset_root(checkpoint_dir: Path) -> tuple[str | None, str | None]:
    """checkpoint의 ``train_config.json``에서 ``dataset.root``를 읽는다.

    Returns:
        (dataset_root 또는 None, 실패 사유 또는 None) - 정확히 하나만 채워진다.
    """
    train_config_path = checkpoint_dir / "train_config.json"
    if not train_config_path.is_file():
        return None, f"train_config.json을 찾을 수 없습니다: {train_config_path}"
    try:
        cfg = json.loads(train_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"train_config.json 파싱 실패: {exc}"
    recorded_root = (cfg.get("dataset") or {}).get("root")
    if not recorded_root:
        return None, "train_config.json에 dataset.root가 없습니다."
    return recorded_root, None


def check_train_dataset_provenance(checkpoint_dir: Path, expected_train_dataset: Path) -> str | None:
    """체크포인트가 실제로 ``expected_train_dataset``으로 학습됐는지 확인한다.

    다르면(또는 확인할 수 없으면) 경고 문자열을 반환한다 - 하드 실패는 아니다(머신마다
    절대경로가 다를 수 있으므로 마지막 경로 성분(basename)만 비교한다). 이 함수는 "과거
    20ep/old 10k 실험 checkpoint를 새 실험이라고 착각해 쓰는" 사고를 막기 위한 안전장치다.
    """
    recorded_root, error = read_checkpoint_train_dataset_root(checkpoint_dir)
    if recorded_root is None:
        return error
    if Path(recorded_root).name != expected_train_dataset.name:
        return (
            f"체크포인트의 학습 dataset({recorded_root!r})이 기대한 학습 dataset"
            f"({expected_train_dataset.name!r})과 다릅니다 - 이 체크포인트가 다른 실험 데이터로 "
            "학습됐을 가능성이 있습니다 (과거 실험과 섞이지 않았는지 확인하세요)."
        )
    return None


def validate_checkpoint_camera_mapping(checkpoint_dir: Path, expected_rename_map: dict[str, str]) -> list[str]:
    """저장된 ``policy_preprocessor.json``이 실제로 ``expected_rename_map``을 담고 있는지
    확인한다 (추측하지 않고 실측한다). 문제가 있으면 경고 문자열 목록을 반환한다(빈 목록이면
    기대대로).
    """
    path = checkpoint_dir / "policy_preprocessor.json"
    if not path.is_file():
        return [f"policy_preprocessor.json을 찾을 수 없습니다: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"policy_preprocessor.json 파싱 실패: {exc}"]

    rename_map = None
    for step in data.get("steps", []):
        if step.get("registry_name") == "rename_observations_processor":
            rename_map = (step.get("config") or {}).get("rename_map")
            break

    if rename_map is None:
        return [
            "preprocessor에 rename_observations_processor 단계가 없습니다 - 카메라 rename이 "
            "저장된 preprocessor에 내장되어 있지 않을 수 있습니다. 이 checkpoint가 원본 데이터셋 "
            "카메라 키(workspace/wrist)를 그대로 기대하는지 별도로 확인하세요."
        ]

    warnings = []
    for src, expected_dst in expected_rename_map.items():
        actual_dst = rename_map.get(src)
        if actual_dst != expected_dst:
            warnings.append(f"카메라 rename 불일치: {src!r} -> {actual_dst!r} (기대: {expected_dst!r})")
    return warnings
