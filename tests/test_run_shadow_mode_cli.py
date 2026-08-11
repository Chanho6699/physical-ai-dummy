"""scripts/run_shadow_mode.py CLI 인자 파싱 테스트 (하드웨어 연결 없이)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_shadow_mode.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_shadow_mode_cli_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_module()


def test_requires_task_and_follower_port(cli_module) -> None:
    with pytest.raises(SystemExit):
        cli_module.parse_args([])


def test_mode_defaults_to_single_step(cli_module) -> None:
    args = cli_module.parse_args(["--task", "pick", "--follower-port", "/dev/ttyACM0"])
    assert args.mode == cli_module.SINGLE_STEP_MODE


def test_mode_rejects_non_single_step(cli_module) -> None:
    with pytest.raises(SystemExit):
        cli_module.parse_args(["--task", "pick", "--follower-port", "/dev/ttyACM0", "--mode", "continuous"])


def test_main_fails_fast_without_server_url(cli_module, monkeypatch) -> None:
    monkeypatch.delenv("VLA_SERVER_URL", raising=False)
    rc = cli_module.main(["--task", "pick", "--follower-port", "/dev/ttyACM0"])
    assert rc == 2


# ---------------------------------------------------------------------------
# --checkpoint (요구사항 1/2번) - 과거 실험 checkpoint를 암묵적으로 쓰지 않는다.
# ---------------------------------------------------------------------------


def test_checkpoint_has_no_implicit_default(cli_module) -> None:
    """--checkpoint를 안 주면 None이어야 한다 - 과거 20ep/old 10k checkpoint 같은 하드코딩된
    기본값이 없어야 한다."""
    args = cli_module.parse_args(["--task", "pick", "--follower-port", "/dev/ttyACM0"])
    assert args.checkpoint is None


def test_checkpoint_and_server_url_are_mutually_exclusive(cli_module) -> None:
    with pytest.raises(SystemExit):
        cli_module.parse_args(
            [
                "--task", "pick", "--follower-port", "/dev/ttyACM0",
                "--checkpoint", "/some/checkpoint",
                "--vla-server-url", "http://desktop:9200",
            ]
        )


def test_checkpoint_alone_is_valid(cli_module) -> None:
    args = cli_module.parse_args(
        ["--task", "pick", "--follower-port", "/dev/ttyACM0", "--checkpoint", "/some/checkpoint"]
    )
    assert args.checkpoint == "/some/checkpoint"
    assert args.vla_server_url is None


def test_main_fails_fast_without_checkpoint_or_server_url_even_with_env_unset(cli_module, monkeypatch) -> None:
    monkeypatch.delenv("VLA_SERVER_URL", raising=False)
    rc = cli_module.main(["--task", "pick", "--follower-port", "/dev/ttyACM0"])
    assert rc == 2


def test_eval_mode_choices_restricted(cli_module) -> None:
    args = cli_module.parse_args(
        ["--task", "pick", "--follower-port", "/dev/ttyACM0", "--eval-mode", "midpoint-shadow"]
    )
    assert args.eval_mode == "midpoint-shadow"
    with pytest.raises(SystemExit):
        cli_module.parse_args(["--task", "pick", "--follower-port", "/dev/ttyACM0", "--eval-mode", "bogus-mode"])


def test_scene_label_and_note_default_to_none(cli_module) -> None:
    args = cli_module.parse_args(["--task", "pick", "--follower-port", "/dev/ttyACM0"])
    assert args.scene_label is None
    assert args.scene_note is None
    assert args.heldout_episode_index is None


def test_heldout_episode_index_parses_as_int(cli_module) -> None:
    args = cli_module.parse_args(
        [
            "--task", "pick", "--follower-port", "/dev/ttyACM0",
            "--eval-mode", "midpoint-shadow", "--scene-label", "T01", "--heldout-episode-index", "0",
        ]
    )
    assert args.scene_label == "T01"
    assert args.heldout_episode_index == 0


# ---------------------------------------------------------------------------
# build_scene_metadata - 순수 함수, 하드웨어 불필요. held-out episode index/label/
# capture_timestamp는 사람이 지정한 값만 담고 물체의 실제 XY는 절대 추정하지 않는다.
# ---------------------------------------------------------------------------


def test_build_scene_metadata_returns_none_when_nothing_given(cli_module) -> None:
    assert cli_module.build_scene_metadata(scene_label=None, scene_note=None, heldout_episode_index=None) is None


def test_build_scene_metadata_carries_label_index_and_capture_timestamp(cli_module) -> None:
    from datetime import datetime, timezone

    fixed_now = datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc)
    meta = cli_module.build_scene_metadata(
        scene_label="T01", scene_note=None, heldout_episode_index=0, now=fixed_now
    )
    assert meta == {
        "label": "T01",
        "note": None,
        "heldout_episode_index": 0,
        "capture_timestamp": "2026-08-08T12:00:00+00:00",
    }


def test_build_scene_metadata_no_xy_keys(cli_module) -> None:
    """물체의 실제 x/y 좌표를 임의로 정의/가정해서 넣지 않는다는 불변식을 고정한다."""
    meta = cli_module.build_scene_metadata(scene_label="T01", scene_note=None, heldout_episode_index=0)
    assert "x" not in meta
    assert "y" not in meta
    assert "position" not in meta
    assert "xy" not in meta


# ---------------------------------------------------------------------------
# build_checkpoint_metadata - 순수 함수, 하드웨어 불필요
# ---------------------------------------------------------------------------


def test_build_checkpoint_metadata_without_expected_dataset(cli_module, tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints" / "007500" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "train_config.json").write_text(json.dumps({"dataset": {"root": "/x/data/so101_cube_xy_grid35_v1"}}))

    meta = cli_module.build_checkpoint_metadata(ckpt, expected_train_dataset=None)
    assert meta["checkpoint_path"] == str(ckpt)
    assert meta["checkpoint_step"] == 7500
    assert meta["train_dataset_provenance"]["recorded_train_dataset_root"] == "/x/data/so101_cube_xy_grid35_v1"
    assert meta["train_dataset_provenance"]["mismatch_warning"] is None


def test_build_checkpoint_metadata_flags_mismatch(cli_module, tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints" / "010000" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "train_config.json").write_text(json.dumps({"dataset": {"root": "/x/data/so101_cube_train_v6"}}))

    meta = cli_module.build_checkpoint_metadata(ckpt, expected_train_dataset=Path("/x/data/so101_cube_xy_grid35_v1"))
    assert meta["train_dataset_provenance"]["mismatch_warning"] is not None
    assert "so101_cube_train_v6" in meta["train_dataset_provenance"]["mismatch_warning"]
