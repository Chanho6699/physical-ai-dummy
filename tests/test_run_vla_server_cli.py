"""scripts/run_vla_server.py CLI 인자 파싱 테스트 (서버 기동 없이)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_vla_server.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_vla_server_cli_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_module()


def test_requires_fake_or_checkpoint(cli_module) -> None:
    with pytest.raises(SystemExit):
        cli_module.parse_args([])


def test_fake_and_checkpoint_are_mutually_exclusive(cli_module) -> None:
    with pytest.raises(SystemExit):
        cli_module.parse_args(["--fake", "--checkpoint", "/some/path"])


def test_fake_alone_is_valid(cli_module) -> None:
    args = cli_module.parse_args(["--fake"])
    assert args.fake is True
    assert args.checkpoint is None


def test_checkpoint_alone_is_valid(cli_module) -> None:
    args = cli_module.parse_args(["--checkpoint", "/some/path"])
    assert args.fake is False
    assert args.checkpoint == "/some/path"


def test_cli_seed_becomes_runner_effective_seed(cli_module, monkeypatch) -> None:
    class Runner:
        def __init__(self, checkpoint, **kwargs):
            self.inference_seed = kwargs["inference_seed"]

    monkeypatch.setattr(cli_module, "SmolVLAPolicyRunner", Runner)
    args = cli_module.parse_args(["--checkpoint", "/some/path", "--inference-seed", "20260815"])
    runner = cli_module.build_policy_runner(args)
    assert runner.inference_seed == args.inference_seed == 20260815


def test_cli_without_seed_preserves_stochastic_mode(cli_module, monkeypatch) -> None:
    class Runner:
        def __init__(self, checkpoint, **kwargs):
            self.inference_seed = kwargs["inference_seed"]

    monkeypatch.setattr(cli_module, "SmolVLAPolicyRunner", Runner)
    args = cli_module.parse_args(["--checkpoint", "/some/path"])
    assert cli_module.build_policy_runner(args).inference_seed is None
