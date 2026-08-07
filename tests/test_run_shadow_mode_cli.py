"""scripts/run_shadow_mode.py CLI 인자 파싱 테스트 (하드웨어 연결 없이)."""

from __future__ import annotations

import importlib.util
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
