"""hardware/safety/calibration_resolution.py 단위 테스트.

우선순위(CLI > local config > `~` 기본 fallback > 거부)와, armed 모드에서 기본
fallback이 비활성화된다는 것을 검증한다. 실제 파일시스템 홈 디렉터리를 건드리지
않는다 - 모든 project_root는 tmp_path로 대체한다.
"""

from __future__ import annotations

import json

import pytest

from hardware.safety import calibration_resolution as calres


def _write_local_config(project_root, *, port=None, robot_id=None, calibration_path=None):
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    robot: dict = {}
    if port is not None:
        robot["port"] = port
    if robot_id is not None:
        robot["id"] = robot_id
    if calibration_path is not None:
        robot["calibration_path"] = calibration_path
    payload = {"robot": robot}
    (project_root / "configs" / "hardware.local.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# 포트 우선순위
# ---------------------------------------------------------------------------


def test_port_cli_arg_wins_over_local_config(tmp_path):
    _write_local_config(tmp_path, port="/dev/from_local_config")
    port, source = calres.resolve_port(cli_port="/dev/from_cli", project_root=tmp_path)
    assert port == "/dev/from_cli"
    assert source == "cli"


def test_port_falls_back_to_local_config(tmp_path):
    _write_local_config(tmp_path, port="/dev/from_local_config")
    port, source = calres.resolve_port(cli_port=None, project_root=tmp_path)
    assert port == "/dev/from_local_config"
    assert source == "local_config"


def test_port_refused_when_nothing_available(tmp_path):
    with pytest.raises(calres.ResolutionError):
        calres.resolve_port(cli_port=None, project_root=tmp_path)


# ---------------------------------------------------------------------------
# calibration 경로 우선순위
# ---------------------------------------------------------------------------


def test_calibration_cli_path_wins_over_everything(tmp_path):
    _write_local_config(tmp_path, robot_id="local_id")
    path, source = calres.resolve_calibration_path(
        cli_calibration_path="/explicit/path.json",
        cli_calibration_id="cli_id",
        project_root=tmp_path,
    )
    assert path == calres.Path("/explicit/path.json")
    assert source == "cli_path"


def test_calibration_cli_id_used_when_no_explicit_path(tmp_path):
    path, source = calres.resolve_calibration_path(
        cli_calibration_path=None, cli_calibration_id="my_follower", project_root=tmp_path
    )
    assert str(path).endswith("my_follower.json")
    assert source == "cli_id"


def test_calibration_falls_back_to_local_config_id(tmp_path):
    _write_local_config(tmp_path, robot_id="local_id")
    path, source = calres.resolve_calibration_path(
        cli_calibration_path=None, cli_calibration_id=None, project_root=tmp_path
    )
    assert str(path).endswith("local_id.json")
    assert source == "local_config"


def test_calibration_local_config_explicit_path_wins_over_local_id(tmp_path):
    _write_local_config(tmp_path, robot_id="local_id", calibration_path="/explicit/from_local.json")
    path, source = calres.resolve_calibration_path(
        cli_calibration_path=None, cli_calibration_id=None, project_root=tmp_path
    )
    assert path == calres.Path("/explicit/from_local.json")
    assert source == "local_config"


def test_calibration_uses_default_fallback_when_nothing_else_available(tmp_path):
    path, source = calres.resolve_calibration_path(
        cli_calibration_path=None,
        cli_calibration_id=None,
        project_root=tmp_path,
        allow_default_fallback=True,
    )
    assert source == "default_fallback"
    assert str(path).endswith(f"{calres.DEFAULT_FOLLOWER_ID_FALLBACK}.json")


def test_calibration_fallback_disabled_raises_when_nothing_else_available(tmp_path):
    with pytest.raises(calres.ResolutionError):
        calres.resolve_calibration_path(
            cli_calibration_path=None,
            cli_calibration_id=None,
            project_root=tmp_path,
            allow_default_fallback=False,
        )


def test_calibration_path_never_hardcoded_as_literal_absolute_string_in_module_source():
    import inspect

    source = inspect.getsource(calres)
    # 사용자 이름/홈 디렉터리가 코드에 그대로 박혀 있으면 안 된다 (템플릿 + id 조합만 허용).
    assert "/home/" not in source
