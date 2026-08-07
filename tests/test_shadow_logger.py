"""runtime/laptop/shadow_logger.py 테스트."""

from __future__ import annotations

import json

import pytest

from runtime.laptop.shadow_logger import RESULT_SHADOW_PASS, build_report, make_session_id, resolve_report_path, write_report


def _minimal_report(**overrides) -> dict:
    kwargs = dict(
        task="pick up the cube",
        backend="realistic_mujoco",
        observation={"schema_valid": True},
        communication={"health_ok": True},
        vla={"raw_action": {}},
        adapter={"mapped_action": {}},
        safety={"decision": "ACCEPT"},
        mujoco={"initial_state": {}},
        validation={"passed": True},
        real_follower_write_count=0,
        result=RESULT_SHADOW_PASS,
        result_reasons=[],
    )
    kwargs.update(overrides)
    return build_report(**kwargs)


def test_build_report_has_required_top_level_keys() -> None:
    report = _minimal_report()
    for key in ("mode", "backend", "real_robot_write_enabled", "task", "observation", "communication", "vla",
                "adapter", "safety", "mujoco", "hardware", "result"):
        assert key in report
    assert report["mode"] == "SHADOW"
    assert report["real_robot_write_enabled"] is False
    assert report["hardware"]["real_follower_write_count"] == 0


def test_build_report_rejects_nonzero_write_count() -> None:
    with pytest.raises(AssertionError):
        _minimal_report(real_follower_write_count=1)


def test_write_report_creates_json_file(tmp_path) -> None:
    report = _minimal_report()
    session_id = make_session_id()
    path = write_report(report, reports_dir=tmp_path, session_id=session_id)
    assert path == resolve_report_path(tmp_path, session_id)
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["result"] == RESULT_SHADOW_PASS


def test_session_id_format() -> None:
    sid = make_session_id()
    assert len(sid) == 15  # YYYYMMDD_HHMMSS
    assert sid[8] == "_"
