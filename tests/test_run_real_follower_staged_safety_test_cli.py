from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import scripts.run_real_follower_staged_safety_test as cli
from runtime.laptop.vla_client import HealthResult

CANDIDATE_B = Path(
    "outputs/pick_drop_v3_v4_combined69/smolvla_pick_drop_v3_v4_combined69_uniform_fresh/checkpoints/010000/pretrained_model"
)


def test_checkpoint_signature_uses_last_four_path_components():
    sig = cli._checkpoint_signature(CANDIDATE_B)
    assert sig == "smolvla_pick_drop_v3_v4_combined69_uniform_fresh/checkpoints/010000/pretrained_model"


def _health(model_id: str | None, ok: bool = True) -> HealthResult:
    return HealthResult(
        ok=ok, status="ok" if ok else None, backend="smolvla", model_loaded=ok, model_id=model_id,
        device="cuda", errors=[], round_trip_ms=5.0,
    )


def test_verify_desktop_checkpoint_passes_on_matching_signature():
    health = _health(model_id=f"/home/desktop-user/models/{cli._checkpoint_signature(CANDIDATE_B)}")
    cli._verify_desktop_checkpoint(health=health, expected_checkpoint=CANDIDATE_B, force=False)  # should not raise


def test_verify_desktop_checkpoint_blocks_on_mismatch_by_default():
    health = _health(model_id="outputs/pick_drop_combined65_reweight2_early/smolvla_.../checkpoints/010000/pretrained_model")
    with pytest.raises(cli.StagedSafetyTestError, match="model_id"):
        cli._verify_desktop_checkpoint(health=health, expected_checkpoint=CANDIDATE_B, force=False)


def test_verify_desktop_checkpoint_blocks_on_missing_model_id():
    health = _health(model_id=None)
    with pytest.raises(cli.StagedSafetyTestError):
        cli._verify_desktop_checkpoint(health=health, expected_checkpoint=CANDIDATE_B, force=False)


def test_verify_desktop_checkpoint_force_bypasses_mismatch(capsys):
    health = _health(model_id="totally-different-checkpoint")
    cli._verify_desktop_checkpoint(health=health, expected_checkpoint=CANDIDATE_B, force=True)  # no raise
    assert "경고" in capsys.readouterr().out


def test_validate_vla_mode_http_requires_server_url():
    class Args:
        vla_mode = "http"
        vla_server_url = None

    with pytest.raises(cli.StagedSafetyTestError, match="vla-server-url"):
        cli._validate_vla_mode_args(Args())


def test_validate_vla_mode_inprocess_rejects_server_url():
    class Args:
        vla_mode = "inprocess"
        vla_server_url = "http://example:9200"

    with pytest.raises(cli.StagedSafetyTestError):
        cli._validate_vla_mode_args(Args())


def test_validate_vla_mode_http_with_url_ok():
    class Args:
        vla_mode = "http"
        vla_server_url = "http://example:9200"

    cli._validate_vla_mode_args(Args())  # should not raise


def test_parse_args_requires_vla_mode():
    with pytest.raises(SystemExit):
        cli.parse_args([
            "--stage", "1", "--hardware-config", "x", "--follower-port", "p", "--follower-id", "i",
        ])


def test_main_stage1_builds_so101_follower_config_with_id_kwarg(tmp_path, monkeypatch):
    """실제 non-dry-run 경로(단, 모든 하드웨어/네트워크 의존성은 mock)를 stage 1로 끝까지
    돌려서 다음을 확인한다:

    1. ``SOFollower(config)``에 전달되는 config는 (bare) ``SOFollowerConfig``가 아니라
       ``SO101FollowerConfig``다 - ``SOFollower.__init__ -> Robot.__init__``이 ``config.id``
       를 읽는데, bare ``SOFollowerConfig``에는 ``id`` 필드 자체가 없어서(``RobotConfig``를
       상속하지 않음) ``AttributeError``가 난다는 것을 실제 설치된 lerobot으로 재현/확인했다.
    2. ``SO101FollowerConfig``에는 ``port``/``id``/``cameras``/``disable_torque_on_disconnect``
       가 전달된다 (``id``는 이제 다시 전달되어야 한다).
    3. ``follower_id``는 여전히 ``ReadOnlyRealFollowerStateSource.from_port``에도 전달된다
       (read-only state/calibration 식별 경로는 그대로 유지).

    lerobot이 설치되어 있지 않은 개발 환경에서도 돌 수 있도록 ``lerobot.robots.so_follower``
    를 가짜 모듈로 ``sys.modules``에 주입한다 - 하드웨어에는 절대 접근하지 않는다
    (fake follower의 connect/get_observation/send_action/disconnect는 전부 no-op).
    """
    import runtime.laptop.camera_source as camera_source_mod
    import runtime.laptop.follower_state_source as state_source_mod
    import runtime.laptop.inprocess_vla_client as inprocess_vla_mod
    import runtime.laptop.staged_real_rollout as rollout_mod

    so101_follower_config_calls: list[dict] = []
    from_port_calls: list[dict] = []

    class FakeSO101FollowerConfig:
        def __init__(self, **kwargs):
            so101_follower_config_calls.append(kwargs)
            self.kwargs = kwargs
            self.id = kwargs.get("id")  # 실제 SOFollowerRobotConfig처럼 id 속성을 노출한다

    class FakeSOFollower:
        def __init__(self, config):
            # 실제 SOFollower.__init__ -> Robot.__init__이 config.id를 읽는 것과 동일하게,
            # 여기서도 config에 id 속성이 없으면 AttributeError가 나야 한다 (진짜 API를 흉내).
            self.id = config.id
            self.config = config

        def connect(self):
            pass

        def get_observation(self):
            return {}

        def send_action(self, action):
            return action

        def disconnect(self):
            pass

    fake_so_follower_module = types.ModuleType("lerobot.robots.so_follower")
    fake_so_follower_module.SOFollower = FakeSOFollower
    fake_so_follower_module.SO101FollowerConfig = FakeSO101FollowerConfig
    fake_lerobot = types.ModuleType("lerobot")
    fake_lerobot_robots = types.ModuleType("lerobot.robots")
    monkeypatch.setitem(sys.modules, "lerobot", fake_lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.robots", fake_lerobot_robots)
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", fake_so_follower_module)

    class FakeCamera:
        def open(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        camera_source_mod.RealCameraObservationSource,
        "from_hardware_config_path",
        classmethod(lambda cls, path: FakeCamera()),
    )

    class FakeStateSource:
        def connect(self):
            pass

        def disconnect(self):
            pass

    def fake_from_port(cls, *, port, follower_id, calibration_path=None):
        from_port_calls.append({"port": port, "follower_id": follower_id, "calibration_path": calibration_path})
        return FakeStateSource()

    monkeypatch.setattr(
        state_source_mod.ReadOnlyRealFollowerStateSource, "from_port", classmethod(fake_from_port)
    )

    monkeypatch.setattr(
        inprocess_vla_mod, "InProcessSmolVLAClient", lambda *, checkpoint, policy_type, device: object()
    )

    class FakeStageResult:
        verdict = "PASS"
        steps = []
        stop_reason = None
        real_follower_write_count = 0

        def to_dict(self):
            return {}

    monkeypatch.setattr(rollout_mod.StagedRealRolloutRunner, "run_stage", lambda self, *, stage, max_steps: FakeStageResult())

    monkeypatch.setattr(cli, "require_interactive_confirmation", lambda: None)

    hardware_config = tmp_path / "hardware.local.json"
    hardware_config.write_text("{}", encoding="utf-8")

    rc = cli.main([
        "--stage", "1",
        "--vla-mode", "inprocess",
        "--hardware-config", str(hardware_config),
        "--follower-port", "/dev/ttyFAKE0",
        "--follower-id", "my_follower",
        "--confirm-physically-present",
        "--receipt-dir", str(tmp_path / "receipts"),
        "--report-dir", str(tmp_path / "reports"),
    ])

    assert rc == 0
    assert so101_follower_config_calls == [
        {"port": "/dev/ttyFAKE0", "id": "my_follower", "cameras": {}, "disable_torque_on_disconnect": True}
    ]
    assert from_port_calls == [
        {"port": "/dev/ttyFAKE0", "follower_id": "my_follower", "calibration_path": None}
    ]


def test_dry_run_http_mode_touches_no_hardware_and_no_network(capsys):
    """dry-run은 vla-server-url이 실제로 존재하지 않아도(연결 안 하므로) 성공해야 한다."""
    rc = cli.main([
        "--stage", "1", "--dry-run", "--vla-mode", "http",
        "--vla-server-url", "http://192.0.2.1:9200",  # TEST-NET-1 (RFC 5737) - 절대 응답 없음
        "--hardware-config", "/nonexistent.json", "--follower-port", "/dev/ttyACM0", "--follower-id", "t",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLA_MODE=http" in out
    assert "192.0.2.1" in out
