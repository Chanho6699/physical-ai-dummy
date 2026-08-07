#!/usr/bin/env python3
"""Shadow Mode v1 실행 CLI - single-step (섹션 19, 21).

실제 fixed/wrist 카메라 + 실제 SO-101 follower state(읽기 전용)를 Desktop SmolVLA에
보내고, 반환된 action을 Safety Gate -> Realistic MuJoCo에서 단 1 step 검증한다.
실제 SO-101 follower에는 어떤 명령도 보내지 않는다 (``REAL_SO101_WRITE=DISABLED``).

실행 예시:
    python scripts/run_shadow_mode.py \\
        --task "Pick up the cube" \\
        --mode single-step \\
        --vla-server-url http://<desktop-tailscale-ip>:9200 \\
        --follower-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14113538-if00 \\
        --follower-id chanho_follower
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.laptop.camera_source import CameraSourceError, RealCameraObservationSource
from runtime.laptop.follower_state_source import FollowerStateSourceError, ReadOnlyRealFollowerStateSource
from runtime.laptop.mujoco_shadow_backend import MuJoCoBackendError, RealisticMuJoCoBackend
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig, SafetyGateConfigError
from runtime.laptop.shadow_logger import RESULT_SHADOW_FAIL, RESULT_SHADOW_PASS_WITH_CLAMP
from runtime.laptop.shadow_mode_runner import ShadowModeRunner
from runtime.laptop.vla_client import VLAClientConfig, VLAHttpClient

DEFAULT_HARDWARE_CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware.local.json"
SINGLE_STEP_MODE = "single-step"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow Mode v1 (single-step)")
    parser.add_argument("--task", required=True, help="VLA에 전달할 task instruction")
    parser.add_argument("--mode", default=SINGLE_STEP_MODE, choices=[SINGLE_STEP_MODE], help="v1은 single-step만 지원합니다.")
    parser.add_argument("--vla-server-url", default=None, help="미지정 시 환경변수 VLA_SERVER_URL 사용")
    parser.add_argument("--vla-timeout-s", type=float, default=15.0)
    parser.add_argument("--vla-api-token", default=None)
    parser.add_argument("--hardware-config", default=str(DEFAULT_HARDWARE_CONFIG_PATH))
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--follower-id", default="chanho_follower")
    parser.add_argument("--follower-calibration-path", default=None)
    parser.add_argument("--mujoco-scene-path", default=None)
    parser.add_argument("--mujoco-profile-path", default=None)
    parser.add_argument("--reports-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 60)
    print("MODE=SHADOW")
    print("VLA_LOCATION=DESKTOP")
    print("EXECUTION_BACKEND=REALISTIC_MUJOCO")
    print("REAL_SO101_WRITE=DISABLED")
    print("=" * 60)

    server_url = args.vla_server_url or os.environ.get("VLA_SERVER_URL")
    if not server_url:
        print("[오류] --vla-server-url 또는 환경변수 VLA_SERVER_URL이 필요합니다.")
        return 2

    camera_source = None
    state_source = None
    vla_client = None
    try:
        camera_source = RealCameraObservationSource.from_hardware_config_path(args.hardware_config)
        camera_source.open()

        state_source = ReadOnlyRealFollowerStateSource.from_port(
            port=args.follower_port, follower_id=args.follower_id, calibration_path=args.follower_calibration_path
        )
        state_source.connect()

        vla_client = VLAHttpClient(
            VLAClientConfig(server_url=server_url, timeout_s=args.vla_timeout_s, api_token=args.vla_api_token)
        )

        safety_gate = SafetyGate(SafetyGateConfig.from_repo_defaults())

        mujoco_backend = RealisticMuJoCoBackend(scene_path=args.mujoco_scene_path, profile_path=args.mujoco_profile_path)
        mujoco_backend.preflight()

        runner = ShadowModeRunner(
            camera_source=camera_source,
            state_source=state_source,
            vla_client=vla_client,
            safety_gate=safety_gate,
            mujoco_backend=mujoco_backend,
            task=args.task,
            reports_dir=Path(args.reports_dir) if args.reports_dir else None,
        )

        print(f"[Shadow] single-step 실행 시작 (task={args.task!r}, server={server_url})")
        run_result = runner.run_once()

    except (CameraSourceError, FollowerStateSourceError, SafetyGateConfigError, MuJoCoBackendError) as exc:
        print(f"[오류] preflight 실패: {exc}")
        return 2
    finally:
        if state_source is not None:
            state_source.disconnect()  # read-only reader의 disconnect - torque write 없음
        if camera_source is not None:
            camera_source.close()
        if vla_client is not None:
            vla_client.close()

    print("=" * 60)
    print(f"RESULT={run_result.result}")
    print(f"session_id={run_result.session_id}")
    print(f"report={run_result.report_path}")
    print(f"real_follower_write_count={run_result.real_follower_write_count}")
    if run_result.reasons:
        print("reasons:")
        for r in run_result.reasons:
            print(f"  - {r}")
    print("=" * 60)

    assert run_result.real_follower_write_count == 0, "치명적 불변식 위반: 실물 follower에 write가 발생했습니다."

    if run_result.result == RESULT_SHADOW_FAIL:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
