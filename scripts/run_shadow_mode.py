#!/usr/bin/env python3
"""Shadow Mode v1 실행 CLI - single-step (섹션 19, 21).

실제 fixed/wrist 카메라 + 실제 SO-101 follower state(읽기 전용)를 SmolVLA에 보내고,
반환된 action을 Safety Gate -> Realistic MuJoCo에서 단 1 step 검증한다. 실제 SO-101
follower에는 어떤 명령도 보내지 않는다 (``REAL_SO101_WRITE=DISABLED``).

# VLA backend 두 가지 (둘 중 정확히 하나를 선택해야 한다 - 기본값 없음)

1. ``--checkpoint``: SmolVLA checkpoint를 이 프로세스가 직접 로딩해 추론한다(네트워크 왕복
   없음, ``runtime.laptop.inprocess_vla_client.InProcessSmolVLAClient``). checkpoint의
   저장된 preprocessor/postprocessor를 그대로 쓴다(카메라 rename이 이미 내장돼 있으면
   중복으로 다시 rename하지 않는다). **이전 실험(예: 20-episode pilot, old 10k/15k/20k
   checkpoint)을 암묵적으로 기본값으로 쓰지 않는다** - ``--checkpoint``를 명시해야만
   동작한다.
2. ``--vla-server-url``: 이미 떠 있는 Desktop VLA HTTP 서버(``scripts/run_vla_server.py
   --checkpoint ...``)와 통신한다(기존 방식, 변경 없음).

실행 예시(in-process, Grid35 checkpoint)::

    python scripts/run_shadow_mode.py \\
        --task "Pick up the cube and place it in the target area." \\
        --checkpoint outputs/grid35/smolvla_grid35_fresh_v1/checkpoints/010000/pretrained_model \\
        --expected-train-dataset data/so101_cube_xy_grid35_v1 \\
        --eval-mode midpoint-shadow --scene-label T01 \\
        --follower-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14113538-if00 \\
        --follower-id chanho_follower

실행 예시(기존 HTTP 방식)::

    python scripts/run_shadow_mode.py \\
        --task "Pick up the cube" \\
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

from runtime.common.checkpoint_provenance import (
    check_train_dataset_provenance,
    infer_checkpoint_step,
    read_checkpoint_train_dataset_root,
)
from runtime.laptop.camera_source import CameraSourceError, RealCameraObservationSource
from runtime.laptop.follower_state_source import FollowerStateSourceError, ReadOnlyRealFollowerStateSource
from runtime.laptop.inprocess_vla_client import InProcessSmolVLAClient, InProcessVLAClientError
from runtime.laptop.mujoco_shadow_backend import MuJoCoBackendError, RealisticMuJoCoBackend
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig, SafetyGateConfigError
from runtime.laptop.shadow_logger import RESULT_SHADOW_FAIL, RESULT_SHADOW_PASS_WITH_CLAMP
from runtime.laptop.shadow_mode_runner import ShadowModeRunner
from runtime.laptop.vla_client import VLAClientConfig, VLAHttpClient

DEFAULT_HARDWARE_CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware.local.json"
SINGLE_STEP_MODE = "single-step"

# 요구사항 7번 - 이번 Grid35/Midpoint10 실험용 평가 모드 두 가지. 둘 다 report에 라벨만
# 남기고 동작 자체(observation/inference/safety/mujoco 파이프라인)는 바꾸지 않는다 - "동일
# scene 반복 실행"/"held-out scene에서 실행"은 운영자가 실제로 로봇 앞의 물체를 어디에
# 두느냐로 결정되는 것이지, 이 스크립트가 물체 위치를 알거나 바꿀 방법이 없다.
EVAL_MODE_FIXED_SCENE_REPEAT = "fixed-scene-repeat"
EVAL_MODE_MIDPOINT_SHADOW = "midpoint-shadow"
EVAL_MODES = (EVAL_MODE_FIXED_SCENE_REPEAT, EVAL_MODE_MIDPOINT_SHADOW)


def build_checkpoint_metadata(checkpoint_dir: Path, *, expected_train_dataset: Path | None) -> dict:
    """checkpoint absolute path/step/학습 dataset provenance를 report용 dict로 만든다.

    ``expected_train_dataset``이 주어지면(``--expected-train-dataset``) 실제 저장된
    ``train_config.json``의 dataset root와 비교해 다르면 경고를 남긴다 - "과거 실험
    checkpoint를 이번 실험이라고 착각해서 쓰는" 사고를 막기 위한 안전장치다(하드 실패는
    아니다 - 머신마다 절대경로가 다를 수 있다).
    """
    recorded_root, read_error = read_checkpoint_train_dataset_root(checkpoint_dir)
    mismatch_warning = None
    if expected_train_dataset is not None:
        mismatch_warning = check_train_dataset_provenance(checkpoint_dir, expected_train_dataset)
    return {
        "checkpoint_path": str(checkpoint_dir),
        "checkpoint_step": infer_checkpoint_step(checkpoint_dir),
        "train_dataset_provenance": {
            "recorded_train_dataset_root": recorded_root,
            "read_error": read_error,
            "expected_train_dataset": str(expected_train_dataset) if expected_train_dataset else None,
            "mismatch_warning": mismatch_warning,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow Mode v1 (single-step)")
    parser.add_argument("--task", required=True, help="VLA에 전달할 task instruction")
    parser.add_argument("--mode", default=SINGLE_STEP_MODE, choices=[SINGLE_STEP_MODE], help="v1은 single-step만 지원합니다.")

    # -- backend 선택 (checkpoint in-process vs HTTP) - 정확히 하나만 -----------------
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="SmolVLA checkpoint 경로(pretrained_model 디렉터리) - 이 프로세스가 직접 로딩합니다. "
        "--vla-server-url과 동시에 쓸 수 없습니다. 기본값이 없습니다 - 과거 실험 checkpoint를 "
        "암묵적으로 쓰지 않기 위함입니다.",
    )
    parser.add_argument("--policy-type", default="smolvla")
    parser.add_argument("--inference-device", default=None, help="--checkpoint 사용 시 cuda/cpu (미지정 시 자동 감지)")
    parser.add_argument(
        "--expected-train-dataset",
        default=None,
        help="(선택) --checkpoint의 train_config.json에 기록된 학습 dataset과 비교해 다르면 "
        "report에 경고를 남깁니다. 예: data/so101_cube_xy_grid35_v1",
    )
    parser.add_argument("--vla-server-url", default=None, help="미지정 시 환경변수 VLA_SERVER_URL 사용")
    parser.add_argument("--vla-timeout-s", type=float, default=15.0)
    parser.add_argument("--vla-api-token", default=None)

    # -- 이번 실험용 평가 모드/scene 메타데이터 (요구사항 7번) --------------------------
    parser.add_argument(
        "--eval-mode",
        default=None,
        choices=EVAL_MODES,
        help=f"이번 세션의 평가 모드 라벨({', '.join(EVAL_MODES)}) - report에만 기록되고 파이프라인 동작은 바꾸지 않습니다.",
    )
    parser.add_argument(
        "--scene-label",
        default=None,
        help="예: T01~T10 (midpoint-shadow에서 어떤 held-out scene인지 사람이 지정) - 물체의 실제 "
        "XY는 자동으로 기록하지 않습니다(현재 dataset metadata에 없음).",
    )
    parser.add_argument("--scene-note", default=None, help="scene에 대한 자유 형식 메모")

    parser.add_argument("--hardware-config", default=str(DEFAULT_HARDWARE_CONFIG_PATH))
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--follower-id", default="chanho_follower")
    parser.add_argument("--follower-calibration-path", default=None)
    parser.add_argument("--mujoco-scene-path", default=None)
    parser.add_argument("--mujoco-profile-path", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument(
        "--no-camera-frame-capture",
        action="store_true",
        help="workspace/wrist 프레임을 JPEG로 저장하지 않습니다 (기본은 저장합니다 - fixed-seed 재평가를 위해).",
    )
    args = parser.parse_args(argv)

    if args.checkpoint and args.vla_server_url:
        parser.error("--checkpoint와 --vla-server-url은 동시에 지정할 수 없습니다 (VLA backend를 하나만 선택하세요).")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 60)
    print("MODE=SHADOW")
    print("VLA_LOCATION=" + ("IN_PROCESS" if args.checkpoint else "DESKTOP"))
    print("EXECUTION_BACKEND=REALISTIC_MUJOCO")
    print("REAL_SO101_WRITE=DISABLED")
    print("=" * 60)

    server_url = args.vla_server_url or os.environ.get("VLA_SERVER_URL")
    if not args.checkpoint and not server_url:
        print("[오류] --checkpoint 또는 --vla-server-url(또는 환경변수 VLA_SERVER_URL) 중 하나가 필요합니다.")
        return 2

    scene_metadata = None
    if args.scene_label or args.scene_note:
        scene_metadata = {"label": args.scene_label, "note": args.scene_note}

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

        checkpoint_metadata: dict | None = None
        if args.checkpoint:
            checkpoint_dir = Path(args.checkpoint).resolve()
            checkpoint_metadata = build_checkpoint_metadata(
                checkpoint_dir,
                expected_train_dataset=Path(args.expected_train_dataset) if args.expected_train_dataset else None,
            )
            mismatch = checkpoint_metadata["train_dataset_provenance"]["mismatch_warning"]
            if mismatch:
                print(f"[경고] checkpoint provenance: {mismatch}")
            vla_client = InProcessSmolVLAClient(
                checkpoint=str(checkpoint_dir), policy_type=args.policy_type, device=args.inference_device
            )
            print(f"[Shadow] in-process SmolVLA checkpoint 로딩 완료: {checkpoint_dir} (device={vla_client._policy_runner.device_label()})")
        else:
            vla_client = VLAHttpClient(
                VLAClientConfig(server_url=server_url, timeout_s=args.vla_timeout_s, api_token=args.vla_api_token)
            )
            # HTTP 모드에서는 이 스크립트가 desktop이 실제로 로딩한 checkpoint를 독립적으로
            # 검증할 수 없다(그건 desktop 프로세스의 책임) - report의 vla.model_id/backend
            # (predict 응답에 이미 포함됨)가 그 신호를 대신한다. 여기서 없는 정보를
            # 만들어내지 않는다.

        safety_gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
        source_summary = safety_gate.config.source_summary()
        if source_summary.get("uses_calibration_fallback"):
            print(
                "[경고] Safety Gate가 실제 팔로워 캘리브레이션 파일이 아니라 "
                f"fallback 범위를 쓰고 있습니다 (calibration_file_path={source_summary.get('calibration_file_path')}). "
                "Real Mode 전에 실제 캘리브레이션 파일을 확인하세요."
            )

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
            checkpoint_metadata=checkpoint_metadata,
            evaluation_mode=args.eval_mode,
            scene_metadata=scene_metadata,
            capture_camera_frames=not args.no_camera_frame_capture,
        )

        print(f"[Shadow] single-step 실행 시작 (task={args.task!r}, backend={'checkpoint' if args.checkpoint else server_url})")
        run_result = runner.run_once()

    except (
        CameraSourceError,
        FollowerStateSourceError,
        SafetyGateConfigError,
        MuJoCoBackendError,
        InProcessVLAClientError,
    ) as exc:
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
