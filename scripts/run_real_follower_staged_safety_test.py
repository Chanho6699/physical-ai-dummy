#!/usr/bin/env python3
"""Candidate B 실물 SO-101 follower staged safety test - **처음으로 이 저장소가 실제
로봇에 write하는 경로**. 반드시 사용자가 로봇 옆에서 직접 지켜보며, 한 번에 stage 하나만
실행한다.

## 절대 규칙 (코드가 강제한다, 이 CLI가 우회할 방법을 만들지 않는다)

1. Safety Gate 임계값(``configs/safety_gate.yaml``, ``configs/follower_safe_mapper.yaml``)은
   전혀 바꾸지 않는다 - ``SafetyGate.evaluate()``를 그대로 호출만 한다.
2. ``ACCEPT`` action만 실제로 write된다. ``WOULD_CLAMP``/``REJECT``는 절대 write하지 않는다.
3. 첫 non-ACCEPT에서 그 stage는 즉시 멈춘다 - 남은 step을 건너뛰지 않는다.
4. stage마다 hard step 상한이 코드에 상수로 박혀 있다 (stage1=1, stage2=3~5, stage3<=15) -
   CLI로 이 상한 자체를 늘릴 수 없다.
5. stage는 자동으로 다음 단계로 넘어가지 않는다 - 이 스크립트를 새로 실행해야 다음 stage가
   시작되고, stage 2/3는 바로 전 stage의 PASS 영수증(receipt) 파일이 없으면 시작을 거부한다.
6. ``--dry-run``은 어떤 시리얼 포트도 열지 않고 어떤 장치와도 통신하지 않는다 - 계획(stage/
   step 수/checkpoint/영수증 여부)만 계산해서 출력하고 끝난다. (예외 클래스 두 개를 최상단에서
   import하기 때문에 ``lerobot.motors`` 모듈 자체는 로딩되지만 - 순수 코드 로딩일 뿐 포트를
   열거나 장치에 접근하지 않는다.) 실행 전에 계획만 확인하고 싶을 때 쓴다.

## VLA 실행 방식 두 가지 (``--vla-mode``, 필수)

- ``inprocess``: 이 laptop 프로세스가 직접 SmolVLA checkpoint를 로딩한다 (torch/transformers/
  CUDA 필요 - 기존 방식, 변경 없음).
- ``http``: laptop은 checkpoint를 전혀 로딩하지 않는다 - Desktop에서 이미 떠 있는
  ``scripts/run_vla_server.py``에 ``runtime/laptop/vla_client.py::VLAHttpClient``(기존
  HTTP client, 새 프로토콜 아님 - ``scripts/run_shadow_mode.py``가 이미 쓰는 것과 완전히
  동일)로 붙는다. stage 시작 전에 ``/health``를 확인하고, 서버가 보고하는 checkpoint가
  candidate B와 다르면(``model_id`` == 서버가 로딩한 ``--checkpoint`` 문자열, 정확히
  일치) 기본적으로 즉시 중단한다.

## 실행 예시 (실제 로봇 앞, 반드시 로봇을 직접 보면서)

    source ~/lerobot/.venv/bin/activate

    ### http 모드 (권장 - laptop에 GPU/checkpoint 불필요) ###
    # 0) Desktop에서 먼저 (별도 머신, GPU 있는 쪽):
    python scripts/run_vla_server.py \\
      --checkpoint outputs/pick_drop_v3_v4_combined69/smolvla_pick_drop_v3_v4_combined69_uniform_fresh/checkpoints/010000/pretrained_model \\
      --host 0.0.0.0 --port 9200

    # 1) laptop에서 dry-run (하드웨어 접근 없음, 서버 연결도 안 함)
    python scripts/run_real_follower_staged_safety_test.py --stage 1 --dry-run \\
      --vla-mode http --vla-server-url http://<desktop-tailscale-ip>:9200 \\
      --hardware-config configs/hardware.local.json --follower-port /dev/ttyACM0 --follower-id my_follower

    # 2) laptop에서 stage 1 실제 실행
    python scripts/run_real_follower_staged_safety_test.py --stage 1 \\
      --vla-mode http --vla-server-url http://<desktop-tailscale-ip>:9200 \\
      --hardware-config configs/hardware.local.json --follower-port /dev/ttyACM0 --follower-id my_follower \\
      --confirm-physically-present

    ### inprocess 모드 (기존 방식, laptop에 GPU/checkpoint 필요) ###
    python scripts/run_real_follower_staged_safety_test.py --stage 1 --dry-run \\
      --vla-mode inprocess \\
      --hardware-config configs/hardware.local.json --follower-port /dev/ttyACM0 --follower-id my_follower

    # 3) stage 1이 PASS면, 로봇을 직접 확인한 뒤에만 stage 2 (3~5 step)
    python scripts/run_real_follower_staged_safety_test.py --stage 2 --steps 3 \\
      --vla-mode http --vla-server-url http://<desktop-tailscale-ip>:9200 \\
      --hardware-config configs/hardware.local.json --follower-port /dev/ttyACM0 --follower-id my_follower \\
      --confirm-physically-present

    # 4) stage 2가 PASS면, stage 3 (short chunk, 최대 15 step)
    python scripts/run_real_follower_staged_safety_test.py --stage 3 --steps 10 \\
      --vla-mode http --vla-server-url http://<desktop-tailscale-ip>:9200 \\
      --hardware-config configs/hardware.local.json --follower-port /dev/ttyACM0 --follower-id my_follower \\
      --confirm-physically-present
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 예외 클래스만 미리 import한다 - 이 두 모듈 자체는 import 시점에 하드웨어를 건드리지 않는다
# (연결은 별도 메서드 호출 시에만 일어남), 그래서 dry-run 경로에서도 안전하게 최상단에 둘 수
# 있다. 이렇게 해 두면 아래 main()의 except 절이 참조하는 이름이 어떤 실패 경로에서도
# NameError 없이 항상 존재한다 (예: 확인 문구 오타로 더 무거운 import 줄까지 못 간 경우도).
from runtime.laptop.camera_source import CameraSourceError  # noqa: E402
from runtime.laptop.follower_state_source import FollowerStateSourceError  # noqa: E402

DEFAULT_CANDIDATE_B_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "pick_drop_v3_v4_combined69" / "smolvla_pick_drop_v3_v4_combined69_uniform_fresh"
    / "checkpoints" / "010000" / "pretrained_model"
)
DEFAULT_TASK = "Pick up the cube and drop it into the bin."
DEFAULT_RECEIPT_DIR = PROJECT_ROOT / "reports" / "real_follower_staged_safety_test_v1" / "receipts"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "real_follower_staged_safety_test_v1" / "reports"

# 요구사항: stage별 step 상한을 CLI로 못 늘리게 상수로 고정한다.
STAGE_FIXED_STEPS = {1: 1}
STAGE_STEP_RANGE = {2: (3, 5), 3: (1, 15)}
STAGE_DEFAULT_STEPS = {2: 3, 3: 10}

CONFIRMATION_PHRASE = "I AM PHYSICALLY PRESENT AND WATCHING THE ROBOT"


class StagedSafetyTestError(RuntimeError):
    pass


def _receipt_path(receipt_dir: Path, stage: int) -> Path:
    return receipt_dir / f"stage{stage}_receipt.json"


def _validate_hardware_config(path: Path) -> None:
    if not path.is_file():
        raise StagedSafetyTestError(f"--hardware-config 파일이 없습니다: {path}")
    if path.name == "hardware.example.json":
        raise StagedSafetyTestError(
            "hardware.example.json은 placeholder 설정입니다 - 실제 하드웨어 설정 파일을 지정하세요."
        )
    text = path.read_text(encoding="utf-8")
    if "REPLACE_WITH_" in text:
        raise StagedSafetyTestError(
            f"{path}에 아직 채워지지 않은 placeholder(REPLACE_WITH_...)가 있습니다 - 실제 값으로 채운 뒤 다시 시도하세요."
        )


def _resolve_max_steps(stage: int, steps_arg: int | None) -> int:
    if stage in STAGE_FIXED_STEPS:
        if steps_arg is not None and steps_arg != STAGE_FIXED_STEPS[stage]:
            raise StagedSafetyTestError(f"stage {stage}은 --steps를 지정할 수 없습니다 (항상 {STAGE_FIXED_STEPS[stage]}회).")
        return STAGE_FIXED_STEPS[stage]

    lo, hi = STAGE_STEP_RANGE[stage]
    steps = steps_arg if steps_arg is not None else STAGE_DEFAULT_STEPS[stage]
    if not (lo <= steps <= hi):
        raise StagedSafetyTestError(f"stage {stage}의 --steps는 {lo}~{hi} 범위여야 합니다 (받은 값: {steps}).")
    return steps


def _validate_vla_mode_args(args: argparse.Namespace) -> None:
    if args.vla_mode == "http" and not args.vla_server_url:
        raise StagedSafetyTestError("--vla-mode http이면 --vla-server-url이 필수입니다.")
    if args.vla_mode == "inprocess" and args.vla_server_url:
        raise StagedSafetyTestError("--vla-mode inprocess에서는 --vla-server-url을 지정할 수 없습니다 (모드가 섞였습니다).")


def _checkpoint_signature(checkpoint: Path) -> str:
    """checkpoint 경로의 마지막 4개 구성요소를 이어붙인 문자열 - Desktop과 laptop의 절대경로
    prefix가 달라도(다른 머신이므로 당연히 다를 수 있음) checkpoint 자체가 같은지 판단할 수
    있는 최소 서명. 정확한 전체 경로 일치를 요구하면 마운트 경로 차이만으로도 항상
    실패하므로 이렇게 하지 않는다."""
    parts = checkpoint.parts
    return "/".join(parts[-4:]) if len(parts) >= 4 else str(checkpoint)


def _verify_desktop_checkpoint(*, health, expected_checkpoint: Path, force: bool) -> None:
    """``/health``의 ``model_id``는 Desktop이 ``--checkpoint``로 받은 문자열 그대로다
    (``runtime/desktop/vla_server.py::SmolVLAPolicyRunner.model_id = checkpoint``) - 그래서
    "가능한 범위에서" 검증이란, 서명(마지막 4개 경로 구성요소)이 그 문자열에 부분 문자열로
    들어있는지 확인하는 것이다."""
    expected_sig = _checkpoint_signature(expected_checkpoint)
    model_id = health.model_id or ""
    if expected_sig not in model_id:
        message = (
            f"Desktop 서버가 보고한 model_id({model_id!r})에 기대한 candidate B checkpoint 서명"
            f"({expected_sig!r})이 없습니다 - 다른 checkpoint가 로딩되어 있을 수 있습니다."
        )
        if not force:
            raise StagedSafetyTestError(f"{message} (진짜 의도한 것이면 --force-checkpoint-mismatch로 우회 가능 - 신중히 판단하세요.)")
        print(f"[경고, 강제 진행] {message}")


def _check_previous_stage_receipt(receipt_dir: Path, stage: int) -> None:
    if stage == 1:
        return
    prev = _receipt_path(receipt_dir, stage - 1)
    if not prev.is_file():
        raise StagedSafetyTestError(
            f"stage {stage}를 시작하려면 stage {stage - 1}의 PASS 영수증이 필요합니다: {prev} (없음). "
            f"먼저 stage {stage - 1}를 실행해 PASS를 받으세요."
        )
    receipt = json.loads(prev.read_text(encoding="utf-8"))
    if receipt.get("verdict") != "PASS":
        raise StagedSafetyTestError(
            f"stage {stage - 1}의 영수증이 PASS가 아닙니다 (verdict={receipt.get('verdict')}) - stage {stage}를 시작할 수 없습니다."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--steps", type=int, default=None, help="stage 2/3에서만 사용 (stage 1은 항상 1회)")
    p.add_argument("--hardware-config", type=Path, required=True, help="카메라 설정 (configs/hardware.local.json 등, example 금지)")
    p.add_argument("--follower-port", required=True)
    p.add_argument("--follower-id", required=True)
    p.add_argument("--follower-calibration-path", default=None)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CANDIDATE_B_CHECKPOINT,
                    help="inprocess 모드: 실제로 로딩할 checkpoint. http 모드: Desktop이 로딩했어야 할 checkpoint (검증용, 로딩하지 않음)")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--vla-mode", required=True, choices=["inprocess", "http"],
                    help="inprocess: 이 프로세스가 직접 checkpoint 로딩(GPU/transformers 필요). http: Desktop VLA 서버에 HTTP로 연결(laptop에 GPU/checkpoint 불필요)")
    p.add_argument("--vla-server-url", default=None, help="--vla-mode http일 때 필수 (예: http://<desktop-tailscale-ip>:9200)")
    p.add_argument("--vla-timeout-s", type=float, default=15.0, help="http 모드 요청 timeout - 기본 VLAClientConfig와 동일")
    p.add_argument("--vla-api-token", default=None, help="http 모드에서 Desktop 서버가 인증을 요구하면 지정")
    p.add_argument("--force-checkpoint-mismatch", action="store_true",
                    help="http 모드에서 Desktop이 보고한 checkpoint가 candidate B와 다르게 보여도 강행 (기본은 차단 - 신중히 사용)")
    p.add_argument("--policy-type", default="smolvla")
    p.add_argument("--inference-device", default=None, help="inprocess 모드 전용 (cuda/cpu)")
    p.add_argument("--post-write-settle-s", type=float, default=0.3)
    p.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--dry-run", action="store_true", help="하드웨어에 전혀 접근하지 않고 계획만 출력한다")
    p.add_argument(
        "--confirm-physically-present", action="store_true",
        help="로봇 옆에서 직접 지켜보고 있다는 것을 명시적으로 확인 (--dry-run이 아니면 필수, 이것만으로 충분하지 않고 실행 시 타이핑 확인도 추가로 요구된다)",
    )
    return p.parse_args(argv)


def print_banner(*, stage: int, max_steps: int, checkpoint: Path, task: str, dry_run: bool, vla_mode: str, vla_server_url: str | None) -> None:
    print("=" * 70)
    print("MODE=REAL_FOLLOWER_STAGED_SAFETY_TEST")
    print(f"STAGE={stage}  MAX_STEPS={max_steps}")
    print(f"VLA_MODE={vla_mode}" + (f"  VLA_SERVER_URL={vla_server_url}" if vla_mode == "http" else "  (in-process, GPU/checkpoint on this machine)"))
    print(f"CHECKPOINT={checkpoint}" + ("  (Desktop이 이 checkpoint를 로딩했는지 /health로 검증됨)" if vla_mode == "http" else ""))
    print(f"TASK={task!r}")
    print("SAFETY_GATE=UNCHANGED (configs/safety_gate.yaml, configs/follower_safe_mapper.yaml)")
    print("WRITE_POLICY=ACCEPT-ONLY (WOULD_CLAMP/REJECT는 절대 실물에 쓰지 않음)")
    print("HALT_POLICY=최초 non-ACCEPT에서 즉시 stage 중단 (건너뛰고 계속하지 않음)")
    print(f"DRY_RUN={dry_run}")
    print("=" * 70)


def require_interactive_confirmation() -> None:
    print()
    print("!!! 지금부터 실제 SO-101 follower에 write를 시도합니다. !!!")
    print("로봇 옆에 물리적으로 있으면서 즉시 개입(전원 차단 등)할 수 있는 상태여야 합니다.")
    typed = input(f"계속하려면 정확히 다음 문구를 입력하세요: {CONFIRMATION_PHRASE}\n> ")
    if typed.strip() != CONFIRMATION_PHRASE:
        raise StagedSafetyTestError("확인 문구가 일치하지 않습니다 - 중단합니다.")


def print_step_report(step) -> None:
    print(f"\n--- step {step.step} ---")
    print(f"  before_state_deg = {step.before_state_deg}")
    if step.step_error:
        print(f"  [중단] step_error = {step.step_error}")
        return
    print(f"  raw_action_deg   = {step.raw_action_deg}")
    print(f"  safety_decision  = {step.safety_decision}  reasons={list(step.safety_reasons)}")
    print(f"  written          = {step.written}")
    if step.written:
        print(f"  sent_action_deg  = {step.sent_action_deg}")
        print(f"  after_state_deg  = {step.after_state_deg}")
        print(f"  delta_deg        = {step.delta_deg}")
    elif step.write_error:
        print(f"  write_error      = {step.write_error}")
    elif step.would_be_safe_action_deg is not None:
        print(f"  (write 안 함) would_be_safe_action_deg = {step.would_be_safe_action_deg}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        max_steps = _resolve_max_steps(args.stage, args.steps)
        _check_previous_stage_receipt(args.receipt_dir, args.stage)
        _validate_vla_mode_args(args)
        if not args.dry_run:
            _validate_hardware_config(args.hardware_config)
    except StagedSafetyTestError as exc:
        print(f"[오류] {exc}")
        return 2

    print_banner(
        stage=args.stage, max_steps=max_steps, checkpoint=args.checkpoint, task=args.task, dry_run=args.dry_run,
        vla_mode=args.vla_mode, vla_server_url=args.vla_server_url,
    )

    if args.dry_run:
        print("\n[dry-run] 하드웨어/서버에 전혀 접근하지 않았습니다. 위 계획이 맞으면 --dry-run 없이 다시 실행하세요.")
        if args.vla_mode == "http":
            print(f"[dry-run] 실제 실행 시 {args.vla_server_url}/health를 먼저 확인하고, "
                  f"checkpoint 서명({_checkpoint_signature(args.checkpoint)!r})이 일치하지 않으면 write 없이 중단합니다.")
        print(f"[dry-run] --confirm-physically-present 플래그와 실행 중 타이핑 확인이 추가로 필요합니다.")
        return 0

    if not args.confirm_physically_present:
        print("[오류] --dry-run이 아니면 --confirm-physically-present가 필수입니다.")
        return 2

    camera_source = None
    state_source = None
    writer = None
    try:
        require_interactive_confirmation()  # 이 try 블록 안이어야 잘못된 문구 입력도 깔끔하게 처리된다

        # 여기서부터만 하드웨어/lerobot 관련 모듈을 import한다 (dry-run 경로는 여기 절대 안 옴).
        from lerobot.robots.so_follower import SOFollower, SOFollowerConfig

        # CameraSourceError/FollowerStateSourceError는 이미 파일 최상단에서 import했다 -
        # 여기서 다시 import하면 (설령 같은 이름이라도) 이 함수 전체에서 그 이름이 지역
        # 변수로 취급되어, 이 줄에 도달하기 *전에* 예외가 나는 경로(예: 확인 문구 오타)에서
        # 아래 ``except`` 절이 ``UnboundLocalError``를 내는 버그가 있었다 - 다시 만들지 않는다.
        from hardware.safety.staged_follower_writer import StagedFollowerArmedWriter
        from runtime.laptop.camera_source import RealCameraObservationSource
        from runtime.laptop.follower_state_source import ReadOnlyRealFollowerStateSource
        from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig
        from runtime.laptop.staged_real_rollout import PASS, StagedRealRolloutRunner

        # -- VLA client 준비: 여기서 http/inprocess가 갈린다 -----------------------------
        # http 모드는 이 if 분기 밖에서 InProcessSmolVLAClient를 단 한 번도 import하지
        # 않는다 - laptop에 transformers/CUDA가 없어도 이 경로는 항상 동작해야 한다는
        # 요구사항을 import 구조로도 강제한다.
        if args.vla_mode == "http":
            from runtime.laptop.vla_client import VLAClientConfig, VLAHttpClient

            vla_client = VLAHttpClient(
                VLAClientConfig(server_url=args.vla_server_url, timeout_s=args.vla_timeout_s, api_token=args.vla_api_token)
            )
            print(f"[준비] Desktop VLA 서버 health 확인 중: {args.vla_server_url}")
            health = vla_client.check_health()
            if not health.ok:
                raise StagedSafetyTestError(
                    f"Desktop VLA 서버({args.vla_server_url})가 정상이 아닙니다 - write를 시도하지 않고 중단합니다. "
                    f"(status={health.status}, raw_error={health.raw_error}, round_trip_ms={health.round_trip_ms})"
                )
            print(f"[준비] health OK: backend={health.backend} model_id={health.model_id} device={health.device}")
            _verify_desktop_checkpoint(health=health, expected_checkpoint=args.checkpoint, force=args.force_checkpoint_mismatch)
        else:
            from runtime.laptop.inprocess_vla_client import InProcessSmolVLAClient

            vla_client = InProcessSmolVLAClient(
                checkpoint=str(args.checkpoint), policy_type=args.policy_type, device=args.inference_device
            )
            print(f"[준비] SmolVLA checkpoint 로딩 완료 (in-process): {args.checkpoint}")

        camera_source = RealCameraObservationSource.from_hardware_config_path(args.hardware_config)
        camera_source.open()

        state_source = ReadOnlyRealFollowerStateSource.from_port(
            port=args.follower_port, follower_id=args.follower_id, calibration_path=args.follower_calibration_path
        )
        state_source.connect()

        safety_gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
        source_summary = safety_gate.config.source_summary()
        if source_summary.get("uses_calibration_fallback"):
            print(
                "[경고] Safety Gate가 실제 캘리브레이션 파일이 아니라 fallback 범위를 쓰고 있습니다 "
                f"(calibration_file_path={source_summary.get('calibration_file_path')}). "
                "실제 캘리브레이션 파일이 있는지 다시 확인하는 것을 강력히 권장합니다."
            )

        follower_config = SOFollowerConfig(
            port=args.follower_port, id=args.follower_id, cameras={}, disable_torque_on_disconnect=True,
        )
        follower = SOFollower(follower_config)
        writer = StagedFollowerArmedWriter(follower=follower, max_write_count=max_steps)
        writer.connect()
        print(f"[준비] follower 연결 완료: port={args.follower_port} id={args.follower_id} (write budget={max_steps})")

        runner = StagedRealRolloutRunner(
            camera_source=camera_source, state_source=state_source, vla_client=vla_client, safety_gate=safety_gate,
            writer=writer, task=args.task, checkpoint_label=str(args.checkpoint),
            post_write_settle_s=args.post_write_settle_s,
        )

        print(f"\n[실행] stage {args.stage} 시작 (max_steps={max_steps})...")
        result = runner.run_stage(stage=args.stage, max_steps=max_steps)

        for step in result.steps:
            print_step_report(step)

        print(f"\n{'=' * 70}")
        print(f"STAGE {args.stage}: {result.verdict}")
        if result.stop_reason:
            print(f"stop_reason: {result.stop_reason}")
        print(f"real_follower_write_count = {result.real_follower_write_count}")
        print(f"{'=' * 70}")

        args.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.report_dir / f"stage{args.stage}_{int(time.time())}.json"
        report_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n리포트 저장: {report_path}")

        if result.verdict == PASS:
            args.receipt_dir.mkdir(parents=True, exist_ok=True)
            receipt_path = _receipt_path(args.receipt_dir, args.stage)
            receipt_path.write_text(json.dumps({
                "stage": args.stage, "verdict": "PASS", "max_steps": max_steps,
                "checkpoint": str(args.checkpoint), "timestamp": time.time(), "report_path": str(report_path),
            }, indent=2), encoding="utf-8")
            print(f"영수증 저장 (다음 stage 시작에 필요): {receipt_path}")
            print(f"\n다음: 로봇을 직접 확인한 뒤에만 stage {args.stage + 1}를 실행하세요." if args.stage < 3 else "\n모든 stage 통과.")
        else:
            print("\n영수증을 쓰지 않았습니다 - 이 stage는 PASS가 아니므로 다음 stage를 시작할 수 없습니다.")

        return 0 if result.verdict == PASS else 1

    except (CameraSourceError, FollowerStateSourceError, StagedSafetyTestError) as exc:
        print(f"[오류] {exc}")
        return 2
    finally:
        if writer is not None:
            writer.disconnect()
        if state_source is not None:
            try:
                state_source.disconnect()
            except Exception:  # noqa: BLE001
                pass
        if camera_source is not None:
            try:
                camera_source.close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
