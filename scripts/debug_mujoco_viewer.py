#!/usr/bin/env python3
"""MuJoCo GUI 렌더링 경로 최소 재현 스크립트 (네트워크/리더암/팔로워암/FastAPI 전부 사용하지 않음).

`scripts/run_remote_mujoco_diagnostic.py`와 완전히 분리된 스크립트다. 목적은 단 하나:
"MuJoCo GUI 창이 이 머신/이 WSLg 세션에서 실제로 렌더링되는가?"만 확인한다.

지원 모드:
    --mode passive          : mujoco.viewer.launch_passive (main thread에서 직접 step + sync)
    --mode launch            : mujoco.viewer.launch (MuJoCo 자체 physics thread가 굴러가는 표준 GUI, 이 프로세스의
                                main thread는 창이 닫힐 때까지 블록된다 - 공식 API 기준 사용법)
    --mode render-offscreen  : GUI 없이 mujoco.Renderer로 프레임을 PNG로 저장 (GUI 창 자체가 실패해도
                                물리 시뮬레이션/scene 로딩 자체는 정상인지 분리해서 확인하기 위함)

모든 모드에서 scene.xml만 로딩하고, 30초 동안(옵션으로 조절 가능) 시뮬레이션을 진행한다.
Ctrl+C로 언제든 종료할 수 있다. 예외는 전부 한글 메시지로 출력한다.

실행 예:
    python scripts/debug_mujoco_viewer.py --mode passive
    python scripts/debug_mujoco_viewer.py --mode launch
    python scripts/debug_mujoco_viewer.py --mode render-offscreen --frames 60 \\
        --output-dir reports/mujoco_gui_debug
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SCENE = PROJECT_ROOT / "simulation" / "mujoco" / "assets" / "scene.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "mujoco_gui_debug"

BAR = "=" * 68


def _print_header(title: str) -> None:
    print(BAR)
    print(f"[진단] {title}")
    print(BAR)


def _print_env() -> None:
    import os

    print(f"[환경] 실행 스레드 = {threading.current_thread().name} (main 여부: {threading.current_thread() is threading.main_thread()})")
    print(f"[환경] DISPLAY = {os.environ.get('DISPLAY')!r}")
    print(f"[환경] WAYLAND_DISPLAY = {os.environ.get('WAYLAND_DISPLAY')!r}")
    print(f"[환경] XDG_RUNTIME_DIR = {os.environ.get('XDG_RUNTIME_DIR')!r}")
    print(f"[환경] MUJOCO_GL = {os.environ.get('MUJOCO_GL')!r} (GUI 창 자체는 이 값과 무관하게 항상 GLFW를 사용함)")
    try:
        import mujoco

        print(f"[환경] MuJoCo = {mujoco.__version__}")
    except Exception as exc:  # pragma: no cover - 진단 스크립트 자체가 실패하는 경우
        print(f"[실패] mujoco import 실패: {exc}")
        raise
    try:
        import glfw

        print(f"[환경] glfw-py = {glfw.get_version_string().decode() if isinstance(glfw.get_version_string(), bytes) else glfw.get_version_string()}")
        if glfw.init():
            platform_id = glfw.get_platform() if hasattr(glfw, "get_platform") else None
            name = {getattr(glfw, "PLATFORM_X11", -1): "X11", getattr(glfw, "PLATFORM_WAYLAND", -2): "Wayland"}.get(
                platform_id, f"알 수 없음({platform_id})"
            )
            print(f"[환경] GLFW가 선택한 플랫폼 = {name}")
    except Exception as exc:
        print(f"[경고] glfw 정보 조회 실패 (치명적이지 않음): {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["passive", "launch", "render-offscreen"], default="passive")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="로딩할 MuJoCo scene XML 경로")
    parser.add_argument("--duration", type=float, default=30.0, help="passive/launch 모드에서 시뮬레이션할 시간(초)")
    parser.add_argument("--frames", type=int, default=60, help="render-offscreen 모드에서 저장할 프레임 수")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="render-offscreen 모드 PNG 저장 디렉터리"
    )
    return parser


def _load(scene: Path):
    import mujoco

    if not scene.is_file():
        raise FileNotFoundError(f"scene 파일을 찾을 수 없습니다: {scene}")
    print(f"[검사] scene.xml 로딩: {scene}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"[통과] 모델 로딩 완료 (joint={model.njnt}, actuator={model.nu}, body={model.nbody})")
    return model, data


def run_passive(scene: Path, duration: float) -> int:
    """공식 권장 패턴: main thread에서 launch_passive -> while is_running(): step; sync()."""
    import mujoco
    import mujoco.viewer as mj_viewer

    assert threading.current_thread() is threading.main_thread(), "GUI는 반드시 main thread에서 생성해야 합니다."

    model, data = _load(scene)

    print("[검사] launch_passive main-thread 실행")
    try:
        viewer_cm = mj_viewer.launch_passive(model, data)
    except Exception as exc:
        print(f"[실패] launch_passive 호출 자체가 예외를 던졌습니다: {exc}")
        return 1

    step_count = 0
    try:
        with viewer_cm as viewer:
            print(f"[통과] 창 생성됨 (is_running={viewer.is_running()})")
            start = time.monotonic()
            last_report = start
            while viewer.is_running() and (time.monotonic() - start) < duration:
                loop_start = time.monotonic()
                mujoco.mj_step(model, data)
                try:
                    viewer.sync()
                except Exception as exc:  # sync 중 예외는 조용히 삼키지 않는다
                    print(f"[실패] viewer.sync() 중 예외 발생: {exc}")
                    raise
                step_count += 1
                now = time.monotonic()
                if now - last_report >= 5.0:
                    print(f"[진행] {now - start:.1f}s 경과, step={step_count}, is_running={viewer.is_running()}")
                    last_report = now
                elapsed = time.monotonic() - loop_start
                sleep_for = model.opt.timestep - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        print(f"[통과] viewer 정상 종료 (총 step={step_count})")
    except KeyboardInterrupt:
        print("\n[중단] Ctrl+C로 종료했습니다.")
    except Exception as exc:
        print(f"[실패] passive 루프 중 예외 발생: {exc}")
        return 1
    return 0


def run_launch(scene: Path, duration: float) -> int:
    """mujoco.viewer.launch(): MuJoCo 자체 physics thread가 도는 표준 GUI. main thread는 창이 닫힐 때까지 블록된다."""
    import mujoco.viewer as mj_viewer

    assert threading.current_thread() is threading.main_thread(), "GUI는 반드시 main thread에서 생성해야 합니다."

    model, data = _load(scene)

    print(f"[검사] launch() 실행 - {duration:.0f}초 후 자동으로 프로세스를 종료합니다 (또는 창을 직접 닫거나 Ctrl+C).")
    print("[안내] launch()는 공식 API 기준으로 blocking 호출이며, 커스텀 loader 콜백은 필수가 아닙니다.")

    timer = threading.Timer(duration, lambda: _force_exit_note())
    timer.daemon = True
    timer.start()
    try:
        mj_viewer.launch(model, data)
        print("[통과] launch() 창이 닫혀 정상적으로 반환되었습니다.")
    except KeyboardInterrupt:
        print("\n[중단] Ctrl+C로 종료했습니다.")
    except Exception as exc:
        print(f"[실패] launch() 중 예외 발생: {exc}")
        return 1
    finally:
        timer.cancel()
    return 0


def _force_exit_note() -> None:
    print(f"\n[안내] duration 경과. launch()는 창을 직접 닫아야 반환됩니다 - 창을 닫아 주세요 (Quit 버튼 또는 Ctrl+C).")


def run_render_offscreen(scene: Path, frames: int, output_dir: Path) -> int:
    """GUI 없이 mujoco.Renderer로 프레임을 PNG로 저장한다. 이게 성공하면 물리/scene은 정상이라는 뜻이다."""
    import mujoco
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        print("[실패] Pillow(PIL)가 설치되어 있지 않아 PNG로 저장할 수 없습니다. `pip install pillow` 후 다시 시도하세요.")
        return 1

    model, data = _load(scene)

    print(f"[검사] offscreen renderer 실행 (frames={frames})")
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception as exc:
        print(f"[실패] mujoco.Renderer 생성 실패: {exc}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    try:
        for i in range(frames):
            mujoco.mj_step(model, data)
            renderer.update_scene(data)
            frame = renderer.render()
            frame_std = float(np.std(frame))
            path = output_dir / f"frame_{i:03d}.png"
            Image.fromarray(frame).save(path)
            saved += 1
            if i == 0:
                print(f"[정보] 첫 프레임 픽셀 표준편차={frame_std:.2f} (0에 가까우면 완전히 단색 = 렌더링 실패 의심)")
    except Exception as exc:
        print(f"[실패] 프레임 렌더링/저장 중 예외 발생 (frame {saved}): {exc}")
        return 1
    finally:
        renderer.close()

    print(f"[통과] PNG {saved}프레임을 {output_dir}에 저장했습니다.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    _print_header("MuJoCo GUI 렌더링 경로 점검 (최소 재현 스크립트)")
    _print_env()
    print(BAR)

    try:
        if args.mode == "passive":
            code = run_passive(args.scene, args.duration)
        elif args.mode == "launch":
            code = run_launch(args.scene, args.duration)
        else:
            code = run_render_offscreen(args.scene, args.frames, args.output_dir)
    except FileNotFoundError as exc:
        print(f"[실패] {exc}")
        return 1
    except Exception as exc:  # 진단 스크립트에서는 모든 예외를 한글로 드러낸다
        print(f"[실패] 예상치 못한 예외: {type(exc).__name__}: {exc}")
        return 1

    print(BAR)
    if code == 0:
        print("[판정] 이 모드는 예외 없이 완료되었습니다. (창 내용이 실제로 보였는지는 사람이 직접 확인해야 합니다 - "
              "이 스크립트는 창 생성/루프 진행만 보증합니다. render-offscreen 모드는 PNG 파일로 시각 검증이 가능합니다.)")
    else:
        print("[판정] 이 모드는 실패했습니다. 위 [실패] 메시지를 확인하세요.")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[중단] Ctrl+C로 종료했습니다.")
        raise SystemExit(130)
