"""한글 CLI 상태 메시지와 진행률 표시.

이 모듈은 오직 "출력 문자열 생성/인쇄"만 담당한다. 계산/판정 로직은 여기 두지 않는다
(safety_checks.py, dataset_action_replay.py 참고). 로그 파일/JSON에는 ANSI escape code를
절대 기록하지 않는다 - 이 모듈의 함수들은 stdout에만 출력하고, 리포트 생성 코드와는
분리되어 있다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from simulation.mujoco.action_mapping import JointMapping
from simulation.mujoco.diagnostic_analysis import DiagnosticEvent
from simulation.mujoco.safety_checks import SafetyEvent


@dataclass
class ConsoleOptions:
    quiet: bool = False
    verbose: bool = False
    use_color: bool = True


_RESET = "\033[0m"
_COLORS = {
    "PASS": "\033[32m",  # green
    "WARN": "\033[33m",  # yellow
    "BLOCKED": "\033[31m",  # red
    "HEADER": "\033[36m",  # cyan
    "DIM": "\033[2m",
}


def resolve_use_color(no_color_flag: bool) -> bool:
    if no_color_flag:
        return False
    return sys.stdout.isatty()


def _c(text: str, tag: str, opts: ConsoleOptions) -> str:
    if not opts.use_color:
        return text
    return f"{_COLORS.get(tag, '')}{text}{_RESET}"


def print_header(
    opts: ConsoleOptions,
    *,
    dataset_root: str,
    episode_index: int,
    total_frames: int,
    fps: int,
    speed: float,
    mode: str,
) -> None:
    if opts.quiet:
        return
    bar = "=" * 68
    print(bar)
    print(_c("[준비] SO-101 MuJoCo 데이터셋 Action Replay", "HEADER", opts))
    print(bar)
    print(f"[데이터셋] {dataset_root}")
    print(f"[에피소드] {episode_index}")
    print(f"[전체 프레임] {total_frames}")
    print(f"[FPS] {fps}")
    print(f"[재생 속도] {speed}배")
    print(f"[실행 모드] {mode}")
    print("[로봇 모델] SO-101 MuJoCo (robotstudio_so101, mujoco_menagerie, Apache-2.0)")
    print("-" * 68)


def print_check(opts: ConsoleOptions, level: str, label: str, message: str) -> None:
    if opts.quiet and level != "BLOCKED":
        return
    tag = {"PASS": "[통과]", "WARN": "[경고]", "BLOCKED": "[차단]"}.get(level, f"[{level}]")
    print(f"{_c(tag, level, opts)} {label}: {message}")


def print_section(opts: ConsoleOptions, label: str) -> None:
    if opts.quiet:
        return
    print(f"[검사] {label}...")


def print_mapping_table(opts: ConsoleOptions, mapping: tuple[JointMapping, ...]) -> None:
    if opts.quiet:
        return
    print("===== Action Mapping =====")
    for i, entry in enumerate(mapping):
        print(
            f"[{i}] {entry.dataset_name:<14s} -> {entry.mujoco_joint_name + '_joint':<20s} "
            f"-> {entry.mujoco_actuator_name + '_actuator'}"
        )
        if opts.verbose:
            print(
                f"      unit={entry.unit} scale={entry.scale:.6f} offset={entry.offset} "
                f"sign={entry.sign:+.1f}"
            )
            print(f"      근거: {entry.basis}")
    print("==========================")


def print_progress(
    opts: ConsoleOptions,
    frame: int,
    total_frames: int,
    safety_level: str,
    *,
    bar_width: int = 20,
) -> None:
    if opts.quiet:
        return
    ratio = frame / total_frames if total_frames else 0.0
    filled = int(bar_width * ratio)
    bar = "█" * filled + "░" * (bar_width - filled)
    tag = {"PASS": "PASS", "WARN": "WARN", "BLOCKED": "BLOCKED"}.get(safety_level, safety_level)
    colored_tag = _c(tag, safety_level, opts)
    line = f"[재생 중] {bar} {ratio * 100:5.1f}% | Frame {frame}/{total_frames} | Safety {colored_tag}"
    print(line)


def print_verbose_frame(opts: ConsoleOptions, frame: int, events: list[SafetyEvent]) -> None:
    if not opts.verbose:
        return
    for evt in events:
        print(f"  [frame {frame}] {evt.level} {evt.code}: {evt.message}")


def print_blocked(opts: ConsoleOptions, event: SafetyEvent) -> None:
    print(_c(f"[차단] {event.message}", "BLOCKED", opts))
    if event.frame is not None:
        print(f"[프레임] {event.frame}")
    if event.value is not None:
        print(f"[입력값] {event.value:.4f}")
    if event.limit is not None:
        print(f"[허용 범위] {event.limit[0]:.4f} ~ {event.limit[1]:.4f}")
    print("[조치] 시뮬레이션을 중지하고 실물 실행을 금지합니다.")


def print_final_summary(
    opts: ConsoleOptions,
    *,
    processed_frames: int,
    total_frames: int,
    joint_limit_violations: int,
    actuator_limit_violations: int,
    max_delta_violations: int,
    collisions: int,
    nan_count: int,
    final_result: str,
    report_path: str,
) -> None:
    bar = "=" * 68
    print(bar)
    print(_c("[완료] MuJoCo Action Replay 종료", "HEADER", opts))
    print(bar)
    print(f"[처리 프레임] {processed_frames} / {total_frames}")
    print(f"[관절 제한 위반] {joint_limit_violations}회")
    print(f"[Actuator 제한 위반] {actuator_limit_violations}회")
    print(f"[최대 프레임 변화량 초과] {max_delta_violations}회")
    print(f"[충돌 감지] {collisions}회")
    print(f"[NaN/Inf 발생] {nan_count}회")
    print(f"[최종 결과] {_c(final_result, final_result, opts)}")
    print(f"[리포트] {report_path}")
    print(bar)


def print_error(message: str) -> None:
    print(f"[오류] {message}", file=sys.stderr)


def print_dry_run_summary(opts: ConsoleOptions, *, would_process_frames: int, precheck_warnings: int) -> None:
    if opts.quiet:
        return
    print("-" * 68)
    print("[dry-run] 실제 actuator 적용 없이 검사만 수행했습니다.")
    print(f"[dry-run] 재생 시 처리될 프레임 수: {would_process_frames}")
    print(f"[dry-run] 사전 검사 경고 수: {precheck_warnings}")


# ---------------------------------------------------------------------------
# scripts/analyze_mujoco_joint_range_mismatch.py 전용 출력
# ---------------------------------------------------------------------------


def print_joint_range_header(opts: ConsoleOptions, *, joint_name: str, dataset_root: str) -> None:
    if opts.quiet:
        return
    bar = "=" * 68
    print(bar)
    print(_c(f"[분석] {joint_name} 데이터셋-시뮬레이터 범위 비교", "HEADER", opts))
    print(bar)
    print(f"[데이터셋] {dataset_root}")


def print_joint_range_summary(opts: ConsoleOptions, analysis, over_amounts_deg) -> None:
    """analysis: JointRangeAnalysis, over_amounts_deg: np.ndarray. (타입은 순환 import 방지를 위해 생략)"""
    lo, hi = analysis.mujoco_joint_range_deg
    print(f"[데이터셋 action 최솟값] {analysis.global_action_min_deg:.4f} deg")
    print(f"[데이터셋 action 최댓값] {analysis.global_action_max_deg:.4f} deg")
    print(f"[데이터셋 state 최솟값] {analysis.global_state_min_deg:.4f} deg")
    print(f"[데이터셋 state 최댓값] {analysis.global_state_max_deg:.4f} deg")
    print(f"[MuJoCo 관절 범위] {lo:.4f} ~ {hi:.4f} deg")
    print(f"[초과 에피소드] {analysis.episodes_exceeding} / {analysis.total_episodes}")
    print(f"[초과 프레임] {analysis.total_exceeding_frames} / {analysis.total_frames}")
    print(f"[최대 초과량] {analysis.max_over_deg:.4f} deg")
    if over_amounts_deg.size:
        print(
            f"[초과량 평균/중앙값] {float(np.mean(over_amounts_deg)):.4f} / "
            f"{float(np.median(over_amounts_deg)):.4f} deg"
        )
    if opts.verbose:
        print("-" * 68)
        print("[episode별 상세]")
        for ep in analysis.episodes:
            seg_text = ", ".join(f"{s}-{e}" for s, e in ep.exceed_segments) or "-"
            corr_text = f"{ep.action_state_correlation:.3f}" if ep.action_state_correlation is not None else "N/A"
            print(
                f"  ep{ep.episode_index:02d} action[{ep.action_min_deg:7.2f}~{ep.action_max_deg:7.2f}] "
                f"state[{ep.state_min_deg:7.2f}~{ep.state_max_deg:7.2f}] "
                f"초과프레임={ep.action_over_count:4d} 최대초과={ep.max_over_deg:6.2f}deg "
                f"상관계수={corr_text} 구간=[{seg_text}]"
            )


def print_hypotheses(opts: ConsoleOptions, findings) -> None:
    if opts.quiet:
        return
    print("-" * 68)
    verdict_tag = {
        "확인됨": "BLOCKED",  # 강한 확신 -> 눈에 띄는 색(빨강) 재사용
        "가능성 높음": "WARN",
        "가능성 낮음": "PASS",
        "확인 불가": "DIM",
    }
    for finding in findings:
        tag = verdict_tag.get(finding.verdict, "DIM")
        print(f"{_c(f'[{finding.verdict}]', tag, opts)} {finding.name}")
        print(f"    근거: {finding.evidence}")


def print_joint_range_footer(opts: ConsoleOptions, *, json_path: str, csv_path: str) -> None:
    bar = "=" * 68
    print(bar)
    print(f"[JSON 리포트] {json_path}")
    print(f"[CSV 리포트] {csv_path}")
    print(bar)


# ---------------------------------------------------------------------------
# simulation/mujoco/remote_diagnostic.py 전용 출력 (원격 SO-101 실시간 진단)
# ---------------------------------------------------------------------------


def print_remote_prepare_header(opts: ConsoleOptions, *, server_url: str) -> None:
    if opts.quiet:
        return
    bar = "=" * 68
    print(bar)
    print(_c("[준비] SO-101 원격 MuJoCo 진단", "HEADER", opts))
    print(bar)
    print(f"[서버] {server_url}")


def print_remote_preflight_footer(opts: ConsoleOptions) -> None:
    if opts.quiet:
        return
    print("=" * 68)


def print_remote_compact(
    opts: ConsoleOptions,
    *,
    server_status: str,
    sample_count: int,
    latency_ms: float,
    joint: str,
    leader_deg: float,
    follower_deg: float,
    difference_deg: float,
    mujoco_target_deg: float,
    mujoco_qpos_deg: float,
    limit_margin_deg: float | None,
    safety_status: str,
) -> None:
    if opts.quiet:
        return
    bar = "=" * 80
    print(bar)
    print(_c("[SO-101 원격 실시간 진단]", "HEADER", opts))
    print(bar)
    print(f"[서버 상태] {server_status}")
    print(f"[샘플] {sample_count}")
    print(f"[지연] {latency_ms:.0f} ms")
    print(f"[관절] {joint}")
    print("-" * 80)
    print(f"리더암       : {leader_deg:7.2f} deg")
    print(f"팔로워암     : {follower_deg:7.2f} deg")
    print(f"차이         : {difference_deg:7.2f} deg")
    print(f"MuJoCo 목표  : {mujoco_target_deg:7.2f} deg")
    print(f"MuJoCo 실제  : {mujoco_qpos_deg:7.2f} deg")
    margin_text = f"{limit_margin_deg:7.2f} deg" if limit_margin_deg is not None else "     N/A"
    print(f"상한 여유    : {margin_text}")
    print(f"Safety       : {_c(safety_status, safety_status, opts)}")
    print(bar)


def print_remote_table(
    opts: ConsoleOptions,
    *,
    server_status: str,
    sample_count: int,
    latency_ms: float,
    rows: list[dict],
) -> None:
    if opts.quiet:
        return
    bar = "=" * 80
    print(bar)
    print(_c("[SO-101 원격 실시간 진단 - 전체 관절]", "HEADER", opts))
    print(bar)
    print(f"[서버 상태] {server_status}  [샘플] {sample_count}  [지연] {latency_ms:.0f} ms")
    print(f"{'관절':<14s}{'리더':>10s}{'팔로워':>12s}{'차이':>10s}{'MuJoCo':>10s}{'Safety':>10s}")
    for row in rows:
        safety_text = _c(f"{row['safety_status']:>10s}", row["safety_status"], opts)
        print(
            f"{row['joint']:<14s}{row['leader_deg']:10.2f}{row['follower_deg']:12.2f}"
            f"{row['difference_deg']:10.2f}{row['mujoco_qpos_deg']:10.2f}{safety_text}"
        )
    print(bar)


def print_remote_blocked(
    opts: ConsoleOptions,
    *,
    joint: str,
    leader_value_deg: float,
    limit_deg: tuple[float, float],
) -> None:
    lo, hi = limit_deg
    print(_c(f"[차단] {joint} 목표값이 MuJoCo 상한을 초과했습니다.", "BLOCKED", opts))
    print(f"[리더 값] {leader_value_deg:.2f} deg")
    print(f"[MuJoCo 범위] {lo:.2f} ~ {hi:.2f} deg")
    print("[처리] 새 target을 적용하지 않고 직전 안전값을 유지합니다.")


def print_remote_recovered(opts: ConsoleOptions, *, joint: str) -> None:
    if opts.quiet:
        return
    print(_c(f"[복구] {joint} 목표값이 다시 MuJoCo 범위 안으로 돌아왔습니다.", "PASS", opts))


def print_remote_pause(opts: ConsoleOptions, *, reason: str, auto_resume: bool) -> None:
    print(_c(f"[일시정지] {reason}", "WARN", opts))
    if not auto_resume:
        print("[일시정지] auto_resume=false 이므로 사용자가 재시작하기 전까지 MuJoCo 갱신을 재개하지 않습니다.")


def print_remote_resume(opts: ConsoleOptions, *, reason: str) -> None:
    print(_c(f"[복구] {reason}", "PASS", opts))


def print_remote_diagnostic_event(opts: ConsoleOptions, event: DiagnosticEvent) -> None:
    label_map = {
        "persistent_difference": "리더-팔로워 차이가 지속되고 있습니다.",
        "follower_saturation_suspected": "팔로워 포화(saturation)가 의심됩니다.",
        "sign_mismatch_suspected": "리더-팔로워 변화 방향(sign)이 반복적으로 어긋납니다.",
        "offset_suspected": "일정한 offset(캘리브레이션 영점 차이)이 의심됩니다.",
        "leader_out_of_mujoco_range": "리더 값만 MuJoCo 관절 range를 벗어났습니다.",
    }
    print(_c(f"[진단] {label_map.get(event.code, event.message)}", "WARN", opts))
    print(f"[관절] {event.joint}")
    print(f"[상세] {event.message}")
    causes = event.details.get("possible_causes")
    if causes:
        print(f"[가능성] {' 또는 '.join(causes)}")


def print_remote_error(message: str) -> None:
    print(f"[오류] {message}", file=sys.stderr)


def print_remote_dry_run_summary(
    opts: ConsoleOptions,
    *,
    joints: list[str],
    would_apply_all_joints: bool,
    report_path: str,
) -> None:
    if opts.quiet:
        return
    print("-" * 68)
    print("[dry-run] 실제 네트워크 호출과 MuJoCo actuator 적용 없이 설정만 검사했습니다.")
    print(f"[dry-run] 진단 대상 관절: {joints}")
    print(f"[dry-run] MuJoCo에는 항상 6개 관절 전체가 적용됩니다: {would_apply_all_joints}")
    print(f"[dry-run] 리포트 저장 예정 경로: {report_path}")


def print_remote_final_summary(
    opts: ConsoleOptions,
    *,
    duration_sec: float,
    sample_count: int,
    latency_mean_ms: float,
    latency_max_ms: float,
    stale_count: int,
    mujoco_blocked_events: int,
    persistent_difference_events: int,
    follower_saturation_events: int,
    final_result: str,
    csv_path: str | None,
    json_path: str,
) -> None:
    bar = "=" * 68
    print(bar)
    print(_c("[완료] SO-101 원격 MuJoCo 진단 종료", "HEADER", opts))
    print(bar)
    print(f"[실행 시간] {duration_sec:.1f}초")
    print(f"[수신 샘플] {sample_count}")
    print(f"[평균 지연] {latency_mean_ms:.1f} ms")
    print(f"[최대 지연] {latency_max_ms:.1f} ms")
    print(f"[stale 발생] {stale_count}회")
    print(f"[MuJoCo 차단] {mujoco_blocked_events}회")
    print(f"[지속 차이 이벤트] {persistent_difference_events}회")
    print(f"[팔로워 포화 의심] {follower_saturation_events}회")
    print(f"[최종 결과] {_c(final_result, final_result, opts)}")
    if csv_path:
        print(f"[CSV] {csv_path}")
    print(f"[JSON] {json_path}")
    print(bar)
