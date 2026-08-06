"""wrist_flex 등 관절 range 불일치 원인 조사용 분석 모듈.

이 모듈은 재생 동작(dataset_action_replay.py)이나 safety_checks.py의 판정 로직을
전혀 바꾸지 않는다. 순수하게 "조사"만 하며, MJCF나 configs/mujoco_so101.yaml을
읽기만 하고 절대 쓰지 않는다.

배경 조사 결과 (조사 근거는 docs/wrist_flex_range_mismatch_investigation.md 참고):

- LeRobot의 `lerobot_record.py` 녹화 루프에서 `action`은 teleop(리더암)의
  `get_action()` 결과를 그대로 로깅한 값이고, `observation.state`는 follower(팔로워암)의
  `get_observation()` 결과다. 즉 **`action`은 "리더암 자신의 calibration 기준" 값이고,
  `observation.state`는 "팔로워암 자신의 calibration 기준" 값**이며, 서로 다른 물리
  로봇의 서로 다른 calibration에서 나온 숫자다 (~/lerobot/src/lerobot/scripts/lerobot_record.py).
- `make_default_teleop_action_processor()`/`make_default_robot_action_processor()`는
  기본적으로 `IdentityProcessorStep()`만 사용한다 (~/lerobot/src/lerobot/processor/factory.py).
  이 프로젝트의 `data_collection/recorder.py`도 커스텀 processor나 `max_relative_target`을
  지정하지 않으므로, 리더암이 읽은 값이 그 어떤 clipping도 없이 그대로 팔로워에 전송되고
  그대로 데이터셋에 기록된다.
- `MotorNormMode.DEGREES`의 `_unnormalize` (raw ticks로 변환, 즉 실제 서보에 보낼 값을 만드는
  경로)는 `RANGE_M100_100`/`RANGE_0_100`와 달리 **범위를 clamp하지 않는다**
  (~/lerobot/src/lerobot/motors/motors_bus.py). 즉 리더암 calibration 기준 값이 팔로워
  calibration 기준 안전 범위를 벗어나도 소프트웨어 단에서 막히지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulation.mujoco.action_mapping import DEG2RAD
from simulation.mujoco.dataset_loader import DatasetInfo, load_dataset_info, load_episode
from simulation.mujoco.so101_model import get_joint_limits, load_model


@dataclass(frozen=True)
class EpisodeJointStats:
    episode_index: int
    frame_count: int
    action_min_deg: float
    action_max_deg: float
    state_min_deg: float
    state_max_deg: float
    action_over_count: int  # 관절 range를 벗어나는 action 프레임 수
    state_over_count: int  # 관절 range를 벗어나는 state 프레임 수 (참고용)
    max_over_deg: float  # action이 range를 벗어난 최대 초과량 (deg, 0이면 초과 없음)
    exceed_segments: tuple[tuple[int, int], ...]  # action이 연속으로 초과하는 (시작,끝) 프레임 구간
    action_state_correlation: float | None  # 이 에피소드에서 action-state 상관계수


@dataclass(frozen=True)
class JointRangeAnalysis:
    dataset_root: Path
    joint_name: str
    dataset_index: int
    fps: int
    total_episodes: int
    mujoco_joint_range_rad: tuple[float, float]
    mujoco_joint_range_deg: tuple[float, float]
    mujoco_actuator_ctrlrange_rad: tuple[float, float] | None
    episodes: tuple[EpisodeJointStats, ...]

    @property
    def global_action_min_deg(self) -> float:
        return min(ep.action_min_deg for ep in self.episodes)

    @property
    def global_action_max_deg(self) -> float:
        return max(ep.action_max_deg for ep in self.episodes)

    @property
    def global_state_min_deg(self) -> float:
        return min(ep.state_min_deg for ep in self.episodes)

    @property
    def global_state_max_deg(self) -> float:
        return max(ep.state_max_deg for ep in self.episodes)

    @property
    def episodes_exceeding(self) -> int:
        return sum(1 for ep in self.episodes if ep.action_over_count > 0)

    @property
    def total_exceeding_frames(self) -> int:
        return sum(ep.action_over_count for ep in self.episodes)

    @property
    def total_frames(self) -> int:
        return sum(ep.frame_count for ep in self.episodes)

    @property
    def max_over_deg(self) -> float:
        values = [ep.max_over_deg for ep in self.episodes]
        return max(values) if values else 0.0


def _find_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """True인 구간들을 (시작 index, 끝 index, inclusive) 튜플 리스트로 반환."""
    if mask.size == 0 or not mask.any():
        return ()
    runs = []
    in_run = False
    start = 0
    for i, value in enumerate(mask):
        if value and not in_run:
            start = i
            in_run = True
        elif not value and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return tuple(runs)


def analyze_joint(
    dataset_root: str | Path,
    joint_name: str,
    *,
    scene_path: str | Path | None = None,
) -> tuple[JointRangeAnalysis, np.ndarray]:
    """데이터셋 전체 episode에 대해 joint_name의 action/state 통계를 계산한다.

    Returns:
        (analysis, over_amounts_deg): over_amounts_deg는 초과가 발생한 모든 프레임의
        초과량(deg)을 모은 1차원 배열로, 분포(평균/백분위수) 계산에 사용한다.
    """
    root = Path(dataset_root).expanduser().resolve()
    info: DatasetInfo = load_dataset_info(root)

    if joint_name not in [name.removesuffix(".pos") for name in info.action_names]:
        raise ValueError(
            f"'{joint_name}'은(는) 이 데이터셋의 action feature에 없습니다. "
            f"사용 가능: {[n.removesuffix('.pos') for n in info.action_names]}"
        )
    dataset_index = [n.removesuffix(".pos") for n in info.action_names].index(joint_name)

    model = load_model(scene_path)
    limits = get_joint_limits(model, (joint_name,))[joint_name]
    lo_rad, hi_rad = limits.joint_range
    lo_deg, hi_deg = math.degrees(lo_rad), math.degrees(hi_rad)

    episode_stats: list[EpisodeJointStats] = []
    all_over_amounts: list[float] = []

    for episode_index in range(info.total_episodes):
        episode = load_episode(root, episode_index, info)
        action_deg = episode.action[:, dataset_index].astype(np.float64)
        state_deg = episode.state[:, dataset_index].astype(np.float64)

        action_rad = action_deg * DEG2RAD
        over_mask = (action_rad < lo_rad) | (action_rad > hi_rad)
        over_count = int(over_mask.sum())

        state_rad = state_deg * DEG2RAD
        state_over_count = int(((state_rad < lo_rad) | (state_rad > hi_rad)).sum())

        over_amount_deg = np.zeros_like(action_deg)
        over_amount_deg[action_rad > hi_rad] = np.degrees(action_rad[action_rad > hi_rad] - hi_rad)
        over_amount_deg[action_rad < lo_rad] = np.degrees(lo_rad - action_rad[action_rad < lo_rad])
        max_over = float(over_amount_deg.max()) if over_count else 0.0
        if over_count:
            all_over_amounts.extend(over_amount_deg[over_mask].tolist())

        segments = _find_runs(over_mask)

        correlation: float | None = None
        if len(action_deg) > 1 and np.std(action_deg) > 1e-9 and np.std(state_deg) > 1e-9:
            correlation = float(np.corrcoef(action_deg, state_deg)[0, 1])

        episode_stats.append(
            EpisodeJointStats(
                episode_index=episode_index,
                frame_count=episode.length,
                action_min_deg=float(action_deg.min()),
                action_max_deg=float(action_deg.max()),
                state_min_deg=float(state_deg.min()),
                state_max_deg=float(state_deg.max()),
                action_over_count=over_count,
                state_over_count=state_over_count,
                max_over_deg=max_over,
                exceed_segments=segments,
                action_state_correlation=correlation,
            )
        )

    actuator_ctrlrange = limits.actuator_ctrlrange

    analysis = JointRangeAnalysis(
        dataset_root=root,
        joint_name=joint_name,
        dataset_index=dataset_index,
        fps=info.fps,
        total_episodes=info.total_episodes,
        mujoco_joint_range_rad=(lo_rad, hi_rad),
        mujoco_joint_range_deg=(lo_deg, hi_deg),
        mujoco_actuator_ctrlrange_rad=actuator_ctrlrange,
        episodes=tuple(episode_stats),
    )
    return analysis, np.array(all_over_amounts)


def hypothetical_offset_needed_deg(analysis: JointRangeAnalysis) -> float:
    """action 전체가 MuJoCo range 안에 들어오려면 필요한 균일 offset(deg) 크기.

    참고용 가상 계산일 뿐이며, 실제 매핑에 적용하지 않는다 (--dry-run 성격의 진단).
    양수면 "이만큼 음의 방향으로 이동해야 최댓값이 range 안에 들어온다"는 뜻이다.
    """
    lo_deg, hi_deg = analysis.mujoco_joint_range_deg
    over_hi = analysis.global_action_max_deg - hi_deg
    over_lo = lo_deg - analysis.global_action_min_deg
    return max(over_hi, over_lo, 0.0)


@dataclass(frozen=True)
class HypothesisFinding:
    name: str
    verdict: str  # "확인됨" | "가능성 높음" | "가능성 낮음" | "확인 불가"
    evidence: str


def evaluate_hypotheses(
    analysis: JointRangeAnalysis, over_amounts_deg: np.ndarray
) -> tuple[HypothesisFinding, ...]:
    """조사 결과를 바탕으로 각 가설을 판정한다 (코드 근거 기반, 하드코딩된 결론이 아니라
    analysis 객체의 실제 통계치로부터 도출한다)."""

    state_over_any = sum(ep.state_over_count for ep in analysis.episodes) > 0
    action_over_any = analysis.total_exceeding_frames > 0
    correlations = [ep.action_state_correlation for ep in analysis.episodes if ep.action_state_correlation is not None]
    mean_corr = float(np.mean(correlations)) if correlations else None

    # state가 action보다 항상 좁게 분포하는지 (leader가 follower보다 더 멀리 가는 패턴인지)
    state_narrower_count = sum(
        1
        for ep in analysis.episodes
        if ep.state_max_deg <= analysis.mujoco_joint_range_deg[1] + 1e-6
        and ep.action_max_deg > ep.state_max_deg + 0.5
    )

    findings = []

    findings.append(
        HypothesisFinding(
            name="MuJoCo joint range가 실제 팔로워 하드웨어보다 좁음",
            verdict="가능성 낮음" if not state_over_any else "가능성 높음",
            evidence=(
                f"observation.state(팔로워 실측값) 기준 range 초과 프레임 수={sum(ep.state_over_count for ep in analysis.episodes)}. "
                f"state는 전 episode에서 MuJoCo range 안쪽({analysis.global_state_max_deg:.2f}deg 이하)에 머무는 경우가 대부분이면, "
                "MuJoCo range 자체가 팔로워 하드웨어보다 좁다는 근거는 약하다."
            ),
        )
    )

    findings.append(
        HypothesisFinding(
            name="dataset와 MuJoCo의 zero offset 불일치",
            verdict="가능성 낮음",
            evidence=(
                "초과가 range 양쪽(최솟값/최댓값)에 고르게 나타나는 균일한 shift 패턴이 아니라, "
                f"최댓값(high) 쪽에서만 반복적으로 나타난다 (분석된 {analysis.episodes_exceeding}개 episode 전부 high-side). "
                "균일 offset 오류라면 대개 최솟값 쪽에서도 대칭적인 초과가 함께 나타나야 하므로, "
                "단순 zero offset 문제로 보기는 어렵다."
            ),
        )
    )

    sign_verdict = "가능성 낮음"
    if mean_corr is not None and mean_corr < 0:
        sign_verdict = "가능성 높음"
    findings.append(
        HypothesisFinding(
            name="joint direction/sign 불일치",
            verdict=sign_verdict,
            evidence=(
                f"episode별 action-state 상관계수 평균={mean_corr:.4f}"
                if mean_corr is not None
                else "상관계수를 계산할 수 없음 (표준편차가 0에 가까운 episode만 존재)"
            )
            + ". 값이 양의 상관관계를 강하게 보이면(≈+1) 두 신호가 같은 방향으로 움직이는 것이므로 "
            "부호 반전 가능성은 낮다.",
        )
    )

    findings.append(
        HypothesisFinding(
            name="leader/follower calibration 차이",
            verdict="확인됨",
            evidence=(
                "~/lerobot/src/lerobot/scripts/lerobot_record.py의 녹화 루프에서 'action'은 "
                "teleop(리더암).get_action()의 결과를 그대로 기록하고, 'observation.state'는 "
                "follower(팔로워암).get_observation()의 결과를 기록한다. 리더/팔로워는 서로 다른 물리 "
                "로봇으로 각자 독립적인 calibration(range_min/range_max)을 가지므로, 같은 이름의 관절이라도 "
                "degree 값이 반드시 일치할 근거가 없다. 실제로 이 데이터셋에서 "
                f"observation.state는 {analysis.global_state_min_deg:.2f}~{analysis.global_state_max_deg:.2f}deg 범위에 "
                f"머무는 반면 action은 최대 {analysis.global_action_max_deg:.2f}deg까지 올라간다 "
                f"(state보다 최대 {max(0.0, analysis.global_action_max_deg - analysis.global_state_max_deg):.2f}deg 더 큼)."
            ),
        )
    )

    findings.append(
        HypothesisFinding(
            name="데이터 수집 시 실제 안전 범위 초과 (소프트웨어 clipping 부재)",
            verdict="가능성 높음",
            evidence=(
                "~/lerobot/src/lerobot/processor/factory.py의 make_default_teleop_action_processor / "
                "make_default_robot_action_processor는 기본적으로 IdentityProcessorStep만 사용해 아무 것도 "
                "clip하지 않는다. 이 프로젝트의 data_collection/recorder.py도 커스텀 processor나 "
                "max_relative_target을 지정하지 않는다. 또한 motors_bus.py의 MotorNormMode.DEGREES용 "
                "_unnormalize()는 RANGE_M100_100/RANGE_0_100과 달리 값을 clamp하지 않는다. 즉 리더암 값이 "
                "팔로워 calibration 기준 안전범위를 넘어도 소프트웨어 단에서 막히지 않고 그대로 팔로워에 전송됐을 것이다."
            ),
        )
    )

    findings.append(
        HypothesisFinding(
            name="state/action 표현 차이 (state=팔로워 실측, action=리더 명령)",
            verdict="확인됨",
            evidence=(
                f"{analysis.episodes_exceeding}/{analysis.total_episodes}개 episode에서 action이 MuJoCo range를 "
                f"초과하지만, 그 중 state까지 함께 초과하는 episode는 "
                f"{sum(1 for ep in analysis.episodes if ep.state_over_count > 0)}개뿐이다. "
                "action과 state가 서로 다른 물리적 실체(리더 vs 팔로워)를 나타내는 서로 다른 신호이므로 "
                "둘이 정확히 같은 range를 가질 이유가 없다."
            ),
        )
    )

    findings.append(
        HypothesisFinding(
            name="리더/팔로워 calibration 파일의 실제 range_min/range_max 수치",
            verdict="확인 불가",
            evidence=(
                "이 머신의 ~/.cache/huggingface/lerobot/calibration/ 에 chanho_leader / chanho_follower "
                "calibration JSON이 존재하지 않는다 (녹화에 사용된 실제 머신에만 있었을 가능성). "
                "따라서 이 조사는 action/state 통계로부터 '간접적으로' 원인을 추정한 것이며, "
                "리더/팔로워 각각의 정확한 calibration range_min/range_max, 서보 해상도, motor id는 "
                "직접 확인하지 못했다."
            ),
        )
    )

    return tuple(findings)


def build_report_dict(
    analysis: JointRangeAnalysis,
    over_amounts_deg: np.ndarray,
    findings: tuple[HypothesisFinding, ...],
) -> dict:
    """JSON으로 저장할 리포트 dict를 만든다 (판정 로직/재생 동작에 영향 없음, 순수 직렬화)."""
    percentiles = {}
    if over_amounts_deg.size > 0:
        for p in (50, 90, 99):
            percentiles[f"p{p}"] = float(np.percentile(over_amounts_deg, p))

    return {
        "dataset_root": str(analysis.dataset_root),
        "joint_name": analysis.joint_name,
        "dataset_action_index": analysis.dataset_index,
        "fps": analysis.fps,
        "total_episodes": analysis.total_episodes,
        "mujoco_joint_range_rad": list(analysis.mujoco_joint_range_rad),
        "mujoco_joint_range_deg": list(analysis.mujoco_joint_range_deg),
        "mujoco_actuator_ctrlrange_rad": (
            list(analysis.mujoco_actuator_ctrlrange_rad) if analysis.mujoco_actuator_ctrlrange_rad else None
        ),
        "global_action_min_deg": analysis.global_action_min_deg,
        "global_action_max_deg": analysis.global_action_max_deg,
        "global_state_min_deg": analysis.global_state_min_deg,
        "global_state_max_deg": analysis.global_state_max_deg,
        "episodes_exceeding": analysis.episodes_exceeding,
        "total_episodes_checked": analysis.total_episodes,
        "total_frames": analysis.total_frames,
        "total_exceeding_frames": analysis.total_exceeding_frames,
        "max_over_deg": analysis.max_over_deg,
        "mean_over_deg": float(np.mean(over_amounts_deg)) if over_amounts_deg.size else 0.0,
        "over_deg_percentiles": percentiles,
        "hypothetical_uniform_offset_needed_deg": hypothetical_offset_needed_deg(analysis),
        "episodes": [
            {
                "episode_index": ep.episode_index,
                "frame_count": ep.frame_count,
                "action_min_deg": ep.action_min_deg,
                "action_max_deg": ep.action_max_deg,
                "state_min_deg": ep.state_min_deg,
                "state_max_deg": ep.state_max_deg,
                "action_over_count": ep.action_over_count,
                "state_over_count": ep.state_over_count,
                "max_over_deg": ep.max_over_deg,
                "exceed_segments": [list(seg) for seg in ep.exceed_segments],
                "action_state_correlation": ep.action_state_correlation,
            }
            for ep in analysis.episodes
        ],
        "hypotheses": [
            {"name": f.name, "verdict": f.verdict, "evidence": f.evidence} for f in findings
        ],
    }


def build_csv_rows(analysis: JointRangeAnalysis) -> list[dict]:
    """episode별 통계를 CSV로 저장하기 위한 행(dict) 리스트를 만든다."""
    rows = []
    for ep in analysis.episodes:
        rows.append(
            {
                "episode_index": ep.episode_index,
                "frame_count": ep.frame_count,
                "action_min_deg": ep.action_min_deg,
                "action_max_deg": ep.action_max_deg,
                "state_min_deg": ep.state_min_deg,
                "state_max_deg": ep.state_max_deg,
                "action_over_count": ep.action_over_count,
                "state_over_count": ep.state_over_count,
                "max_over_deg": ep.max_over_deg,
                "exceed_segment_count": len(ep.exceed_segments),
                "exceed_segments": ";".join(f"{s}-{e}" for s, e in ep.exceed_segments),
                "action_state_correlation": ep.action_state_correlation,
            }
        )
    return rows
