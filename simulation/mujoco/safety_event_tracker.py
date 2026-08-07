"""WARN/BLOCKED 발생 원인을 추적/병합해 웹 화면과 JSON/CSV 리포트로 남기는 모듈.

**이 모듈은 어떤 safety 판정도 새로 내리지 않는다.** PASS/WARN/BLOCKED 여부와 실제로
`data.ctrl`에 적용되는 target은 여전히 전적으로 `safety_checks.py`(`check_frame_targets`,
`check_simulation_state`, `filter_active_blocking`)가 결정한다 - 이 파일은 그 결과(과 render
루프가 이미 계산해 둔 stale/sequence/mode 등 네트워크 상태)를 **관찰만** 해서:

1. 사람이 이해하기 쉬운 "원인 코드"로 분류하고 (기존 결과에서 확정 불가능하면 반드시
   ``UNKNOWN_SAFETY_REASON``)
2. 같은 관절/같은 원인이 연속으로 발생하면 하나의 이벤트로 병합하고
3. 끝난 뒤에도 일정 시간(``sticky_display_sec``) 웹 화면에 남겨두고
4. 세션 종료 시 JSON/CSV로 저장한다.

``NEAR_JOINT_LIMIT``만 예외적으로 이 모듈이 직접 계산하는 값(margin_deg가
``near_limit_margin_deg`` 미만)을 근거로 만든다 - 이건 `data.ctrl`에 적용되는 값이나
BLOCKED/WARN 판정 자체를 전혀 바꾸지 않는 **순수 표시/기록용** 정보이며, 기존
safety_checks.py의 joint_limit_tolerance_rad 등 "적용을 막는" threshold와는 다른 값이다
(``configs/remote_mujoco_diagnostic.yaml``의 ``safety_event_tracking`` 섹션에서 설정한다 -
`configs/mujoco_so101.yaml`의 safety threshold는 이 모듈이 읽지도, 바꾸지도 않는다).
"""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from simulation.mujoco.safety_checks import SafetyEvent

# 사람이 이해하기 쉬운 고정 원인 코드 목록. 이 밖의 코드는 만들지 않는다 - 기존 검사
# 결과에서 확정할 수 없으면 반드시 UNKNOWN_SAFETY_REASON.
REASON_CODES: tuple[str, ...] = (
    "JOINT_RANGE_LOW",
    "JOINT_RANGE_HIGH",
    "NEAR_JOINT_LIMIT",
    "FRAME_DELTA_HIGH",
    "REMOTE_STALE",
    "SEQUENCE_STALLED",
    "INVALID_VALUE",
    "MISSING_JOINT",
    "MODE_NOT_READ_ONLY",
    "UNKNOWN_SAFETY_REASON",
)

# 관절 단위가 아니라 연결 전체에 대한 판정(REMOTE_STALE/SEQUENCE_STALLED/MODE_NOT_READ_ONLY)에
# 붙이는 sentinel 관절 이름. 실제 관절 이름과 절대 겹치지 않도록 고정 문자열을 쓴다.
CONNECTION_WIDE_JOINT = "(connection)"

CSV_FIELDNAMES: tuple[str, ...] = (
    "event_id",
    "severity",
    "reason_code",
    "joint",
    "started_at",
    "ended_at",
    "duration_ms",
    "sample_count",
    "first_remote_sequence",
    "last_remote_sequence",
    "requested_target_deg",
    "applied_target_deg",
    "joint_min_deg",
    "joint_max_deg",
    "margin_deg",
    "delta_deg",
    "remote_age_ms",
    "stale",
)


@dataclass(frozen=True)
class SafetyEventTrackerConfig:
    """``configs/remote_mujoco_diagnostic.yaml``의 ``safety_event_tracking`` 섹션과 대응.

    여기 있는 값들은 "얼마나 오래/몇 샘플이나 지나야 이벤트를 합치거나 화면에서 지울지"를
    정하는 **표시/기록용** 설정이지, safety 판정 자체를 바꾸는 threshold가 아니다.
    """

    clear_after_samples: int = 3
    sticky_display_sec: float = 10.0
    # 참고용 "한계 근접" 표시 threshold. data.ctrl 적용이나 BLOCKED/WARN 판정에는 전혀
    # 영향을 주지 않는다 (기존 joint_limit_tolerance_rad와는 완전히 별개).
    near_limit_margin_deg: float = 5.0

    def __post_init__(self) -> None:
        if self.clear_after_samples < 1:
            raise ValueError("clear_after_samples는 1 이상이어야 합니다.")
        if self.sticky_display_sec < 0:
            raise ValueError("sticky_display_sec는 0 이상이어야 합니다.")
        if self.near_limit_margin_deg < 0:
            raise ValueError("near_limit_margin_deg는 0 이상이어야 합니다.")


@dataclass
class SafetyIssue:
    """이번 샘플(프레임)에서 관측된 "문제 1건". severity가 None이면 PASS(문제 없음)."""

    joint: str
    severity: str | None  # "WARN" | "BLOCKED" | None(PASS)
    reason_code: str | None  # severity가 None이면 reason_code도 None
    requested_target_deg: float | None = None
    applied_target_deg: float | None = None
    joint_min_deg: float | None = None
    joint_max_deg: float | None = None
    margin_deg: float | None = None
    delta_deg: float | None = None
    remote_age_ms: float | None = None
    stale: bool = False


@dataclass
class SafetyEventRecord:
    event_id: int
    severity: str
    reason_code: str
    joint: str
    started_at: float  # time.time() (wall clock, 사람이 읽는 시각/리포트용)
    ended_at: float | None  # None이면 아직 진행 중
    sample_count: int
    first_remote_sequence: int | None
    last_remote_sequence: int | None
    requested_target_deg: float | None
    applied_target_deg: float | None
    joint_min_deg: float | None
    joint_max_deg: float | None
    margin_deg: float | None
    delta_deg: float | None
    remote_age_ms: float | None
    stale: bool

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else self.started_at
        return max(0.0, (end - self.started_at) * 1000.0)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "severity": self.severity,
            "reason_code": self.reason_code,
            "joint": self.joint,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": round(self.duration_ms, 1),
            "sample_count": self.sample_count,
            "first_remote_sequence": self.first_remote_sequence,
            "last_remote_sequence": self.last_remote_sequence,
            "requested_target_deg": self.requested_target_deg,
            "applied_target_deg": self.applied_target_deg,
            "joint_min_deg": self.joint_min_deg,
            "joint_max_deg": self.joint_max_deg,
            "margin_deg": self.margin_deg,
            "delta_deg": self.delta_deg,
            "remote_age_ms": self.remote_age_ms,
            "stale": self.stale,
            "active": self.ended_at is None,
        }


# ---------------------------------------------------------------------------
# 원인 코드 분류 (기존 SafetyEvent.code만 보고 판단 - 확정 안 되면 UNKNOWN_SAFETY_REASON)
# ---------------------------------------------------------------------------


def classify_frame_event_reason(evt: SafetyEvent) -> str:
    """safety_checks.py가 이미 낸 SafetyEvent 하나를 사람이 읽는 원인 코드로 분류한다.

    새로운 판정을 하지 않는다 - evt.code/value/limit 등 이미 계산된 값만 본다.
    """
    if evt.code in ("joint_limit", "actuator_limit"):
        if evt.value is not None and evt.limit is not None:
            lo, hi = evt.limit
            if evt.value < lo:
                return "JOINT_RANGE_LOW"
            if evt.value > hi:
                return "JOINT_RANGE_HIGH"
        return "UNKNOWN_SAFETY_REASON"  # value/limit이 없으면 어느 쪽인지 확정할 수 없음
    if evt.code == "max_delta":
        return "FRAME_DELTA_HIGH"
    if evt.code == "simulation_nan":
        return "INVALID_VALUE"
    # velocity_limit / self_collision / table_collision / contact_spike / simulation_divergence
    # 등은 이 고정 코드 목록에 해당하는 게 없다 - 추측해서 새 코드를 만들지 않는다.
    return "UNKNOWN_SAFETY_REASON"


_SEVERITY_RANK = {"BLOCKED": 2, "WARN": 1}


def pick_most_severe_event(events: list[SafetyEvent]) -> SafetyEvent | None:
    """한 관절에 이번 샘플 여러 이벤트가 겹치면(예: joint_limit + max_delta) 더 심각한 것 하나만
    고른다 - 요구사항 스키마가 "샘플당 관절당 원인 코드 1개"를 전제하기 때문이다."""
    if not events:
        return None
    return max(events, key=lambda e: _SEVERITY_RANK.get(e.level, 0))


# ---------------------------------------------------------------------------
# 추적기
# ---------------------------------------------------------------------------


class SafetyEventTracker:
    """스레드 세이프. render 스레드가 매 프레임 ``observe()``를 호출하고, HTTP 스레드가
    ``current_safety()``/``recent_events()``/``event_counts()``를 읽는다."""

    def __init__(self, config: SafetyEventTrackerConfig | None = None) -> None:
        self._config = config or SafetyEventTrackerConfig()
        self._lock = threading.Lock()
        self._next_event_id = 1
        # (joint, reason_code) -> 진행 중인 이벤트
        self._active: dict[tuple[str, str], SafetyEventRecord] = {}
        # (joint, reason_code) -> 연속으로 "발생 안 함"이 관측된 샘플 수
        self._clear_counts: dict[tuple[str, str], int] = {}
        self._completed: list[SafetyEventRecord] = []  # 세션 전체 이력 (리포트용)
        self._last_current: dict[str, dict] = {}  # 이번 샘플에 관측된 문제만 (current_safety용)

    # -- 관측 --------------------------------------------------------------

    def observe(self, *, now_wall: float, remote_sequence: int | None, issues: list[SafetyIssue]) -> None:
        with self._lock:
            occurring_keys: set[tuple[str, str]] = set()
            current: dict[str, dict] = {}

            for issue in issues:
                if issue.severity is None:
                    continue
                assert issue.reason_code is not None
                key = (issue.joint, issue.reason_code)
                occurring_keys.add(key)
                current[issue.joint] = {
                    "severity": issue.severity,
                    "reason_code": issue.reason_code,
                    "margin_deg": issue.margin_deg,
                }
                self._clear_counts[key] = 0

                record = self._active.get(key)
                if record is None:
                    record = SafetyEventRecord(
                        event_id=self._next_event_id,
                        severity=issue.severity,
                        reason_code=issue.reason_code,
                        joint=issue.joint,
                        started_at=now_wall,
                        ended_at=None,
                        sample_count=0,
                        first_remote_sequence=remote_sequence,
                        last_remote_sequence=remote_sequence,
                        requested_target_deg=None,
                        applied_target_deg=None,
                        joint_min_deg=None,
                        joint_max_deg=None,
                        margin_deg=None,
                        delta_deg=None,
                        remote_age_ms=None,
                        stale=False,
                    )
                    self._next_event_id += 1
                    self._active[key] = record

                record.sample_count += 1
                record.last_remote_sequence = remote_sequence
                record.ended_at = now_wall  # "마지막으로 관측된 시각"으로 계속 갱신
                # severity는 이번 이벤트 동안 더 심각한 쪽으로만 올라간다(WARN -> BLOCKED는
                # 새 이벤트가 아니라 심각도 상승으로 취급, BLOCKED -> WARN은 유지: 한 번이라도
                # BLOCKED였다면 그 이벤트는 BLOCKED로 남긴다).
                if issue.severity == "BLOCKED":
                    record.severity = "BLOCKED"
                record.requested_target_deg = issue.requested_target_deg
                record.applied_target_deg = issue.applied_target_deg
                record.joint_min_deg = issue.joint_min_deg
                record.joint_max_deg = issue.joint_max_deg
                record.margin_deg = issue.margin_deg
                record.delta_deg = issue.delta_deg
                record.remote_age_ms = issue.remote_age_ms
                record.stale = issue.stale

            # 이번 샘플에 나타나지 않은 진행 중인 이벤트들은 clear_after_samples만큼
            # 연속으로 빠지면 종료 처리한다 (병합 정책 - 순간적으로 한 샘플만 빠진 건
            # 새 이벤트로 쪼개지 않는다).
            for key in list(self._active.keys()):
                if key in occurring_keys:
                    continue
                self._clear_counts[key] = self._clear_counts.get(key, 0) + 1
                if self._clear_counts[key] >= self._config.clear_after_samples:
                    record = self._active.pop(key)
                    self._completed.append(record)
                    self._clear_counts.pop(key, None)

            self._last_current = current

    def finalize(self, *, now_wall: float) -> None:
        """세션 종료 시 호출 - 아직 진행 중인 이벤트를 전부 닫는다."""
        with self._lock:
            for record in self._active.values():
                if record.ended_at is None:
                    record.ended_at = now_wall
                self._completed.append(record)
            self._active.clear()
            self._clear_counts.clear()

    # -- 조회 (HTTP 스레드에서 호출) -----------------------------------------

    def current_safety(self) -> dict:
        with self._lock:
            level = "PASS"
            if any(v["severity"] == "BLOCKED" for v in self._last_current.values()):
                level = "BLOCKED"
            elif any(v["severity"] == "WARN" for v in self._last_current.values()):
                level = "WARN"
            return {"level": level, "joints": dict(self._last_current)}

    def recent_events(self, *, now_wall: float) -> list[dict]:
        """진행 중인 이벤트 + sticky_display_sec 이내에 끝난 이벤트를 최신순으로 반환."""
        with self._lock:
            all_records = list(self._active.values()) + list(self._completed)
            visible = [
                r
                for r in all_records
                if r.ended_at is None or (now_wall - r.ended_at) <= self._config.sticky_display_sec
            ]
            visible.sort(key=lambda r: (r.ended_at or now_wall), reverse=True)
            return [r.to_dict() for r in visible]

    def event_counts(self, *, now_wall: float) -> dict[str, int]:
        counts = {"WARN": 0, "BLOCKED": 0}
        for evt in self.recent_events(now_wall=now_wall):
            if evt["severity"] in counts:
                counts[evt["severity"]] += 1
        return counts

    def all_events(self) -> list[SafetyEventRecord]:
        """세션 전체 이력 (진행 중인 것 포함) - 리포트 저장용. finalize() 이후 호출 권장."""
        with self._lock:
            return list(self._active.values()) + list(self._completed)


# ---------------------------------------------------------------------------
# 리포트 저장 (reports/remote_mujoco_diagnostic/safety_events_<timestamp>.json/.csv)
# ---------------------------------------------------------------------------


def make_event_session_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def resolve_safety_event_paths(reports_dir: Path, session_id: str) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"safety_events_{session_id}.json"
    csv_path = reports_dir / f"safety_events_{session_id}.csv"
    return json_path, csv_path


def write_safety_events_json(path: Path, events: list[SafetyEventRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_count": len(events),
        "reason_code_counts": _count_by(events, lambda e: e.reason_code),
        "severity_counts": _count_by(events, lambda e: e.severity),
        "events": [e.to_dict() for e in events],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_safety_events_csv(path: Path, events: list[SafetyEventRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            writer.writerow({key: e.to_dict().get(key, "") for key in CSV_FIELDNAMES})


def _count_by(events: list[SafetyEventRecord], key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        k = key_fn(e)
        counts[k] = counts.get(k, 0) + 1
    return counts
