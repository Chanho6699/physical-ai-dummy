"""Policy intent validation before MotionGuard.

Intent answers only whether a raw policy target is structurally valid and free of
gross anomalies. A mechanically valid far target is accepted and MotionGuard owns
velocity, acceleration, and jerk limiting. The validator is stateful so severe raw
target discontinuities can be rejected without comparing normal trajectory distance
against the current encoder state.

Small mechanical endpoint overshoot is left for Final Safety saturation. Schema,
non-finite values, gross mechanical violations, initial multi-joint spikes, and
severe temporal discontinuities fail closed as REJECT. Intent never emits
WOULD_CLAMP.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.action_adapter import adapt_vla_action
from runtime.laptop.safety_gate import SafetyDecision, SafetyGate

DEFAULT_INTENT_GROSS_OUTLIER_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "intent_gross_outlier.yaml"
)


class IntentValidationConfigError(RuntimeError):
    """Intent gross-outlier calibration artifact is missing or malformed."""


@dataclass(frozen=True)
class IntentGrossOutlierConfig:
    initial_multi_joint: dict[str, float]
    temporal_single_hard: dict[str, float]
    temporal_multi_joint: dict[str, float]
    simultaneous_joint_count: int = 3
    provenance: dict[str, object] | None = None


    @staticmethod
    def from_yaml(
        path: str | Path = DEFAULT_INTENT_GROSS_OUTLIER_CONFIG_PATH,
    ) -> "IntentGrossOutlierConfig":
        import yaml
        resolved = Path(path)
        try:
            raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise IntentValidationConfigError(f"Intent calibration load failed: {resolved}: {exc}") from exc
        parsed = {}
        for section in ("initial_multi_joint", "temporal_single_hard", "temporal_multi_joint"):
            values = raw.get(section)
            if not isinstance(values, dict) or any(j not in values for j in JOINT_ORDER):
                raise IntentValidationConfigError(f"{resolved}: invalid {section} mapping")
            parsed[section] = {joint: float(values[joint]) for joint in JOINT_ORDER}
            if any(not math.isfinite(v) or v <= 0 for v in parsed[section].values()):
                raise IntentValidationConfigError(f"{resolved}: {section} values must be finite and positive")
        count = int(raw.get("simultaneous_joint_count", 3))
        if not 2 <= count <= len(JOINT_ORDER):
            raise IntentValidationConfigError(f"{resolved}: invalid simultaneous_joint_count={count}")
        provenance = raw.get("provenance")
        return IntentGrossOutlierConfig(
            initial_multi_joint=parsed["initial_multi_joint"],
            temporal_single_hard=parsed["temporal_single_hard"],
            temporal_multi_joint=parsed["temporal_multi_joint"],
            simultaneous_joint_count=count,
            provenance=provenance if isinstance(provenance, dict) else None,
        )


@dataclass(frozen=True)
class IntentValidationResult:
    """``PolicyIntentValidator.check_intent()``의 결과. ``SafetyDecision``과 필드가
    비슷해 보이지만 의미가 다르다는 걸 명시하기 위해 별도 타입으로 둔다(요구사항 -
    semantic confusion 방지)."""

    valid: bool  # True == decision == "ACCEPT" (이 raw target을 신뢰 가능한 의도로 받아들임)
    decision: str  # Intent emits only "ACCEPT" or "REJECT".
    reasons: tuple[str, ...]


class PolicyIntentValidator:
    """Reject only gross raw-policy anomalies before MotionGuard.

    SafetyGate remains the source of truth for schema, finite-value, and mechanical
    validation. Calibrated initial/temporal envelopes add anomaly detection, and
    only the most recent accepted raw target is retained as temporal history.
    """

    def __init__(
        self, safety_gate: SafetyGate, config: IntentGrossOutlierConfig | None = None,
    ) -> None:
        self._safety_gate = safety_gate
        self._config = config or IntentGrossOutlierConfig.from_yaml()
        self._previous_valid_raw_target: dict[str, float] | None = None

    def reset_history(self) -> None:
        self._previous_valid_raw_target = None

    def check_intent(
        self, *, raw_target_deg: dict[str, float], current_state_deg: dict[str, float]
    ) -> IntentValidationResult:
        adapted = adapt_vla_action(raw_target_deg)
        mechanical: SafetyDecision = self._safety_gate.evaluate(
            adapted_action=adapted, current_state_deg=current_state_deg, observation_valid=True,
            check_excessive_step=False, check_mechanical_range=True,
        )
        # Small endpoint overshoot is recoverable downstream saturation. Gross mechanical
        # violations remain fail-closed before MotionGuard.
        if mechanical.decision == "REJECT":
            return IntentValidationResult(valid=False, decision="REJECT", reasons=mechanical.reasons)

        command = adapted.command_deg
        previous = self._previous_valid_raw_target
        if previous is None:
            initial_spikes = [
                joint for joint in JOINT_ORDER
                if abs(command[joint] - current_state_deg[joint])
                > self._config.initial_multi_joint[joint]
            ]
            if len(initial_spikes) >= self._config.simultaneous_joint_count:
                return IntentValidationResult(
                    valid=False,
                    decision="REJECT",
                    reasons=(f"INTENT_MULTI_JOINT_SPIKE: joints={initial_spikes}",),
                )
        else:
            deltas = {joint: abs(command[joint] - previous[joint]) for joint in JOINT_ORDER}
            hard = [
                joint for joint in JOINT_ORDER
                if deltas[joint] > self._config.temporal_single_hard[joint]
            ]
            multi = [
                joint for joint in JOINT_ORDER
                if deltas[joint] > self._config.temporal_multi_joint[joint]
            ]
            if hard or len(multi) >= self._config.simultaneous_joint_count:
                return IntentValidationResult(
                    valid=False,
                    decision="REJECT",
                    reasons=(
                        "INTENT_SEVERE_TEMPORAL_DISCONTINUITY: "
                        f"hard={hard}, multi={multi}",
                    ),
                )

        self._previous_valid_raw_target = dict(command)
        return IntentValidationResult(valid=True, decision="ACCEPT", reasons=())
