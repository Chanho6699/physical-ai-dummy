"""wrist_roll 안전 시험 도구의 포트/calibration 경로 우선순위 해석.

우선순위(요구사항 4번):
    1) CLI 인자
    2) local config (``configs/hardware.local.json``)
    3) ``~`` 기반 표준 LeRobot 캐시 경로 (calibration만 해당 - 기본 follower id 사용)
    4) 없으면 실행 거부

경로를 코드에 절대 경로로 고정하지 않는다 - 3번 항목도 "LeRobot 표준 캐시 경로
템플릿 + 기본 id" 조합으로 구성할 뿐, 특정 사용자의 절대 경로 문자열을 그대로
박아두지 않는다 (``scripts/run_hardware_state_server.py``의
``DEFAULT_FOLLOWER_CALIBRATION_TEMPLATE``과 동일한 패턴).

armed 모드에서는 ``allow_default_fallback=False``로 호출해 3번 fallback을 막아야
한다 (요구사항 4번 "실제 armed 모드에서는 calibration fallback을 허용하지 않는
방향으로 설계").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# scripts/run_hardware_state_server.py의 DEFAULT_FOLLOWER_CALIBRATION_TEMPLATE와 동일한
# LeRobot 표준 캐시 경로 템플릿.
FOLLOWER_CALIBRATION_PATH_TEMPLATE = "~/.cache/huggingface/lerobot/calibration/robots/so_follower/{id}.json"

# 3순위(마지막 fallback)에서만 쓰이는 기본 follower id. armed 모드에서는 이 fallback을
# 허용하지 않는다.
DEFAULT_FOLLOWER_ID_FALLBACK = "chanho_follower"

DEFAULT_LOCAL_CONFIG_RELATIVE_PATH = Path("configs") / "hardware.local.json"


class ResolutionError(RuntimeError):
    """포트 또는 calibration 경로를 확정할 수 없을 때 (실행을 거부해야 함)."""


def _load_local_config(project_root: Path) -> dict | None:
    path = project_root / DEFAULT_LOCAL_CONFIG_RELATIVE_PATH
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_port(*, cli_port: str | None, project_root: Path) -> tuple[str, str]:
    """포트 문자열과 어디서 왔는지("cli" | "local_config")를 반환한다."""
    if cli_port:
        return cli_port, "cli"

    local = _load_local_config(project_root)
    if local:
        port = (local.get("robot") or {}).get("port")
        if port:
            return port, "local_config"

    raise ResolutionError(
        "follower 포트를 확정할 수 없습니다. --port를 지정하거나 "
        "configs/hardware.local.json의 robot.port를 설정하세요."
    )


def resolve_calibration_path(
    *,
    cli_calibration_path: str | None,
    cli_calibration_id: str | None,
    project_root: Path,
    allow_default_fallback: bool = True,
) -> tuple[Path, str]:
    """calibration 파일 경로와 어디서 왔는지를 반환한다.

    반환되는 두 번째 값(source)은 ``"cli_path"``, ``"cli_id"``, ``"local_config"``,
    ``"default_fallback"`` 중 하나다. 이 함수는 파일이 실제로 존재하는지 확인하지
    않는다 - 그건 호출부가 ``hardware/state_server/calibration_loader.py``의
    ``load_calibration_file``로 로드를 시도하면서 확인한다 (없으면 명확히 실패).
    """
    if cli_calibration_path:
        return Path(cli_calibration_path).expanduser(), "cli_path"
    if cli_calibration_id:
        return (
            Path(FOLLOWER_CALIBRATION_PATH_TEMPLATE.format(id=cli_calibration_id)).expanduser(),
            "cli_id",
        )

    local = _load_local_config(project_root)
    if local:
        robot = local.get("robot") or {}
        explicit_path = robot.get("calibration_path")
        if explicit_path:
            return Path(explicit_path).expanduser(), "local_config"
        local_id = robot.get("id")
        if local_id:
            return Path(FOLLOWER_CALIBRATION_PATH_TEMPLATE.format(id=local_id)).expanduser(), "local_config"

    if allow_default_fallback:
        return (
            Path(FOLLOWER_CALIBRATION_PATH_TEMPLATE.format(id=DEFAULT_FOLLOWER_ID_FALLBACK)).expanduser(),
            "default_fallback",
        )

    raise ResolutionError(
        "calibration 경로를 확정할 수 없습니다 (armed 모드에서는 기본 fallback이 허용되지 않습니다). "
        "--calibration-path/--calibration-id를 지정하거나 configs/hardware.local.json의 "
        "robot.id/robot.calibration_path를 설정하세요."
    )


@dataclass(frozen=True)
class ResolvedPaths:
    port: str
    port_source: str
    calibration_path: Path
    calibration_source: str
