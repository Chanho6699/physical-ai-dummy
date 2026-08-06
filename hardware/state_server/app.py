"""FastAPI 읽기 전용 상태 조회 API.

이 모듈에는 읽기(GET) 엔드포인트만 존재한다: ``/health``, ``/state``, ``/calibration``.
POST/PUT/DELETE 라우트는 하나도 정의하지 않으며, ``/action``·``/move``·``/command``·
``/teleop`` 같은 제어 엔드포인트도 존재하지 않는다 - 정의되지 않은 경로/메서드는 FastAPI
기본 동작에 따라 404를 반환한다 (``tests/test_hardware_state_server_api.py``에서 검증).

CORS: 의도적으로 ``CORSMiddleware``를 추가하지 않았다. 이 서버는 로컬 네트워크/Tailscale
안에서만 사용할 예정이며, 전체 허용(``allow_origins=["*"]``) CORS는 명시적으로 금지되어
있다. 브라우저 기반 크로스 오리진 요청이 필요해지면 그때 허용 origin을 명시적으로
지정해서 추가해야 한다 (지금은 그런 요구가 없으므로 추가하지 않았다).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

from hardware.state_server.state_service import StatePoller

API_DESCRIPTION = (
    "실물 SO-101 리더암/팔로워암의 현재 관절 상태를 조회하는 **읽기 전용** API입니다.\n\n"
    "이 서버는 어떤 형태로도 실물에 명령을 전송하지 않습니다 "
    "(send_action / sync_write / torque enable 등 없음). "
    "제어가 필요한 요청(POST /action, /move, /command, /teleop 등)은 지원하지 않으며, "
    "존재하지 않는 경로/메서드로 요청하면 404가 반환됩니다."
)


class ArmStateModel(BaseModel):
    connected: bool
    positions_deg: dict[str, float] | None = None
    raw_ticks: dict[str, int] | None = None
    stale: bool
    age_ms: float | None = None
    read_error_count: int


class StateResponseModel(BaseModel):
    timestamp: float
    sequence: int
    mode: str
    leader: ArmStateModel
    follower: ArmStateModel
    difference_deg: dict[str, float]
    warnings: list[str]


class HealthResponseModel(BaseModel):
    status: str
    mode: str
    leader_connected: bool
    follower_connected: bool
    write_enabled: bool
    timestamp: float
    errors: list[str] = []


class JointCalibrationModel(BaseModel):
    homing_offset: int
    range_min: int
    range_max: int


class CalibrationResponseModel(BaseModel):
    leader: dict[str, JointCalibrationModel]
    follower: dict[str, JointCalibrationModel]


def create_app(
    *,
    poller: StatePoller,
    calibration_public: dict[str, dict[str, dict[str, int]]],
    api_token: str | None = None,
) -> FastAPI:
    """읽기 전용 FastAPI 앱을 만든다.

    Args:
        poller: 이미 생성된(연결 여부는 무관) StatePoller. 이 함수는 poller의 생명주기를
            관리하지 않는다 - start()/stop()/connect_all()/disconnect_all()은 호출자
            (scripts/run_hardware_state_server.py)가 관리한다.
        calibration_public: ``{"leader": {...}, "follower": {...}}`` 형태의 공개용
            캘리브레이션 dict (calibration_loader.to_public_dict 결과). 파일 경로 등
            민감한 정보는 포함하지 않아야 한다.
        api_token: 지정하면 모든 GET 엔드포인트에 ``Authorization: Bearer <token>``
            헤더를 요구한다. None이면 인증 없이 허용한다 (로컬/Tailscale 전용 기본값).
    """

    app = FastAPI(
        title="SO-101 읽기 전용 관절 상태 서버",
        description=API_DESCRIPTION,
        version="0.1.0",
    )

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if api_token is None:
            return
        expected = f"Bearer {api_token}"
        if authorization != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="인증 토큰이 없거나 올바르지 않습니다.",
            )

    def _to_arm_model(view) -> ArmStateModel:
        return ArmStateModel(
            connected=view.connected,
            positions_deg=view.positions_deg,
            raw_ticks=view.raw_ticks,
            stale=view.stale,
            age_ms=view.age_ms,
            read_error_count=view.read_error_count,
        )

    @app.get("/health", response_model=HealthResponseModel, tags=["read-only"])
    def get_health(_: None = Depends(require_token)) -> HealthResponseModel:
        snap = poller.health()
        return HealthResponseModel(
            status=snap.status,
            mode=snap.mode,
            leader_connected=snap.leader_connected,
            follower_connected=snap.follower_connected,
            write_enabled=snap.write_enabled,
            timestamp=snap.timestamp,
            errors=snap.errors,
        )

    @app.get("/state", response_model=StateResponseModel, tags=["read-only"])
    def get_state(_: None = Depends(require_token)) -> StateResponseModel:
        snap = poller.snapshot()
        return StateResponseModel(
            timestamp=snap.timestamp,
            sequence=snap.sequence,
            mode=snap.mode,
            leader=_to_arm_model(snap.leader),
            follower=_to_arm_model(snap.follower),
            difference_deg=snap.difference_deg,
            warnings=snap.warnings,
        )

    @app.get("/calibration", response_model=CalibrationResponseModel, tags=["read-only"])
    def get_calibration(_: None = Depends(require_token)) -> CalibrationResponseModel:
        return CalibrationResponseModel(
            leader=calibration_public["leader"],
            follower=calibration_public["follower"],
        )

    return app
