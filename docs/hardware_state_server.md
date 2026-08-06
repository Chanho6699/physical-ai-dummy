# SO-101 읽기 전용 관절 상태 서버

## 1. 목적

노트북에 연결된 SO-101 **리더암**과 **팔로워암**의 현재 관절값을 실시간으로 읽어
HTTP API(`GET /health`, `GET /state`, `GET /calibration`)로 제공한다. 이 서버의 출력은
이후 **데스크탑의 MuJoCo SO-101 진단 클라이언트**가 소비할 예정이다.

이번 작업에서는 MuJoCo, SmolVLA, YOLO, ROS2를 연결하지 않는다. 이 서버는 순수하게
"실물 관절 상태를 HTTP로 노출"하는 역할만 한다.

## 2. 노트북·데스크탑 분리 구조

```text
리더암 state ─┐
              ├→ 노트북 Read-Only State Server (0.0.0.0:8001)
팔로워 state ─┘
                     ↓ HTTP (LAN 또는 Tailscale)
              데스크탑 MuJoCo 진단 클라이언트 (이번 작업 범위 밖)
```

- 노트북: USB serial로 리더암·팔로워암과 직접 통신하며, 이 저장소의
  `hardware/state_server/`와 `scripts/run_hardware_state_server.py`를 실행한다.
- 데스크탑: 이 서버의 `/state`를 주기적으로 폴링해 MuJoCo에 반영한다 (별도 작업).
- 두 머신은 로컬 네트워크 또는 Tailscale로 연결된다고 가정한다 (아래 8절 참고).

## 3. 읽기 전용 보장

### 3.1 왜 `lerobot.robots.so_follower.SOFollower` / `lerobot.teleoperators.so_leader.SOLeader`를 그대로 쓰지 않는가

`~/lerobot/src/lerobot/robots/so_follower/so_follower.py`,
`~/lerobot/src/lerobot/teleoperators/so_leader/so_leader.py`를 직접 읽고 확인했다
(LeRobot은 이 저장소에 없고 `~/lerobot`에 별도 관리됨 - README.md 참고).

1. **`connect()`가 무조건 쓰기를 발생시킨다.** `SOFollower.connect()` /
   `SOLeader.connect()`는 내부에서 항상 `self.configure()`를 호출한다.
   `configure()`는 `bus.torque_disabled()` 컨텍스트 안에서이긴 하지만
   `bus.write("Operating_Mode", ...)`, `P/I/D_Coefficient`,
   (follower 한정) `Max_Torque_Limit`/`Protection_Current`/`Overload_Torque`까지
   모터 레지스터에 `write()`를 실제로 실행한다.
2. **캘리브레이션 불일치 시 더 위험한 경로로 빠진다.** 모터에 저장된 캘리브레이션이
   전달한 캘리브레이션 파일과 다르면(`is_calibrated == False`) `connect()`가
   `self.calibrate()`까지 실행한다. `calibrate()`는 `bus.disable_torque()`,
   `input()`으로 사용자에게 팔을 움직이라고 요구, `set_half_turn_homings()`(모터의
   `Homing_Offset`을 새로 write), `record_ranges_of_motion()`, `write_calibration()`을
   수행할 수 있다 - 읽기 전용 서버에서는 절대 일어나서는 안 되는 경로다.
3. `SOLeader`에는 별도로 공개된 `enable_torque()`/`disable_torque()` 메서드도 있다.

따라서 이 서버는 로봇/텔레오퍼레이터 클래스를 완전히 우회하고,
`lerobot.motors.feetech.FeetechMotorsBus`를 직접 사용하는 별도 reader
(`hardware/state_server/readonly_so101_reader.py`의 `ReadOnlySO101Reader`)를 구현했다.

### 3.2 실제 사용한 read path

| 단계 | 호출 | 확인한 근거 (쓰기 여부) |
|---|---|---|
| 버스 생성 | `FeetechMotorsBus(port=..., motors=..., calibration=...)` | 통신을 열지 않는 단순 객체 생성. |
| 연결 | `bus.connect()` (`motors_bus.py`의 `SerialMotorsBus.connect` → `FeetechMotorsBus._handshake`) | 포트를 열고 각 모터에 `ping()` + `_assert_same_firmware()`(펌웨어 버전 read)만 수행. **write 없음.** |
| 정규화 위치 읽기 | `bus.sync_read("Present_Position", ..., normalize=True)` | `GroupSyncRead` 기반 읽기 명령. 모터에 값을 쓰지 않는다. |
| raw tick 읽기 | `bus.sync_read("Present_Position", ..., normalize=False)` | 동일한 읽기 명령을 정규화 없이 호출한 것뿐이라 **안전하게 지원 가능** (아래 3.4 참고). |
| 해제 | `bus.disconnect(disable_torque=False)` | 포트만 닫는다. 기본 인자 `disable_torque=True`로 두면 `disable_torque()`가 `Torque_Enable`/`Lock` 레지스터에 `write()`를 실행하므로(`feetech.py` 확인) **반드시 False로 고정**했다. |

`ReadOnlySO101Reader`가 절대 호출하지 않는(그리고 공개 메서드로 노출하지도 않는)
메서드: `write`, `sync_write`, `enable_torque`, `disable_torque`, `write_calibration`,
`reset_calibration`, `set_half_turn_homings`, `calibrate`, `configure`, `setup_motor`,
`send_action`, `send_feedback`, `teleop_step`. `tests/test_readonly_so101_reader.py`가
① 클래스의 공개 속성 이름 목록을 감사(audit)하고 ② 가짜 버스에 이 메서드들을 두어
호출되는 즉시 테스트가 실패하도록 만들어 둘 다 검증한다.

### 3.3 connect/disconnect가 torque 상태를 바꾸는가

- **로봇/텔레오퍼레이터 클래스(`SOFollower`/`SOLeader`)를 썼다면**: 그렇다. `connect()`가
  `configure()`를 통해 모터 설정 레지스터를 쓰고, 필요 시 `calibrate()`까지 실행된다.
  이것이 이 클래스들을 쓰지 않은 핵심 이유다 (3.1절).
- **이 서버(`ReadOnlySO101Reader`)는**: 아니다. `connect()`는 `bus.connect()`만
  호출하고, `disconnect()`는 `bus.disconnect(disable_torque=False)`만 호출한다. torque
  활성/비활성 레지스터에 어떤 write도 발생하지 않는다. (참고: `disable_torque=True`가
  기본값인 이유는 로봇 정지 시 안전을 위해서지만, 그 자체가 `write()` 호출이기 때문에
  "쓰기 명령을 절대 보내지 않는다"는 이 서버의 원칙을 예외 없이 지키기 위해 의도적으로
  꺼 두었다.)

### 3.4 raw tick 지원 여부

지원한다. `bus.sync_read(..., normalize=False)`는 `bus.sync_read(..., normalize=True)`와
완전히 동일한 통신 경로(`GroupSyncRead`)를 쓰고 차이는 값을 도(degree)/0~100으로
스케일링하는지 여부뿐이라, 읽기 자체는 동등하게 안전하다. 그래서 스펙의
"[미지원] 안전한 읽기 전용 raw tick 조회 경로를 확인하지 못했습니다" 문구를 쓰지
않았다 - 실제로 확인된 안전한 경로가 있었기 때문이다. `GET /state`의
`leader.raw_ticks` / `follower.raw_ticks`에 0~4095 범위의 정수로 채워진다.

### 3.5 gripper 단위 관련 주의

LeRobot의 SO-101 설정(`so_follower.py`/`so_leader.py`)은 `use_degrees=True`(기본값)여도
gripper 모터만은 항상 `MotorNormMode.RANGE_0_100`(0~100)으로 정규화하고, 나머지 5개
관절(`shoulder_pan`~`wrist_roll`)만 degree로 정규화한다. 이 서버도 동일하게
재현했다. 따라서 API 응답의 `positions_deg["gripper"]`는 이름과 달리 **실제로는
"열림 비율(0~100)"이며 각도(degree)가 아니다.** 응답 스키마의 다른 필드와 통일성을
위해 키 이름은 `positions_deg`로 유지했다.

## 4. 실행 방법

### 4.1 사전 준비 (설치)

이 프로젝트의 다른 도구들과 마찬가지로 `~/lerobot` venv를 그대로 사용한다
(`docs/data_collection_tools.md`, `docs/mujoco_action_replay.md`와 동일한 패턴).
`fastapi`/`uvicorn`/`pytest`가 이 venv에 없어서 새로 설치했다 (실제로 설치해
동작을 확인함):

```bash
source ~/lerobot/lerobot/bin/activate
pip install "fastapi>=0.110" "uvicorn>=0.27" pytest
```

설치된 버전(이 작업 시점 기준): `fastapi==0.141.1`, `uvicorn==0.52.1`,
`pytest==9.1.1`, `pydantic==2.13.4`. `pyserial`, `feetech-servo-sdk`(scservo_sdk),
`PyYAML`, `deepdiff`, `httpx`는 이미 `~/lerobot` venv에 설치되어 있어 추가하지
않았다.

> **참고**: `docs/mujoco_action_replay.md`는 `~/lerobot/.venv`를 언급하지만, 이 작업
> 시점에는 그 경로가 존재하지 않았고(재생성/이름 변경된 것으로 보임) 실제로 동작하는
> venv는 `~/lerobot/lerobot`이었다 (`data_collection/recorder.py`,
> `docs/data_collection_tools.md`가 안내하는 `source ~/lerobot/lerobot/bin/activate`와
> 동일). 이번 작업은 이 경로를 사용했다.

### 4.2 dry-run (하드웨어 연결 없이 설정만 확인)

```bash
source ~/lerobot/lerobot/bin/activate
cd ~/Projects/physical-ai-dummy

python scripts/run_hardware_state_server.py --dry-run --verbose
```

캘리브레이션 파일이 존재/파싱 가능한지, 설정값(host/port/rate_hz/...)이 무엇인지만
확인하고 종료한다 (`exit code 0`). 하드웨어에 어떤 통신도 시도하지 않는다.

### 4.3 실제 서버 실행

```bash
python scripts/run_hardware_state_server.py \
  --leader-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14029966-if00 \
  --follower-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14113538-if00 \
  --leader-id chanho_leader \
  --follower-id chanho_follower \
  --host 0.0.0.0 \
  --port 8001 \
  --rate-hz 30
```

`--leader-id`/`--follower-id`의 기본값이 각각 `chanho_leader`/`chanho_follower`이고,
`--leader-calibration-path`/`--follower-calibration-path`를 생략하면 LeRobot 표준
캐시 경로에서 id로 자동으로 찾으므로, 이 노트북에서는 다음처럼 포트만 넘겨도 된다:

```bash
python scripts/run_hardware_state_server.py \
  --leader-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14029966-if00 \
  --follower-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14113538-if00
```

추가 옵션: `--leader-calibration-path`, `--follower-calibration-path`,
`--stale-after-ms`(기본 500), `--max-read-errors`(기본 3), `--api-token`,
`--config`(기본 `configs/hardware_state_server.yaml` - 있으면 기본값으로 병합되고,
실제로 넘긴 CLI 플래그가 항상 우선한다), `--quiet`, `--verbose`, `--no-color`.

### 4.4 종료

`Ctrl+C`(SIGINT) 또는 `kill <PID>`(SIGTERM) 모두 다음을 순서대로 수행한다:

1. 백그라운드 상태 출력 스레드 정지.
2. `[종료] 서버 종료 처리 중...` 출력.
3. 폴링 스레드(`StatePoller`) 정지 (`join`으로 완전히 끝날 때까지 대기).
4. 리더/팔로워 reader의 `disconnect()` 호출 (`bus.disconnect(disable_torque=False)` -
   포트만 닫고 목표값을 전송하지 않음). 실패해도 예외를 삼키지 않고 `[오류]`로
   출력한 뒤 나머지 정리를 계속한다.
5. `[종료] 완료` 출력.

> **구현 메모 (SIGTERM 관련 버그와 수정)**: 초기 구현에서 `uvicorn.Server.run()`은
> graceful shutdown을 완료한 뒤 원래 시그널 핸들러를 복원하고 같은 시그널을
> `signal.raise_signal()`로 다시 보낸다. SIGINT는 Python 기본 핸들러가
> `KeyboardInterrupt`를 던져 우리 `try/except`까지 안전하게 전파되지만, SIGTERM의 OS
> 기본 동작은 프로세스를 즉시 죽이는 것이라 실제로 테스트해보니(`kill -TERM`) 위 4단계
> 정리 코드가 전혀 실행되지 않고 프로세스가 종료됐다(`$? == 143`). 그래서
> `scripts/run_hardware_state_server.py`에 SIGTERM을 `KeyboardInterrupt`로 변환하는
> 핸들러를 `server.run()` 호출 직전에 등록해 SIGINT와 동일한 정상 종료 경로를 타도록
> 수정했다. 수정 후 `kill -TERM`으로 실제 검증했다 (exit code 0, 정리 로그 모두 출력 -
> 7절 참고).

## 5. `GET /health`

```json
{
  "status": "ok",
  "mode": "read_only",
  "leader_connected": true,
  "follower_connected": true,
  "write_enabled": false,
  "timestamp": 1786020000.0,
  "errors": []
}
```

`status`는 리더/팔로워 중 하나라도 연결되지 않았거나, `read_error_count`가
`--max-read-errors` 이상이거나, 값이 stale이면 `"degraded"`가 된다. `errors`에는
한글 메시지(`"팔로워암 연결 실패"`, `"리더암 state 읽기 3회 연속 실패"` 등)가 담긴다.

## 6. `GET /state`

```json
{
  "timestamp": 1786020000.0,
  "sequence": 152,
  "mode": "read_only",
  "leader": {
    "connected": true,
    "positions_deg": {
      "shoulder_pan": 3.2, "shoulder_lift": -12.4, "elbow_flex": 31.8,
      "wrist_flex": 82.4, "wrist_roll": 4.1, "gripper": 15.1
    },
    "raw_ticks": {
      "shoulder_pan": 2087, "shoulder_lift": 1450, "elbow_flex": 2201,
      "wrist_flex": 2482, "wrist_roll": 2048, "gripper": 2200
    },
    "stale": false,
    "age_ms": 12.4,
    "read_error_count": 0
  },
  "follower": { "...": "leader와 동일한 구조" },
  "difference_deg": {
    "shoulder_pan": 0.2, "shoulder_lift": -0.4, "elbow_flex": 0.6,
    "wrist_flex": 2.6, "wrist_roll": 0.1, "gripper": 0.4
  },
  "warnings": []
}
```

- `difference_deg[joint] = leader.positions_deg[joint] - follower.positions_deg[joint]`.
  둘 중 하나라도 값이 없으면 `difference_deg`는 빈 객체가 되고 `warnings`에 이유가
  담긴다.
- `stale`/`age_ms`/`read_error_count`는 스펙 예시(`"상태 읽기 실패 시"` 절)에 맞춰
  `leader`/`follower` 블록에 상시 포함시켰다 (마지막 정상값을 계속 제공하면서도
  얼마나 오래됐는지 클라이언트가 판단할 수 있게 하기 위함).
- HTTP 요청이 들어올 때마다 하드웨어를 다시 읽지 않는다. 이 값은 백그라운드
  폴링 스레드가 채워둔 캐시를 즉시 반환한 것이다.

## 7. `GET /calibration`

```json
{
  "leader": {
    "wrist_flex": { "homing_offset": -1419, "range_min": 1019, "range_max": 3225 }
  },
  "follower": {
    "wrist_flex": { "homing_offset": 1716, "range_min": 1052, "range_max": 2977 }
  }
}
```

캘리브레이션 파일 절대 경로 등 민감한 시스템 경로는 응답에 포함하지 않는다
(`calibration_loader.to_public_dict`가 `homing_offset`/`range_min`/`range_max`만
남긴다).

## 8. Tailscale 또는 LAN 접근 예시

노트북에서 `--host 0.0.0.0 --port 8001`로 서버를 띄운 뒤, 데스크탑에서:

```bash
# LAN (같은 공유기에 연결된 경우)
curl http://<노트북의 LAN IP>:8001/health

# Tailscale (별도 네트워크에 있어도 동작)
curl http://<노트북의 tailscale IP 또는 MagicDNS 이름>:8001/health
```

데스크탑 MuJoCo 진단 클라이언트가 사용할 기본 base URL은 다음과 같다 (실제 값은
노트북의 네트워크 설정에 따라 다르며, 이번 작업에서 확정하지 않았다):

```text
http://<노트북 주소>:8001
```

방화벽에서 8001 포트를 노트북 쪽에서 허용해야 한다. Tailscale을 쓰면 두 기기가 같은
tailnet에 있기만 하면 별도 포트 포워딩 없이 접근 가능하다 (Tailscale 자체의 ACL
설정은 이번 작업 범위 밖).

## 9. 보안 주의사항

- **CORS 전체 허용을 추가하지 않았다.** `CORSMiddleware`를 아예 등록하지 않아
  브라우저 기반 크로스 오리진 요청에 대한 `Access-Control-Allow-Origin` 헤더가
  없다 (`tests/test_hardware_state_server_api.py::test_no_wildcard_cors_headers_on_cross_origin_request`로
  검증).
- **제어 API가 없다.** `/action`, `/move`, `/command`, `/teleop`으로 POST하면 404,
  정의된 GET 경로(`/health`, `/state`, `/calibration`)에 POST하면 405가 반환된다.
  Swagger(`/docs`)에도 GET 3개만 노출된다.
- **request body를 받는 라우트가 없다.** 모든 엔드포인트가 GET이며 쿼리 파라미터도
  받지 않는다.
- **캘리브레이션 파일을 수정하는 API가 없다.** 읽기만 한다.
- **임의 경로 탐색/파일 읽기 API가 없다.** 캘리브레이션 경로는 서버 시작 시 CLI
  인자로만 고정되며, 요청 시점에 클라이언트가 경로를 지정할 방법이 없다.
- **임의 Python 실행 기능이 없다.**
- **`--api-token`(선택)**: 지정하면 모든 GET 요청에
  `Authorization: Bearer <token>` 헤더가 필요하다. `--api-token` 대신 환경변수
  `SO101_STATE_SERVER_TOKEN`으로도 지정할 수 있다 (커밋될 수 있는 YAML/스크립트
  인자보다 안전). 토큰 값은 로그/콘솔 출력에 절대 나타나지 않는다 (`--verbose`
  출력도 `설정됨`/`없음`으로만 표시).
- 이 서버는 로컬 네트워크 또는 Tailscale 전용으로 설계했다. 공인 인터넷에 그대로
  노출하는 것은 권장하지 않는다 (TLS 종료, rate limit 등이 없음).

## 10. 오류 처리

- **연결 실패**: 리더/팔로워 중 하나가 연결에 실패해도 다른 하나는 계속 연결을
  시도한다 (부분 연결 허용). 실패한 쪽은 `/health`의 `leader_connected`/
  `follower_connected`가 `false`가 되고 `status`가 `"degraded"`가 된다.
- **읽기 실패**: 마지막 정상값과 해당 timestamp를 계속 보존하고 `stale`로 표시한다.
  거짓 정상값을 만들지 않는다 (`ArmPollState.last_good`이 검증(NaN/Inf, 관절 누락,
  shape 불일치 거부)을 통과한 샘플로만 갱신됨 - `state_models.validate_positions_deg`).
- **연속 실패 임계값**: `--max-read-errors`(기본 3)회 연속 실패하면 콘솔에
  `[경고] ... N회 연속 실패` + `[조치] 마지막 정상값을 stale 상태로 제공합니다.`를
  1회만 출력하고(도배 방지), `/health`가 `"degraded"`로 전환된다. 이후 성공하면
  `[통과] ... 연결 복구`를 출력한다.
- **종료 중 오류**: `disconnect()` 중 예외가 발생해도 숨기지 않고 `[오류]`로
  출력하며, 남은 팔의 정리는 계속 진행한다.

## 11. 알려진 제한사항 (확인되지 않음)

- **실측 30Hz 안정성**: `--rate-hz 30`을 기본값으로 뒀지만, 이는 이 프로젝트의 기존
  카메라/데이터셋 fps 관례(30fps)를 따른 것이며 이 SO-101 Feetech 버스가 두 팔(리더+
  팔로워, 총 12개 모터)을 한 스레드에서 순서대로 `sync_read`할 때 실제로 30Hz를
  안정적으로 유지하는지는 **이번 작업에서 실물로 측정하지 않았다**. 통신 지연에 따라
  실측 주기가 이보다 낮을 수 있다.
- **`stale_after_ms`/`max_read_errors` 기본값**: 500ms/3회는 보수적으로 고른 값이며
  실물 튜닝을 거치지 않았다.
- **동시 접속자 수**: 여러 클라이언트가 동시에 폴링할 때의 성능은 검증하지 않았다
  (FastAPI/uvicorn 기본 동작에 의존).
- **`mujoco` 미설치**: 이 저장소의 기존 `tests/test_mujoco_*.py` 4개는 현재 이
  venv(`~/lerobot/lerobot`)에 `mujoco` 패키지가 없어 수집 단계에서 실패한다. 이는
  이번 작업 이전부터 있던 환경 문제이며(`docs/mujoco_action_replay.md`가 언급하는
  `~/lerobot/.venv`가 현재 존재하지 않음), 이번 작업의 범위(MuJoCo 미연결)와도
  무관해 손대지 않았다.
- **실물 하드웨어 검증**: 이 문서와 코드는 `--dry-run`, mock reader 기반 단위
  테스트, 그리고 (연결되어 있다면) `/health`/`/state`의 짧은 실물 조회로 검증했다.
  장시간 연속 운용, 실제 USB 재연결/케이블 분리 시나리오, 여러 시간대의 안정성은
  검증하지 않았다.
