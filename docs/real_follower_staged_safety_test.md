# Real follower staged safety test (Candidate B) — 사용 안내

이 저장소가 실제 SO-101 follower에 처음으로 write하는 경로다. 반드시 **로봇 옆에 물리적으로
있으면서 즉시 개입(전원 차단 등)할 수 있는 상태**에서만 실행한다.

이 세션(cloud/원격 환경)에는 시리얼 장치가 전혀 연결되어 있지 않아 실제 하드웨어로 검증하지
못했다. 대신 다음 두 가지로 검증했다:
- fake follower 기반 단위/통합 테스트 (`tests/test_staged_follower_writer.py`,
  `tests/test_staged_real_rollout.py`, `tests/test_run_real_follower_staged_safety_test_cli.py` —
  27개 전부 통과).
- **실제 `scripts/run_vla_server.py --fake` HTTP 서버를 로컬에 띄워 진짜 HTTP 왕복**으로
  health check/checkpoint 불일치 감지/연결 실패 처리를 직접 확인했다 (아래 "http 모드 검증
  이력" 참고). 실제 candidate B checkpoint를 실제 로봇으로 실행하는 것은 사용자가 직접 해야
  한다.

## 두 가지 VLA 실행 방식 (`--vla-mode`, 필수)

| | `inprocess` | `http` |
|---|---|---|
| checkpoint 로딩 위치 | 이 스크립트를 실행하는 laptop | Desktop (별도 머신, GPU) |
| laptop에 필요한 것 | torch/transformers/CUDA | 없음 (requests만) |
| 재사용 코드 | `runtime/laptop/inprocess_vla_client.InProcessSmolVLAClient` (기존, 변경 없음) | `runtime/laptop/vla_client.VLAHttpClient` (기존, `scripts/run_shadow_mode.py`가 이미 쓰는 것과 동일 - 새 프로토콜 아님) |
| stage 시작 전 검증 | checkpoint 로딩 성공 여부 | `/health` 응답 + `model_id`가 candidate B checkpoint 서명을 포함하는지 |

`http` 모드에서는 `InProcessSmolVLAClient`를 **한 번도 import하지 않는다** (import 구조
자체로 laptop이 GPU/checkpoint 없이도 항상 동작하게 강제함 - 코드 리뷰로 확인 가능,
`scripts/run_real_follower_staged_safety_test.py`의 `if args.vla_mode == "http":` 분기 참고).

## 실행 순서

### 0) Desktop에서 (GPU 있는 머신) — candidate B VLA 서버 실행

```bash
source ~/lerobot/.venv/bin/activate
python scripts/run_vla_server.py \
  --checkpoint outputs/pick_drop_v3_v4_combined69/smolvla_pick_drop_v3_v4_combined69_uniform_fresh/checkpoints/010000/pretrained_model \
  --host 0.0.0.0 --port 9200
```

(기본 포트는 **9200**이다 - `scripts/run_shadow_mode.py`의 기존 예시와 동일한 관례. 사용자가
제시한 8000은 이 저장소의 기존 관례와 다른 placeholder였다 - 실제 서버를 어떤 포트로
띄우든 `--vla-server-url`에 그 값을 그대로 맞추면 된다.)

### 1) laptop에서 dry-run (하드웨어/서버 접근 없음)

```bash
python scripts/run_real_follower_staged_safety_test.py --stage 1 --dry-run \
  --vla-mode http --vla-server-url http://<desktop-tailscale-ip>:9200 \
  --hardware-config configs/hardware.local.json \
  --follower-port /dev/serial/by-id/<실제 팔로워 포트> --follower-id <실제 follower id>
```

### 2) laptop에서 stage 1 실제 실행 (로봇을 직접 보면서)

```bash
python scripts/run_real_follower_staged_safety_test.py --stage 1 \
  --vla-mode http --vla-server-url http://<desktop-tailscale-ip>:9200 \
  --hardware-config configs/hardware.local.json \
  --follower-port /dev/serial/by-id/<...> --follower-id <...> \
  --confirm-physically-present
```

실행하면 `--confirm-physically-present` 플래그에 더해, 실행 중 다음 문구를 정확히 타이핑해야
한다: `I AM PHYSICALLY PRESENT AND WATCHING THE ROBOT`.

이 명령이 하는 일의 순서(실패하면 그 즉시 멈추고, 이후 단계로 넘어가지 않음):
1. `http://<desktop-tailscale-ip>:9200/health` 확인 - 실패/timeout이면 **하드웨어를 전혀
   열지 않고** 바로 중단한다.
2. health가 정상이면 응답의 `model_id`(Desktop이 실제로 로딩한 `--checkpoint` 문자열
   그대로)가 candidate B checkpoint 서명을 포함하는지 확인 - 다르면 기본적으로 중단한다
   (`--force-checkpoint-mismatch`로만 우회 가능, 신중히 판단할 것).
3. 카메라 연결 (`configs/hardware.local.json`).
4. 팔로워 state 읽기 연결 (read-only).
5. Safety Gate 준비 (임계값 불변).
6. 팔로워 write 연결 (`SOFollower`, 여기서부터 실제 write가 **가능**해진다 - 4번 참고).
7. stage 실행: 매 step 관측→추론→Action Adapter→Safety Gate 순으로 판정하고, **`ACCEPT`일
   때만** 실제로 write한다.

### 3) stage 1이 PASS면, 로봇을 직접 확인한 뒤에만 stage 2

```bash
python scripts/run_real_follower_staged_safety_test.py --stage 2 --steps 3 \
  --vla-mode http --vla-server-url http://<desktop-tailscale-ip>:9200 \
  --hardware-config configs/hardware.local.json \
  --follower-port /dev/serial/by-id/<...> --follower-id <...> \
  --confirm-physically-present
```

### 4) stage 2가 PASS면 stage 3 (short chunk, 최대 15 step)

```bash
python scripts/run_real_follower_staged_safety_test.py --stage 3 --steps 10 \
  --vla-mode http --vla-server-url http://<desktop-tailscale-ip>:9200 \
  --hardware-config configs/hardware.local.json \
  --follower-port /dev/serial/by-id/<...> --follower-id <...> \
  --confirm-physically-present
```

### inprocess 모드로 대체 실행하고 싶다면 (laptop에 GPU/checkpoint 있을 때만)

```bash
python scripts/run_real_follower_staged_safety_test.py --stage 1 \
  --vla-mode inprocess \
  --hardware-config configs/hardware.local.json \
  --follower-port /dev/serial/by-id/<...> --follower-id <...> \
  --confirm-physically-present
```

## 어떤 과정에서 real write가 가능한가

`hardware/safety/staged_follower_writer.StagedFollowerArmedWriter.write_action_once()` 단
한 곳뿐이다 - 그리고 그 메서드는 `runtime/laptop/staged_real_rollout.py`의 파이프라인에서
`decision.decision == "ACCEPT"`일 때만 호출된다(코드 레벨로 강제됨, 이 문서가 아니라
`StagedRealRolloutRunner._run_one_step`을 직접 읽어 확인 가능). 그 앞의 모든 단계
(VLA health check, checkpoint 서명 검증, 카메라 연결, state 읽기, Safety Gate 판정)는 전부
write 발생 여부에 영향을 줄 수는 있어도(예: 어느 하나라도 실패하면 중단되어 write가
일어나지 않음) 그 자체로 write를 수행하지는 않는다. `SOFollower.connect()`(위 순서 6번)
시점에 LeRobot 표준 `configure()`가 내부적으로 도는데, 이는 `Goal_Position` write가 아니라
PID/Operating_Mode 설정이다(`hardware/diagnostics/instrumented_teleop.py`에 이미 문서화된
LeRobot 표준 동작 - 이 세션이 새로 추가한 write가 아니다).

## http 모드 검증 이력 (이 세션에서 실제로 수행함)

로컬에서 `python scripts/run_vla_server.py --fake --host 127.0.0.1 --port 9250`으로 진짜
FastAPI 서버를 띄우고, `run_real_follower_staged_safety_test.py`를 실제 HTTP 요청으로
붙여서 다음을 확인했다 (전부 실제 소켓 왕복, mock 아님):
- `/health` 정상 확인 → `model_id`가 fake 서버 값("fake-identity-v1")이라 candidate B
  서명과 불일치 → **기본값으로 정확히 차단됨**(하드웨어 연결 시도 전).
- `--force-checkpoint-mismatch`를 주면 경고를 찍고 통과 → 다음 단계(카메라 연결)로 정상
  진행하다가 이 환경엔 실제 카메라가 없어 카메라 오류로 멈춤(팔로워는 여전히 안 건드림).
- 존재하지 않는 포트로 연결 시도 → timeout 후 정확히 exit code 2로 중단, 하드웨어 미접근.

## ABNORMAL_STOPPED가 나오면

절대 같은 stage를 바로 재시도하지 말 것. 리포트 JSON에서 `stop_reason`과 마지막 step의
`safety_decision`/`safety_reasons`를 먼저 읽고, 왜 non-ACCEPT가 나왔는지 파악한 뒤에 판단한다.

## 아직 하지 않은 것 (요청 범위 밖)

cube 접근 테스트는 이 3-stage가 전부 PASS한 뒤에만, 별도 요청 시 진행한다.
