# SO-101 원격 MuJoCo 실시간 진단

노트북에 연결된 SO-101 리더암/팔로워암의 상태를 **읽기 전용**으로 조회하는 상태 서버
(`hardware/state_server/`, `docs/hardware_state_server.md`)와, 데스크탑의 기존 MuJoCo
안전 검증 파이프라인(`simulation/mujoco/`)을 연결해 실시간으로 비교/진단하는 도구다.

```text
노트북 리더암 state ───────────┐
                              ├→ 데스크탑 실시간 비교기
노트북 팔로워암 state ─────────┘
                                      ↓
리더암 state
→ 기존 Action Mapping (simulation/mujoco/action_mapping.py)
→ 기존 Safety Gate (simulation/mujoco/safety_checks.py)
→ MuJoCo SO-101 (simulation/mujoco/so101_model.py)
                                      ↓
한글 상태 화면 + CSV/JSON 진단 리포트
```

## 1. 왜 안전한가 (읽기 전용 구조)

- 데스크탑은 노트북 서버에 **GET만** 보낸다 (`GET /health`, `GET /state`, `GET /calibration`).
  `simulation/mujoco/remote_state_client.py`에는 POST/PUT/PATCH/DELETE 메서드 자체가
  정의되어 있지 않다 - 코드에 없으니 실수로도 호출할 수 없다.
- `/action`, `/move`, `/command`, `/teleop` 같은 제어 경로는 아예 호출하지 않는다
  (노트북 서버 자체도 그런 경로를 구현하지 않는다 - `docs/hardware_state_server.md` 참고).
- 팔로워암 값은 리더-팔로워 **비교 용도로만** 쓰인다. MuJoCo 목표값은 항상 리더암
  `positions_deg`에서만 온다.
- 실물 팔로워암에는 이 프로그램의 어떤 코드 경로로도 명령이 전달되지 않는다 - 데스크탑은
  노트북과 HTTP GET으로만 통신하고, 노트북 서버 자체가 팔로워에 쓰기를 하지 않는다
  (읽기 전용 서버이므로 데스크탑에서 무엇을 보내든 애초에 팔로워를 움직일 방법이 없다).
- MuJoCo 관절 range, 프레임간 변화량, 속도, 접촉 임계값은 **새로 정의하지 않고**
  기존 `configs/mujoco_so101.yaml` + `simulation/mujoco/safety_checks.py`를 그대로
  재사용한다. 관절 range를 벗어난 값은 절대 clamp하지 않고, 해당 관절만 직전 안전
  target을 유지한다 (아래 4번 참고).
- SmolVLA 체크포인트, YOLO, ROS2는 이 도구 어디에도 연결되어 있지 않다.

## 2. 서버 URL 설정 (LAN / Tailscale)

노트북에서 `scripts/run_hardware_state_server.py`를 먼저 실행해 상태 서버를 띄운다
(자세한 내용은 `docs/hardware_state_server.md`). 기본 포트는 `8001`이다.

같은 LAN에 있다면:

```bash
python scripts/run_remote_mujoco_diagnostic.py \
  --server-url http://<노트북_LAN_IP>:8001 --headless --joint wrist_flex --duration 20
```

Tailscale을 쓰면 노트북의 Tailscale IP(보통 `100.x.x.x`)나 MagicDNS 이름을 그대로 쓸 수 있다:

```bash
python scripts/run_remote_mujoco_diagnostic.py \
  --server-url http://100.x.x.x:8001 --headless --joint wrist_flex --duration 20
```

노트북 서버가 `--api-token`(또는 `SO101_STATE_SERVER_TOKEN` 환경변수)으로 인증을 켰다면,
데스크탑에서도 같은 토큰을 `--api-token` 또는 동일한 환경변수로 넘겨야 한다. 토큰은
CLI 인자·환경변수로만 받고, 콘솔 로그·CSV·JSON 리포트 어디에도 기록하지 않는다
(`simulation/mujoco/remote_state_client.py`의 `RemoteClientConfig.__repr__`,
`simulation/mujoco/diagnostic_report.py`의 `write_json_report`가 방어적으로 걸러낸다).

## 3. 실행 방법

### dry-run (네트워크/실물 없이 파이프라인만 확인)

```bash
python scripts/run_remote_mujoco_diagnostic.py --server-url http://127.0.0.1:8001 --dry-run
```

다음만 확인하고 끝난다: 설정 파일, CLI 인자, MuJoCo 모델 로딩, 관절 mapping, safety 설정,
mock state(0deg)로 mapping→safety gate 파이프라인을 1회 통과시켜보는 것, 리포트 저장
경로. **노트북에 어떤 HTTP 요청도 보내지 않고, MuJoCo actuator에도 값을 적용하지 않는다.**

### headless

```bash
python scripts/run_remote_mujoco_diagnostic.py \
  --server-url http://<노트북_IP>:8001 --headless --joint wrist_flex --duration 20
```

### 모든 관절

```bash
python scripts/run_remote_mujoco_diagnostic.py \
  --server-url http://<노트북_IP>:8001 --headless --all-joints --duration 30
```

`--joint`는 여러 번 지정해 두 개 이상의 관절을 함께 볼 수도 있다. `--all-joints`와는
동시에 쓸 수 없다 (둘 중 하나만). `--joint`/`--all-joints` 어느 쪽이든 **MuJoCo에는
항상 리더암 6개 관절 전체가 적용된다** - 이 옵션은 "무엇을 화면/리포트에서 강조해서
볼지"만 제어하고, MuJoCo가 실제로 받는 관절 수를 줄이지 않는다 (한쪽 관절만 갱신하면
나머지 관절이 부자연스럽게 굳어 있어 오히려 시각적으로 오해를 줄 수 있기 때문).

### GUI

```bash
python scripts/run_remote_mujoco_diagnostic.py \
  --server-url http://<노트북_IP>:8001 --gui --joint wrist_flex --duration 20
```

WSLg 환경에서는 MuJoCo GUI viewer가 segfault를 일으킬 수 있다 (알려진 제한사항 참고).
**반드시 headless로 먼저 검증한 뒤에만 GUI를 실행할 것** (15번 "실제 검증 순서" 참고).

### 그 외 옵션

| 옵션 | 의미 | 기본값 |
| --- | --- | --- |
| `--rate-hz` | 노트북 폴링 주기(Hz) | 20 (설정 파일) |
| `--timeout-ms` | HTTP 요청 timeout | 500 |
| `--stale-after-ms` | 이 나이(ms)를 넘으면 stale로 간주 | 500 |
| `--max-retries` | 요청 실패 시 최대 재시도 횟수 (무한 재시도 없음) | 3 |
| `--api-token` | 노트북 서버 인증 토큰 | 없음 (또는 `SO101_STATE_SERVER_TOKEN`) |
| `--record` | 샘플별 CSV 상세 기록 저장 (기본은 JSON 요약만) | 꺼짐 |
| `--report-path` | JSON 리포트 저장 경로 (CSV는 같은 이름에 `.csv`) | `reports/remote_mujoco_diagnostic/session_<시각>.json` |
| `--quiet` / `--verbose` / `--no-color` | 출력 상세도/색상 제어 | 모두 꺼짐 |

## 4. 상태 표시 의미

### 준비 단계 (요구사항 6번 순서 그대로)

```text
[준비] SO-101 원격 MuJoCo 진단
[서버] http://100.x.x.x:8001
[통과] 서버 연결          <- GET /health 성공
[통과] 서버 상태          <- status == ok (degraded면 WARN으로 표시하되 계속 진행)
[통과] READ ONLY 모드 확인 <- mode == read_only, 아니면 즉시 BLOCKED
[통과] 쓰기 기능 비활성화 확인 <- write_enabled == false, 아니면(모르면 포함) 즉시 BLOCKED
[통과] 리더암 연결
[통과] 팔로워암 연결
[통과] 최신 state 수신     <- GET /state, stale==false, 관절 6개 확인
[통과] MuJoCo 모델 로딩
[통과] 관절 mapping 확인
```

`mode != read_only` 또는 `write_enabled != false`(값을 모르는 경우 포함)이면 그 즉시
BLOCKED로 중단하고 실행 자체를 시작하지 않는다 - 다른 어떤 검사보다 우선한다.

### 실시간 비교 화면

```text
[SO-101 원격 실시간 진단]
[서버 상태] 정상
[샘플] 184
[지연] 21 ms
[관절] wrist_flex
리더암       :  82.35 deg
팔로워암     :  79.81 deg
차이         :   2.54 deg
MuJoCo 목표  :  82.35 deg   <- 리더 값을 mapping한 rad/deg 목표 (BLOCKED가 아니면 리더 값과 같음)
MuJoCo 실제  :  81.92 deg   <- 물리 스텝 이후 실제 qpos (목표와 다를 수 있음)
상한 여유    :  12.65 deg   <- min(목표-하한, 상한-목표). 음수면 이미 범위를 벗어난 상태
Safety       : PASS
```

화면 갱신은 초당 4회로 제한한다 (요구사항: 초당 2~5회 이하). 매 네트워크 요청마다
줄을 찍지 않는다. `--verbose`이거나 관절을 2개 이상 선택하면(`--all-joints` 포함)
전체 관절 표로 표시된다.

### PASS / WARN / BLOCKED

- **PASS**: 이상 없음.
- **WARN**: `difference_warn_deg`(기본 3deg)를 넘는 리더-팔로워 차이, 프레임간 변화량
  초과, 속도 초과, 접촉 등 - 실행은 계속된다.
- **BLOCKED**: 해당 관절이 MuJoCo 관절/actuator range를 벗어남 - 그 관절만 직전 안전
  target을 유지하고, 값을 clamp하지 않는다. 예:

  ```text
  [차단] wrist_flex 목표값이 MuJoCo 상한을 초과했습니다.
  [리더 값] 96.82 deg
  [MuJoCo 범위] -95.00 ~ 95.00 deg
  [처리] 새 target을 적용하지 않고 직전 안전값을 유지합니다.
  ```

  이 도구는 **관찰/진단 도구**라 BLOCKED가 발생해도 세션 전체를 중단하지 않는다
  (`simulation/mujoco/dataset_action_replay.py`와의 차이점 - 그쪽은 재생 도구라 즉시
  멈춘다). 세션이 끝난 뒤 최종 결과는 BLOCKED 발생 횟수를 반영해 **WARN**으로
  표시된다. 최종 결과가 **BLOCKED**인 경우는 준비 단계 자체가 실패해 진단을 시작하지도
  못했다는 뜻이다 (mode 위반, 연결 실패 등).

## 5. 네트워크 오류 처리

`/state` timeout, stale, sequence 정지, malformed 응답, leader/follower disconnected,
NaN/Inf, 관절 누락, 서버 mode 변화, write_enabled 위반 중 하나라도 감지되면 그 즉시
MuJoCo 갱신을 멈추고 직전 안전 target을 유지한다 (`[일시정지]` 출력).

기본값(`configs/remote_mujoco_diagnostic.yaml`의 `safety.auto_resume: false`)은
**한 번 멈추면 이 세션 동안 자동으로 재개하지 않는** 보수적 정책이다. `mode`/
`write_enabled` 위반은 이 설정과 무관하게 항상 영구 정지(fatal)로 처리한다 - 안전
정책이 서버 쪽에서 실제로 깨졌다는 뜻이라 세션 내에서 복구를 시도하지 않는다.

`auto_resume: true`로 바꾸면 연속 정상 응답이 `resume_after_consecutive_ok`(기본 5)회
쌓였을 때만 재개한다 (단, mode/write_enabled 위반은 여전히 재개하지 않는다).

## 6. 진단 이벤트 의미 (`simulation/mujoco/diagnostic_analysis.py`)

| 코드 | 의미 | 판정 방식 |
| --- | --- | --- |
| `persistent_difference` | 리더-팔로워 차이가 같은 방향으로 오래 지속 | `abs(diff) > persistent_difference_deg`가 `persistent_duration_sec` 이상 같은 부호로 유지 |
| `follower_saturation_suspected` | 팔로워 포화(saturation) 의심 | `saturation_duration_sec` 동안 리더는 `leader_motion_delta_deg` 이상 움직였는데 팔로워는 `follower_stationary_delta_deg` 이하로만 움직이고 차이는 벌어짐 |
| `sign_mismatch_suspected` | 리더-팔로워 변화 방향(부호) 불일치 | 최근 `sign_mismatch_window_size`개 구간 중 `sign_mismatch_min_count`개 이상에서 부호가 반대 |
| `offset_suspected` | 고정 offset(캘리브레이션 영점 차이) 의심 | `offset_window_sec` 동안 리더가 `offset_pose_variation_deg` 이상 여러 자세를 거치는데 차이의 표준편차가 `offset_stability_deg` 이하 |
| `leader_out_of_mujoco_range` | 리더만 MuJoCo 관절 range를 벗어남 (팔로워는 range 안쪽) | 이번 wrist_flex 문제(`docs/wrist_flex_range_mismatch_investigation.md`)를 실시간으로도 잡아내기 위한 이벤트. `leader_out_of_mujoco_range`가 계속 뜨는 관절은 팔로워는 정상인데 리더 값만 이상하거나, 리더/팔로워 calibration의 range가 실제로 다르다는 뜻이다 |

`persistent_difference`/`follower_saturation_suspected`/`sign_mismatch_suspected`/
`offset_suspected`는 조건이 처음 성립하는 순간에만 이벤트를 낸다(edge-triggered) -
콘솔이 같은 문제로 도배되지 않게 하기 위함이다. 반면 `leader_out_of_mujoco_range`는
조건이 성립하는 **모든 샘플**에서 판정 결과를 남긴다 (CSV의 `event_code` 열이 매
프레임을 정확히 반영해야 하기 때문). 콘솔에는 새로 발생했을 때만 찍는다.

이 threshold들은 모두 실측이 아닌 추정치다 (아래 "알려진 제한사항" 참고).

## 7. CSV / JSON 리포트 구조

`--record`를 주면 `reports/remote_mujoco_diagnostic/session_<timestamp>.csv`에
샘플×관절 단위로 한 행씩 기록한다 (열: `local_timestamp`, `remote_timestamp`,
`sequence`, `network_latency_ms`, `state_age_ms`, `joint_name`, `leader_position_deg`,
`follower_position_deg`, `difference_deg`, `leader_raw_tick`, `follower_raw_tick`,
`mujoco_target_deg`, `mujoco_qpos_deg`, `mujoco_limit_margin_deg`, `safety_status`,
`event_code`, `blocked_reason`). `--record` 없이도 JSON 요약은 항상
`reports/remote_mujoco_diagnostic/session_<timestamp>.json`에 저장된다 (세션 통계:
샘플 수, 지연, stale/timeout 횟수, 진단 이벤트 횟수, 최종 결과 등 -
`simulation/mujoco/diagnostic_report.py:build_json_summary` 시그니처가 필드 목록의
단일 출처다). API 토큰은 어떤 리포트에도 기록되지 않는다 (저장 직전 방어적으로 제거).

## 8. 실물 실행 체크리스트 (요구사항 15번)

1. 데스크탑에서 `pytest tests/test_remote_state_client.py tests/test_diagnostic_analysis.py tests/test_diagnostic_report.py tests/test_remote_diagnostic.py` 전체 통과 확인
2. `--dry-run`으로 파이프라인 확인
3. 노트북에서 `curl http://<노트북_IP>:8001/health` 로 read_only/write_enabled=false 확인
4. `curl http://<노트북_IP>:8001/state` 로 leader/follower 정상 확인
5. 데스크탑에서 `--headless --joint wrist_flex --duration 5` 짧게 실행
6. `--headless --joint wrist_flex --duration 10`으로 조금 더 길게 실행
7. **리더암을 중앙 부근에서 아주 조금만** 움직여본다
8. 팔로워암이 전혀 움직이지 않는지 육안으로 확인한다 (이 도구는 팔로워에 아무 것도 쓰지
   않지만, 실물 확인은 항상 별도로 한다)
9. MuJoCo 화면(또는 CSV의 `mujoco_qpos_deg`)만 움직이는지 확인한다
10. CSV/JSON 리포트를 열어 값이 그럴듯한지 확인한다
11. 위 과정이 모두 문제없을 때만 `--gui`를 실행한다 (WSLg segfault 위험 - headless 우선)

**실물 팔로워암이 조금이라도 움직이면 즉시 모든 프로그램(이 진단 도구, 노트북 상태
서버)을 종료하고 안전 실패로 보고할 것.**

## 9. 알려진 제한사항

- `diagnostic` 섹션의 모든 threshold(`persistent_difference_deg`, `saturation_duration_sec`,
  `offset_*`, `sign_mismatch_*` 등)는 실물 데이터로 튜닝되지 않은 추정치다. 오탐/미탐이
  있을 수 있으니 실제 세션 CSV로 재보정이 필요하다.
- `max_joint_delta_per_frame`/`max_velocity`(기존 `configs/mujoco_so101.yaml`)는 30fps
  데이터셋 재생 기준으로 산출된 값이라, `--rate-hz`가 다르면 (특히 낮으면) 프레임간
  변화량 WARN이 실제보다 더 자주/덜 발생할 수 있다. 이 도구가 그 값을 새로 추정하지는
  않는다.
- `network_latency_ms`/`state_age_ms`는 데스크탑-노트북 간 클럭 스큐를 보정하지 않는다.
  `state_age_ms`는 노트북이 자체적으로 계산해 보낸 `age_ms`를 그대로 쓰므로 스큐에
  영향받지 않지만, `remote_timestamp`(노트북 wall clock)와 `local_timestamp`(데스크탑
  wall clock)를 직접 빼서 지연을 구하면 안 된다.
- `sequence` 정지 판정(`SequenceWatchdog`)은 노트북 서버가 폴링 스레드를 멈추지 않고
  같은 sequence를 반복 응답하는 극단적 오류만 잡는다. 서버가 아예 죽어 연결 자체가
  끊기는 경우는 timeout/connection error 경로로 별도 처리된다.
- GUI 모드는 WSLg 환경에서 segfault가 보고된 적이 있어 (`docs/mujoco_action_replay.md`
  참고) headless 검증 없이 바로 실행하지 않는다.
- 이 문서와 CLI는 SO-101 6개 관절(`shoulder_pan`, `shoulder_lift`, `elbow_flex`,
  `wrist_flex`, `wrist_roll`, `gripper`) 전제로 작성되었다. `gripper`는
  `readonly_so101_reader.py`의 관례상 각도가 아니라 0~100 정규화 값일 수 있다
  (`docs/hardware_state_server.md` 참고) - 이 도구는 그 값을 그대로 degree 단위로
  다루므로, gripper 관절의 "deg" 표시는 실제로는 열림 비율일 수 있다는 점에 유의한다.
- `[확인되지 않음]` 이 프로젝트 환경에서 실제 노트북 서버 + 데스크탑 조합으로 장시간
  (수십 분 이상) 연속 실행했을 때의 메모리/지연 추세는 검증되지 않았다.
