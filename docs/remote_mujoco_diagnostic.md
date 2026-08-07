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

WSLg 환경에서는 MuJoCo GUI viewer가 segfault를 일으킬 수 있다 (알려진 제한사항, 10절 참고).
**반드시 headless로 먼저 검증한 뒤에만 GUI를 실행할 것** (15번 "실제 검증 순서" 참고).

### offscreen (GUI 창 없이 PNG/MP4로 저장 - WSLg GUI 대체 경로)

GUI 창 렌더링을 눈으로 직접 확인하기 어렵거나(원격 세션, WSLg 등) `--gui`가 불안정할 때
쓰는 경로다. 네트워크/리더암/safety gate 파이프라인은 headless와 완전히 동일하고, 차이는
"화면 대신 PNG/MP4로 저장한다"는 것뿐이다 - **실제 팔로워암에는 여전히 아무 것도 쓰지
않는다.**

```bash
python scripts/run_remote_mujoco_diagnostic.py \
  --server-url http://<노트북_IP>:8001 \
  --offscreen --joint wrist_flex --duration 10 \
  --save-frames reports/remote_mujoco_diagnostic/frames

python scripts/run_remote_mujoco_diagnostic.py \
  --server-url http://<노트북_IP>:8001 \
  --offscreen --joint wrist_flex --duration 10 \
  --video-output reports/remote_mujoco_diagnostic/wrist_flex.mp4
```

`--save-frames`/`--video-output` 중 최소 하나는 지정해야 한다 (`--offscreen`만 주면
CLI에서 바로 오류로 막는다). PNG 디렉터리에는 `frame_NNNNNN.png`와 함께 각 프레임의
`local_timestamp`/`remote_sequence`/`remote_timestamp`를 기록한 `frames_manifest.json`이
같이 저장된다. `--offscreen-width`/`--offscreen-height`(기본 640x480), `--offscreen-fps`
(기본: `--rate-hz`와 동일)로 조절할 수 있다. MP4 저장에는 `opencv-python`이, PNG 저장에는
`Pillow`가 필요하다 - 없으면 조용히 넘어가지 않고 한글 오류로 즉시 알린다.

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
| `--save-frames` | (offscreen 전용) PNG 저장 디렉터리 | 없음 |
| `--video-output` | (offscreen 전용) MP4 저장 경로 | 없음 |
| `--offscreen-width`/`--offscreen-height` | (offscreen 전용) 프레임 해상도 | 640x480 |
| `--offscreen-fps` | (offscreen 전용) MP4 fps | `--rate-hz`와 동일 |
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

## 10. GUI 렌더링 문제 조사 (WSLg) - "창은 뜨는데 내용이 안 보인다"

`--gui`(내부적으로 `mujoco.viewer.launch_passive`)가 작업표시줄에 창은 띄우지만 내용이
비어 보인다는 문제를 이 환경에서 조사한 기록이다. 조사에 쓴 도구:

- `scripts/debug_mujoco_viewer.py` - 이 저장소/네트워크와 완전히 분리된 최소 재현
  스크립트. `--mode passive|launch|render-offscreen` 세 경로를 각각 검증할 수 있다.
- `scripts/run_mujoco_gui_diagnostics.sh` - `MUJOCO_GL`/`LIBGL_ALWAYS_SOFTWARE`/
  `WAYLAND_DISPLAY` 조합을 자동으로 실행하고, `xwd`/`xwininfo`가 있으면 GUI 창을 실제로
  캡처해서 픽셀 표준편차로 "단색(블랭크)인지 실제 콘텐츠인지"까지 자동 판정한다.

**이 세션에서 확인된 것 (스크린샷으로 실제 검증됨, "본 사람이 없어 확인 못 함" 상태 해소):**

1. `launch_passive`/`launch` 둘 다, 그리고 실제 `run_remote_mujoco_diagnostic.py --gui`
   전체 경로도 **실제로 로봇 팔이 렌더링된 창 내용을 만들어낸다** (창을 `xwd`로 캡처해
   PNG로 변환 → 체크보드 바닥 위 SO-101 팔이 정상적으로 보임, 리더 관절값을 흉내낸 mock
   서버로 움직였을 때 프레임이 실제로 바뀌는 것도 확인). `viewer.sync()` 호출, main
   thread에서의 생성, `while` 루프 구조 모두 문제없이 동작한다 - **이 저장소의 GUI 루프
   코드 자체에 구조적 결함은 없었다** (3/4/5/10번 후보는 이 환경에서는 기각됨).
2. 대신, **GLFW 창을 만든 프로세스가 종료될 때 약 30~50% 확률로 SIGSEGV가 발생한다**
   (`dmesg`에 `libgallium-25.2.8-*.so`, 즉 Mesa **llvmpipe** 소프트웨어 렌더러 내부의
   general protection fault로 기록됨). 이 크래시는 `mujoco.Renderer` 기반 단일 스레드
   오프스크린 렌더링에서는 수십 회 반복해도 한 번도 재현되지 않았다 - **GLFW 창 +
   백그라운드 렌더 스레드가 있는 경로에서만** 재현된다.
3. 근본 원인은 **이 WSL2 인스턴스에 OpenGL용 GPU 패스스루가 활성화돼 있지 않다는
   것**이다: `/dev/dri/render*` 노드가 아예 없고, `glxinfo -B`/`eglinfo` 모두 `Accelerated:
   no`로 llvmpipe를 보고한다 (반면 `nvidia-smi`/CUDA는 정상 동작 - 즉 컴퓨트 패스스루는
   되지만 그래픽 패스스루는 안 되는 상태). `LIBGL_ALWAYS_SOFTWARE=1`을 껐다 켜거나
   `WAYLAND_DISPLAY`를 지워 X11을 강제해도(`scripts/run_mujoco_gui_diagnostics.sh`로
   자동 비교) 크래시 발생 여부는 바뀌지 않았다 - **backend 선택 문제가 아니라 llvmpipe
   자체의 멀티스레드 정리 코드 안정성 문제로 보인다.**
4. 크래시 시점은 비결정적이다. 이번 세션에서는 항상 "루프가 정상적으로 다 끝난 뒤,
   프로세스 종료 시점"에만 재현되어 화면 자체는 매번 정상 렌더링됐지만, 같은 버그가 더
   이른 타이밍(첫 프레임을 그리기 전)에 발생하면 "창은 떴는데 내용이 안 보인다"는 원래
   증상과 동일하게 보일 수 있다 - **다만 이건 가능성이지 확정은 아니다.**
   `[확인되지 않음]` 원래 보고된 증상이 정확히 이 버그였는지는 이 세션에서 재현하지
   못했으므로 단정할 수 없다.

**권장 조치:**

- 지금 당장 시각 확인이 필요하면 `--offscreen` + `--save-frames`/`--video-output`을 쓴다
  (위 "offscreen" 절 참고) - GUI 창 경로를 아예 타지 않으므로 이 크래시의 영향을 받지
  않는다.
- `--gui`를 계속 쓰려면, 실행이 끝난 뒤 터미널에 `Segmentation fault (core dumped)`가
  보이는 것 자체는 (지금까지 관찰된 범위에서는) 렌더링 결과나 리포트 저장에 영향을 주지
  않는다 - 리포트/최종 요약은 크래시 이전에 이미 저장·출력된다. 하지만 이 크래시가 더
  이른 타이밍에 발생할 가능성을 배제할 수 없으므로 안심 근거로 삼지 말 것.
- 근본 해결은 이 저장소 코드 밖의 영역이다: 이 WSL2 배포판에서 `/dev/dri` render node가
  생성되도록 GPU 그래픽 패스스루(Mesa D3D12/Dozen)를 활성화해야 llvmpipe를 벗어날 수
  있다. 이건 Windows/WSL 설정 문제이며, 이 조사에서는 시스템 드라이버/패키지를 설치하거나
  변경하지 않았다 (요구사항: 코드 문제로 위장하지 않는다).
- `scripts/run_mujoco_gui_diagnostics.sh`를 재실행하면 (같은 llvmpipe 상황이 유지되는 한)
  같은 패턴이 재현될 가능성이 높다 - 재검증 시 참고.

## 11. 실시간 웹 뷰어 (WSLg GUI 창 대신 브라우저로 확인)

`mujoco.viewer` 창 자체를 아예 쓰지 않고, `mujoco.Renderer`로 만든 프레임을 MJPEG로
스트리밍해 Windows Chrome에서 실시간으로 보는 경로다. 10절의 GLFW/llvmpipe 문제와 완전히
무관하다 - GLFW 창을 만들지 않으므로 그 크래시의 영향을 받지 않는다.

```bash
python scripts/run_remote_mujoco_web_viewer.py \
  --server-url http://<노트북_IP>:8001 --joint wrist_flex --host 0.0.0.0 --port 8080 --fps 20
```

```bash
python scripts/run_remote_mujoco_web_viewer.py \
  --server-url http://<노트북_IP>:8001 --all-joints --host 0.0.0.0 --port 8080 --fps 20
```

실행하면 다음이 출력된다 (WSL IP는 `wsl hostname -I`로도 확인 가능):

```text
[웹 뷰어] Windows 브라우저에서 여세요:
http://localhost:8080
[대체] localhost forwarding이 안 되면: http://<WSL_IP>:8080
```

구조 (`simulation/mujoco/live_web_viewer.py`):

```text
[network 스레드] RemoteSO101StateClient.get_state() 반복 폴링 (GET만, rate_hz)
      -> 스레드 세이프 "최신 state 1개"만 보관 (오래된 값은 자동 버려짐)
[render 스레드]  최신 state 읽기 -> action_mapping.py + safety_checks.py(기존 그대로)
      -> mj_step -> mujoco.Renderer -> JPEG 인코딩 -> "최신 프레임 1장" 발행 (fps)
[HTTP 서버]      표준 라이브러리 http.server(ThreadingHTTPServer)만 사용, 추가 의존성 없음
      /            상태표 + <img src="/stream.mjpg">가 있는 HTML
      /stream.mjpg multipart/x-mixed-replace MJPEG (Chrome이 <img>에서 네이티브 지원)
      /frame.jpg   정지 프레임 1장 (도구/테스트용)
      /status      JSON (서버 상태/sequence/지연/관절값/safety/fps/stale)
      /health      단순 liveness 체크
```

안전 동작:

- BLOCKED 관절은 `safety_checks.py`를 그대로 재사용해 직전 안전 target을 유지한다 (10절/기존
  원격 진단과 동일한 정책 - 이 파일에서 range를 새로 정의하지 않는다).
- 리더/팔로워 값이 stale하거나 네트워크가 끊기면 MuJoCo target 갱신만 멈추고(마지막 pose를
  계속 보여줌), `/status`의 `stale`/`server_status`에 반영된다.
- 서버 `mode`가 `read_only`가 아니게 되면(팀원이 실수로 teleop을 켜는 등) 그 순간부터
  **영구적으로** target 갱신을 멈춘다 (`server_status`에 "영구 정지"로 표시) - 이후 다시
  `read_only`로 돌아와도 자동으로 재개하지 않는다 (기존 `remote_diagnostic.py`의 mode
  violation 정책과 동일).
- 팔로워암에는 이 경로 전체에서 어떤 것도 쓰지 않는다 - `remote_state_client.py`는 GET만
  가지고 있다.
- 브라우저 탭을 닫아도(=HTTP 연결이 끊겨도) network/render 스레드와 HTTP 서버 자체는 계속
  동작한다 - 다시 브라우저를 열면 바로 재접속된다. 프로세스를 완전히 끝내려면 터미널에서
  Ctrl+C.

옵션: `--host`(기본 `0.0.0.0`), `--port`(기본 `8080`), `--fps`(기본 20, 권장 15~30),
`--rate-hz`(네트워크 폴링 주기, 렌더링과 무관), `--frame-width`/`--frame-height`(기본
640x480), `--jpeg-quality`(기본 80), `--debug-control`(1초마다 제어 진단 출력),
`--config`(기본 `configs/remote_mujoco_diagnostic.yaml`), `--clear-after-samples`/
`--sticky-display-sec`/`--near-limit-margin-deg`/`--sequence-stall-warn-after-s`/
`--sequence-stall-block-after-s`(11.2절 - safety 이벤트 추적/표시 설정, threshold 아님),
`--events-report-dir`(safety 이벤트 JSON/CSV 저장 위치, 기본 `reports/remote_mujoco_diagnostic`).
`--joint`/`--all-joints`는 기존 원격 진단과 같은 규칙 - 표시 대상만 제한하고 MuJoCo에는
항상 6개 관절 전체가 적용된다.

**알려진 제한사항**: 이 환경(llvmpipe 소프트웨어 렌더링, `Accelerated: no`)에서는 640x480
기준 실측 렌더 FPS가 목표(15~20)에 못 미치고 6~8fps 정도로 나오는 경우가 있었다 - 물리
정확도에는 영향 없지만(물리 스텝은 목표 fps 기준으로 계산됨) 화면이 매끄럽지 않게 보일 수
있다. `--frame-width`/`--frame-height`를 줄이면 완화된다. 실제 노트북 서버 연결로는
검증하지 않았다 (fake 서버로만 종단 테스트함, 9번 참고).

### 11.1. `--all-joints`에서 로봇이 안 움직이는 것처럼 보이는 경우

`--joint wrist_flex`는 정상 추종하는데 `--all-joints`에서는 화면 수치(leader/follower/target
등)는 바뀌는데 렌더링된 로봇이 안 움직이는 것처럼 보인다는 보고가 있어 조사했다.

**코드 감사 + 재현 테스트 결과**: `args.joints`(표시 대상 목록)는 `mapping`/물리 적용
(`mj_step`, `data.ctrl`)에 전혀 영향을 주지 않는다 - `mapping`은 `--joint`/`--all-joints`와
무관하게 항상 6개 관절 전체로 고정 생성된다. `tests/test_live_web_viewer.py`의
`test_all_joints_ctrl_qpos_and_frame_change_over_time`(6개 관절이 모두 정상 range 안에서
움직일 때)은 현재 코드에서 통과한다 - 즉 "`--all-joints`가 구조적으로 항상 멈춘다"는 재현은
되지 않았다.

대신 재현에 성공한 시나리오는: **사용자가 직접 움직이지 않는 나머지 관절들의 실물 정지
위치가 이 MuJoCo 모델의 실제 관절 range를 벗어나 있으면, 그 관절들만 (설계대로) BLOCKED되어
최초 안전 target(대개 0도)에 멈춘다.** 각 관절은 서로 독립적으로 처리되므로(한 관절 BLOCKED가
다른 관절을 막지 않음, `test_out_of_range_leader_value_is_blocked_and_previous_target_held`류
테스트로 검증됨) wrist_flex 자체는 정상적으로 계속 움직이지만, 5/6 관절이 멈춰 있으면
전체적으로 "로봇이 거의 안 움직인다"는 인상을 줄 수 있다 - 특히 이전 화면에는 어떤 관절이
왜 멈췄는지 표시가 없어서 이 상태와 진짜 정지 버그를 구분하기 어려웠다.

**이번에 한 최소 수정** (safety 완화/clamp/range 수정 없음):

1. `/status`와 웹 화면에 **Requested target**(safety gate 통과 "전" 계산값)과 **Applied
   target**(`data.ctrl`에 실제로 쓰인 값)을 분리해서 보여준다. BLOCKED된 관절은 행 배경이
   빨갛게 강조되고 "BLOCKED (이전값 유지)"로 표시된다 - 둘이 다르면 그 관절이 멈춰 있다는
   뜻이다.
2. `--debug-control` 옵션(또는 `WebViewerArgs.debug_control=True`)을 추가했다 - 1초마다
   `selected_joints`/`mapped_targets`/`blocking_joints`/`applied_targets`/`data.ctrl`/
   `data.qpos`를 stdout에 출력한다 (토큰/인증정보/원격 응답 원문은 출력하지 않음).
3. `tests/test_live_web_viewer.py`에 회귀 테스트를 추가했다: `--all-joints`에서 6개 관절의
   `data.ctrl`/`data.qpos`/렌더 프레임이 실제로 변하는지 확인하는 테스트, 그리고 한 관절만
   BLOCKED일 때 requested/applied가 갈라지고 나머지는 정상 추종하는지 확인하는 테스트.

실제 노트북 서버로 재현되면 `--debug-control`을 켜고 `blocking_joints`가 비어 있지 않은지
확인하는 것이 가장 빠른 진단 경로다. 이 조사에서 시뮬레이션한 것보다 더 광범위한 BLOCKED
(예: 6개 전부)가 재현되면, 이는 이 MuJoCo 모델의 관절 range와 실물 캘리브레이션이 크게
어긋난다는 뜻이므로 (docs/mujoco_action_replay.md의 기존 미해결 TODO) 이 웹 뷰어가 아니라
그 근본 원인(모델 range vs 실물 range)을 봐야 한다 - 이 세션에서는 range 자체를 수정하지
않았다.

### 11.2. WARN/BLOCKED 원인 추적 (safety 이벤트 로그, sticky 표시)

`--debug-control`은 "지금 이 순간"만 보여준다. 순간적으로 나타났다 사라지는 WARN/BLOCKED의
원인을 놓치지 않도록 `simulation/mujoco/safety_event_tracker.py`가 매 프레임 결과를
관찰해 기록한다 (safety 판정 자체는 그대로 `safety_checks.py`가 낸다 - 이 모듈은 판정하지
않고 분류/병합/기록만 한다).

**원인 코드**: `JOINT_RANGE_LOW`/`JOINT_RANGE_HIGH`(관절 range 초과, 어느 쪽인지는
`SafetyEvent.value`/`limit`로 확정), `FRAME_DELTA_HIGH`(프레임간 변화량 초과),
`NEAR_JOINT_LIMIT`(margin이 `near_limit_margin_deg` 미만 - **판정에 영향 없는 순수 표시용**),
`REMOTE_STALE`/`SEQUENCE_STALLED`/`MODE_NOT_READ_ONLY`/`INVALID_VALUE`/`MISSING_JOINT`(네트워크/
연결 상태), `UNKNOWN_SAFETY_REASON`(velocity_limit/self_collision/table_collision/
simulation_divergence 등 위 목록에 없는 경우 - 추측해서 새 코드를 만들지 않는다).

**Sticky 표시**: 같은 (관절, 원인 코드)가 연속되면 하나의 이벤트로 병합하고,
`clear_after_samples`(기본 3)번 연속 정상이어야 종료 처리한다. 종료된 뒤에도
`sticky_display_sec`(기본 10초) 동안 웹 화면(`/status`의 `recent_safety_events`)에 남아있다 -
한 샘플만 BLOCKED여도 놓치지 않는다. 화면에는 "현재 상태"(이번 샘플만)와 "최근 이벤트"
(sticky 목록)를 분리해서 보여준다.

**설정** (`configs/remote_mujoco_diagnostic.yaml`의 `safety_event_tracking` 섹션 -
safety threshold가 아니라 표시/기록 설정):

```yaml
safety_event_tracking:
  clear_after_samples: 3
  sticky_display_sec: 10
  near_limit_margin_deg: 5.0
```

CLI에서 `--clear-after-samples`/`--sticky-display-sec`/`--near-limit-margin-deg`로 개별
override 가능. `--config`로 다른 YAML을 지정할 수도 있다 (기본:
`configs/remote_mujoco_diagnostic.yaml`, `safety.sequence_stall_warn_after_s`/
`block_after_s`도 여기서 그대로 재사용한다 - 새로 정의하지 않음).

**리포트**: 웹 뷰어 종료(Ctrl+C) 시 `reports/remote_mujoco_diagnostic/safety_events_<timestamp>.json`
+`.csv`를 자동 저장한다. 각 이벤트에 `event_id`/`severity`/`reason_code`/`joint`/
`started_at`/`ended_at`/`duration_ms`/`sample_count`/`first_remote_sequence`/
`last_remote_sequence`/`requested_target_deg`/`applied_target_deg`/`joint_min_deg`/
`joint_max_deg`/`margin_deg`/`delta_deg`/`remote_age_ms`/`stale`가 포함된다. 인증
토큰이나 원격 응답 원문은 어디에도 담기지 않는다.

**API**: `/status`에 `current_safety`/`recent_safety_events`/`safety_event_counts`가
추가됐고, 읽기 전용 `/events`(GET만)에서 같은 내용을 바로 볼 수 있다.

## 12. Follower-safe 명령 매퍼 (`--command-source follower-safe`)

**실물 팔로워에는 여전히 어떤 것도 쓰지 않는다.** 이 모드는 "실제 팔로워에 보낼 예정인"
안전 명령을 시뮬레이션해서 MuJoCo에만 적용해 미리 검증하기 위한 것이다
(`simulation/mujoco/follower_safe_mapper.py`, 설정: `configs/follower_safe_mapper.yaml`).

```bash
python scripts/run_remote_mujoco_web_viewer.py \
  --server-url http://<노트북IP>:8001 --all-joints \
  --command-source follower-safe --safe-mapper-config configs/follower_safe_mapper.yaml \
  --host 0.0.0.0 --port 8080 --fps 10
```

기본값은 `--command-source raw-leader`(기존 동작, 리더 값을 MuJoCo safety gate만 거쳐
그대로 적용)이며 그대로 유지된다.

**단계 (A~F, 전부 분리)**: A) raw leader state → B) follower 좌표 매핑(`leader_to_follower_sign`,
기본 항등·미확인) → C) 팔로워 안전 range 확인(**clamp 아님, hold**) → D) rate limit(시간
기반, `max_step = rate_deg_per_sec × elapsed_sec`) → E) stale/sequence/mode/connection/invalid
hold(직전 안전 명령 유지) → F) 최종 명령.

**캘리브레이션**: `calibration_file_path`(노트북 실제 경로, 데스크탑엔 없을 수 있음) → 없으면
`fallback_raw_range`(요구사항에 명시된 실제 팔로워 값) → 둘 다 없으면 실행 중단. raw tick ->
degree 변환은 LeRobot `MotorNormMode.DEGREES` 공식(`mid=(min+max)/2`,
`max_res=motor_resolution-1`, `deg=(raw-mid)*360/max_res`)을 그대로 재사용했다
(`~/lerobot/src/lerobot/motors/motors_bus.py` 조사 결과) - `homing_offset`은 서보 펌웨어에
이미 반영돼 있어 소프트웨어에서 다시 빼지 않는다. **gripper는 이 프로젝트 전체에서
`RANGE_0_100`(퍼센트) 정규화를 쓰기 때문에(도가 아님) `UNVERIFIED_RANGE`로 표시하고 실제
출력 후보에서 제외한다** - 추측으로 단위를 맞추지 않았다.

**rate limit 초기값**(`configs/follower_safe_mapper.yaml`) - **미검증, 보수적인 임시값**,
실물 테스트 후 반드시 재조정 필요:

```yaml
rate_limit_deg_per_sec:
  shoulder_pan: 20
  shoulder_lift: 15
  elbow_flex: 20
  wrist_flex: 15
  wrist_roll: 25
  gripper: 30
```

**초기화**: 첫 유효 snapshot에서 관절별로 (1) 팔로워 현재 위치(read-only 서버의
`follower.positions_deg`)를 우선 쓰고, (2) 없으면 첫 리더 값을 그대로 써서(점프할 "이전
값"이 없는 프레임 0이라 점프가 아님) 시작한다. 이후에는 항상 rate-limited 수렴만 한다.

**hold 정책**: `REMOTE_STALE`/`SEQUENCE_STALLED`/`INVALID_LEADER_STATE`/`MISSING_JOINT`/
`MODE_NOT_READ_ONLY`/`CONNECTION_LOST`/`INVALID_VALUE`/`RANGE_VIOLATION`/`UNVERIFIED_RANGE`
중 하나면 직전 안전 명령을 그대로 유지한다. 복구되면 마지막 명령에서부터 rate limit을 통해
다시 수렴한다(즉시 점프 없음).

**웹 화면**: 상단에 `Command source`/`Hold`/`Hold reason`/`Intervention count`, 관절별 표에
`Leader raw`/`Follower current`/`Requested(mapped)`/`Applied(limited)`/`MuJoCo actual`/
`margin`/`safety`(rate-limited·hold 사유 포함)를 표시한다.

**리포트**: 종료 시 `reports/remote_mujoco_diagnostic/follower_safe_mapper_<timestamp>.json`
+`.csv`를 저장한다 - 요구사항에 명시된 샘플별 14개 필드와 요약(`total_samples`,
`rate_limit_interventions`, `range_holds`, `stale_holds`, `sequence_holds`, `invalid_holds`,
`max_raw_delta_by_joint`, `max_limited_delta_by_joint`)을 그대로 포함한다.

### 12.1. "gripper가 hold되면 팔 5개 관절도 안 움직인다" 조사

실물 노트북 연결로 `--command-source follower-safe`를 실행했을 때 gripper가 처음부터
`UNVERIFIED_RANGE`(설계대로 - 11절 참고)로 hold였고, 동시에 팔 5개 관절도 MuJoCo에서 안
움직였다는 보고가 있어 "gripper의 hold가 전역 hold로 승격된다"는 가설을 조사했다.

**코드 감사 결과**: `FollowerSafeMapper._step_joint()`는 관절마다 완전히 독립적으로
실행된다 - `self._calibrations[joint]`/`self._initialized[joint]`만 참조하고, 관절 간에
공유되는 mutable 상태가 없다. `live_web_viewer.py`의 `data.ctrl` 적용 루프도 `mapping`의
각 엔트리(관절)마다 자신의 `sample.limited_command_deg`만 써서 개별적으로 판단한다.
**구조적으로 한 관절의 hold가 다른 관절에 전파될 방법이 없다** - `gripper의 hold_reason이
"UNVERIFIED_RANGE"인 것과, 다른 관절의 hold_reason이 무엇인지는 서로 완전히 독립이다.

**재현 시도**: gripper를 항상 45(퍼센트, 영구 UNVERIFIED_RANGE)로 고정하고 나머지 5개
관절을 실제로 계속 움직이는 fake 서버로 종단 테스트한 결과, 5개 관절은 gripper와 무관하게
정상적으로(target/qpos 모두) 계속 갱신됐다 - `tests/test_live_web_viewer.py`의
`test_gripper_unverified_range_isolated_arm_joints_actively_update`로 고정 회귀 테스트화함.

**가장 유력한 실제 설명**: gripper의 (설계상 항상 발생하는) `UNVERIFIED_RANGE` hold와,
팔 5개 관절에 영향을 준 **별개의 전역 hold**(`REMOTE_STALE`/`SEQUENCE_STALLED` 등, 예를 들어
리더를 움직이지 않고 화면만 보고 있으면 sequence가 정지한 것처럼 보일 수 있음)가 우연히
동시에 관측되어, 마치 gripper가 원인인 것처럼 보였을 가능성이 가장 높다. 기존에는 "Hold"
필드 하나만 있어 이 둘을 구분해서 볼 방법이 없었다 - 이번에 고친 것은 이 **관측성 부족**이다
(아래).

**수정한 것 (판정 로직 자체는 바꾸지 않음 - 이미 올바르게 분리되어 있었음)**:

1. `simulation/mujoco/follower_safe_mapper.py`에 `ARM_WIDE_HOLD_REASONS`(전역 hold 4가지:
   `REMOTE_STALE`/`SEQUENCE_STALLED`/`MODE_NOT_READ_ONLY`/`CONNECTION_LOST`)와
   `summarize_hold()`를 추가해 "global_hold는 이 4가지 중 하나가 있을 때만 true"임을
   명시적인 코드로 고정했다.
2. `/status`(및 `--debug-control` 출력)에 `global_hold`/`global_hold_reason`/
   `active_joint_count`/`held_joint_count`/`held_joints`(관절→사유 dict)를 추가해, 다음에
   같은 상황이 재현되면 즉시 "이건 전역 문제인지 관절 하나만의 문제인지"를 구분할 수 있게
   했다.
3. 웹 화면 상단을 "Global Hold"로 명확히 하고, 관절별 hold 목록과
   "전체 제어 정지 아님 - N개 관절 격리됨, M개 관절 활성" 안내를 추가했다.
4. `--debug-control`을 follower-safe 모드에서 켜면 1초마다 관절별
   `raw_leader`/`mapped_target`/`limited_command`/`joint_hold`/`joint_hold_reason`과
   `global_hold`/`global_hold_reason`/`held_joints`를 stdout에 출력한다.

실물에서 다시 이 증상이 재현되면 `--debug-control`을 켜고 각 관절의 `joint_hold_reason`을
직접 확인하는 것이 가장 빠른 진단 경로다 - `held_joints`가 `{"gripper": "UNVERIFIED_RANGE"}`
하나뿐인데도 팔이 안 움직인다면 이 저장소의 코드 문제이므로 재조사가 필요하지만, 다른
관절도 같이 들어 있다면(특히 `REMOTE_STALE`/`SEQUENCE_STALLED`) 그게 진짜 원인이다.
