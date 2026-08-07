# SO-101 MuJoCo 데이터셋 Action Replay (1단계 안전 검증)

## 1. 목적

LeRobot 실물 데이터셋(`data/so101_cube_xy_train_v1`, `data/so101_cube_train_v6`)에 기록된
**action을 SO-101 MuJoCo 시뮬레이션에서 재생**하여, 다음 두 가지를 검증한다.

1. 데이터셋 action이 SO-101 MuJoCo 모델의 관절/actuator와 올바르게 매핑되는가 (이름, 단위,
   부호, 순서).
2. 그 action을 그대로 재생했을 때 관절 range, 속도, 충돌 등 물리적으로 위험한 상태가
   발생하지 않는가.

## 2. 현재 지원 범위 (1단계)

**포함**:
- LeRobot v3.0 형식 데이터셋의 action/state/timestamp/frame_index를 직접 파싱해 읽음
  (이미지/비디오는 읽지 않음).
- 데이터셋 action feature 이름 -> MuJoCo joint/actuator 매핑, 단위 변환(deg -> rad).
- 실행 전 정적 검사 + 재생 중 동적 안전 검사 (PASS/WARN/BLOCKED).
- headless / GUI 재생, dry-run, JSON 리포트 생성.

**포함하지 않음 (의도적으로 범위 밖)**:
- 실물 SO-101 하드웨어 제어 (USB serial 포트에 접근하지 않음).
- ROS2 연동.
- SmolVLA 등 정책 추론 연결 (`outputs/pilot/smolvla_cube_1000steps/...` 체크포인트는
  이 코드에서 로딩하지 않는다).
- 큐브/타겟 존 등 조작 대상 오브젝트의 물리 모델링 (바닥만 "table_surface"로 근사).

## 3. 설치

기존 `~/lerobot/.venv`를 그대로 사용한다. 새로 설치가 필요했던 패키지는 다음과 같다
(둘 다 실제로 설치해 동작을 확인했다).

| 패키지 | 버전 | 이유 |
|---|---|---|
| `mujoco` | 3.11.0 | SO-101 MJCF 로딩/시뮬레이션. 설치 전에는 `~/lerobot/.venv`에 없었음. |
| `pytest` | 9.1.1 | `tests/` 실행. 설치 전에는 없었음. |

```bash
source ~/lerobot/.venv/bin/activate
pip install "mujoco==3.11.0" pytest
```

`numpy`, `pandas`, `pyarrow`, `PyYAML`, `torch`는 이미 `~/lerobot/.venv`에 설치되어 있어
추가하지 않았다. `mujoco` 설치 시 함께 들어온 의존성(`absl-py`, `etils`, `glfw`, `pyopengl`,
`zipp`)도 버전 충돌 없이 설치되었다 (`pip install` 로그 상 기존 패키지와 충돌 없음).

> **참고(환경 특이사항)**: 이 시스템 셸에는 `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages`가
> 전역으로 설정되어 있다. 이 값이 있으면 `pytest`가 ROS2의 `launch_testing` pytest 플러그인을
> 자동 로딩하려다 `lark` 모듈 누락으로 즉시 크래시한다 (우리 코드와 무관한 환경 문제). 테스트
> 실행 시에는 `env -u PYTHONPATH pytest ...`처럼 해당 변수만 비우고 실행하면 된다. 이 프로젝트는
> ROS2를 전혀 사용하지 않는다.

## 4. 데이터셋 요구사항

`meta/info.json`의 `codebase_version`이 `"v3.0"`인 LeRobot 데이터셋을 대상으로 하며, 다음
파일이 있어야 한다.

```
meta/info.json
meta/episodes/chunk-*/file-*.parquet
data/chunk-*/file-*.parquet
```

action 차원은 코드에서 6으로 하드코딩하지 않고 `meta/info.json`의
`features.action.shape`/`features.action.names`를 그대로 읽는다
(`simulation/mujoco/dataset_loader.py`).

## 5. 실행

### GUI

```bash
source ~/lerobot/.venv/bin/activate
python scripts/replay_dataset_action_mujoco.py \
  --dataset-root data/so101_cube_xy_train_v1 \
  --episode-index 0 \
  --speed 1.0 \
  --gui
```

### Headless

```bash
python scripts/replay_dataset_action_mujoco.py \
  --dataset-root data/so101_cube_xy_train_v1 \
  --episode-index 0 \
  --speed 1.0 \
  --headless
```

### Dry-run (actuator에 값을 적용하지 않고 검사만)

```bash
python scripts/replay_dataset_action_mujoco.py \
  --dataset-root data/so101_cube_xy_train_v1 \
  --episode-index 0 \
  --headless --dry-run
```

### 그 외 옵션

```
--max-frames N          최대 처리 프레임 수
--start-frame N          재생 시작 프레임 index
--report-path PATH       JSON 리포트 저장 경로 (기본: reports/mujoco_replay/episode_XXX.json)
--config PATH             safety 설정 YAML (기본: configs/mujoco_so101.yaml)
--continue-on-warning     WARN이 발생해도 재생을 계속 진행 (BLOCKED는 이 옵션과 무관하게 항상 중단)
--quiet                   최종 결과와 오류만 출력
--verbose                 mapping 근거, 프레임별 상세 진단까지 출력
--no-color                ANSI 색상 끔 (색상이 없어도 내용을 이해할 수 있게 되어 있음)
```

## 6. 출력 메시지 의미

- `[준비]` / `[검사]` / `[통과]` / `[경고]` / `[차단]` / `[완료]`: 진행 단계 표시.
- `[재생 중] ████░░░░ 20.0% | Frame 1/5 | Safety WARN`: 진행률 표시줄. 기본 1초 또는 30프레임마다
  한 번만 갱신되며 매 프레임 출력하지 않는다 (`configs/mujoco_so101.yaml`의
  `console.progress_interval_*`로 조정 가능).
- `[차단] ... [프레임] ... [입력값] ... [허용 범위] ... [조치] 시뮬레이션을 중지하고 실물 실행을 금지합니다.`:
  BLOCKED 발생 시 원인을 구체적으로 보여준다.

## 7. PASS / WARN / BLOCKED 기준

| 수준 | 의미 | 재생 계속 여부 |
|---|---|---|
| PASS | 문제 없음 | 계속 |
| WARN | 주의가 필요하지만 물리적으로 즉시 위험하지는 않은 상태 | **기본값: 재생 중단** (검토를 위해). `--continue-on-warning`을 주면 계속 진행 |
| BLOCKED | 관절/actuator 제한 초과, NaN/Inf, 시뮬레이션 발산, 과도한 contact 등 | 항상 즉시 중단 (옵션으로 우회 불가) |

이 도구는 안전 검증 목적이므로 **WARN도 기본적으로 재생을 멈춘다**. `--continue-on-warning`은
"경고를 무시하고 끝까지 훑어보고 싶을 때"만 쓰는 분석용 옵션이며, BLOCKED는 이 옵션으로도
우회되지 않는다.

### 실행 전 정적 검사 (`simulation/mujoco/safety_checks.py::run_static_checks`)

- action shape 불일치, 빈 action, NaN/Inf → BLOCKED
- 관절 이름 누락/중복, actuator mapping 누락 → BLOCKED
- timestamp 역전, frame_index 불연속 → BLOCKED
- action 값이 MuJoCo 관절 range를 벗어나는 프레임이 있는지 사전 스캔 → WARN (정확한 프레임별
  판정은 재생 중 동적 검사가 담당)

### 재생 중 동적 검사 (`run_dynamic_frame_check` 계열)

- 관절 range 초과(`joint_limit`), actuator ctrlrange 초과(`actuator_limit`) → **BLOCKED**
  (모델(scene.xml)에서 직접 읽은 값과 비교. `configs/mujoco_so101.yaml`에는 중복 기재하지 않음)
- 프레임간 최대 변화량 초과(`max_delta`) → WARN
- 관절 속도 초과(`velocity_limit`) → WARN
- MuJoCo qpos/qvel NaN·Inf, 수치 발산(`simulation_nan`, `simulation_divergence`) → **BLOCKED**
- table 접촉(`table_collision`), 로봇 자기 자신 접촉(`self_collision`) → WARN
- 동시 contact 개수가 임계값을 넘는 경우(`contact_spike`) → 개수에 따라 WARN 또는 **BLOCKED**

`configs/mujoco_so101.yaml`의 `safety.stop_on_joint_limit` / `stop_on_nan` / `stop_on_collision`을
`false`로 바꾸면 해당 카테고리의 BLOCKED가 WARN으로 완화된다 (기본값은 모두 `true`).

## 8. JSON 리포트 구조

기본 저장 위치: `reports/mujoco_replay/episode_XXX.json` (이미 있으면 timestamp suffix를 붙여
덮어쓰지 않는다).

```json
{
  "dataset_root": "data/so101_cube_xy_train_v1",
  "episode_index": 0,
  "task": "Pick up the cube and place it in the target area.",
  "frame_count": 897,
  "processed_frames": 333,
  "fps": 30,
  "playback_speed": 1.0,
  "mode": "headless",
  "dry_run": false,
  "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
  "action_shape": [897, 6],
  "action_min": {"...": "..."},
  "action_max": {"...": "..."},
  "action_unit": "deg (dataset 원본 단위, mujoco 반영 시 rad로 변환)",
  "max_frame_delta": {"...": "..."},
  "max_frame_delta_unit": "rad",
  "joint_limit_violations": [{"level": "BLOCKED", "code": "joint_limit", "frame": 333, "joint": "wrist_flex", "value": 1.6632, "limit": [-1.658063, 1.658063]}],
  "actuator_limit_violations": ["..."],
  "velocity_violations": ["..."],
  "collisions": ["..."],
  "simulation_nan_count": 0,
  "warnings": ["..."],
  "blocked_reason": "wrist_flex가 허용 범위를 초과했습니다.",
  "warning_stop_reason": null,
  "final_result": "BLOCKED"
}
```

`action_min`/`action_max`는 데이터셋 원본 단위(degree)로, `max_frame_delta`는
`safety.max_joint_delta_per_frame`과 비교 가능하도록 radian으로 기록한다. 터미널에 출력한
최종 요약(`[관절 제한 위반] N회` 등)은 이 JSON의 `*_violations` 배열 길이와 항상 일치한다.

## 9. Action mapping 근거

`simulation/mujoco/action_mapping.py`에 자세히 적어 두었다. 요약:

| 항목 | 값 | 근거 |
|---|---|---|
| 이름 대응 | 데이터셋 `X.pos` == MuJoCo joint/actuator `X` | `meta/info.json`의 `features.action.names`와 `simulation/mujoco/assets/robotstudio_so101/so101.xml`의 joint/actuator 선언을 직접 대조 |
| 단위 | degree (데이터셋) → radian (MuJoCo) | `~/lerobot/src/lerobot/robots/so_follower/config_so_follower.py`의 `use_degrees=True` 기본값 + `~/lerobot/src/lerobot/motors/motors_bus.py`의 `MotorNormMode.DEGREES` 변환식 |
| scale | `pi/180` | 위와 동일 |
| offset | `0.0` | DEGREES 정규화는 캘리브레이션 중앙값을 이미 0으로 맞춤 |
| sign | `+1.0` | **[확인되지 않음]** 실물 캘리브레이션 방향과 이 MuJoCo 모델의 방향이 같다는 보장은 없음. 이름이 같으면 방향도 같다는 상식적 가정일 뿐, 하드웨어로 직접 검증하지 않았다 |

## 10. MuJoCo 모델 출처

`simulation/mujoco/assets/robotstudio_so101/`는 [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Google DeepMind)의 `robotstudio_so101` 패키지를 **수정 없이 그대로** 벤더링한 것이다. 원본은
[The Robot Studio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)의
공식 MJCF에서 파생되었다. 라이선스는 **Apache License 2.0**이며 원문을 그대로 보존했다
(`simulation/mujoco/assets/robotstudio_so101/LICENSE`). 자세한 내용과, 최상위
`simulation/mujoco/assets/scene.xml`을 별도로 작성한 이유(MuJoCo 3.11.0에서 발견된 다단계
`<include>` 경로 결합 버그 우회)는 `simulation/mujoco/assets/ATTRIBUTION.md`에 적어 두었다.

## 11. 알려진 제한사항

1. **wrist_flex 관절이 이 MuJoCo 모델의 range(±95.0도)를 실제 데이터셋에서 반복적으로
   초과한다.** `so101_cube_xy_train_v1`의 20개 에피소드를 전부 재생한 결과 16개 에피소드가
   `wrist_flex`의 `joint_limit` BLOCKED로 중단되었다 (나머지 4개는 끝까지 재생되었지만 gripper
   자기접촉 WARN이 있었다). 최대 초과폭은 약 2도(≈0.035 rad)로, 매 에피소드에서 공통적으로
   관측되어 우연한 노이즈로 보이지 않는다. **원인을 `scripts/analyze_mujoco_joint_range_mismatch.py`와
   `docs/wrist_flex_range_mismatch_investigation.md`에서 별도로 조사했다.** 가장 유력한 원인은
   캘리브레이션 자체의 미세한 오차가 아니라, **`action`(리더암 자신의 calibration 기준 값)과
   `observation.state`(팔로워암 자신의 calibration 기준 값)가 애초에 서로 다른 물리 로봇의
   서로 다른 좌표계에서 나온 값**이라는 구조적인 이유다 (`observation.state`는 20개 에피소드
   전부에서 range를 단 한 번도 벗어나지 않았다). **의도적으로 이 값을 완화하거나 threshold를
   넓혀 가리지 않았다** — 이 차이를 있는 그대로 드러내는 것이 이번 1단계 안전 검증의 목적이기
   때문이다. 실물 재생 전에는 반드시 실물 로봇의 wrist_flex 캘리브레이션과 이 MuJoCo 모델의
   range를 대조해야 한다.
2. **[확인되지 않음]** action의 부호(sign)가 실물과 정확히 일치하는지 하드웨어로 검증하지
   못했다 (9절 참고).
3. **[확인되지 않음]** `gripper` 값과 MuJoCo actuator 값의 정확한 선형성 — degree 단위라는 것은
   소스코드로 확인했지만, gripper는 특히 기구학적으로 비선형적일 수 있는 여지가 있고, 이번
   범위에서는 실물 gripper 개폐량과의 대조 실험은 하지 않았다.
4. **self_collision 판정이 모든 비-table 접촉을 뭉뚱그려 WARN으로 처리한다.** 벤더링된 MJCF의
   gripper 충돌 geometry 상당수가 `name` 속성이 없어(`so101.xml` 원본 그대로), 코드에서
   "그리퍼가 물체를 정상적으로 쥘 때 발생하는 자기 접촉"과 "진짜 위험한 자기 충돌"을 이름으로
   구분하지 못한다. 실제로 `so101_cube_xy_train_v1`을 재생하면 그리퍼가 거의 닫힌 자세일 때
   1~2개의 미세한(0.4mm 이하) contact가 항상 감지되어 WARN이 뜬다 — 물리적으로는 정상이다.
   추후 명명된 collision geom을 기준으로 "그리퍼 자체 접촉"과 "그 외 self-collision"을
   분리하는 개선이 필요하다.
5. **contact 개수 임계값(`safety.contact.*`)은 추정치다.** 이 데이터셋에는 큐브 등 조작
   대상 오브젝트가 없어 정상적인 grasp 상황의 실측 contact 통계를 얻을 수 없었다. 값은 보수적인
   기본값이며 재보정이 필요하다.
6. **속도 제한(`safety.max_velocity`)은 MJCF에 명시된 값이 아니라 이 프로젝트에서 데이터
   통계 기반으로 추가한 값이다.** 실물 STS3215 서보의 정격 속도와 대조 검증하지 않았다.
7. **물리 스텝 타이밍은 근사치다.** `fps=30`, MuJoCo `timestep=0.005`이므로 프레임당
   `round((1/30)/0.005)=7` 스텝을 밟는다. `1/30 ≈ 0.0333`초와 `7×0.005=0.035`초 사이에 약
   5% 오차가 있다 (프레임이 누적될수록 실제 경과 시간과 살짝 어긋남). action replay 자체의
   정확도(각 프레임에서 목표 위치가 맞는지)에는 영향이 없지만, 절대 시간 동기화가 중요한
   후속 작업에서는 고려해야 한다.
8. **GUI 뷰어는 이 실행 환경(WSLg)에서 프레임 재생 자체는 끝까지 성공했지만, 뷰어 종료
   시점에 segmentation fault가 발생했다** (`mujoco.viewer.launch_passive`의 GLFW 정리 코드
   추정). 리포트/최종 요약은 크래시 이전에 이미 정상적으로 저장·출력된 뒤였다. 실제 화면이
   시각적으로 올바르게 렌더링되었는지는 이 세션에서 직접 확인하지 못했다 (아래 12절 참고).
   네이티브 GPU 디스플레이 환경에서 재검증이 필요하다.

   **[후속 확인, `scripts/debug_mujoco_viewer.py`/`scripts/run_mujoco_gui_diagnostics.sh`
   조사]** 이후 세션에서 `xwd`로 GUI 창을 직접 캡처해 실제 렌더링 내용을 확인한 결과,
   위 8번 항목의 "실제 화면 렌더링 확인 못 함" 부분은 해소되었다 - `launch_passive`/
   `launch` 둘 다 창 내용이 실제로 정상 렌더링된다(스크린샷 픽셀 표준편차로 확인, 단색
   블랭크 아님). 다만 segfault 자체는 여전히 재현된다: `mujoco.viewer.launch_passive`/
   `launch`로 만든 GLFW 창을 반복 실행하면 프로세스 종료 시 약 30~50% 확률로
   `libgallium-*.so`(Mesa **llvmpipe** 소프트웨어 렌더러) 내부에서 SIGSEGV가 발생한다.
   이 머신은 `/dev/dri` render node가 없어(=WSLg GPU 그래픽 패스스루 비활성, `nvidia-smi`는
   정상 동작하므로 CUDA passthrough와는 별개 문제) OpenGL이 GLX/EGL 모두 llvmpipe로
   폴백되고 있고, 크래시는 GLFW 창 + 백그라운드 렌더 스레드가 있는 경로에서만 재현되며
   (`mujoco.Renderer` 단일 스레드 오프스크린 경로는 수십 회 반복해도 크래시 없음),
   `LIBGL_ALWAYS_SOFTWARE`/`WAYLAND_DISPLAY` 조합을 바꿔도 사라지지 않는다 - 즉 backend
   선택 문제가 아니라 llvmpipe 자체의 멀티스레드 안정성 문제로 보인다. 크래시 타이밍이
   비결정적이므로, 이 프로젝트에서 원래 보고된 "창은 뜨는데 내용이 안 보인다" 증상도 같은
   버그가 첫 프레임을 그리기 전에 발생하는 경우로 설명 가능하다(확정은 아님). 자세한
   내용/재현 절차는 `docs/remote_mujoco_diagnostic.md`의 "GUI 렌더링 문제 조사" 절 참고.
9. SmolVLA 추론은 이번 1단계에 연결하지 않았다 (요구사항대로).

## 12. 실물 실행 전 반드시 확인할 항목

- [ ] 11절의 wrist_flex 캘리브레이션 불일치 원인을 규명하고, 실물 로봇의 실제 관절 range로
      MuJoCo 모델(또는 safety 설정)을 갱신할 것.
- [ ] action 부호(sign)를 실물 로봇 1개 관절씩 저속으로 움직여 방향이 일치하는지 확인할 것.
- [ ] 이 도구가 만든 리포트에서 `final_result: "BLOCKED"`인 에피소드는 원인이 해결되기 전까지
      실물에서 재생하지 말 것.
- [ ] GUI 화면을 네이티브 디스플레이 환경에서 직접 열어 로봇 동작이 시각적으로도 타당한지
      확인할 것 (이 세션에서는 [미검증]).
- [ ] gripper 관련 self_collision WARN이 실제 grasp 상황에서도 안전한 수준인지 오브젝트를
      포함한 시나리오로 재검증할 것.
