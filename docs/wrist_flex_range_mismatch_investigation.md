# wrist_flex 관절 range 불일치 원인 조사

> 이 문서는 조사 결과만 담는다. MJCF나 `configs/mujoco_so101.yaml`의 값은 이 조사 과정에서
> 변경하지 않았고, `dataset_action_replay.py`의 재생/판정 동작도 그대로다.

## 1. 현상

`so101_cube_xy_train_v1` 20개 episode 중 16개가 wrist_flex `joint_limit` BLOCKED로
중단되었고 (`docs/mujoco_action_replay.md` 참고), `so101_cube_train_v6` 5개 중 1개도
같은 이유로 중단되었다. 예: episode 0, frame 333, action 1.6632 rad(=95.31°)가
MuJoCo range `[-1.658063, 1.658063]`(=±95.0°)를 벗어남.

## 2. 데이터셋 전체 통계 (`scripts/analyze_mujoco_joint_range_mismatch.py`로 산출)

### so101_cube_xy_train_v1 (20 episodes, 17,935 프레임)

| 항목 | 값 |
|---|---|
| action 최솟값 / 최댓값 | 37.36° / **97.14°** |
| observation.state 최솟값 / 최댓값 | 38.37° / **84.44°** |
| MuJoCo wrist_flex range | -95.00° ~ 95.00° |
| range 초과 episode | **16 / 20** |
| range 초과 프레임 | 7,387 / 17,935 (41.2%) |
| 최대 초과량 | 2.14° (action 기준) |
| 평균/중앙값 초과량 | 1.58° / 1.79° |
| state 기준 초과 프레임 | **0** |

### so101_cube_train_v6 (5 episodes, 4,484 프레임)

| 항목 | 값 |
|---|---|
| action 최솟값 / 최댓값 | (episode별로 다름) / **96.88°** |
| observation.state 최댓값 | **84.35°** |
| range 초과 episode | 1 / 5 |
| state 기준 초과 프레임 | **0** |

**핵심 패턴**: 두 데이터셋 모두에서 `observation.state`(팔로워 실측값)는 wrist_flex가
84.35~84.44° 이상으로 올라간 적이 **단 한 프레임도 없다** (episode마다 최댓값이 84.35xx
또는 84.44xx로 소수점 넷째 자리까지 거의 동일 — 반복적으로 같은 지점에서 멈춘다는 뜻).
반면 `action`(리더암 명령값)은 여러 episode에서 95° 이상까지 올라가고, 한 번 넘으면
**episode 끝까지 그 상태로 지속**된다 (예: episode 0의 초과 구간은 `[333, 896]`, 즉 897
프레임짜리 episode의 마지막 프레임까지 연속). 이는 짧은 노이즈성 튐이 아니라 리더암이
자기 자신의 한쪽 끝까지 밀어붙인 뒤 거기 머무는 패턴이다.

action과 state는 episode별로 강한 양의 상관관계(평균 상관계수 ≈0.97~0.98)를 보여, 두
신호가 반대 방향으로 움직이는 부호 오류의 흔적은 없다.

## 3. MuJoCo MJCF 조사

- `simulation/mujoco/assets/robotstudio_so101/so101.xml`의 wrist_flex:
  `<joint axis="0 0 1" name="wrist_flex" type="hinge" range="-1.658063 1.658063" class="sts3215"/>`
  (주석: "5-degree calibration offset applied to joint range" — 이건 **elbow_flex**에 달린
  주석이며 wrist_flex에는 없다. wrist_flex range는 어떤 오프셋 계산 없이 ±1.658063 rad로
  선언되어 있다.)
- actuator ctrlrange: `-1.65806 1.65806` — joint range와 사실상 동일(소수점 반올림 차이만 있음).
- 초기 qpos: MJCF에 wrist_flex용 `<key>`(keyframe)가 없어 MuJoCo 기본값 0으로 시작한다.
  (재생 시에는 `dataset_action_replay.py`가 이를 `observation.state[0]`으로 덮어써서 시작하므로
  이번 조사와는 무관하다.)
- **벤더링된 사본과 원본이 같은지**: `diff -rq`로 `simulation/mujoco/assets/robotstudio_so101`와
  시스템에 이미 있던 `physical-ai-recycling-cell/third_party/mujoco_menagerie/robotstudio_so101`를
  비교한 결과 **완전히 동일**(바이트 단위)하다. 즉 이번 프로젝트에서 MJCF를 수정한 적이 없다.
- range 값 자체(±95.0°)는 아주 깔끔하게 대칭인 "설계 스펙"처럼 보인다 (`robotstudio_so101`
  패키지는 TheRobotStudio의 공식 CAD 기반 MJCF에서 파생된 것으로, 실측이 아니라 CAD 상의
  기구학적 한계로 짐작된다 — 이 정확한 유래는 `mujoco_menagerie`의 커밋 로그에서 명시적으로
  확인하지 못했다. [확인 불가]).

## 4. LeRobot SO-101 구현 조사

### 4.1 `use_degrees=True`, `MotorNormMode.DEGREES`

`SOFollowerConfig.use_degrees` 기본값은 `True`
(`~/lerobot/src/lerobot/robots/so_follower/config_so_follower.py:42`), 이 값이 `True`이면
`MotorNormMode.DEGREES`로 정규화한다 (`so_follower.py:50`). `MotorsBus._normalize`/`_unnormalize`
(`~/lerobot/src/lerobot/motors/motors_bus.py:855-909`)의 DEGREES 분기:

```python
# _normalize (raw ticks -> degree, 읽기용)
bounded_val = min(max_, max(min_, val))   # raw ticks를 calibration [min_, max_]로 clamp
mid = (min_ + max_) / 2
normalized = (bounded_val - mid) * 360 / (motor_resolution - 1)

# _unnormalize (degree -> raw ticks, 쓰기용/명령용)
mid = (min_ + max_) / 2
unnormalized = int(val * (motor_resolution - 1) / 360 + mid)   # <- clamp 없음!
```

**중요한 비대칭**: 읽을 때(`_normalize`)는 raw 값을 calibration 범위로 clamp한 뒤 degree로
바꾸지만, 쓸 때(`_unnormalize`, 명령을 보낼 때 사용)는 **clamp를 하지 않는다**. `RANGE_M100_100`/
`RANGE_0_100` 모드는 `_unnormalize`에서도 `min(100, max(-100, val))`로 clamp하는데, `DEGREES`
모드만 이 clamp가 없다. 이는 이 프로젝트만의 버그가 아니라 **LeRobot 0.6.2 자체의 동작**이다.

### 4.2 calibration이 action 값에 반영되는 방식

`mid = (range_min + range_max) / 2`가 "0도"의 기준이 된다. `range_min`/`range_max`는
로봇별 calibration 절차(각 관절을 손으로 끝까지 움직여 raw tick 최소/최대값을 기록)로
정해지는 **그 개별 로봇 유닛 고유의 값**이다. 즉 "0도"는 CAD상의 기구학적 중립 자세가
아니라 "이 특정 팔이 실제로 도달 가능한 두 끝점의 중간"으로 정의된다.

### 4.3 leader/follower의 관계 — **가장 중요한 발견**

`~/lerobot/src/lerobot/scripts/lerobot_record.py`의 녹화 루프 (271~336행):

```python
obs = robot.get_observation()                  # <- 팔로워에서 읽음
...
act = teleop.get_action()                       # <- 리더에서 읽음 (SOLeader.get_action)
act_processed_teleop = teleop_action_processor((act, obs))
action_values = act_processed_teleop             # <- 이 값이 데이터셋에 "action"으로 기록됨
robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
...
_sent_action = robot.send_action(robot_action_to_send)   # <- 팔로워에 실제 전송
...
action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
```

- **`observation.state` = 팔로워(follower) 자신의 calibration으로 정규화한, 팔로워가 실제로
  도달한 위치.**
- **`action` = 리더(leader) 자신의 calibration으로 정규화한, 리더 오퍼레이터가 움직인 위치**
  (`teleop.get_action()` → `SOLeader`가 자기 자신의 calibration으로 degree 변환). `teleop_action_processor`/
  `robot_action_processor`는 기본값이 `make_default_processors()`가 반환하는
  `RobotProcessorPipeline(steps=[IdentityProcessorStep()])`뿐이다
  (`~/lerobot/src/lerobot/processor/factory.py:46-65`) — 즉 리더 값이 **그 어떤 변환도 없이**
  그대로 로깅되고 그대로 팔로워에 전송된다.
- 이 프로젝트의 `data_collection/recorder.py`도 커스텀 processor나
  `SOFollowerConfig.max_relative_target`을 지정하지 않는다 (`grep` 결과 두 스크립트 어디에도
  없음) — 즉 소프트웨어 단의 clipping이 전혀 없다.

**결론**: `action`과 `observation.state`는 **서로 다른 물리적 로봇(리더 vs 팔로워)의, 서로
다른 calibration 기준 좌표계**에서 나온 값이다. 이름이 같다고 해서 두 값이 같은 range를
가질 이유가 전혀 없다.

### 4.4 실물 모터의 min/max 저장 위치, 현재 calibration 정보 존재 여부

LeRobot은 calibration을 `HF_LEROBOT_CALIBRATION`(기본
`~/.cache/huggingface/lerobot/calibration/{robots,teleoperators}/<type>/<id>.json`)에
JSON으로 저장한다 (`~/lerobot/src/lerobot/{robots/robot.py, teleoperators/teleoperator.py}`).
이 시스템에서 확인한 결과:

```
find ~/.cache/huggingface/lerobot -maxdepth 5   ->  (없음, 디렉터리 자체가 없음)
find ~ -iname "*calibration*"                    ->  이 프로젝트/데이터셋과 무관한 파일들뿐
```

**`chanho_leader`/`chanho_follower`(`configs/hardware.example.json`에 적힌 id)의 calibration
JSON은 이 머신에 없다.** 즉 실제 녹화에 쓰인 정확한 `range_min`/`range_max`/모터 해상도 값은
**[확인 불가]** — 이번 조사는 그 값들을 직접 읽지 못하고, action/state 통계로부터 간접적으로
결론을 낸 것이다.

## 5. 가설별 판정

| 가설 | 판정 | 핵심 근거 |
|---|---|---|
| MuJoCo joint range가 실제 팔로워 하드웨어보다 좁음 | **가능성 낮음** | `observation.state`(팔로워 실측)는 두 데이터셋 전부에서 range 초과가 0건. 팔로워 실측 기준으로는 MuJoCo range가 좁다는 증거가 없다. |
| dataset와 MuJoCo의 zero offset 불일치 | **가능성 낮음** | 초과가 항상 high-side(최댓값 쪽)에서만 발생하고, 한 번 초과하면 episode 끝까지 지속되는 "포화(saturation)" 패턴이다. 균일한 zero offset 오류라면 최솟값 쪽에서도 대칭적으로 나타나야 하는데 그렇지 않다. |
| joint direction/sign 불일치 | **가능성 낮음** | action-state episode별 상관계수 평균 0.97~0.98로 강한 양의 상관관계. 부호가 반대라면 음의 상관관계가 나타나야 한다. |
| **leader/follower calibration 차이** | **확인됨** | 소스 코드로 직접 추적: action=리더 자신의 calibration 기준값, state=팔로워 자신의 calibration 기준값. 서로 다른 물리 로봇의 서로 다른 calibration이므로 수치가 다를 근거가 명확하다 (4.3절). |
| 데이터 수집 시 실제 안전 범위 초과 (소프트웨어 clipping 부재) | **가능성 높음** | `IdentityProcessorStep` 기본값 + `max_relative_target` 미설정 + `MotorNormMode.DEGREES`의 `_unnormalize`에 clamp가 없음(4.1절) → 리더 값이 팔로워 안전범위를 넘어도 소프트웨어가 막지 않고 그대로 전송됐을 것이다. (실제 서보에 도달했는지, 서보 자체 하드웨어 리밋에 막혔는지는 [확인 불가]) |
| state/action 표현 차이 | **확인됨** | leader/follower calibration 차이의 필연적 결과. 16/20 episode가 action 기준으로는 초과지만 state 기준으로는 단 1건도 초과가 없다는 사실 자체가 두 신호가 다른 실체를 가리킨다는 직접 증거. |
| (추가) 리더/팔로워 실제 calibration range_min/range_max 수치 | **확인 불가** | 이 머신에 calibration JSON이 없어 정확한 수치를 확인하지 못함 (4.4절). |

**가장 가능성 높은 원인**: `action`(리더암 값)과 `observation.state`(팔로워암 값)가 원래
서로 다른 calibration 기준의 값이며, 이 replay 도구는 `action`을 팔로워 모양의 MuJoCo
모델에 그대로 명령으로 흘려보낸다. 리더암이 (팔로워보다 넓은 실제 가동범위를 갖고 있거나,
단순히 오퍼레이터가 리더를 더 멀리 밀어서) 95°를 넘는 값을 종종 만들어내고, 소프트웨어
clipping이 없어 그 값이 그대로 팔로워에 명령으로 전송되고 데이터셋에도 그대로 남는다.
실제 팔로워는 (아마도 물리적 하드스톱이나 서보 자체 리밋에 의해) 84.4° 부근에서 멈췄다.

## 6. 아직 확인되지 않은 부분

1. `chanho_leader`/`chanho_follower`의 실제 calibration JSON (range_min/range_max, 모터 모델별
   해상도) — 이 머신에 없음.
2. 팔로워가 84.4° 부근에서 멈추는 **물리적** 원인 — 손목 조립체의 기구학적 하드스톱인지,
   와이어/카메라 마운트 간섭인지, STS3215 서보 자체의 EEPROM 각도 제한인지는 실물을 봐야 안다.
3. 리더암이 실제로 95°를 초과하는 물리적 자세까지 갈 수 있는지 (리더 쪽 카메라 마운트가 없어
   더 넓게 움직일 수 있다는 것은 추정이며 실측하지 않았다).
4. `robot_action_to_send`(팔로워에 실제로 전송된 값)이 데이터셋에 저장되지 않으므로, 서보에
   실제로 어떤 raw tick 명령이 갔는지는 로그만으로 재구성할 수 없다.

## 7. 수정안 비교

| 수정안 | 장점 | 위험 | 필요한 근거 | 실물 안전성 영향 |
|---|---|---|---|---|
| **mapping에 offset 적용** (action에 고정 offset을 빼서 range 안에 맞춤) | 구현이 간단하고 BLOCKED가 줄어듦 | **위험**: 실제로는 offset 문제가 아니라 리더/팔로워가 다른 로봇이라는 구조적 문제이므로, offset을 적용하면 "리더 값을 억지로 팔로워 range 안으로 욱여넣는" 것이 되어 오히려 실제 팔로워 동작과 더 멀어질 수 있다. 또한 초과량이 episode마다 다르므로(1.58~2.14°) 고정 offset 하나로는 모든 episode를 정확히 설명하지 못한다. | 리더/팔로워 실제 calibration 값 확인 후, "리더→팔로워 좌표 변환"을 제대로 유도해야 함 (단순 뺄셈이 아닐 수 있음) | **부정적** — 근거 없이 값을 조작해 실제로는 안전하지 않은 명령을 "안전한 것처럼" 보이게 할 수 있음. **채택하지 않음.** |
| **MJCF joint range 수정** (±95°를 더 넓게) | wrist_flex BLOCKED가 사라짐 | **위험**: MJCF range가 좁다는 근거가 오히려 "가능성 낮음"으로 나왔다 (state는 84.4° 이상 간 적이 없음). Range를 넓히면 팔로워가 실제로는 도달 못 하거나 손상 위험이 있는 각도까지 시뮬레이션이 "정상"으로 통과시키게 됨 | 팔로워의 **실측** 기구학적 한계 (실물 측정 또는 제조사 스펙) | **부정적** — 안전 마진을 근거 없이 없애는 것과 같음. 지시사항에서도 명시적으로 금지. **채택하지 않음.** |
| **actuator ctrlrange 수정** | 위와 동일 | 위와 동일 | 위와 동일 | 위와 동일. **채택하지 않음.** |
| **safety tolerance 추가** (예: ±2~3° 여유) | 구현 간단, 지금 당장 16/20 episode가 통과하게 됨 | **위험**: tolerance는 "부동소수점 오차"를 흡수하기 위한 것이지, 지금처럼 "리더가 팔로워 range를 실제로 넘어서는 값을 명령했다"는 사실 자체를 가리는 데 쓰면 안 됨. 이번 조사로 밝혀진 진짜 원인(리더/팔로워 불일치)을 숨기는 결과가 됨 | 없음 — 오히려 이 조사 결과가 tolerance를 넓히지 말아야 한다는 근거임 | **부정적**. **채택하지 않음** (`configs/mujoco_so101.yaml`의 `joint_limit_tolerance_rad=0.005`는 그대로 유지). |
| **해당 데이터 재수집** (리더 range를 팔로워 range 안으로 제한하거나, `max_relative_target` 설정 후 재녹화) | 근본 원인을 데이터 소스에서 해결. 가장 "정직한" 해결책 | 시간/자원 비용. 실물 로봇 접근 필요 | 리더/팔로워 calibration 재확인, 녹화 파이프라인에 clipping 추가 | **중립~긍정적** — 재수집 자체는 안전에 영향 없음. 장기적으로 가장 신뢰할 수 있는 해결책. |
| **해당 frame/episode 제외** (BLOCKED episode를 학습/재생 대상에서 제외하거나, 초과 구간 이후를 잘라냄) | 지금 당장 실행 가능, 원인 데이터를 왜곡하지 않음 | 데이터 손실 (16/20 episode가 영향받아 학습 데이터가 크게 줄어듦). 초과 이후 구간을 자르면 grasp/place 등 과업 후반부가 통째로 사라질 수 있음 | 없음 (이미 갖고 있는 정보로 즉시 적용 가능) | **긍정적** — 실물 재생/학습에서 위험 구간을 원천적으로 제외하므로 가장 보수적이고 안전. |

**권장**: 단기적으로는 "해당 frame/episode 제외"(또는 최소한 BLOCKED로 표시된 episode를 실물
재생 대상에서 제외)를 유지하고, 중기적으로는 "해당 데이터 재수집"(리더/팔로워 calibration을
재점검하고 `max_relative_target`을 설정한 뒤 재녹화)을 진행하는 것을 권장한다. MJCF/tolerance를
건드리는 방안은 모두 위험이 더 크므로 권장하지 않는다.

## 8. 실물에서 직접 확인해야 할 사항

- [ ] `chanho_leader`, `chanho_follower` calibration JSON을 실제 녹화에 쓰인 머신에서 찾아
      `range_min`/`range_max`/모터 모델을 확인할 것.
- [ ] 팔로워 wrist_flex를 손으로 천천히 돌려 84.4° 부근에서 실제로 하드스톱이 있는지, 있다면
      그 원인(조립체 간섭·카메라 마운트·서보 EEPROM 리밋)을 확인할 것.
- [ ] 리더 wrist_flex가 실제로 95° 이상 물리적으로 움직이는지 확인할 것.
- [ ] `lerobot_record.py`를 다시 실행할 계획이 있다면 `SOFollowerConfig.max_relative_target`을
      설정해 리더 값이 팔로워에 그대로 꽂히지 않도록 할 것.
