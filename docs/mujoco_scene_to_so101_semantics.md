# MuJoCo scene ↔ 실제 SO-101 semantics 대응 (pick&drop full-rollout benchmark용)

`reports/mujoco_full_rollout_candidate_comparison_v1`을 위해 무엇이 실물에서 검증된 것이고
무엇이 이번 benchmark를 위해 새로 만든 synthetic 근사인지 명확히 구분해 기록한다.
`docs/mujoco_action_replay.md`(1단계 안전 검증 도구)의 후속 문서이며, 그 문서의 관절
mapping/부호/단위 근거는 그대로 유지된다 — 여기서는 **이번에 새로 추가된 부분**만 다룬다.

## 1. 파일 구조

- `simulation/mujoco/assets/scene.xml`: 기존 파일, **수정하지 않음**. 로봇 body chain(관절
  이름/range/actuator/`gripperframe`/`wrist_cam` 전부 포함)만 있고 cube/bin/workspace 카메라는
  없다. 기존 모든 도구(`replay_dataset_action_mujoco.py`, `live_web_viewer.py`, Shadow Mode,
  `tests/test_mujoco_*`)는 이 파일을 계속 그대로 쓴다.
- `simulation/mujoco/assets/scene_pick_drop.xml`: 새 파일. `scene.xml`의 텍스트를 그대로 복사한
  뒤 `</worldbody>` 직전에 cube/bin/`workspace_cam`만 추가로 삽입한 것
  (`scripts/generate_mujoco_pick_drop_scene.py`가 생성, 수작업 복사 아님 — 벤더링된 291줄 XML을
  손으로 옮겨적다 생기는 오류를 피하기 위해 텍스트 치환으로 생성한다). 로봇 body chain 자체는
  1바이트도 바뀌지 않는다.

## 2. 실물에서 검증됨 / 상속됨 (바꾸지 않음)

| 항목 | 근거 |
|---|---|
| 관절 이름/순서/range, actuator ctrlrange | `so101.xml` 원본 그대로 (`docs/mujoco_action_replay.md` §9) |
| action mapping (deg→rad, scale=π/180, sign=+1 **[미확인]**) | 동일 문서 §9, 이번에도 재사용 |
| `gripperframe` site (EE 기준점) | so101.xml에 이미 존재 (`pos="0.012 -0.000218 -0.098127"`), 이번 benchmark가 새로 만든 것 아님 |
| `wrist_cam` | 실물 카메라 마운트 CAD(`wrist_roll_follower_so101_camera_mount.stl`)에서 파생된 위치. 단, sign/방향 자체는 여전히 하드웨어로 직접 검증되지 않음 (기존 캐비어트 유지) |
| Safety Gate 임계값 (`configs/safety_gate.yaml`, `configs/follower_safe_mapper.yaml`) | 이번 benchmark에서 **일절 변경하지 않음** |

## 3. 이번에 새로 추가한 것 (synthetic, 실물 대응 없음)

| 항목 | 값 | 근거/방법 | 신뢰도 |
|---|---|---|---|
| cube 크기 | 24mm 정육면체 (half-size 12mm), 질량 15g | 저장소 어디에도 실제 cube 치수가 기록되어 있지 않음(전체 grep으로 확인) — SO-101 gripper가 쥘 수 있는 합리적인 소형 큐브 크기로 근사 | 낮음 — 실측 아님 |
| cube 10개 위치 | `configs/mujoco_rollout_scenes_v1.json`의 `cube_xy` | **로봇의 실제 관절 range를 `mujoco.mj_forward`로 스윕**해 `gripperframe` site가 실제로 도달하는 점 구름(cloud)을 만들고, 그 구름에서 3.3cm 이내인 점만 채택 (`scripts/generate_mujoco_pick_drop_scene.py` 실행 로그: 25×15×15×5 grid sweep, 3475개 도달 가능 점 확보 후 밀도가 높은 중앙 구역에서 10개 선정, 전부 최근접 거리 ≤3.3cm로 검증). **임의 좌표 아님** — 하지만 실물 historical T01-T10과는 무관 | 중간 — 기구학적으로는 근거 있음, 실물 좌표는 아님 |
| bin 위치/크기 | 중심 (0.30, -0.30), 내부 10×10cm, 벽 높이 3.5cm | 동일한 FK 스윕으로 도달 가능성 확인(최근접 1.8cm) | 중간 |
| `workspace_cam` | pos `(0.55, 0, 0.45)`, 아래를 내려다보는 고정 각도 | 실물 workspace 카메라의 실제 포즈를 이 저장소에서 찾을 수 없었음(전체 grep 확인) — "정면 위쪽에서 테이블을 내려다보는" 일반적인 배치를 임의로 선택 | **낮음 — 실물 대응 없음, 아래 4절 한계로 별도 강조** |
| 로봇 초기 자세 (scene당) | `configs/mujoco_rollout_scenes_v1.json`의 `initial_pose_deg` | **실측값**: `data/so101_cube_xy_midpoint_test10_v2_clean`의 10개 episode 각각의 실제 기록된 frame-0 `observation.state` (지어낸 값 아님, 어느 episode의 몇 번째 frame인지 `initial_pose_source`에 명시) | 높음 — 실제 기록된 데이터 |
| cube/gripper 접촉 물리(마찰/조건수) | `condim=4`, `friction="1.0 0.01 0.001"` 등 MuJoCo 기본값에 가까운 보수적 설정 | 실측 grasp contact 통계가 이 저장소에 전혀 없음(`docs/mujoco_action_replay.md` §11.5와 동일한 한계) — 튜닝되지 않은 값 | 낮음 |

## 4. 알려진 한계 — 반드시 결과 해석 시 감안할 것

1. **Visual domain gap (Secondary/synthetic closed-loop track에만 해당)**: MuJoCo가 렌더링한
   이미지(텍스처/조명/기하)는 SmolVLA가 학습한 실제 카메라 영상과 근본적으로 다르다. 특히
   `workspace_cam`은 실물 대응 포즈가 없는 완전히 새로 만든 카메라다. 이 때문에 synthetic
   closed-loop rollout에서 관찰되는 policy 행동은 "실물에서 이렇게 할 것이다"를 예측하는 것이
   아니라 "MuJoCo 렌더링이라는 낯선 입력 분포에서 policy가 어떻게 반응하는가"에 가깝다 —
   그래서 이 benchmark의 주 비교는 Primary track(실제 카메라 이미지 사용)이 담당하고, synthetic
   closed-loop는 참고/exploratory로만 보고한다.
2. **Physics/contact 모델 불확실성 (Secondary track의 물리 성공 판정에만 영향)**: grasp contact
   임계값, gripper collision geometry 일부가 이름 없음(`docs/mujoco_action_replay.md` §11.4와
   동일 한계) 등으로 "물리적으로 진짜 grasp가 성공했는가" 판정은 낮은 신뢰도다. 이 때문에
   `pick_drop_eval.py`는 Track A(kinematic, contact 불필요)를 주 지표로, Track B(physics
   contact)를 부차 지표로 명확히 분리한다.
3. **cube/bin 위치는 기구학적으로 도달 가능함만 검증되었고, 실물 데이터로 검증되지 않았다.**
   Primary track에서도 이 cube/bin은 "실제 그 episode에 물체가 있던 위치"가 아니라 접근/그립/
   운반 방향을 채점하기 위한 고정 참조 zone으로만 쓰인다 — 매 리포트에 이 사실을 명시한다.
4. **로봇 초기 자세 10개가 서로 거의 동일하다.** 실측값 그대로 가져온 결과이며(3절), 원본
   데이터셋 자체가 매 episode를 고정된 teleoperation "home" 자세에서 시작했기 때문이다 — 이번
   benchmark가 다양성을 줄인 것이 아니라 원본 데이터의 실제 특성이다. Scene 간 차이는 주로 cube
   위치에서 나온다.
