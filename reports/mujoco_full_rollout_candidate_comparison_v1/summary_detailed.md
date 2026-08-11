# MuJoCo full-rollout candidate comparison - summary

- total rollouts: 120
- real_follower_write_count (전체 합): 0 (항상 0이어야 함)

**중요**: MuJoCo 환경은 실제 SO-101 환경과 동일하지 않다. 아래 physics 성공률(contact 기반)을 SmolVLA policy quality와 동일시하지 말 것 - 이 benchmark의 1차 목적은 policy가 만든 action trajectory가 운동학적/의미론적으로 올바른지 검증하는 것이다 (Track A/kinematic이 주 지표, Track B/physics는 부차 지표).

## Track별/candidate별 핵심 지표

| track | candidate | n | kinematic 성공 | physics 성공 | approach | grasp pose | lift | bin 근접 | safety reject | clamp-free | mean approach dist | mean EE-bin dist | mean jerk |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | A | 30 | 0% | 0% | 33% | 13% | 63% | 0% | 13% | 0% | 0.101m | 0.287m | 1.29deg |
| primary | B | 30 | 0% | 0% | 37% | 20% | 73% | 0% | 3% | 0% | 0.099m | 0.292m | 1.43deg |
| secondary | A | 30 | 0% | 0% | 27% | 10% | 7% | 0% | 0% | 0% | 0.099m | 0.327m | 0.88deg |
| secondary | B | 30 | 0% | 0% | 27% | 10% | 10% | 0% | 0% | 0% | 0.092m | 0.318m | 0.95deg |

## Primary track (real observation replay) - 주 비교

### Candidate A (V2+V3 reweight2:1 @10000, accuracy-oriented)

- n=30, kinematic 성공률=0%, safety reject율=13%, clamp-free율=0%
- mean approach min-dist=0.101m (참고 zone까지 - 실제 그 episode의 진짜 물체 위치가 아님, 아래 한계 참고)
- mean trajectory jerk proxy=1.29deg, mean max single-step delta=18.55deg
- seed sensitivity(within-scene std of kinematic success)=0.000
- failure reasons: {'failed_approach': 18, 'safety_reject': 4, 'missed_grasp': 4, 'wrong_direction': 4}

### Candidate B (V3+V4 uniform @10000, safety-oriented)

- n=30, kinematic 성공률=0%, safety reject율=3%, clamp-free율=0%
- mean approach min-dist=0.099m (참고 zone까지 - 실제 그 episode의 진짜 물체 위치가 아님, 아래 한계 참고)
- mean trajectory jerk proxy=1.43deg, mean max single-step delta=21.37deg
- seed sensitivity(within-scene std of kinematic success)=0.000
- failure reasons: {'failed_approach': 18, 'wrong_direction': 6, 'missed_grasp': 5, 'safety_reject': 1}

## Secondary / exploratory track (synthetic closed-loop) - 참고용

**주의**: MuJoCo가 렌더링한 이미지는 SmolVLA가 학습한 실제 카메라 영상과 다르다 (visual domain gap, `docs/mujoco_scene_to_so101_semantics.md` §4). 아래 수치는 "MuJoCo 렌더링이라는 낯선 입력에서 policy가 어떻게 반응하는가"이지 실물 성능 예측이 아니다.

### Candidate A (V2+V3 reweight2:1 @10000, accuracy-oriented)

- n=30, kinematic 성공률=0%, physics 성공률=0%, safety reject율=0%
- failure reasons: {'failed_approach': 22, 'missed_grasp': 8}

### Candidate B (V3+V4 uniform @10000, safety-oriented)

- n=30, kinematic 성공률=0%, physics 성공률=0%, safety reject율=0%
- failure reasons: {'failed_approach': 22, 'missed_grasp': 8}
