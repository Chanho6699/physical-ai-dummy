# Grid35 SmolVLA - Held-out Midpoint10 Offline Evaluation

- eval dataset (held-out, 학습에 미사용): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_xy_midpoint_test10_v2_clean`
- train dataset (provenance 확인용, 이 평가에는 사용 안 함): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_pick_drop_combined65_v1`
- task: Pick up the cube and drop it into the bin.
- seed: 42
- Safety Gate: 재사용됨 (`runtime/laptop/safety_gate.py`, write 없음)

> **중요**: 이 offline metric(action MAE/delta)은 실제 task success를 단정하지 않는다. train35(학습 데이터) 성능은 이번 checkpoint 선택 기준이 아니며, 아래는 전부 unseen midpoint10 기준이다. 이 표의 목적은 Shadow Mode에 올릴 후보 1~2개를 거르는 것이다.

## 핵심 비교 표 (unseen midpoint10 기준)

| checkpoint | frames | episodes | action MAE | delta median | delta p95 | delta max | WOULD_PASS | WOULD_CLAMP | WOULD_REJECT |
|---|---|---|---|---|---|---|---|---|---|
| 2500 | 3288 | 10 | 4.7102 | 3.6116 | 16.1574 | 39.3168 | 218 | 3038 | 32 |
| 5000 | 3288 | 10 | 3.6342 | 2.5413 | 15.1207 | 37.9967 | 634 | 2645 | 9 |
| 7500 | 3288 | 10 | 3.5673 | 2.3900 | 14.8036 | 35.1661 | 722 | 2546 | 20 |
| 10000 | 3288 | 10 | 3.6793 | 2.4398 | 15.9478 | 39.7450 | 711 | 2536 | 41 |

## Ground-truth demonstration state->action delta (checkpoint 무관, 참고용 baseline)

- median=0.7912, p95=6.6374, max=15.1702

## Joint별 action MAE

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 4.5739 | 6.6862 | 7.1292 | 3.6823 | 0.2878 | 5.9015 |
| 5000 | 3.1713 | 5.4089 | 5.7888 | 2.6322 | 0.1787 | 4.6250 |
| 7500 | 2.9292 | 4.9231 | 5.7634 | 3.0025 | 0.1866 | 4.5988 |
| 10000 | 2.8312 | 5.1489 | 6.0424 | 3.2558 | 0.1724 | 4.6253 |

## Joint별 WOULD_CLAMP count

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 1357 | 1720 | 2216 | 1264 | 0 | 1049 |
| 5000 | 750 | 1522 | 1626 | 739 | 0 | 593 |
| 7500 | 721 | 1373 | 1670 | 822 | 0 | 566 |
| 10000 | 747 | 1386 | 1712 | 958 | 0 | 611 |

## 한계 (읽는 사람이 반드시 알아야 함)

- teacher forcing 평가다: 매 프레임 policy에게 넣는 state/이미지는 데이터셋에 실제로 기록된 값이지, policy 자신이 이전 스텝에서 만든 action을 따라간 결과가 아니다. 즉 진짜 closed-loop rollout이 아니라 "기록된 궤적을 그대로 재생하면서 각 시점에서 policy라면 무엇을 예측했을까"를 보는 것이다.
- SmolVLA는 action chunk(＝chunk_size)를 한 번에 생성하고 재사용한다 - 실제로 매 프레임마다 새로 forward하지 않고, chunk가 소진될 때만(이번 설정 기준 매 50프레임마다) 새로 샘플링한다. 이는 실제 배포(`runtime/desktop/vla_server.py`)와 동일한 동작이다.
- action MAE/delta는 offline metric이다 - 실제 task success(큐브를 집어서 target에 놓았는지)를 이 수치만으로 단정할 수 없다.

