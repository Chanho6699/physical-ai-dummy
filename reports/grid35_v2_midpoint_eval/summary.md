# Grid35 SmolVLA - Held-out Midpoint10 Offline Evaluation

- eval dataset (held-out, 학습에 미사용): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_xy_midpoint_test10_v2_clean`
- train dataset (provenance 확인용, 이 평가에는 사용 안 함): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_xy_grid35_v2_clean`
- task: Pick up the cube and place it in the target area.
- seed: 42
- Safety Gate: 재사용됨 (`runtime/laptop/safety_gate.py`, write 없음)

> **중요**: 이 offline metric(action MAE/delta)은 실제 task success를 단정하지 않는다. train35(학습 데이터) 성능은 이번 checkpoint 선택 기준이 아니며, 아래는 전부 unseen midpoint10 기준이다. 이 표의 목적은 Shadow Mode에 올릴 후보 1~2개를 거르는 것이다.

## 핵심 비교 표 (unseen midpoint10 기준)

| checkpoint | frames | episodes | action MAE | delta median | delta p95 | delta max | WOULD_PASS | WOULD_CLAMP | WOULD_REJECT |
|---|---|---|---|---|---|---|---|---|---|
| 2500 | 3288 | 10 | 5.3875 | 3.7038 | 20.6367 | 50.3875 | 133 | 2893 | 262 |
| 5000 | 3288 | 10 | 4.2909 | 2.8811 | 16.7852 | 45.2020 | 500 | 2663 | 125 |
| 7500 | 3288 | 10 | 3.9841 | 2.5641 | 15.6096 | 40.4257 | 547 | 2644 | 97 |
| 10000 | 3288 | 10 | 4.2210 | 2.5022 | 17.1280 | 48.3630 | 569 | 2565 | 154 |

## Ground-truth demonstration state->action delta (checkpoint 무관, 참고용 baseline)

- median=0.7912, p95=6.6374, max=15.1702

## Joint별 action MAE

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 4.7338 | 7.6445 | 9.4915 | 4.9422 | 0.4658 | 5.0470 |
| 5000 | 3.4568 | 6.0798 | 7.1219 | 4.7001 | 0.2145 | 4.1723 |
| 7500 | 3.1811 | 5.7285 | 6.2682 | 4.1102 | 0.2121 | 4.4045 |
| 10000 | 3.0400 | 5.9863 | 7.1881 | 4.6097 | 0.2141 | 4.2878 |

## Joint별 WOULD_CLAMP count

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 1303 | 1851 | 2274 | 1402 | 98 | 635 |
| 5000 | 708 | 1561 | 1928 | 1191 | 3 | 501 |
| 7500 | 595 | 1579 | 1792 | 1024 | 0 | 621 |
| 10000 | 557 | 1514 | 1836 | 1041 | 1 | 606 |

## 한계 (읽는 사람이 반드시 알아야 함)

- teacher forcing 평가다: 매 프레임 policy에게 넣는 state/이미지는 데이터셋에 실제로 기록된 값이지, policy 자신이 이전 스텝에서 만든 action을 따라간 결과가 아니다. 즉 진짜 closed-loop rollout이 아니라 "기록된 궤적을 그대로 재생하면서 각 시점에서 policy라면 무엇을 예측했을까"를 보는 것이다.
- SmolVLA는 action chunk(＝chunk_size)를 한 번에 생성하고 재사용한다 - 실제로 매 프레임마다 새로 forward하지 않고, chunk가 소진될 때만(이번 설정 기준 매 50프레임마다) 새로 샘플링한다. 이는 실제 배포(`runtime/desktop/vla_server.py`)와 동일한 동작이다.
- action MAE/delta는 offline metric이다 - 실제 task success(큐브를 집어서 target에 놓았는지)를 이 수치만으로 단정할 수 없다.

