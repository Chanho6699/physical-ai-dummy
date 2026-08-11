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
| 2500 | 3288 | 10 | 4.8672 | 4.1492 | 17.4509 | 34.2920 | 73 | 3180 | 35 |
| 5000 | 3288 | 10 | 4.1088 | 2.5526 | 17.5875 | 38.5081 | 712 | 2499 | 77 |
| 7500 | 3288 | 10 | 3.5877 | 2.4248 | 15.0655 | 39.2265 | 751 | 2503 | 34 |
| 10000 | 3288 | 10 | 3.5278 | 2.3498 | 15.0053 | 40.7564 | 741 | 2517 | 30 |

## Ground-truth demonstration state->action delta (checkpoint 무관, 참고용 baseline)

- median=0.7912, p95=6.6374, max=15.1702

## Joint별 action MAE

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 4.8242 | 6.1783 | 8.2451 | 4.0023 | 0.2877 | 5.6657 |
| 5000 | 3.1551 | 6.6178 | 6.8861 | 3.0312 | 0.1914 | 4.7711 |
| 7500 | 2.6765 | 5.1295 | 5.7327 | 3.2307 | 0.1798 | 4.5769 |
| 10000 | 2.6506 | 4.9127 | 5.6552 | 3.1078 | 0.1709 | 4.6695 |

## Joint별 WOULD_CLAMP count

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 1597 | 1665 | 2501 | 1393 | 4 | 1469 |
| 5000 | 824 | 1513 | 1553 | 868 | 0 | 546 |
| 7500 | 701 | 1466 | 1548 | 858 | 0 | 516 |
| 10000 | 702 | 1379 | 1648 | 874 | 0 | 609 |

## 한계 (읽는 사람이 반드시 알아야 함)

- teacher forcing 평가다: 매 프레임 policy에게 넣는 state/이미지는 데이터셋에 실제로 기록된 값이지, policy 자신이 이전 스텝에서 만든 action을 따라간 결과가 아니다. 즉 진짜 closed-loop rollout이 아니라 "기록된 궤적을 그대로 재생하면서 각 시점에서 policy라면 무엇을 예측했을까"를 보는 것이다.
- SmolVLA는 action chunk(＝chunk_size)를 한 번에 생성하고 재사용한다 - 실제로 매 프레임마다 새로 forward하지 않고, chunk가 소진될 때만(이번 설정 기준 매 50프레임마다) 새로 샘플링한다. 이는 실제 배포(`runtime/desktop/vla_server.py`)와 동일한 동작이다.
- action MAE/delta는 offline metric이다 - 실제 task success(큐브를 집어서 target에 놓았는지)를 이 수치만으로 단정할 수 없다.

