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
| 2500 | 3288 | 10 | 4.6534 | 3.2749 | 14.2557 | 41.3681 | 244 | 3025 | 19 |
| 5000 | 3288 | 10 | 4.4293 | 3.1157 | 18.0357 | 43.0667 | 340 | 2837 | 111 |
| 7500 | 3288 | 10 | 3.4023 | 2.4117 | 13.8089 | 38.9136 | 704 | 2577 | 7 |
| 10000 | 3288 | 10 | 3.6108 | 2.3938 | 15.4082 | 39.6204 | 664 | 2573 | 51 |

## Ground-truth demonstration state->action delta (checkpoint 무관, 참고용 baseline)

- median=0.7912, p95=6.6374, max=15.1702

## Joint별 action MAE

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 4.6781 | 7.4178 | 6.7588 | 3.4403 | 0.3345 | 5.2908 |
| 5000 | 4.0191 | 6.0260 | 7.3514 | 3.8788 | 0.2237 | 5.0767 |
| 7500 | 2.8848 | 4.6121 | 5.2596 | 2.8969 | 0.1805 | 4.5797 |
| 10000 | 2.8331 | 4.9846 | 5.8946 | 3.3657 | 0.1695 | 4.4175 |

## Joint별 WOULD_CLAMP count

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 1518 | 1863 | 1357 | 1137 | 22 | 1186 |
| 5000 | 1196 | 1549 | 1987 | 974 | 0 | 928 |
| 7500 | 647 | 1266 | 1707 | 841 | 0 | 532 |
| 10000 | 717 | 1347 | 1736 | 935 | 0 | 536 |

## 한계 (읽는 사람이 반드시 알아야 함)

- teacher forcing 평가다: 매 프레임 policy에게 넣는 state/이미지는 데이터셋에 실제로 기록된 값이지, policy 자신이 이전 스텝에서 만든 action을 따라간 결과가 아니다. 즉 진짜 closed-loop rollout이 아니라 "기록된 궤적을 그대로 재생하면서 각 시점에서 policy라면 무엇을 예측했을까"를 보는 것이다.
- SmolVLA는 action chunk(＝chunk_size)를 한 번에 생성하고 재사용한다 - 실제로 매 프레임마다 새로 forward하지 않고, chunk가 소진될 때만(이번 설정 기준 매 50프레임마다) 새로 샘플링한다. 이는 실제 배포(`runtime/desktop/vla_server.py`)와 동일한 동작이다.
- action MAE/delta는 offline metric이다 - 실제 task success(큐브를 집어서 target에 놓았는지)를 이 수치만으로 단정할 수 없다.

