# Grid35 SmolVLA - Held-out Midpoint10 Offline Evaluation

- eval dataset (held-out, 학습에 미사용): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_pick_drop_v4_heldout10`
- train dataset (provenance 확인용, 이 평가에는 사용 안 함): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_pick_drop_v3_v4_combined69_v1`
- task: Pick up the cube and drop it into the bin.
- seed: 42
- Safety Gate: 재사용됨 (`runtime/laptop/safety_gate.py`, write 없음)

> **중요**: 이 offline metric(action MAE/delta)은 실제 task success를 단정하지 않는다. train35(학습 데이터) 성능은 이번 checkpoint 선택 기준이 아니며, 아래는 전부 unseen midpoint10 기준이다. 이 표의 목적은 Shadow Mode에 올릴 후보 1~2개를 거르는 것이다.

## 핵심 비교 표 (unseen midpoint10 기준)

| checkpoint | frames | episodes | action MAE | delta median | delta p95 | delta max | WOULD_PASS | WOULD_CLAMP | WOULD_REJECT |
|---|---|---|---|---|---|---|---|---|---|
| 2500 | 1968 | 6 | 5.6533 | 3.4570 | 17.4162 | 41.8549 | 93 | 1621 | 254 |
| 5000 | 1968 | 6 | 4.4145 | 2.5236 | 16.3760 | 51.2886 | 193 | 1609 | 166 |
| 7500 | 1968 | 6 | 3.5897 | 1.9901 | 12.3104 | 34.4732 | 504 | 1357 | 107 |
| 10000 | 1968 | 6 | 3.6092 | 2.0389 | 13.0263 | 35.3180 | 427 | 1397 | 144 |

## Ground-truth demonstration state->action delta (checkpoint 무관, 참고용 baseline)

- median=0.7912, p95=5.8901, max=10.8942

## Joint별 action MAE

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 5.6003 | 8.9524 | 9.2056 | 3.1161 | 2.1220 | 4.9233 |
| 5000 | 3.9471 | 6.7374 | 7.6118 | 2.7868 | 1.3365 | 4.0673 |
| 7500 | 3.2314 | 5.7327 | 5.2916 | 2.1527 | 1.2053 | 3.9244 |
| 10000 | 3.1029 | 5.7178 | 5.4194 | 2.1572 | 1.2487 | 4.0091 |

## Joint별 WOULD_CLAMP count

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 1014 | 1093 | 1040 | 525 | 772 | 453 |
| 5000 | 642 | 852 | 1086 | 445 | 560 | 295 |
| 7500 | 424 | 630 | 749 | 316 | 386 | 292 |
| 10000 | 425 | 647 | 865 | 322 | 379 | 272 |

## 한계 (읽는 사람이 반드시 알아야 함)

- teacher forcing 평가다: 매 프레임 policy에게 넣는 state/이미지는 데이터셋에 실제로 기록된 값이지, policy 자신이 이전 스텝에서 만든 action을 따라간 결과가 아니다. 즉 진짜 closed-loop rollout이 아니라 "기록된 궤적을 그대로 재생하면서 각 시점에서 policy라면 무엇을 예측했을까"를 보는 것이다.
- SmolVLA는 action chunk(＝chunk_size)를 한 번에 생성하고 재사용한다 - 실제로 매 프레임마다 새로 forward하지 않고, chunk가 소진될 때만(이번 설정 기준 매 50프레임마다) 새로 샘플링한다. 이는 실제 배포(`runtime/desktop/vla_server.py`)와 동일한 동작이다.
- action MAE/delta는 offline metric이다 - 실제 task success(큐브를 집어서 target에 놓았는지)를 이 수치만으로 단정할 수 없다.

