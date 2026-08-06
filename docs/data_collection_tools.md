# SO-101 데이터 수집 도구

## 역할

- `scripts/record_episodes.py`: 장치 검사 후 공식 `lerobot-record` 실행
- `scripts/validate_dataset.py`: LeRobot 데이터셋 구조·프레임·state/action·영상 자동 검증
- `configs/hardware.local.json`: 이 노트북의 로봇팔/카메라 장치 경로

LeRobot이 실제 동기화와 저장을 담당합니다. 프로젝트 코드는 이를 다시 구현하지 않습니다.

## 녹화 전 명령 검사

```bash
source ~/lerobot/lerobot/bin/activate
cd ~/Projects/physical-ai-dummy

python scripts/record_episodes.py \
  --dataset-name so101_cube_train_v3 \
  --episodes 5 \
  --episode-seconds 30 \
  --reset-seconds 20 \
  --dry-run
```

## 실제 녹화

```bash
python scripts/record_episodes.py \
  --dataset-name so101_cube_train_v3 \
  --episodes 5 \
  --episode-seconds 30 \
  --reset-seconds 20
```

프로그램은 한국어로 전체 녹화 규칙을 표시하고, Enter 확인 후 `lerobot-record`를 실행합니다. 음성 안내는 항상 꺼져 있습니다.

## 데이터셋 검증

```bash
python scripts/validate_dataset.py \
  data/so101_cube_train_v2 \
  --expected-episodes 5
```

검증 결과 JSON은 기본적으로 아래에 저장됩니다.

```text
reports/dataset_validation/so101_cube_train_v2.json
```

## 중요한 한계

자동 검증은 파일 존재, 프레임 수, 영상 규격, 관절 벡터, 인덱스와 타임스탬프를 검사합니다. 큐브를 실제로 잡고 목표에 놓았는지는 영상으로 사람이 검수해야 합니다.
