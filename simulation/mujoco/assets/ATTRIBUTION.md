# MuJoCo SO-101 모델 출처

## robotstudio_so101/

`robotstudio_so101/` 디렉터리는 **수정 없이 그대로** 벤더링(vendoring)한 외부 파일이다.

- 출처: [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Google DeepMind
  관리) 저장소의 `robotstudio_so101` 패키지.
- 원본 MJCF 파생 경로: [The Robot Studio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)
  저장소의 공식 `so101_new_calib.xml` (commit `608122e9ac330a753735f2e18aee73338e9ac407`)에서
  파생됨. 자세한 변경 이력은 [`robotstudio_so101/CHANGELOG.md`](robotstudio_so101/CHANGELOG.md) 참고.
- 라이선스: **Apache License 2.0**. 원문은 [`robotstudio_so101/LICENSE`](robotstudio_so101/LICENSE)에
  그대로 보존되어 있다.
- 로컬 확보 경로: 이 저장소에는 인터넷에서 새로 내려받지 않고, 같은 시스템에 이미 체크아웃되어 있던
  `~/Projects/physical-ai-recycling-cell/third_party/mujoco_menagerie/robotstudio_so101`를
  파일 그대로 복사했다 (STL 메쉬 포함, 바이트 단위 동일).
- 확인된 사양: 6개 revolute 관절(`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
  `wrist_roll`, `gripper`), 각 관절에 대응하는 `position` actuator 1개씩, 관절 range/actuator
  ctrlrange는 라디안 단위로 XML에 명시되어 있다 (`so101.xml` 참고).

## scene.xml (이 저장소에서 작성)

`simulation/mujoco/assets/scene.xml`은 이 저장소에서 새로 작성한 파일이며, 로봇 본체 정의
(관절/actuator/충돌 geometry, 질량/관성 등)는 **`robotstudio_so101/so101.xml`의 내용을 그대로
복사한 것으로, 수치를 임의로 변경하지 않았다.**

값을 바꾼 부분은 다음 두 가지뿐이다.

1. `<compiler meshdir="...">` 값을 `assets` → `robotstudio_so101/assets`로 변경 (파일 위치가
   달라졌으므로 메쉬 상대경로만 보정).
2. 바닥 평면(`table_surface`)과 조명(`light`), 스카이박스 텍스처를 추가 (원본 `scene.xml`의
   해당 부분을 참고해 동일하게 구성).

**왜 `<include>`를 쓰지 않았는가**: 처음에는
`<include file="robotstudio_so101/so101.xml"/>` + `<include file="robotstudio_so101/scene.xml"/>`
형태로 조립하려 했다. 그러나 설치된 MuJoCo 3.11.0에서, 서로 다른 디렉터리에 걸쳐 2단계 이상 중첩된
상대경로 `<include>`를 사용하면 mesh 파일 경로가 디렉터리를 중복 결합해 잘못 계산되는 문제가
재현되었다 (`assets/simulation/mujoco/assets/robotstudio_so101/....stl` 형태로 깨짐). 이를
피하기 위해 로봇 본체 XML을 `scene.xml`에 직접 병합했다. `robotstudio_so101/so101.xml`,
`robotstudio_so101/scene.xml` 원본은 수정하지 않고 그대로 두었으므로, 필요하면 언제든 원본과
`diff`로 비교해 값이 바뀌지 않았음을 확인할 수 있다.

## table_surface geom

`scene.xml`의 `table_surface`라는 이름의 평면(plane) geom은 실제 하드웨어 실험대(table)를
단순화한 대리물(proxy)이다. 이번 1단계 범위에서는 큐브나 타겟 존 같은 조작 대상 오브젝트를
모델링하지 않으므로(=action replay만 검증), "table collision" 안전 검사는 로봇 geometry가
이 평면과 접촉하는지 여부로 판정한다.
