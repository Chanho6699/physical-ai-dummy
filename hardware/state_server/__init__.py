"""SO-101 리더암/팔로워암 읽기 전용 관절 상태 서버.

목적: 노트북에 연결된 실물 SO-101 리더암·팔로워암의 현재 관절값을 주기적으로 읽어
HTTP API(`GET /health`, `GET /state`, `GET /calibration`)로 제공한다. 이 서버는 실물에
어떤 명령도 쓰지 않는다 (자세한 근거는 `readonly_so101_reader.py`와
`docs/hardware_state_server.md` 참고).

이 패키지가 절대 하지 않는 일:
  - 목표 위치/action 전송, torque 제어, teleoperation.
  - MuJoCo, SmolVLA, YOLO, ROS2 연동.
  - 캘리브레이션 파일 쓰기.
"""
