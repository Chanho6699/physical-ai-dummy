"""정상 LeRobot SO-101 teleoperation 경로를 그대로 사용하는 계측(instrumentation) 도구 모음.

``hardware/safety/``와 달리 이 패키지의 도구는 실제로 follower를 움직인다 - 단, 움직임
자체는 전부 ``lerobot.robots.so_follower.SOFollower``/``lerobot.teleoperators.so_leader.
SOLeader``의 정상 ``connect()``/``configure()``/``get_action()``/``send_action()``/
``disconnect()`` 경로로만 발생한다. 이 패키지 어디에도 별도의 저수준 servo write(직접
``FeetechMotorsBus.write()``/``sync_write()`` 호출)는 없다. 자세한 조사 근거는
``hardware/diagnostics/instrumented_teleop.py`` 모듈 docstring 참고.
"""
