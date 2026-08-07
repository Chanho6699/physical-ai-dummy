"""Laptop(SO-101 follower/카메라/Safety/MuJoCo) 측 Shadow Mode 컴포넌트.

이 패키지 안에는 실물 SO-101 follower에 write하는 코드가 없다. 실물 follower는
``follower_state_source.ReadOnlyRealFollowerStateSource`` 로만 접근하며, 이 클래스는
read-only인 ``hardware.state_server.readonly_so101_reader.ReadOnlySO101Reader``를 그대로
감싼 것뿐이다 - ``send_action``/``sync_write`` 경로 자체가 이 패키지 안에 존재하지 않는다.
"""
