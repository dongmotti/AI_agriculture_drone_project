from djitellopy import Tello
from time import sleep

# Tello 드론 객체 생성
tello = Tello()
tello.connect()

# 미션 패드 감지 활성화
tello.enable_mission_pads()

# 미션 패드 감지 방향 설정 (1은 전방 감지)
tello.set_mission_pad_detection_direction(2)

# 이륙
tello.takeoff()

try:
    while True:
        # 현재 미션 패드 ID 확인
        pad_id = tello.get_mission_pad_id()
        print("Detected Mission Pad ID:", pad_id)

        if pad_id == 1:
            # 미션 패드를 발견하면 이동 중지
            tello.send_rc_control(0, 0, 0, 0)

            # 미션 패드까지의 거리 및 방향 계산
            pad_x = tello.get_mission_pad_distance_x()  # X 축 거리
            pad_y = tello.get_mission_pad_distance_y()  # Y 축 거리
            sleep(1)
            print(f"Pad X: {pad_x}, Pad Y: {pad_y}")

            # 미션 패드까지 이동
            tello.send_rc_control(pad_x, pad_y,0,0)
            sleep(1)
            # 착륙
            tello.land()
            break  # 루프 종료

        # 미션 패드를 발견하지 못한 경우 계속 진행
        else:
            tello.move_forward(50)

            # 미션 패드를 발견하지 않았을 때 계속해서 확인을 위해 잠시 대기
            sleep(1)

except KeyboardInterrupt:
    # 사용자가 프로그램을 종료하면 드론을 안전하게 착륙시킴
    tello.land()

finally:
    # 미션 패드 감지 비활성화 및 연결 종료
    tello.disable_mission_pads()
    tello.end()
