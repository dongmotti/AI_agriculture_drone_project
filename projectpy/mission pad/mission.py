from djitellopy import Tello
import time
from djitellopy import Tello

# Tello 드론 객체 생성
tello = Tello()
tello.connect()

# configure drone
tello.enable_mission_pads()
tello.set_mission_pad_detection_direction(2)  # forward detection only

tello.takeoff()

pad = tello.get_mission_pad_id()

# detect and react to pads until we see pad #1
while pad != 1:
    if pad == 3:
        tello.move_back(30)
        tello.rotate_clockwise(90)

    if pad == 4:
        tello.move_up(30)
        tello.land()

    pad = tello.get_mission_pad_id()

# graceful termination
tello.disable_mission_pads()
tello.land()
tello.end()

