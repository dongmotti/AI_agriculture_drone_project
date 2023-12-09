from djitellopy import tello
from time import sleep
import cv2
import threading

drone = tello.Tello()
drone.connect()
print(drone.get_battery())

# 카메라 스트림 시작
drone.streamon()

# 동기화를 위한 Event 객체 생성
camera_ready_event = threading.Event()
camera_ready_event.clear()



def camera_stream():
    while True:
        img = drone.get_frame_read().frame
        img = cv2.resize(img, (854, 480))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        # Event가 설정되어 있으면 카메라 작업을 일시 중단
        if not camera_ready_event.is_set():
            camera_ready_event.wait()

# 카메라 스레드 시작
camera_thread = threading.Thread(target=camera_stream)
camera_thread.start()

# 카메라 스레드 재개
camera_ready_event.set()
sleep(5)

#캡쳐
camera_ready_event.clear()  # 카메라 작업 일시 중단
sleep(1)  # 캡쳐를 위해 잠시 대기
img = drone.get_frame_read().frame  # 현재 프레임 캡쳐
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("6.jpg", img)  # 이미지 저장
camera_ready_event.set()  # 카메라 작업 재개
sleep(2)

# 스레드 종료 및 창 닫기
camera_thread.join()
drone.streamoff()
cv2.destroyAllWindows()
