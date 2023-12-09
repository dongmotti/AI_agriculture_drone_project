from djitellopy import tello
from time import sleep
import cv2
import threading

# Tello 드론 초기화 및 연결
drone = tello.Tello()
drone.connect()
print(drone.get_battery())

# 카메라 스트림 시작
drone.streamon()

# 동기화를 위한 Event 객체 생성
camera_ready_event = threading.Event()
camera_ready_event.clear()

# 카메라 스레드 함수
def camera_stream():
    while True:
        img = drone.get_frame_read().frame
        img = cv2.resize(img, (854, 480))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        if not camera_ready_event.is_set():
            camera_ready_event.wait()

# 카메라 스레드 시작
camera_thread = threading.Thread(target=camera_stream)
camera_thread.start()

sleep(2)

# 카메라 스레드 재개
camera_ready_event.set()

sleep(2)

#캡쳐 1
camera_ready_event.clear()  # 카메라 작업 일시 중단
sleep(1)  # 캡쳐를 위해 잠시 대기
img = drone.get_frame_read().frame  # 현재 프레임 캡쳐
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("11111.jpg", img)  # 이미지 저장
camera_ready_event.set()  # 카메라 작업 재개

cv2.imshow(f"11111.jpg", img)

# 결과 이미지 저장
result_image_path = f"11111.jpg"
cv2.imwrite(result_image_path, img)
print(f"Result image saved as {result_image_path}")

cv2.waitKey(1)