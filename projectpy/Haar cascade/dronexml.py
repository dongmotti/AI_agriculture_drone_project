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

def detect_and_draw_objects(image, cascade, label, color):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(objects) == 0:
        print(f"{label} 탐지 실패")
        return

    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        object_roi = image[y:y+h, x:x+w]  # 객체의 ROI(Region of Interest) 추출

        # 외각선 추출
        gray_roi = cv2.cvtColor(object_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, threshold1=30, threshold2=100)
        cv2.imshow(f"{label} Edges", edges)

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

drone.takeoff()
sleep(2)

# 카메라 스레드 재개
camera_ready_event.set()
sleep(2)
drone.move_down(30)
sleep(2)

#캡쳐
camera_ready_event.clear()  # 카메라 작업 일시 중단
sleep(1)  # 캡쳐를 위해 잠시 대기
img = drone.get_frame_read().frame  # 현재 프레임 캡쳐
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("6.jpg", img)  # 이미지 저장
camera_ready_event.set()  # 카메라 작업 재개
sleep(2)

# 이미지 파일 경로
image_paths = ["6.jpg"]

for image_path in image_paths:# 이미지 로드
    image = cv2.imread(image_path)

    # 사과 탐지
    apple_cascade = cv2.CascadeClassifier('apple_cascade.xml')  # 필요한 경우 적절한 XML 파일을 다운로드하고 경로를 지정하세요
    detect_and_draw_objects(image, apple_cascade, "사과", (0, 255, 0))

    # 결과 이미지 표시
    cv2.imshow(f"Image - {image_path}", image)

    # 결과 이미지 저장
    result_image_path = f"result_{image_path}"
    cv2.imwrite(result_image_path, image)
    print(f"Result image saved as {result_image_path}")

    cv2.waitKey(1)

sleep(2)
drone.land()



# 스레드 종료 및 창 닫기
camera_thread.join()
drone.streamoff()
cv2.destroyAllWindows()
