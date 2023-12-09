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

drone.takeoff()
sleep(2)

# 카메라 스레드 재개
camera_ready_event.set()


drone.move_down(30)
sleep(3)
drone.move_right(45)

#캡쳐 apple 1
camera_ready_event.clear()  # 카메라 작업 일시 중단
sleep(1)  # 캡쳐를 위해 잠시 대기
img = drone.get_frame_read().frame  # 현재 프레임 캡쳐
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("1.jpg", img)  # 이미지 저장
camera_ready_event.set()  # 카메라 작업 재개

drone.send_rc_control(0, 0, 0, 0)  # 비행 제어 멈춤
sleep(2)  # 2만큼 대기

# 이미지 파일 경로
image_paths = ["1.jpg"]

for image_path in image_paths:
    # 이미지 로드
    img = cv2.imread(image_path)

    # HSV 변환
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 빨간색 범위 설정
    lower_red1 = (0, 50, 50)
    upper_red1 = (10, 255, 255)
    lower_red2 = (160, 50, 50)
    upper_red2 = (180, 255, 255)


    # 이미지 필터링
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # 컨투어 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 사과 탐지 여부 판단
    apple_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # 컨투어 면적이 일정 크기 이상일 때 질병으로 판단
            apple_detected = True
            # 질병이 탐지된 부분에 네모 표시
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(img, "normal", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # 사과 탐지가 안되었을 경우의 처리
    if not apple_detected:
        # 원래 정해진 동작 수행 (예: 이동, 회전 등)
        pass

    # 결과 이미지 표시
    cv2.imshow(f"Image - {image_path}", img)

    # 결과 이미지 저장
    result_image_path = f"result_{image_path}"
    cv2.imwrite(result_image_path, img)
    print(f"Result image saved as {result_image_path}")

    cv2.waitKey(1)

drone.move_right(35)
sleep(2)

#캡쳐 apple 2
camera_ready_event.clear()  # 카메라 작업 일시 중단
sleep(1)  # 캡쳐를 위해 잠시 대기
img = drone.get_frame_read().frame  # 현재 프레임 캡쳐
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("2.jpg", img)  # 이미지 저장
camera_ready_event.set()  # 카메라 작업 재개

drone.send_rc_control(0, 0, 0, 0)  # 비행 제어 멈춤
sleep(2)  # 2만큼 대기

# 이미지 파일 경로
image_paths = ["2.jpg"]

for image_path in image_paths:
    # 이미지 로드
    img = cv2.imread(image_path)

    # HSV 변환
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 빨간색 범위 설정
    lower_red1 = (0, 50, 50)
    upper_red1 = (10, 255, 255)
    lower_red2 = (160, 50, 50)
    upper_red2 = (180, 255, 255)


    # 이미지 필터링
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # 컨투어 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 사과 탐지 여부 판단
    apple_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # 컨투어 면적이 일정 크기 이상일 때 질병으로 판단
            apple_detected = True
            # 사과가 탐지된 부분에 네모 표시
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(img, "detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # 사과 탐지가 안되었을 경우의 처리
    if not apple_detected:
        # 원래 정해진 동작 수행 (예: 이동, 회전 등)
        pass

    # 결과 이미지 표시
    cv2.imshow(f"Image - {image_path}", img)

    # 결과 이미지 저장
    result_image_path = f"result_{image_path}"
    cv2.imwrite(result_image_path, img)
    print(f"Result image saved as {result_image_path}")

    cv2.waitKey(1)

drone.move_forward(20)
sleep(3)

#캡쳐 close
camera_ready_event.clear()  # 카메라 작업 일시 중단
sleep(2)  # 캡쳐를 위해 잠시 대기
img = drone.get_frame_read().frame  # 현재 프레임 캡쳐
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("22.jpg", img)  # 이미지 저장
camera_ready_event.set()  # 카메라 작업 재개

# 이미지 파일 경로
image_paths = ["22.jpg"]

for image_path in image_paths:
    # 이미지 로드
    img = cv2.imread(image_path)

    # HSV 변환
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 검은색 범위
    lower_black = (0, 0, 0)
    upper_black = (180, 255, 50)  # 검은색은 채도와 밝기에 따라 범위를 조정가능

    # 이미지 필터링
    mask_black = cv2.inRange(img_hsv, lower_black, upper_black)
    masked_img = cv2.bitwise_and(img, img, mask=mask_black)

    # 컨투어 추출
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 검은색 탐지 여부 판단
    black_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # 컨투어 면적이 일정 크기 이상일 때 질병으로 판단
            black_detected = True

            # 컨투어의 외곽선을 찾습니다.
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 외곽선 따기
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

    # 검은색 탐지가 되었을 경우의 처리
    if black_detected:
        # 외곽선 주위에 "탄저병 detected" 메시지 표시
        cv2.putText(img, "Anthracnose detected!!", (approx[0][0][0] - 200,approx[0][0][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
        cv2.putText(img, "Bogard spray", (0, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    # 결과 이미지 표시
    cv2.imshow(f"Image - {image_path}", img)

    # 결과 이미지 저장
    result_image_path = f"result_{image_path}"
    cv2.imwrite(result_image_path, img)
    print(f"Result image saved as {result_image_path}")

    cv2.waitKey(1)

sleep(3)
drone.move_back(20)
sleep(3)


drone.rotate_clockwise(180)

sleep(2)

drone.land()
camera_thread.join()
drone.streamoff()
drone.end()