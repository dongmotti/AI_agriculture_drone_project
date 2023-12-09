from djitellopy import tello
from time import sleep
import cv2
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 예시 데이터셋 폴더 경로 설정 (사용자가 실제 데이터셋 경로로 변경해야 함)
good_apple_folder_path = r"C:\Users\MSI\PycharmProjects\pythonProjec\pythonProject\Tello\Normal_Apple"
bad_apple_folder_path = r"C:\Users\MSI\PycharmProjects\pythonProjec\pythonProject\Tello\Scab_Apple"

# 폴더에서 파일 리스트 가져오기
good_apple_files = [os.path.join(good_apple_folder_path, file) for file in os.listdir(good_apple_folder_path)]
bad_apple_files = [os.path.join(bad_apple_folder_path, file) for file in os.listdir(bad_apple_folder_path)]

# 데이터셋 로드
good_apple_images = [cv2.imread(file) for file in good_apple_files]
bad_apple_images = [cv2.imread(file) for file in bad_apple_files]

# 이미지 크기 조정 및 정규화
image_size = (224, 224)
good_apple_images = [cv2.resize(image, image_size) for image in good_apple_images]
bad_apple_images = [cv2.resize(image, image_size) for image in bad_apple_images]

# 이미지 데이터와 레이블 추출
good_apple_labels = np.ones(len(good_apple_images))  # 1은 정상적인 사과를 나타냄
bad_apple_labels = np.zeros(len(bad_apple_images))   # 0은 문제가 있는 사과를 나타냄

# 데이터 합치기
all_images = np.concatenate((np.array(good_apple_images), np.array(bad_apple_images)), axis=0)
all_labels = np.concatenate((good_apple_labels, bad_apple_labels), axis=0)

# 모델 구성
model = models.Sequential([
    tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(all_images, all_labels, epochs=10, batch_size=32, validation_split=0.2)

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
drone.send_rc_control(0, 0, 0, 0)
sleep(2)

# 카메라 스레드 재개
camera_ready_event.set()


drone.send_rc_control(0, 0, 0, 0)
sleep(3)

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
    # 테스트할 이미지 로드
    image_path = cv2.imread(image_path)
    image_path = cv2.resize(image_path, image_size)

    # 테스트 이미지에 대한 예측
    prediction = model.predict(np.array([image_path]))

    # 결과 이미지 출력
    image = cv2.imread(image_path)
    class_label = "normal apple" if prediction[0] > 0.2 else "diseased apple"
    cv2.putText(image, f'Predicted class: {class_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Apple Detection", image)
    cv2.waitKey(0)

    # 예측 결과 출력
    print(class_label)

    # 결과 이미지 저장
    result_image_path = f"result_{image_path}"
    cv2.imwrite(result_image_path, img)
    print(f"Result image saved as {result_image_path}")
    cv2.waitKey(1)

drone.send_rc_control(0, 0, 0, 0)
sleep(2)
drone.land()