import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2

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

# 테스트할 이미지 로드
test_image_path = "apple.jpg"  # 사용자가 자신의 테스트 이미지 경로로 바꿔주세요
test_image = cv2.imread(test_image_path)
test_image = cv2.resize(test_image, image_size)

# 테스트 이미지에 대한 예측
prediction = model.predict(np.array([test_image]))

# 결과 이미지 출력
image = cv2.imread(test_image_path)
class_label = "normal apple" if prediction[0] > 0.2 else "diseased apple"
cv2.putText(image, f'Predicted class: {class_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Apple Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 예측 결과 출력
print(class_label)