import cv2
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models

# 데이터셋 경로 설정
good_apple_folder_path = r"C:\Users\MSI\PycharmProjects\pythonProjec\pythonProject\Tello\Normal_Apple"
bad_apple_folder_path = r"C:\Users\MSI\PycharmProjects\pythonProjec\pythonProject\Tello\Scab_Apple"

# 데이터 로드 및 전처리
def load_and_preprocess_data(image_paths, label):
    data = []
    labels = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        data.append(img)
        labels.append(label)
    return np.array(data), np.array(labels)

# 정상 사과 데이터 로드 및 전처리
good_apple_images = glob.glob(os.path.join(good_apple_folder_path, "*.jpg"))
good_data, good_labels = load_and_preprocess_data(good_apple_images, 1)

# 문제가 있는 사과 데이터 로드 및 전처리
bad_apple_images = glob.glob(os.path.join(bad_apple_folder_path, "*.jpg"))
bad_data, bad_labels = load_and_preprocess_data(bad_apple_images, 0)

# 데이터를 합쳐서 섞음
data = np.concatenate((good_data, bad_data), axis=0)
labels = np.concatenate((good_labels, bad_labels), axis=0)

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
model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)
