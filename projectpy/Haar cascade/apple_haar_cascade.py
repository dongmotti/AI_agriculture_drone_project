import cv2

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

# 이미지 불러오기
image = cv2.imread('33.jpg')

# 사과 탐지
apple_cascade = cv2.CascadeClassifier('apple_cascade.xml')  # 필요한 경우 적절한 XML 파일을 다운로드하고 경로를 지정하세요
detect_and_draw_objects(image, apple_cascade, "사과", (0, 255, 0))

# 결과 이미지 출력
cv2.imshow("Apple and Pest Analysis", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
