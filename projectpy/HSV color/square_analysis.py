import cv2

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

    # 컨투어 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 사과 탐지 여부 판단
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # 컨투어 면적이 일정 크기 이상일 때 사과로 판단
            # 사과가 탐지된 부분에 네모 표시
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # 네모 안의 부분을 추출
            detected_region = img[y:y+h, x:x+w]

            # HSV 변환
            detected_region_hsv = cv2.cvtColor(detected_region, cv2.COLOR_BGR2HSV)

            # 검은색 범위
            lower_black = (0, 0, 0)
            upper_black = (180, 70, 50)  # 검은색은 채도와 밝기에 따라 범위를 조정가능

            # 검은색 여부 판단
            mask_black = cv2.inRange(detected_region_hsv, lower_black, upper_black)
            black_detected = cv2.countNonZero(mask_black) > 0


            # 흰색 범위 설정
            lower_white = (0, 0, 180)
            upper_white = (180, 30, 220)

            # 흰색 여부 판단
            mask_white = cv2.inRange(detected_region_hsv, lower_white, upper_white)
            white_detected = cv2.countNonZero(mask_white) > 0

            # 흰색이 감지되면 "detected", 아니면 "normal" 출력
            if white_detected:
                cv2.putText(img, "detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif black_detected:
                cv2.putText(img, "detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(img, "normal", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


    # 결과 이미지 표시
    cv2.imshow(f"Image - {image_path}", img)

    # 결과 이미지 저장
    result_image_path = f"result_{image_path}"
    cv2.imwrite(result_image_path, img)
    print(f"Result image saved as {result_image_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
