from ultralytics import YOLO
import os
import cv2

# 결과 이미지를 저장할 폴더 생성
result_folder = "result"
os.makedirs(result_folder, exist_ok=True)

model = YOLO("yolov8s.pt")  

image_paths = [
    "/Users/park_sh/Desktop/what-desk/back/image/[3%쿠폰]_제로데스크_에보_테이블_컴퓨터_책상_1000~2000size_2Colors.jpg"
    # 추가 가능
]

for image_path in image_paths:
    result = model.predict(image_path, save=False, conf=0.25)

    # 확장자를 제외한 파일 이름 가져오기
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    # "result" 폴더에 결과 이미지 저장
    save_path = os.path.join(result_folder, f"{file_name}_result.jpg")

    # 결과 이미지를 시각화하여 확인
    plots = result[0].plot()
    cv2.imshow("plot", plots)
    
    # 결과 이미지를 파일로 저장
    cv2.imwrite(save_path, plots)

    cv2.waitKey(0)


# 모든 창 닫기
cv2.destroyAllWindows()
