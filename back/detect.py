from ultralytics import YOLO
import cv2


model = YOLO("yolov8s.pt")  

image_paths = [
    # "/Users/park_sh/Desktop/what-desk/back/desk1.jpeg",
    # "/Users/park_sh/Desktop/what-desk/back/desk2.jpeg",  
    # "/Users/park_sh/Desktop/what-desk/back/desk3.jpeg"
    "/Users/park_sh/Desktop/what-desk/back/image/[3%쿠폰]_제로데스크_에보_테이블_컴퓨터_책상_1000~2000size_2Colors.jpg"
    # 추가 가능
]

for image_path in image_paths:
    result = model.predict(image_path, save=True, conf=0.5)
    plots = result[0].plot()

    # 결과 이미지 출력
    cv2.imshow("plot", plots)
    cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()
