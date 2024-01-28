from ultralytics import YOLO
import os
import cv2

def detect(image_path, model, result_folder, conf_threshold=0.25):
    # Load model
    if not hasattr(model, "predict"):
        raise ValueError("The provided model does not have a 'predict' method.")

    # Predict
    result = model.predict(image_path, save=False, conf=conf_threshold)

    # Get the filename without extension
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save the result image in the specified folder
    save_path = os.path.join(result_folder, f"{file_name}_result.jpg")

    # Visualize and save the result image
    plots = result[0].plot()
    cv2.imwrite(save_path, plots)

    return save_path

if __name__ == "__main__":
    # 결과 이미지를 저장할 폴더 생성
    result_folder = "result"
    os.makedirs(result_folder, exist_ok=True)

    # YOLO 모델 로드
    model = YOLO("yolov8s.pt")

    # 이미지 경로
    image_paths = [
        # "/Users/park_sh/Desktop/what-desk/back/image/[3%쿠폰]_제로데스크_에보_테이블_컴퓨터_책상_1000~2000size_2Colors.jpg"
        "/Users/park_sh/Desktop/what-desk/back/image/_[누적판매_1만5천]_오테카_원목_책상.jpg"
        # 추가 가능
    ]

    for image_path in image_paths:
        result_image_path = detect(image_path, model, result_folder)
