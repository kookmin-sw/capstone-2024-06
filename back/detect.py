from ultralytics import YOLO
import os
import cv2
import csv
def detect(image_path, result_folder, conf_threshold=0.25):
    from ultralytics import YOLO
    # Load model
    model = YOLO("yolov8x.pt")
    # model = ""
    # Predict
    result = model.predict(image_path, save=False, conf=conf_threshold, classes =[41,56,58,59,60,62,63,64,66,73,74,75])

    # 이미지 저장 경로
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    # 저장 경로
    save_path = os.path.join(result_folder, f"{file_name}_result.jpg")

    # 저장
    plots = result[0].plot()
    cv2.imwrite(save_path, plots)

    return save_path

def count_class(image_path, conf_threshold=0.25):

    # YOLO 모델 로드
    model = YOLO("yolov8x.pt")

    # 이미지 예측
    target_classes = [41,56,58,59,60,62,63,64,66,73,74,75]
    cls_to_idx = {x: i for i, x in enumerate(target_classes)}

    result = model.predict(image_path, save=False, conf=0.25, classes=target_classes)
    object_counter = [0] * len(target_classes)

    for box in result[0].boxes:
        object_counter[cls_to_idx[box.cls.item()]] += 1
    # print(object_counter)

    return object_counter

def img_obj2csv(data_dir, output_csv):
    # CSV 파일 헤더 작성
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        target_classes = [41,56,58,59,60,62,63,64,66,73,74,75]
        writer.writerow(["Image Path"] + [f"Class {cls}" for cls in target_classes] + ["total"])

    # 해당 폴더가 디렉터리인지 확인
    if os.path.isdir(data_dir):
        print("Processing folder:", data_dir)

        # 각 이미지 파일에 대해 클래스별 객체 수 확인
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            # 이미지 파일인지 확인
            if file_name.endswith((".jpg", ".jpeg", ".png")):
                class_counts = count_class(file_path)
                # 클래스별 객체 수가 6개를 넘을 때만 CSV 파일에 기록
                # if sum(class_counts) > 6:
                with open(output_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([file_path] + class_counts + [sum(class_counts)])
                # else:
                #     print("Deleting image:", file_name)
                #     os.remove(file_path)

# if __name__ == "__main__":
#     data_dir = "./images/train"
#     csv_path = "./class_count.csv"
#     img_obj2csv(data_dir,csv_path)
#     # 결과 이미지를 저장할 폴더 생성
#     result_folder = "./result"
#     os.makedirs(result_folder, exist_ok=True)

#     # YOLO 모델 로드
#     model = YOLO("yolov8x.pt")

#     # 이미지 경로
#     image_paths = '/Users/park_sh/Desktop/backend/back/train_image/독서실책상'

#     for file_name in os.listdir(image_paths):
#         if file_name.endswith((".jpg", ".jpeg", ".png")):
#             image_path = os.path.join(image_paths, file_name)
#             result_image_path = detect(image_path, result_folder,0.1)
#             class_counts = count_class(image_path)
#             print(class_counts)