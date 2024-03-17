import os
import numpy as np
from tensorflow.keras.models import load_model

# 모델 로드
model = load_model("my_vector_model.keras")

# 벡터로 변환된 이미지 폴더 경로 설정
vector_image_folder_path = "./test_vector/h형책상"

# 벡터로 변환된 이미지 폴더 내의 이미지 파일 목록 가져오기
vector_image_files = os.listdir(vector_image_folder_path)

# 정답 클래스
true_class_index = 0  # 정답 클래스 설정

# 정답 개수 초기화
correct_count = 0

# 이미지 파일들을 반복하며 예측 수행
for vector_image_file in vector_image_files:
    # 벡터로 변환된 이미지 파일 경로 생성
    vector_image_path = os.path.join(vector_image_folder_path, vector_image_file)

    # 이미지 벡터 로드
    image_vector = np.load(vector_image_path)
    image_vector = np.expand_dims(image_vector, axis=0)  # 모델 입력 형태에 맞게 차원 확장

    # 모델로 이미지 예측
    prediction = model.predict(image_vector)

    # 예측 결과 출력
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]

    # 예측 결과 확인 및 정답 여부 판별
    if predicted_class_index == true_class_index:
        correct_count += 1

    print("Image:", vector_image_path)
    print("Predicted Class Index:", predicted_class_index)
    print("Confidence:", confidence)
    print()

# 정확도 출력
total_images = len(vector_image_files)
accuracy = correct_count / total_images
print("Total Images:", total_images)
print("Correct Predictions:", correct_count)
print("Accuracy:", accuracy)
