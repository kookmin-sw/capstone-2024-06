import cv2
import numpy as np
import pandas as pd
from detect import count_class

def load_csv(csv_path):
    # CSV 파일을 읽어들이는 함수
    df = pd.read_csv(csv_path)
    return df

def calculate_similarity(class1, class2, total_objects):
    
    # 클래스별로 가중치를 주기? 결과 보면서 추가하기

    normalized_class1 = class1 / total_objects
    normalized_class2 = class2 / total_objects
    return np.sum(normalized_class1 * normalized_class2)

def find_similar_images(detected_classes, csv_data, top_n):
    # 검출된 오브젝트와 유사한 클래스를 가진 이미지를 찾는 함수
    total_objects = csv_data.iloc[:, 1:-1].sum(axis=1).values[0]
    similarities = []
    for index, row in csv_data.iterrows():
        similarity = calculate_similarity(detected_classes, row[1:-1], total_objects)
        similarities.append((row['Image Path'], similarity))
    
    # 유사도가 가장 높은 이미지를 찾습니다.
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 N개의 유사한 이미지를 반환합니다.
    top_similar_images = similarities[:top_n]
    
    return top_similar_images

def object_similar(image_path, csv_path, top_n):
    # 이미지에서 오브젝트 검출
    detected_classes = count_class(image_path)

    # CSV 파일 로드
    csv_data = load_csv(csv_path)

    # 가장 유사한 이미지 찾기
    top_similar_images = find_similar_images(detected_classes, csv_data, top_n)

    # 결과를 담을 배열 초기화
    similar_images_info = []

    # 유사한 이미지의 정보를 배열에 담기
    for idx, (image_path, similarity) in enumerate(top_similar_images, 1):
        similar_images_info.append({"image_path": image_path, "similarity": similarity})

    return similar_images_info

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

# if __name__ == "__main__":
#     # 이미지 경로와 CSV 파일 경로 설정
#     image_path = "/Users/park_sh/Desktop/backend/back/images/test_image/자주스 책상.jpeg"  # 테스트할 이미지 경로
#     csv_path = "./class_count.csv"  # CSV 파일 경로
#     top_n = 10

#     # 메인 함수 실행
#     similar_images_info = object_similar(image_path, csv_path, top_n)

#     # 반환된 유사한 이미지 정보 출력
#     for idx, info in enumerate(similar_images_info, 1):
#         print(f"{idx}. Image Path: {info['image_path']}, Similarity: {info['similarity']}")

