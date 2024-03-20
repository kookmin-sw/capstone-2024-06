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

def main(image_path, csv_path, top_n):
    # 이미지에서 오브젝트 검출
    detected_classes = count_class(image_path)

    # CSV 파일 로드
    csv_data = load_csv(csv_path)

    # 가장 유사한 이미지 찾기
    top_similar_images = find_similar_images(detected_classes, csv_data, top_n)

    # 결과 출력
    print(f"Top {top_n} most similar images:")
    for idx, (image_path, similarity) in enumerate(top_similar_images, 1):
        print(f"{idx}. Image Path: {image_path}, Similarity: {similarity}")

if __name__ == "__main__":
    # 이미지 경로와 CSV 파일 경로 설정
    image_path = "/Users/park_sh/Desktop/backend/back/images/test_image/독서실책상/스터디플랜A_풀옵션_독서실책상_스터디카페_타공패널+LED+각도조절_경사판_820.png"  # 테스트할 이미지 경로
    csv_path = "./class_count.csv"  # CSV 파일 경로
    top_n = 3

    # 메인 함수 실행
    main(image_path, csv_path, top_n)
