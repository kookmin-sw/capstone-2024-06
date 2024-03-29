from detect import count_class
import os
from vector_similar import find_similar_vectors
from object_similar import cosine_similarity
from PIL import Image


def recommend_image(image_path, top_n):
    # csv_path = "./class_count.csv"  
    vector_folder = "./vectors/train_vector" 
    user_img_obj = count_class(image_path)
    # 이미지 벡터 유사도를 계산하여 유사한 이미지 정보 가져오기
    vector_similar_images = find_similar_vectors(vector_folder, image_path, top_n)
    vector_similar_images_info = [{"image_path": image_path, "vector_similarity": similarity} for image_path, similarity in vector_similar_images]
    
    final_score = []
    for vec_info in vector_similar_images_info:
        # 이미지의 벡터 유사도가 가장 높은 이미지를 기준으로 오브젝트 유사도를 계산
        detected_classes = count_class(vec_info["image_path"])  # 벡터 유사도가 가장 높은 이미지의 오브젝트 정보 가져오기
        cosine_sim = cosine_similarity(user_img_obj, detected_classes)  # 오브젝트 유사도 계산
        total_similarity = vec_info["vector_similarity"] * cosine_sim  # 벡터 유사도와 오브젝트 유사도를 곱하여 최종 유사도 계산
        final_score.append({"image_path": vec_info["image_path"], "total_similarity": total_similarity})

    # 최종 추천 이미지를 유사도를 기준으로 정렬
    final_score.sort(key=lambda x: x["total_similarity"], reverse=True)

    # 상위 N개의 최종 추천 이미지 정보 반환
    return final_score[:top_n//2]

# 테스트할 이미지 경로와 상위 N개의 추천 이미지 설정
image_path = "/Users/park_sh/Desktop/backend/back/images/test_image/학사책상.jpeg"  # 테스트할 이미지 경로2
top_n = 10  # 상위 N개의 추천 이미지

# 최종 추천 이미지 순위를 가져오기
ranked_recommendations = recommend_image(image_path, top_n)

print("Ranked Recommendations:")
for idx, recommendation in enumerate(ranked_recommendations, 1):
    print(f"{idx}. Image Path: {recommendation['image_path']}, Total Similarity: {recommendation['total_similarity']}")
    # 이미지 표시
    img = Image.open(recommendation['image_path'])
    img.show()