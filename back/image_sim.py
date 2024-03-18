import os
from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_ssim(image1_path, image2_path):
    # 이미지 읽기
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # 이미지 크기를 동일하게 조정
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])
    image1 = cv2.resize(image1, (min_width, min_height))
    image2 = cv2.resize(image2, (min_width, min_height))

    # 이미지를 그레이스케일로 변환
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # SSIM 계산
    similarity_index, _ = ssim(image1_gray, image2_gray, full=True)

    return similarity_index

def get_all_images_similarity(image_folder, fixed_image_path):
    # 이미지 폴더 내의 이미지 파일 경로 가져오기
    image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith((".jpg", ".jpeg", ".png"))]

    # 이미지 1 (고정 이미지)
    image1_path = fixed_image_path

    # 모든 이미지 쌍에 대한 유사도를 계산하고 저장
    similarity_scores = []
    for img2_path in image_paths:
        similarity_index = calculate_ssim(image1_path, img2_path)
        similarity_scores.append((img2_path, similarity_index))

    # 유사도를 내림차순으로 정렬
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    return similarity_scores

if __name__ == "__main__":
    # 이미지 폴더 경로
    image_folder = "image"

    # 이미지 1 (고정 이미지) 경로
    fixed_image_path = "/Users/park_sh/Desktop/what-desk/back/KakaoTalk_Photo_2024-02-01-22-10-52.jpeg"

    # 모든 이미지와의 유사도 계산
    all_similarity_scores = get_all_images_similarity(image_folder, fixed_image_path)

    # 결과 출력 (상위 3개 이미지)
    for i, (img2_path, similarity_index) in enumerate(all_similarity_scores[:3], start=1):
        print(f"Rank {i}:")
        print(f"Fixed Image: {fixed_image_path}")
        print(f"Image 2: {img2_path}")
        print(f"Similarity Index: {similarity_index}")
        print()
