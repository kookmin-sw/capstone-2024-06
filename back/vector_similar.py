import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import faiss
import ssl
from image_to_vector import image_to_vector
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

def search_similar_vectors(image_vector, index, k=1):
    D, I = index.search(np.array([image_vector]), k)
    return D[0], I[0]

def create_index(vectors):
    # Faiss 인덱스를 생성하고 벡터를 추가합니다.
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

def load_vectors(vector_folder):
    vectors = []
    vector_paths = []
    
    # 벡터 폴더 내의 각 벡터 파일에 대해 반복합니다.
    for root, _, files in os.walk(vector_folder):
        for file in files:
            if file.lower().endswith(".npy"):
                vector_path = os.path.join(root, file)
                # 벡터를 로드합니다.
                vector = np.load(vector_path)
                vectors.append(vector)
                vector_paths.append(vector_path)
    
    return np.array(vectors), vector_paths

def load_images(image_folder):
    images = []
    image_paths = []
    
    # 이미지 폴더 내의 모든 이미지 파일에 대해 반복합니다.
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                # 이미지를 로드합니다.
                image = Image.open(image_path).convert('RGB')
                images.append(image)
                image_paths.append(image_path)
    
    return images, image_paths

def find_similar_vectors(vector_folder, image_path, k=3):
    new_image_vector = image_to_vector(image_path)
    vectors, vector_paths = load_vectors(vector_folder)
    index = create_index(vectors)
    distances, indices = search_similar_vectors(new_image_vector, index, k)
    similar_images = []
    for dist, idx in zip(distances, indices):
        # 벡터 파일의 경로를 이미지 파일의 경로로 변환
        image_file_path = vector_path_to_image_path(vector_paths[idx], image_folder="./images/train")
        similar_images.append((image_file_path, dist))
    return similar_images

def vector_path_to_image_path(vector_path, image_folder):
    # 벡터 파일의 이름과 확장자를 추출
    vector_file_name = os.path.basename(vector_path)

    # 이미지 폴더에서 동일한 이름을 가진 이미지 파일의 경로 생성
    for file_name in os.listdir(image_folder):
        if os.path.splitext(file_name)[0] == os.path.splitext(vector_file_name)[0]:
            return os.path.join(image_folder, file_name)

    # 이미지 파일을 찾지 못한 경우 None 반환
    return None

# # 새로운 이미지를 벡터로 변환합니다.
# image_path = "/Users/park_sh/Desktop/backend/back/images/test_image/자주스 책상.jpeg"

# # 기존 이미지 벡터가 저장된 폴더를 지정합니다.
# vector_folder = "./vectors/train_vector"

# # 유사 벡터 검색
# similar_images = find_similar_vectors(vector_folder, image_path, k=5)
# similar_images_sorted = sorted(similar_images, key=lambda x: x[1], reverse=True) # 내림차순

# # 원본 이미지를 출력합니다.
# print("원본 이미지:")
# plt.imshow(Image.open(image_path))
# plt.axis('off')
# plt.show()

# # 유사한 이미지와 유사도를 출력합니다.
# for i, (image_path, similarity) in enumerate(similar_images_sorted):
#     print(f"\n가장 유사한 이미지 {i+1}: {image_path}, 유사도: {similarity}")
#     # similar_image = Image.open(image_path).convert('RGB')
#     # plt.imshow(similar_image)
#     # plt.axis('off')
#     # plt.show()
