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

# 새로운 이미지를 벡터로 변환합니다.
new_image_path = "/Users/park_sh/Desktop/backend/back/test_image/독서실책상/105_가정용_독서실책상.png"
new_image_vector = image_to_vector(new_image_path)

# 기존 이미지 벡터가 저장된 폴더를 지정합니다.
vector_folder = "./train_vector/"

# 기존 이미지 벡터를 로드합니다.
vectors, vector_paths = load_vectors(vector_folder)

# Faiss 인덱스를 생성합니다.
index = create_index(vectors)

# 유사한 이미지를 검색합니다. 여기서는 가장 유사한 이미지 3개를 출력합니다.
distances, indices = search_similar_vectors(new_image_vector, index, k=3)

# 원본 이미지를 출력합니다.
print("원본 이미지:")
plt.imshow(Image.open(new_image_path))
plt.axis('off')
plt.show()

# 유사한 이미지와 유사도를 출력합니다.
for i, (dist, idx) in enumerate(zip(distances, indices)):
    print(f"\n가장 유사한 이미지 {i+1}: {vector_paths[idx]}, 유사도: {dist}")
    print("유사한 이미지:")
    # # 이미지 파일의 경로를 가져옵니다.
    # similar_image_path = vector_paths[idx]
    # # 이미지를 로드하고 출력합니다.
    # similar_image = Image.open(similar_image_path).convert('RGB')
    # plt.imshow(similar_image)
    # plt.axis('off')
    # plt.show()
