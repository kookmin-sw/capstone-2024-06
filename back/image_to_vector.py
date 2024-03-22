import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import faiss
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def image_to_vector(image_path, model):
    try:
        # 이미지를 읽어옵니다.
        image = Image.open(image_path).convert('RGB')
    except OSError as e:
        print(f"Error reading image {image_path}: {e}")
        return None

    # 이미지가 정상적으로 열리지 않은 경우 None을 반환합니다.
    if image is None:
        return None
    
    # 이미지를 PyTorch의 텐서로 변환합니다.
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # 배치 차원 추가
    
    # 사전 학습된 모델을 사용하여 이미지를 벡터로 변환합니다.
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        vector = model(image_tensor)
    
    # 벡터를 numpy 배열로 변환하여 반환합니다.
    return vector.squeeze().numpy()

def create_vector_files_from_images(image_folder, output_folder):
    # VGG16 모델을 불러옵니다.
    model = models.vgg16(pretrained=True)
    
    # 이미지 폴더 내의 각 이미지를 벡터로 변환하여 파일로 저장합니다.
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path) and image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            # 이미지를 벡터로 변환합니다.
            vector = image_to_vector(image_path, model)
            if vector is not None:
                # 벡터 파일을 저장합니다.
                output_vector_path = os.path.join(output_folder, f"{image_name.split('.')[0]}.npy")
                np.save(output_vector_path, vector)

def merge_index_files(vector_folder, index_file_path):
    # Faiss 인덱스 생성
    d = 1000  # 벡터 차원 (VGG16 모델에서 출력되는 특징 벡터의 크기)
    index = faiss.IndexFlatL2(d)

    # 벡터 폴더 내의 각 클래스 폴더에 대해 반복합니다.
    for class_name in os.listdir(vector_folder):
        class_folder = os.path.join(vector_folder, class_name)
        if os.path.isdir(class_folder):
            print(f"Processing class folder: {class_name}")
            # 클래스 폴더 내의 각 벡터 파일에 대해 반복합니다.
            for vector_filename in os.listdir(class_folder):
                vector_path = os.path.join(class_folder, vector_filename)
                if os.path.isfile(vector_path) and vector_filename.lower().endswith(".npy"):
                    # 벡터를 로드합니다.
                    vector = np.load(vector_path)
                    # 벡터의 차원을 확인합니다.
                    print(f"Vector shape: {vector.shape}")
                    # Faiss 인덱스에 벡터를 추가합니다.
                    index.add(np.expand_dims(vector, axis=0))

    # Faiss 인덱스를 디스크에 저장합니다.
    faiss.write_index(index, index_file_path)
    print("Faiss index has been saved to:", index_file_path)

# 이미지 폴더와 벡터 파일을 저장할 폴더를 지정합니다.
image_folder = "./images/train"
vector_folder = "./vectors/train_vector"
index_file_path = "image_index.index"

# 이미지 폴더 내의 이미지를 벡터 파일로 변환하여 저장합니다.
create_vector_files_from_images(image_folder, vector_folder)

# 이미지 벡터를 Faiss 인덱스에 추가하고 저장합니다.
merge_index_files(vector_folder, index_file_path)
