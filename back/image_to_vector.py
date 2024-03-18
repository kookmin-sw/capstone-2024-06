import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
from torchvision import models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def image_to_vector(image_path):
    # 이미지를 읽어옵니다.
    image = Image.open(image_path).convert('RGB')
    
    # 이미지를 PyTorch의 텐서로 변환합니다.
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # 배치 차원 추가
    
    # 사전 학습된 VGG16 모델을 사용하여 이미지를 벡터로 변환합니다.
    model = models.vgg16(pretrained=True)
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        vector = model(image_tensor)
    
    # 벡터를 numpy 배열로 변환하여 반환합니다.
    return vector.squeeze().numpy()

def create_vector_files_from_images(image_folder, output_folder):
    # 이미지 폴더 내의 각 클래스 폴더에 대해 반복합니다.
    for class_name in os.listdir(image_folder):
        class_folder = os.path.join(image_folder, class_name)
        # 해당 폴더가 디렉터리인지 확인합니다.
        if os.path.isdir(class_folder):
            print(f"Processing class folder: {class_name}")
            # 클래스 폴더 내의 각 이미지 파일에 대해 반복합니다.
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                # 이미지 파일인지 확인합니다.
                if os.path.isfile(image_path) and image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    # 이미지를 벡터로 변환합니다.
                    image_vector = image_to_vector(image_path)
                    # 벡터 파일을 저장합니다.
                    vector_filename = f"{image_name.split('.')[0]}.npy"
                    output_class_folder = os.path.join(output_folder, class_name)
                    if not os.path.exists(output_class_folder):
                        os.makedirs(output_class_folder)
                    output_vector_path = os.path.join(output_class_folder, vector_filename)
                    np.save(output_vector_path, image_vector)

# 이미지 폴더와 벡터 파일을 저장할 폴더를 지정합니다.
image_folder = "./test_image"
output_folder = "./test_vector"

# 이미지 폴더 내의 이미지를 벡터 파일로 변환하여 저장합니다.
create_vector_files_from_images(image_folder, output_folder)
