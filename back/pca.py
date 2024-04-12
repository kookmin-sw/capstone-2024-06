import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# CSV 파일에서 데이터 불러오기
data = pd.read_csv('transformed_data_with_labels.csv')

# 라벨 추출
labels = data['label']

# 벡터 데이터 추출 (라벨 열 제외)
vectors = data.drop('label', axis=1).values

# 시각화를 위한 색상 설정
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']  # 라벨에 따른 색상 설정

# 시각화
plt.figure(figsize=(10, 6))
pca = PCA(n_components=2)  # PCA 객체 생성
transformed_vectors = pca.fit_transform(vectors)  # PCA를 적용하여 벡터 데이터 변환

for i in range(len(transformed_vectors)):
    transformed_vector = transformed_vectors[i]
    label = labels[i]

    # 시각화
    plt.scatter(transformed_vector[0], transformed_vector[1], c=colors[label], label=f'Label {label}')

plt.title('PCA Projection of Vectors with Different Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
