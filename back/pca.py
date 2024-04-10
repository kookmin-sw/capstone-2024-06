from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt

def load_vectors(vector_folder):
    user_vectors = []
    for vector_filename in os.listdir(vector_folder):
        vector_path = os.path.join(vector_folder, vector_filename)
        if os.path.isfile(vector_path) and vector_filename.lower().endswith(".npy"):
            user_vector = np.load(vector_path)
            if len(user_vector) > 0:  # Check if the vector is not empty
                user_vectors.append(user_vector)
            else:
                print(f"Ignored empty vector file: {vector_path}")
    return user_vectors


# 이미지 벡터를 로드합니다.
vector_folder = "./vectors/train"
image_vectors = load_vectors(vector_folder)

# 적절한 차원 수를 결정합니다.
n_components = 2

# PCA를 사용하여 이미지 벡터를 2차원으로 축소합니다.
pca = PCA(n_components=n_components)
image_vectors_pca = pca.fit_transform(image_vectors)

# 시각화를 위해 플롯합니다.
plt.figure(figsize=(10, 6))
plt.scatter(image_vectors_pca[:, 0], image_vectors_pca[:, 1])
plt.title('PCA Projection of Image Vectors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
