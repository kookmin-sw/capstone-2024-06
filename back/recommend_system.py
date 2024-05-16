import os
import random
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def load_vectors(vector_folder):
    vectors = {}
    for file_name in os.listdir(vector_folder):
        file_path = os.path.join(vector_folder, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.npy'):
            vector = np.load(file_path)
            vectors[file_name] = vector
    return vectors

def save_user_vector(user_id, user_vector, vector_folder):
    user_vector_folder = os.path.join(vector_folder, "user")
    if not os.path.exists(user_vector_folder):
        os.makedirs(user_vector_folder)
    file_path = os.path.join(user_vector_folder, f"{user_id}.npy")
    np.save(file_path, user_vector)

def load_user_vector(user_id, vector_folder, vectors):
    file_path = os.path.join(vector_folder, "user", f"{user_id}.npy")
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        random_image_vector = random.choice(list(vectors.values()))
        return random_image_vector

def initialize_user_vector(vectors):
    random_image_vector = random.choice(list(vectors.values()))
    return random_image_vector

def update_user_vector(selected_image_vectors, selected_ratings, user_vector):
    # 각 이미지 벡터와 선호도에 따른 가중치 계산
    weighted_vectors = []
    for vector, rating in zip(selected_image_vectors, selected_ratings):
        if rating == 1:
            weight = -0.2
        elif rating == 2:
            weight = 0.2
        elif rating == 3:
            weight = 1.0
        elif rating == 4:
            weight = 1.2
        else:  # rating == 5
            weight = 2.0
        weighted_vectors.append(weight * vector)
 
    # 가중 평균 계산
    weighted_mean_vector = np.mean(weighted_vectors, axis=0)

    # 사용자 벡터, 평균 중 중간에 위치하게 업데이트
    updated_user_vector = (0.0*user_vector + 2.0*weighted_mean_vector) * 0.5

    return updated_user_vector


def recommend_images(user_vector, vectors):
    user_vector_array = user_vector.reshape(1, -1)
    similarities = cosine_similarity(user_vector_array, list(vectors.values()))
    sorted_indices = np.argsort(similarities[0])[::-1]
    recommended_images = [(list(vectors.keys())[index], similarities[0][index]) for index in sorted_indices]
    return recommended_images

def show_recommendations(recommendations):
    num_recommendations = min(len(recommendations), 3)
    for i, (image_file, similarity) in enumerate(recommendations[:num_recommendations], start=1):
        print(f"이미지: {image_file}, 유사도: {similarity:.2f}")

def plot_vectors(user_vector, recommended_images, vectors, selected_indices=None, selected_ratings=None):
    recommended_image_vectors = [vectors[filename] for filename, _ in recommended_images]

    pca = PCA(n_components=2)
    combined_vectors = np.vstack([user_vector] + recommended_image_vectors)
    pca_result = pca.fit_transform(combined_vectors)

    user_pca_result = pca_result[0]
    recommended_pca_result = pca_result[1:]

    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    plt.scatter(recommended_pca_result[:, 0], recommended_pca_result[:, 1], c='b', marker='o', label='Recommended Image Vectors')
    plt.scatter(user_pca_result[0], user_pca_result[1], c='r', marker='s', label='User Vector')
    
    if selected_indices is not None and selected_ratings is not None:
        selected_image_vectors = [list(vectors.values())[idx] for idx in selected_indices]
        selected_pca_result = pca.transform(selected_image_vectors)
        for idx, rating in zip(selected_pca_result, selected_ratings):
            if rating == 1:
                plt.scatter(idx[0], idx[1], c='orange', marker='o', label='Selected Image Vectors')
            elif rating == 2:
                plt.scatter(idx[0], idx[1], c='yellow', marker='o', label='Selected Image Vectors')
            elif rating == 3:
                plt.scatter(idx[0], idx[1], c='green', marker='o', label='Selected Image Vectors')
            elif rating == 4:
                plt.scatter(idx[0], idx[1], c='gray', marker='o', label='Selected Image Vectors')
            else:
                plt.scatter(idx[0], idx[1], c='purple', marker='o', label='Selected Image Vectors')

    plt.title('User Vector and Recommended Image Vectors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()


def recommend_by_uservector(user_id, selections): # 추후에 유저 id 기반으로 db에서 유저 벡터를 가져오도록 수정
    vector_folder = "./vectors"
    vectors = load_vectors(os.path.join(vector_folder, "train"))
    user_vector = load_user_vector(user_id, vector_folder, vectors)

    if user_vector is None:
        user_vector = initialize_user_vector(vectors)

    selected_image_vectors = []
    selected_ratings = []  # 사용자가 선택한 이미지의 선호도 저장
    selected_indices = []  # 사용자가 선택한 이미지의 인덱스 저장
    for image_index, rating in selections:
        selected_vector = list(vectors.values())[image_index]
        selected_image_vectors.append(selected_vector)
        selected_indices.append(image_index)
        selected_ratings.append(rating)

    # Update user vector based on selected image vectors and ratings
    user_vector = update_user_vector(selected_image_vectors, selected_ratings, user_vector)
    
    save_user_vector(user_id, user_vector, vector_folder)

    recommended_images = recommend_images(user_vector, vectors)
    return recommended_images, selected_indices, selected_ratings


user_id = "user5"
selections = [(0, 5),(560,1),(360,1),(200,5),(100,1)]  # Selected image indices and ratings
recommended_images, selected_indices, selected_ratings = recommend_by_uservector(user_id, selections)
show_recommendations(recommended_images)

vectors = load_vectors("./vectors/train")
user_vector = load_user_vector(user_id, "./vectors", vectors)
plot_vectors(user_vector, recommended_images, vectors, selected_indices, selected_ratings)
