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

def update_user_vector(selected_image_vectors, user_vector):
    # 선택한 이미지들의 평균 벡터 계산
    mean_selected_vector = np.mean(selected_image_vectors, axis=0)

    # 사용자 벡터와 평균 벡터의 중간값 계산
    updated_user_vector = 0.5 * (user_vector + mean_selected_vector)

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

def plot_vectors(user_vector, recommended_images, vectors):
    recommended_image_vectors = [vectors[filename] for filename, _ in recommended_images]

    pca = PCA(n_components=2)
    combined_vectors = np.vstack([user_vector] + recommended_image_vectors)
    pca_result = pca.fit_transform(combined_vectors)

    user_pca_result = pca_result[0]
    recommended_pca_result = pca_result[1:]

    plt.figure(figsize=(10, 6))
    plt.scatter(recommended_pca_result[:, 0], recommended_pca_result[:, 1], c='b', marker='o', label='Recommended Image Vectors')
    plt.scatter(user_pca_result[0], user_pca_result[1], c='r', marker='s', label='User Vector')
    plt.title('User Vector and Recommended Image Vectors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def recommend_by_uservector(user_id, selections):
    vector_folder = "./vectors"
    vectors = load_vectors(os.path.join(vector_folder, "train"))
    user_vector = load_user_vector(user_id, vector_folder, vectors)

    if user_vector is None:
        user_vector = initialize_user_vector(vectors)

    selected_image_vectors = []
    for image_index, rating in selections:
        selected_vector = list(vectors.values())[image_index]
        selected_image_vectors.append(selected_vector)

    # Update user vector based on selected image vectors
    user_vector = update_user_vector(selected_image_vectors, user_vector)
    
    save_user_vector(user_id, user_vector, vector_folder)

    recommended_images = recommend_images(user_vector, vectors)
    return recommended_images

user_id = "user5"
selections = [(0, 1),(5,5),(660,5),(770,5),(880,5),(990,5),(200,5),(100,5)]  # Selected image indices and ratings
recommended_images = recommend_by_uservector(user_id, selections)
show_recommendations(recommended_images)

vectors = load_vectors("./vectors/train")
user_vector = load_user_vector(user_id, "./vectors", vectors)
plot_vectors(user_vector, recommended_images, vectors)
