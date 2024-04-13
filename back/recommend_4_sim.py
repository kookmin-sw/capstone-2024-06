import os
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

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

def load_user_vector(user_id, vector_folder):
    file_path = os.path.join(vector_folder, "user", f"{user_id}.npy")
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        return None

def initialize_user_vector(vectors):
    random_image_vector = random.choice(list(vectors.values()))
    return random_image_vector

def update_user_vector(selected_vector, rating, user_vector):
    # Define weight based on rating
    if rating == 3:
        weight = 1.0  # No change if rating is neutral (3)
    elif rating <= 2:
        weight = 0.5  # Reduce the impact of selected vector if rating is low
    else:
        weight = 1.5  # Increase the impact of selected vector if rating is high

    # Update user vector with weighted selected vector
    updated_user_vector = user_vector + weight * (selected_vector - user_vector)
    return updated_user_vector

def recommend_images(user_vector, vectors):
    # Convert dictionary to list of vectors and filenames
    vector_list = list(vectors.values())

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)  # Dimensionality set to 2 for visualization
    reduced_vectors = pca.fit_transform(vector_list)
    reduced_user_vector = pca.transform(user_vector.reshape(1, -1))

    # Initialize nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=4, algorithm='auto', metric='euclidean')
    nn_model.fit(reduced_vectors)

    # Search for similar images
    distances, indices = nn_model.kneighbors(reduced_user_vector)

    # Exclude selected image from recommendations
    recommended_images = []
    for index, distance in zip(indices[0], distances[0]):
        image_file = list(vectors.keys())[index]
        similarity = 1 / (1 + distance)  # Similarity based on distance
        recommended_images.append((image_file, similarity))
    return recommended_images

def show_recommendations(recommendations):
    num_recommendations = min(len(recommendations), 3)
    for i, (image_file, similarity) in enumerate(recommendations[:num_recommendations], start=1):
        print(f"이미지: {image_file}, 유사도: {similarity:.2f}")

def plot_vectors(reduced_vectors, reduced_user_vector, selected_image_vectors):
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='k', label='Image Vectors')
    plt.scatter(reduced_user_vector[:, 0], reduced_user_vector[:, 1], c='r', label='User Vector')
    for vector in selected_image_vectors:
        plt.scatter(vector[0], vector[1], c='b', label='Selected Image Vectors')
    plt.title('User Vector and Image Vectors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def recommend_by_uservector(user_id, selections):
    vector_folder = "./vectors"
    vectors = load_vectors(os.path.join(vector_folder, "train"))
    user_vector = load_user_vector(user_id, vector_folder)

    if user_vector is None:
        user_vector = initialize_user_vector(vectors)

    # Update user vector based on selections
    selected_image_vectors = []
    for image_index, rating in selections:
        selected_vector = list(vectors.values())[image_index]
        selected_image_vectors.append(selected_vector)
        user_vector = update_user_vector(selected_vector, rating, user_vector)
    
    save_user_vector(user_id, user_vector, vector_folder)

    # Recommend images
    recommended_images = recommend_images(user_vector, vectors)

    # Return recommended images with similarities
    return recommended_images

# # Test the recommendation system
# user_id = "user4"
# selections = [(0, 5), (2, 5)]  # Selected image indices and ratings
# recommended_images = recommend_by_uservector(user_id, selections)
# show_recommendations(recommended_images)
