import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader

class ImageRecommender:
    def __init__(self, image_folder, vector_folder):
        self.image_folder = image_folder
        self.vector_folder = vector_folder
        self.image_files = os.listdir(image_folder)
        self.reader = Reader(line_format='user item rating', rating_scale=(0, 1))
        self.df = self.prepare_data()
        self.dataset = Dataset.load_from_df(self.df, self.reader)
        self.trainset = self.dataset.build_full_trainset()

    def prepare_data(self):
        data = [('user1', image_file, 0) for image_file in self.image_files]
        return pd.DataFrame(data, columns=['user', 'item', 'rating'])

    def choose_selection_method(self):
        method = input("이미지 선택 방법을 선택하세요 (1: 한 장씩 선택, 2: 3개 중 하나 선택): ")
        if method == '1':
            return self.select_image_multiple_times
        elif method == '2':
            return self.select_images
        else:
            print("잘못된 선택입니다. 다시 선택해주세요.")
            return self.choose_selection_method()

    def select_images(self):
        selected_images = []
        ratings = []

        for _ in range(3):
            print("\n이미지를 선택하세요:")
            selected_files = np.random.choice(self.image_files, size=3, replace=False)
            fig, axes = plt.subplots(1, 3)
            for i, (image_file, ax) in enumerate(zip(selected_files, axes), start=1):
                image_path = os.path.join(self.image_folder, image_file)
                image = Image.open(image_path)
                ax.imshow(image)
                ax.set_title(f"Image {i}")
                ax.axis("off")
            plt.show()
            selected_index = int(input("선택한 이미지 번호를 입력하세요 (1, 2, 3 중 하나): ")) - 1

            for i in range(3):
                rating = 1 if i == selected_index else 0
                ratings.append(rating)

            selected_images.extend(selected_files)

        for image_file, rating in zip(selected_images, ratings):
            self.df.loc[self.df['item'] == image_file, 'rating'] = rating

        return selected_images, ratings

    def select_image_multiple_times(self):
        selected_images = []
        ratings = []
        for _ in range(3):
            print("\n이미지를 선택하세요:")
            selected_file = np.random.choice(self.image_files)
            image_path = os.path.join(self.image_folder, selected_file)
            image = Image.open(image_path)
            plt.imshow(image)
            plt.title("Selected Image")
            plt.axis("off")
            plt.show()
            rating = int(input("선택한 이미지의 선호도를 1부터 5까지 입력하세요: "))
            selected_images.append(selected_file)
            ratings.append(rating / 5)

        return selected_images, ratings

    def load_user_vector(self, selected_images):
        user_vectors = []
        for image_file in selected_images:
            vector_file = os.path.join(self.vector_folder, f"{os.path.splitext(image_file)[0]}.npy")
            user_vector = np.load(vector_file)
            user_vectors.append(user_vector)
        return user_vectors

    def recommend_images(self, selected_images, user_vectors):
        # Calculate the average user vector
        user_vector = np.mean(user_vectors, axis=0)

        # Load all image vectors
        all_image_vectors = []
        for image_file in self.image_files:
            vector_file = os.path.join(self.vector_folder, f"{os.path.splitext(image_file)[0]}.npy")
            image_vector = np.load(vector_file)
            all_image_vectors.append(image_vector)
        all_image_vectors = np.array(all_image_vectors)

        # Calculate distances between the user's average vector and all image vectors
        distances = np.linalg.norm(all_image_vectors - user_vector, axis=1)

        # Calculate the mean distance
        mean_distance = np.mean(distances)

        # Initialize nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=4, algorithm='auto', metric='euclidean')
        nn_model.fit(all_image_vectors)

        # Search for similar images
        distances, indices = nn_model.kneighbors([user_vector])

        # Exclude selected images from recommendations and calculate similarity based on distance from the mean
        recommended_images = []
        for index, distance in zip(indices[0], distances[0]):
            image_file = self.image_files[index]
            if image_file not in selected_images:
                similarity = 1 / (1 + abs(mean_distance - distance) / mean_distance)
                recommended_images.append((image_file, similarity))
        return recommended_images

    def show_recommendations(self, recommendations):
        num_recommendations = min(len(recommendations), 3)
        fig, axes = plt.subplots(1, 3, figsize=(20, 4))
        for i, (image_file, similarity) in enumerate(recommendations[:num_recommendations], start=1):
            image_path = os.path.join(self.image_folder, image_file)
            image = Image.open(image_path)
            axes[i-1].imshow(image)
            axes[i-1].set_title(f"Similarity: {similarity:.2f}")
            axes[i-1].axis("off")
        for j in range(num_recommendations, 3):
            axes[j].axis("off")
            axes[j].text(0.5, 0.5, f"선호도: {recommendations[j][1]:.2f}", ha='center', va='center', fontsize=12, color='gray')
        plt.tight_layout()
        plt.show()

    def run(self):
        select_method = self.choose_selection_method()
        selected_images, ratings = select_method()
        user_vectors = self.load_user_vector(selected_images)
        recommendations = self.recommend_images(selected_images, user_vectors)
        self.show_recommendations(recommendations)
        
        # Visualize selected images and their average vector
        avg_user_vector = np.mean(user_vectors, axis=0)
        all_image_vectors = np.array([np.load(os.path.join(self.vector_folder, f"{os.path.splitext(image_file)[0]}.npy")) for image_file in self.image_files])
        
        plt.figure(figsize=(8, 6))
        for image_vector, image_file in zip(all_image_vectors, self.image_files):
            plt.scatter(image_vector[0], image_vector[1], color='black')
        plt.scatter(avg_user_vector[0], avg_user_vector[1], color='red', label='Average User Vector')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Feature Space')
        plt.legend()
        plt.grid(True)
        plt.show()
# 이미지 추천기 객체 생성
image_recommender = ImageRecommender("./images/train", "./vectors/train")
# 이미지 추천 실행
image_recommender.run()
