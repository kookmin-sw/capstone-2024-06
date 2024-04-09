import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from PIL import Image
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

class ImageRecommender:
    def __init__(self, image_folder, vector_folder):
        self.image_folder = image_folder
        self.vector_folder = vector_folder
        self.image_files = os.listdir(image_folder)
        self.reader = Reader(line_format='user item rating', rating_scale=(0, 1))
        self.df = self.prepare_data()
        self.dataset = Dataset.load_from_df(self.df, self.reader)
        self.trainset, self.testset = train_test_split(self.dataset, test_size=0.2, random_state=42)
        self.model = SVD()

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

    def train_model(self):
        self.model.fit(self.trainset)

    def predict_ratings(self, user_vectors):
        predictions = []
        seen_images = set()  # 중복된 이미지를 필터링하기 위한 집합
        for user_vector in user_vectors:
            for image_file in self.image_files:
                if image_file not in seen_images:
                    vector_file = os.path.join(self.vector_folder, f"{os.path.splitext(image_file)[0]}.npy")
                    image_vector = np.load(vector_file)
                    predicted_rating = self.model.predict('user1', image_file).est
                    predictions.append((image_file, predicted_rating))
                    seen_images.add(image_file)
        return predictions

    def show_recommendations(self, predictions):
        predictions.sort(key=lambda x: x[1], reverse=True)
        num_recommendations = min(len(predictions), 5)
        for i, (image_file, rating) in enumerate(predictions[:num_recommendations], start=1):
            image_path = os.path.join(self.image_folder, image_file)
            image = Image.open(image_path)
            plt.imshow(image)
            plt.title(f"Rank {i}, Rating: {rating:.2f}")
            plt.axis("off")
            plt.show()

    def save_user_vector(self, user_vectors):
        with open("./vectors/user/user1.npy", "wb") as f:
            np.save(f, np.array(user_vectors))

    def load_saved_user_vector(self):
        user_vectors = []
        if os.path.exists("./vectors/user/user1.npy"):
            with open("./vectors/user/user1.npy", "rb") as f:
                user_vectors = np.load(f)
        return user_vectors.tolist()

    def run(self):
        select_method = self.choose_selection_method()
        selected_images, ratings = select_method()
        user_vectors = self.load_user_vector(selected_images)
        self.train_model()
        predictions = self.predict_ratings(user_vectors)
        self.show_recommendations(predictions)
        self.save_user_vector(user_vectors)

# 이미지 추천기 객체 생성
image_recommender = ImageRecommender("./images/train", "./vectors/train")
# 이미지 추천 실행
image_recommender.run()
