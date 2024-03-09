import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 이미지를 0~1 범위로 정규화
    return img_array

def predict_image_class(image_path, model):
    preprocessed_img = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction[0][predicted_class]

def predict_images(image_paths, model):
    predictions = []
    for image_path in image_paths:
        predicted_class, confidence = predict_image_class(image_path, model)
        predictions.append((image_path, predicted_class, confidence))
    return predictions

def get_image_paths(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpeg', '.jpg', '.png')):  
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

# 모델 로드
model = tf.keras.models.load_model("back/my_model")

# {'h형책상': 0, '독서실책상': 1, '일자형책상': 2, '컴퓨터책상': 3, '코너형책상': 4}
# reset check
image_folder_path = "/Users/park_sh/Desktop/backend/back/test_image/독서실책상"  # 대상 이미지 폴더
image_paths = get_image_paths(image_folder_path)
predictions = predict_images(image_paths, model)

# 예측 결과 출력
for image_path, predicted_class, confidence in predictions:
    print("Image:", image_path)
    print("Predicted Class:", predicted_class)
    print("Confidence:", confidence)
    print()
