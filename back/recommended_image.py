from detect import count_class
# from image_classification_run import predict_image_class
# from image_classification_run import get_image_paths
import tensorflow as tf
import os
import random

def recommend_image(image):
    # 이미지 내 클래스 또는 레이블 수 세기
#     class_count = count_class(image)
    
#     # 이미지 모양 분류
#     model = tf.keras.models.Sequential([
#     tf.keras.layers.TFSMLayer('./my_model', call_endpoint='serving_default')
# ])    
#     predictions = predict_image_class(image, model)
#     image_shape = predictions[0]
#     conf = predictions[1]

    # 추천 이미지 리스트 초기화
    recommended_images = []
    shapes = ['h형책상', '독서실책상', '일자형책상', '컴퓨터책상', '코너형책상']
    
    # 해당 모양의 폴더에서 랜덤하게 이미지 선택하여 추천 이미지 리스트에 추가
    shape_folder = "./result"
    shape_folder_path = "./result"
    if os.path.isdir(shape_folder_path):
        images_in_folder = os.listdir(shape_folder_path)
        if len(images_in_folder) >= 3:
            recommended_images.extend(random.sample(images_in_folder, 3))
    
    # 추천 이미지 리턴
    return [os.path.join(shape_folder_path, image_name) for image_name in recommended_images]


# image = "/Users/park_sh/Desktop/backend/back/test_image/h형책상/라인_H형_책상세트_LNDE01.png"

# # 이미지와 함께 recommend_image 함수를 호출하여 추천 이미지 가져오기
# recommended_images = recommend_image(image)

# # 결과 출력
# for img in recommended_images:
#     print("추천된 이미지:", img)
