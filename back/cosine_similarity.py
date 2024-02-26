from ultralytics import YOLO

from glob import glob
import numpy as np
import os


def calculate_cosine_simularity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    dot_product = np.dot(vector_a, vector_b)
    if dot_product == 0:
        return 0
    
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    try:
        return dot_product / (magnitude_a * magnitude_b)
    except:
        return 0


def count_object(image_path, model):
    target_classes = [24,25,26,39,41,56,58,59,60,62,63,64,66,73,74]
    cls_to_idx = {x: i for i, x in enumerate(target_classes)}

    result = model.predict(image_path, save=False, conf=0.25, classes=target_classes)
    object_counter = [0] * len(target_classes)

    for box in result[0].boxes:
        object_counter[cls_to_idx[box.cls.item()]] += 1

    return object_counter

model = YOLO("yolov8s.pt")
image_dict = dict()

for image_path in glob("./image/*"):
    object_counter = count_object(image_path, model)
    image_dict[os.path.basename(image_path)] = object_counter

base_key, base_value = next(iter(image_dict.items()))
for key, value in image_dict.items():
    print(key)
    print(value)
    print(calculate_cosine_simularity(base_value, value))
    print("_" * 30)