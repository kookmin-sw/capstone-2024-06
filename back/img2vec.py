import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from config_loader import config


class Img2Vec:
    def get_vector(self, image_path):
        self.get_vectors(self, [image_path])

    def get_vectors(self, image_paths): ...


class FeatureExtractor(Img2Vec):
    def __init__(self, batch_size=32, verbose=False):
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = torch.device("cude" if torch.cuda.is_available() else "cpu")

        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.model.classifier = self.model.classifier[:-1]
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_vectors(self, image_paths):
        iterator = range(0, len(image_paths), self.batch_size)
        if self.verbose:
            iterator = tqdm(iterator)

        vectors = []
        for i in iterator:
            batch_image_paths = image_paths[i : i + self.batch_size]
            batch_images = torch.stack(
                [
                    self.preprocess(Image.open(path).convert("RGB")).to(self.device)
                    for path in batch_image_paths
                ]
            )

            with torch.no_grad():
                batch_vectors = self.model(batch_images)
            vectors.extend(batch_vectors)

        vectors = np.array(vectors)
        vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        return vectors


class ObjectCounter(Img2Vec):
    def __init__(self, conf_threshold=0.25, verbose=False):
        self.conf_threshold = conf_threshold
        self.verbose = verbose
        self.model = YOLO("yolov8x.pt")
        self.classes = [41, 56, 58, 59, 60, 62, 63, 64, 66, 73, 74, 75]
        self.cls_to_idx = {x: i for i, x in enumerate(self.classes)}

    def get_vectors(self, image_paths):
        iterator = image_paths
        if self.verbose:
            iterator = tqdm(iterator)
        
        vectors = []
        for image_path in iterator:
            result = self.model(
                image_path, conf=self.conf_threshold, classes=self.classes, verbose=False
            )
            counter = [0] * len(self.classes)

            for box in result[0].boxes:
                counter[self.cls_to_idx[box.cls.item()]] += 1
                vectors.append(counter)

        vectors = np.array(vectors, dtype=np.float32)
        return vectors


if __name__ == "__main__":

    def save_vectors(vectors, file_path):
        d = vectors.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(vectors)
        faiss.write_index(index, file_path)

    
    # how to use feature extrator
    feature_extractor = FeatureExtractor(verbose=True)

    image_dir = config["PATH"]["train"]
    image_paths = [
        os.path.join(image_dir, file_name) for file_name in os.listdir(image_dir)
    ]
    vectors = feature_extractor.get_vectors(image_paths)
    save_vectors(vectors, "vectors/vgg_features.index")


    # how to use object counter
    object_counter = ObjectCounter(verbose=True)

    image_dir = config["PATH"]["train"]
    image_paths = [
        os.path.join(image_dir, file_name) for file_name in os.listdir(image_dir)
    ]
    vectors = object_counter.get_vectors(image_paths)
    print(vectors)
    save_vectors(vectors, "vectors/object_counts.index")