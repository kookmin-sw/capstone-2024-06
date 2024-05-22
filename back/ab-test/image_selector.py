import os
import random
import numpy as np
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
import torchvision.models as models
from image_label import ImageLabel

from database import crud
from dependencies import *
from img2vec import Feat2Vec


class ImageSelector(QWidget):
    clicked = pyqtSignal(bool)

    def __init__(self, image_dir):
        super().__init__()
        self.image_dir = image_dir
        self.correct_on_left = True

        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier = model.classifier[:-1]
        self.feat2vec = Feat2Vec(model, resize=(224, 224), from_url=True)
        self.feature_mat = np.load("vectors/features.npy")

        self.query_image_label = ImageLabel()
        query_label = QLabel("Query")
        query_layout = QVBoxLayout()
        query_label.setAlignment(Qt.AlignCenter)
        query_layout.addWidget(query_label)
        query_layout.addWidget(self.query_image_label)

        self.output_image_label = ImageLabel()
        self.output_image_label.mousePressEvent = lambda event: self.image_clicked(True)
        self.random_image_label = ImageLabel()
        self.random_image_label.mousePressEvent = lambda event: self.image_clicked(False)

        layout = QHBoxLayout()
        layout.addLayout(query_layout)
        layout.addWidget(self.output_image_label)
        layout.addWidget(self.random_image_label)
        self.setLayout(layout)

        self.load_image()
    
    def image_clicked(self, correct):
        self.load_image()
        self.clicked.emit(correct)

    def load_image(self):
        db = next(get_db())
        query_image, random_image = crud.read_random_design_images_sync(db, 2)

        query_vec = self.feat2vec.get_vector(query_image.src_url)
        query_mat = query_vec.reshape(1, -1)
        similarity_mat = query_mat @ self.feature_mat.T
        output_image = crud.read_design_images_sync(db, int(similarity_mat[0].argsort()[-2]))

        self.query_image_label.set_pixmap(query_image.src_url)
        self.output_image_label.set_pixmap(output_image.src_url)
        self.random_image_label.set_pixmap(random_image.src_url)

        self.switch_image_position()
        db.close()

    def choice_random_image_path(self):
        image_name = random.choice(os.listdir(self.image_dir))
        return os.path.join(self.image_dir, image_name)

    def switch_image_position(self):
        if random.random() < 0.5:
            target_label = (
                self.output_image_label
                if self.correct_on_left
                else self.random_image_label
            )

            layout = self.layout()
            layout.removeWidget(target_label)
            layout.addWidget(target_label)
            self.correct_on_left = False if self.correct_on_left else True
