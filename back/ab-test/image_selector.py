import os
import random
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
import faiss
from image_label import ImageLabel
from img2vec import Feat2Vec


class ImageSelector(QWidget):
    clicked = pyqtSignal(bool)

    def __init__(self, image_dir):
        super().__init__()
        self.image_dir = image_dir
        self.correct_on_left = True

        self.feat2vec = Feat2Vec()
        self.feat_idx = faiss.read_index("vectors/vgg_features.index")

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
        query_image_path = self.choice_random_image_path()

        feat_vec = self.feat2vec.get_vector(query_image_path)
        _, feat_result = self.feat_idx.search(feat_vec, 1)

        output_image_name = os.listdir(self.image_dir)[feat_result[0][0]]
        output_image_path = os.path.join(self.image_dir, output_image_name)

        while True:
            random_image_path = self.choice_random_image_path()
            if (
                random_image_path != query_image_path
                and random_image_path != output_image_path
            ):
                break

        self.query_image_label.set_pixmap(query_image_path)
        self.output_image_label.set_pixmap(output_image_path)
        self.random_image_label.set_pixmap(random_image_path)

        self.switch_image_position()
    
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
