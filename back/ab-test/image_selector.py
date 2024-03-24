import os
from random import sample
from PyQt5.QtWidgets import QWidget, QHBoxLayout
from image_label import ImageLabel


class ImageSelector(QWidget):
    def __init__(self, image_dir):
        super().__init__()
        self.image_dir = image_dir
        self.load_image()
    
    def load_image(self):
        layout = QHBoxLayout()
        for image_filename in sample(os.listdir(self.image_dir), 3):
            image_path = os.path.join(self.image_dir, image_filename)
            image_label = ImageLabel(image_path)
            layout.addWidget(image_label)
        self.setLayout(layout)