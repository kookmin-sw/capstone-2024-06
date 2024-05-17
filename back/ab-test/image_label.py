from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(500, 500)
        self.setStyleSheet("border: 2px solid black")
        self.setAlignment(Qt.AlignCenter)

    def set_pixmap(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(500, 500, aspectRatioMode=True)
        self.setPixmap(pixmap)