import sys
import os
import random

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config_loader import config


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('AB Test')
        self.setGeometry(100, 100, 400, 400)
        

        layout = QHBoxLayout()
        for image_path in random.sample(os.listdir(config["PATH"]["train"]), 3):
            image_label = QLabel()

            image_path = os.path.join(config["PATH"]["train"], image_path)
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)

            image_label.setPixmap(pixmap)
            image_label.mousePressEvent = lambda event: self.on_image_click(event, image_path)
            layout.addWidget(image_label, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def on_image_click(self, event, image_path):
        print(image_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
