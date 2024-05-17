import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSlot

from image_selector import ImageSelector
from progress_summary import ProgressSummary

from config_loader import config


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AB Test")
        self.setGeometry(100, 100, 400, 400)
        self.showMaximized()

        self.image_selector = ImageSelector(config["PATH"]["train"])
        self.image_selector.clicked.connect(self.handle_answer)
        self.progress_summary = ProgressSummary()

        layout = QVBoxLayout()
        layout.addWidget(self.image_selector)
        layout.addWidget(self.progress_summary)
        self.setLayout(layout)

    @pyqtSlot(bool)
    def handle_answer(self, correct):
        self.progress_summary.add_answer(correct)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
