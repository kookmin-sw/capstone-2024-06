from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import requests
from PyQt5.QtCore import QThread, pyqtSignal

class DownloadThread(QThread):
    finished = pyqtSignal(bytes)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            self.finished.emit(response.content)
        except requests.RequestException as e:
            print(f"Error: {e}")

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(500, 500)
        self.setStyleSheet("border: 2px solid black")
        self.setAlignment(Qt.AlignCenter)

    def set_pixmap(self, url):
        self.download_thread = DownloadThread(url + "?w=500&h=500&q=80")
        self.download_thread.finished.connect(self.on_download_finished)
        self.download_thread.start()
    
    def on_download_finished(self, data):
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        pixmap = pixmap.scaled(500, 500, aspectRatioMode=True)
        self.setPixmap(pixmap)