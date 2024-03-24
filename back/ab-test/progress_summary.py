from PyQt5.QtWidgets import QWidget, QVBoxLayout, QProgressBar
from plot_view import PlotView


class ProgressSummary(QWidget):
    def __init__(self):
        super().__init__()
    
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(50)
        self.plot_view = PlotView()

        layout = QVBoxLayout()
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.plot_view)
        self.setLayout(layout)