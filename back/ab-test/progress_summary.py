from PyQt5.QtWidgets import QWidget, QVBoxLayout, QProgressBar
from plot_view import PlotView


class ProgressSummary(QWidget):
    def __init__(self):
        super().__init__()
        self.answers = []
    
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(10000)
        # self.progress_bar.setFormat("%.02f%%" % (self.step))
        self.progress_bar.setValue(7253)

        self.plot_view = PlotView()

        layout = QVBoxLayout()
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.plot_view)
        self.setLayout(layout)
    
    def add_answer(self, correct):
        self.answers.append(correct)
        print(self.answers)