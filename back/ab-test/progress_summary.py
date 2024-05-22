from PyQt5.QtWidgets import QWidget, QVBoxLayout, QProgressBar
from plot_view import PlotView


class ProgressBar(QProgressBar):
    def __init__(self, *args):
        super().__init__(*args)
        self.float_value = 0
        self.valueChanged.connect(self.onValueChanged)

    def onValueChanged(self):
        self.setFormat("%.02f%%" % self.float_value)

    def setValue(self, value):
        self.float_value = value
        super().setValue(int(value))


class ProgressSummary(QWidget):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.correct_count = 0

        self.progress_bar = ProgressBar()
        self.plot_view = PlotView()

        layout = QVBoxLayout()
        layout.addWidget(self.progress_bar)
        # layout.addWidget(self.plot_view)
        self.setLayout(layout)

    def add_answer(self, correct):
        self.total += 1
        if correct:
            self.correct_count += 1

        accuracy = self.correct_count / self.total * 100
        self.progress_bar.setValue(accuracy)
        self.plot_view.append_y(accuracy)