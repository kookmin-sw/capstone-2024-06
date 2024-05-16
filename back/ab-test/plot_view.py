from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go


class PlotView(QWebEngineView):
    def __init__(self):
        super().__init__()

        self.y = []
        self.fig = go.FigureWidget()
        self.fig.add_scatter(y=self.y)
        
        self.update_view()

    def append_y(self, y):
        self.y.append(y)
        self.update_view()

    def update_view(self):
        self.fig.data[0].y = self.y
        self.setHtml(self.fig.to_html(include_plotlyjs="cdn"))