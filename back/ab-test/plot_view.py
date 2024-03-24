from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.express as px


class PlotView(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)

        df = px.data.tips()
        fig = px.box(df, x="day", y="total_bill", color="smoker")
        fig.update_traces(quartilemethod="inclusive")

        self.setHtml(fig.to_html(include_plotlyjs="cdn"))
