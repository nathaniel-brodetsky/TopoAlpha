import sys
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QTimer
import ccxt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
plt.style.use('dark_background')

from tda_core import TDAAnalyzer



class TopoAlphaEngine(QMainWindow):
    def __init__(self, symbol='BTC/USDT', timeframe='1m'):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.binance()

        self.prices = []
        self.stress_history = []

        self.tau = 5
        self.dim = 3

        self.tda = TDAAnalyzer()

        self.init_ui()
        self.preload_data()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(2000)

    def init_ui(self):
        self.setWindowTitle(f"TopoAlpha Engine - {self.symbol}")
        self.resize(1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()

        self.plot_price = pg.PlotWidget(title=f"Price: {self.symbol}")
        self.plot_price.showGrid(x=True, y=True)
        self.price_curve = self.plot_price.plot(pen=pg.mkPen('g', width=2))
        left_layout.addWidget(self.plot_price, stretch=2)

        self.plot_stress = pg.PlotWidget(title="Topological Stress (H1 Max Persistence)")
        self.plot_stress.showGrid(x=True, y=True)
        self.stress_curve = self.plot_stress.plot(pen=pg.mkPen('r', width=2), fillLevel=0, brush=(255, 0, 0, 50))
        left_layout.addWidget(self.plot_stress, stretch=1)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        layout.addWidget(left_widget, stretch=1)

        self.fig = plt.figure()
        self.fig.patch.set_facecolor('#000000')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title('3D Market Attractor', color='white')
        self.ax.set_facecolor('#000000')

        layout.addWidget(self.canvas, stretch=1)

    def preload_data(self):
        print(f"Loading history {self.symbol}...")
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=500)
        self.prices = [candle[4] for candle in ohlcv]
        self.stress_history = [0.0] * len(self.prices)
        self.refresh_plots()

    def update_data(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=1)
            self.prices.append(ohlcv[0][4])

            if len(self.prices) > 500:
                self.prices.pop(0)
                if len(self.stress_history) > 500:
                    self.stress_history.pop(0)

            self.refresh_plots()
        except Exception as e:
            print(f"API Error: {e}")

    def get_takens_embedding(self):
        n = len(self.prices)
        if n < (self.dim - 1) * self.tau + 1:
            return np.zeros((1, 3))

        data = np.array(self.prices)
        data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10

        embedded = np.vstack([
            data[:-(2 * self.tau)],
            data[self.tau:-self.tau],
            data[2 * self.tau:]
        ]).T
        return embedded

    def refresh_plots(self):
        self.price_curve.setData(self.prices)

        embedded_data = self.get_takens_embedding()

        if len(embedded_data) > 0:
            stress = self.tda.get_topological_stress(embedded_data[-50:])

            if len(self.stress_history) < len(self.prices):
                self.stress_history.append(stress)
            else:
                self.stress_history[-1] = stress

            self.stress_curve.setData(self.stress_history)

            self.ax.clear()
            self.ax.set_title('3D Market Attractor', color='white')
            self.ax.grid(True, color='#333333')

            c = np.linspace(0, 1, len(embedded_data))
            self.ax.scatter(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2],
                            c=c, cmap='cool', s=10, alpha=0.8)

            self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    engine = TopoAlphaEngine()
    engine.show()
    sys.exit(app.exec())