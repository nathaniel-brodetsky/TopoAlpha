import sys
import os
import numpy as np
import pyqtgraph as pg
import matplotlib
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

os.environ["QT_QPA_PLATFORM"] = "xcb"
matplotlib.use('Qt5Agg')
plt.style.use('dark_background')

from data_feeder import RobustDataFeeder
from tda_core import TDAAnalyzer
from ml_model import TopoBooster
from paper_trader import PaperTrader


class WorkerThread(QThread):
    data_ready = pyqtSignal(list, list, float, np.ndarray, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, feeder, tda, ml, symbol, timeframe, tau, dim, prices, timestamps, stress_history):
        super().__init__()
        self.feeder = feeder
        self.tda = tda
        self.ml = ml
        self.symbol = symbol
        self.timeframe = timeframe
        self.tau = tau
        self.dim = dim
        self.prices = prices
        self.timestamps = timestamps
        self.stress_history = stress_history
        self.running = True
        self.prob_history = []

    def run(self):
        while self.running:
            try:
                ohlcv = self.feeder.fetch_updates()
                if ohlcv:
                    for candle in ohlcv:
                        t, o, h, l, c, v = candle
                        if len(self.timestamps) == 0 or t > self.timestamps[-1]:
                            self.timestamps.append(t)
                            self.prices.append(c)
                            self.stress_history.append(self.stress_history[-1] if self.stress_history else 0.0)
                        elif t == self.timestamps[-1]:
                            self.prices[-1] = c

                    if len(self.prices) > 500:
                        self.timestamps = self.timestamps[-500:]
                        self.prices = self.prices[-500:]
                        self.stress_history = self.stress_history[-500:]

                    embedded_data = self.get_takens_embedding()
                    current_stress = 0.0

                    if len(embedded_data) > 0:
                        current_stress = self.tda.get_topological_stress(embedded_data[-50:])
                        self.stress_history[-1] = current_stress

                        prob_up = 0.5
                        if self.ml.is_trained:
                            raw_prob = self.ml.predict(self.prices, self.stress_history)
                            self.prob_history.append(raw_prob)
                            if len(self.prob_history) > 5:
                                self.prob_history.pop(0)
                            prob_up = sum(self.prob_history) / len(self.prob_history)

                        self.data_ready.emit(self.prices, self.stress_history, prob_up, embedded_data, current_stress)

            except Exception as e:
                self.error_occurred.emit(str(e))

            self.msleep(2000)

    def get_takens_embedding(self):
        n = len(self.prices)
        if n < (self.dim - 1) * self.tau + 1:
            return np.zeros((1, 3))
        data = np.array(self.prices)
        data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
        embedded = np.vstack([data[:-(2 * self.tau)], data[self.tau:-self.tau], data[2 * self.tau:]]).T
        return embedded

    def stop(self):
        self.running = False
        self.wait()


class TopoAlphaEngine(QMainWindow):
    def __init__(self, symbol='BTC/USDT', timeframe='1m'):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.feeder = RobustDataFeeder(symbol, timeframe)
        self.timestamps = []
        self.prices = []
        self.stress_history = []
        self.tau = 5
        self.dim = 3
        self.tda = TDAAnalyzer()
        self.ml = TopoBooster()
        self.trader = PaperTrader(initial_balance=10000.0, horizon=5)

        self.alpha_stress_threshold = 1.5
        self.active_trade_horizon = None

        self.init_ui()
        self.preload_data()

        self.worker = WorkerThread(self.feeder, self.tda, self.ml, self.symbol, self.timeframe,
                                   self.tau, self.dim, self.prices, self.timestamps, self.stress_history)
        self.worker.data_ready.connect(self.update_ui)
        self.worker.start()

    def init_ui(self):
        self.setWindowTitle(f"TopoAlpha Engine - {self.symbol} | Quant Fund Mode")
        self.resize(1600, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()

        self.portfolio_label = QLabel("BALANCE: $10000.00 | PNL: 0.00% | POS: NONE")
        self.portfolio_label.setStyleSheet(
            "color: #00FFFF; font-size: 20px; font-weight: bold; padding: 5px; background-color: #001122; border-radius: 5px;")
        self.portfolio_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.portfolio_label)

        self.signal_label = QLabel("ML STATUS: INITIALIZING...")
        self.signal_label.setStyleSheet(
            "color: #888888; font-size: 24px; font-weight: bold; padding: 10px; background-color: #111111; border-radius: 5px;")
        self.signal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.signal_label)

        self.plot_price = pg.PlotWidget(title=f"Live Price: {self.symbol}")
        self.plot_price.showGrid(x=True, y=True)
        self.price_curve = self.plot_price.plot(pen=pg.mkPen('g', width=2))

        self.horizon_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', width=2, style=Qt.DashLine))
        self.horizon_line.hide()
        self.plot_price.addItem(self.horizon_line)

        left_layout.addWidget(self.plot_price, stretch=2)

        self.buy_scatter = pg.ScatterPlotItem(size=14, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 255),
                                              symbol='t1')
        self.sell_scatter = pg.ScatterPlotItem(size=14, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255),
                                               symbol='t')
        self.plot_price.addItem(self.buy_scatter)
        self.plot_price.addItem(self.sell_scatter)

        self.buy_markers_x = []
        self.buy_markers_y = []
        self.sell_markers_x = []
        self.sell_markers_y = []

        self.plot_stress = pg.PlotWidget(title="Live Topological Stress (Alpha Finder)")
        self.plot_stress.showGrid(x=True, y=True)
        self.stress_curve = self.plot_stress.plot(pen=pg.mkPen('r', width=2), fillLevel=0, brush=(255, 0, 0, 50))

        self.stress_threshold_line = pg.InfiniteLine(pos=self.alpha_stress_threshold, angle=0,
                                                     pen=pg.mkPen('m', width=2, style=Qt.DashLine))
        self.plot_stress.addItem(self.stress_threshold_line)

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
        ohlcv = self.feeder.fetch_initial(limit=500)
        for candle in ohlcv:
            self.timestamps.append(candle[0])
            self.prices.append(candle[4])
            self.stress_history.append(0.0)

        for i in range(350, len(self.prices)):
            window = self.prices[i - 50:i]
            data = np.array(window)
            data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
            embedded = np.vstack([data[:-(2 * self.tau)], data[self.tau:-self.tau], data[2 * self.tau:]]).T
            if len(embedded) > 0:
                self.stress_history[i] = self.tda.get_topological_stress(embedded)

        self.ml.train(self.prices, self.stress_history)

    def update_ui(self, prices, stress_history, prob_up, embedded_data, current_stress):
        self.price_curve.setData(prices)
        self.stress_curve.setData(stress_history)

        current_x = len(prices) - 1
        current_price = prices[-1]

        trade_result = self.trader.update(current_price, current_x)
        if trade_result:
            self.active_trade_horizon = None
            self.horizon_line.hide()
            print(f"TRADE CLOSED: {trade_result}")

        unrealized_pnl = self.trader.get_unrealized_pnl(current_price)
        pos_str = self.trader.position if self.trader.position else "NONE"
        pnl_color = "#00FF00" if unrealized_pnl >= 0 else "#FF0000"

        self.portfolio_label.setText(
            f"BALANCE: ${self.trader.balance:.2f} | PNL: <font color='{pnl_color}'>{unrealized_pnl:.3f}%</font> | POS: {pos_str}")

        if current_stress >= self.alpha_stress_threshold:
            alpha_status = "🔥 TOPOLOGY BROKEN! ALPHA FOUND!"
            bg_color = "#330033"
        else:
            alpha_status = "Topology Stable."
            bg_color = "#111111"

        if prob_up >= 0.60 and current_stress >= self.alpha_stress_threshold and self.trader.position is None:
            self.signal_label.setText(f"🚀 EXECUTING LONG (UP Prob: {prob_up:.1%}) | {alpha_status}")
            self.signal_label.setStyleSheet(
                f"color: #00FF00; font-size: 24px; font-weight: bold; background-color: {bg_color};")

            if self.trader.execute_trade('LONG', current_price, current_x):
                self.buy_markers_x.append(current_x)
                self.buy_markers_y.append(current_price * 0.9995)
                self.active_trade_horizon = current_x + 5
                self.horizon_line.setPos(self.active_trade_horizon)
                self.horizon_line.show()

        elif prob_up <= 0.40 and current_stress >= self.alpha_stress_threshold and self.trader.position is None:
            self.signal_label.setText(f"🩸 EXECUTING SHORT (UP Prob: {prob_up:.1%}) | {alpha_status}")
            self.signal_label.setStyleSheet(
                f"color: #FF0000; font-size: 24px; font-weight: bold; background-color: {bg_color};")

            if self.trader.execute_trade('SHORT', current_price, current_x):
                self.sell_markers_x.append(current_x)
                self.sell_markers_y.append(current_price * 1.0005)
                self.active_trade_horizon = current_x + 5
                self.horizon_line.setPos(self.active_trade_horizon)
                self.horizon_line.show()

        else:
            self.signal_label.setText(f"⚖️ WAITING FOR ALPHA (UP Prob: {prob_up:.1%}) | {alpha_status}")
            self.signal_label.setStyleSheet(
                f"color: #FFFF00; font-size: 24px; font-weight: bold; background-color: {bg_color};")

        self.buy_scatter.setData(self.buy_markers_x, self.buy_markers_y)
        self.sell_scatter.setData(self.sell_markers_x, self.sell_markers_y)

        self.ax.clear()
        self.ax.set_title('3D Market Attractor', color='white')
        self.ax.grid(True, color='#333333')
        if len(embedded_data) > 0:
            c = np.linspace(0, 1, len(embedded_data))
            self.ax.scatter(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], c=c, cmap='cool', s=10,
                            alpha=0.8)
        self.canvas.draw()

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    engine = TopoAlphaEngine()
    engine.show()
    sys.exit(app.exec())