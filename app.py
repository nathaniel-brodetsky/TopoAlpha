import sys
import logging
import numpy as np
import pyqtgraph as pg
import matplotlib
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use('Qt5Agg')
plt.style.use('dark_background')

from data_feeder import RobustDataFeeder
from tda_core import TDAAnalyzer
from ml_model import TopoBooster
from paper_trader import PaperTrader

logging.basicConfig(level=logging.INFO, format='%(asctime)s[%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("TopoAlpha")


class MarketStreamer(QThread):
    data_ready = pyqtSignal(list, list, list, float, np.ndarray, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, feeder, tda, ml, tau, dim, prices, timestamps, stress_history):
        super().__init__()
        self.feeder = feeder
        self.tda = tda
        self.ml = ml
        self.tau = tau
        self.dim = dim
        self.prices = prices
        self.timestamps = timestamps
        self.stress_history = stress_history
        self.running = True
        self.prob_history = []

    def run(self):
        logger.info("MarketStreamer started. Listening for live ticks...")
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

                    embedded_data = self._get_embedding()
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

                        self.data_ready.emit(self.timestamps, self.prices, self.stress_history, prob_up, embedded_data,
                                             current_stress)

            except Exception as e:
                self.error_occurred.emit(str(e))
            self.msleep(2000)

    def _get_embedding(self):
        n = len(self.prices)
        if n < (self.dim - 1) * self.tau + 1:
            return np.zeros((1, 3))
        data = np.array(self.prices)
        data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
        return np.vstack([data[:-(2 * self.tau)], data[self.tau:-self.tau], data[2 * self.tau:]]).T

    def stop(self):
        self.running = False
        self.wait()


class DashboardUI(QWidget):
    def __init__(self, symbol, stress_threshold):
        super().__init__()
        self.symbol = symbol
        self.stress_threshold = stress_threshold
        self.buy_markers_x = []
        self.buy_markers_y = []
        self.sell_markers_x = []
        self.sell_markers_y = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()

        self.portfolio_label = QLabel("BALANCE: $10000.00 | PNL: 0.00% | POS: NONE")
        self.portfolio_label.setStyleSheet(
            "color:#00FFFF; font-size:18px; font-weight:bold; background:#001122; padding: 5px;")
        left_layout.addWidget(self.portfolio_label)

        self.signal_label = QLabel("ML STATUS: INITIALIZING...")
        self.signal_label.setStyleSheet("color:#888; font-size:20px; font-weight:bold; background:#111; padding: 5px;")
        left_layout.addWidget(self.signal_label)

        self.plot_price = pg.PlotWidget(title=f"Live Price ({self.symbol})", axisItems={'bottom': pg.DateAxisItem()})
        self.plot_price.showGrid(x=True, y=True)
        self.price_curve = self.plot_price.plot(pen=pg.mkPen('g', width=2))
        left_layout.addWidget(self.plot_price, stretch=2)

        self.plot_stress = pg.PlotWidget(title="Topological Stress", axisItems={'bottom': pg.DateAxisItem()})
        self.plot_stress.showGrid(x=True, y=True)
        self.stress_curve = self.plot_stress.plot(pen=pg.mkPen('r', width=2), fillLevel=0, brush=(255, 0, 0, 50))
        self.plot_stress.setXLink(self.plot_price)
        left_layout.addWidget(self.plot_stress, stretch=1)

        self.stress_threshold_line = pg.InfiniteLine(pos=self.stress_threshold, angle=0,
                                                     pen=pg.mkPen('m', width=2, style=Qt.DashLine))
        self.plot_stress.addItem(self.stress_threshold_line)

        self.buy_scatter = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(0, 255, 0), symbol='t1')
        self.sell_scatter = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(255, 0, 0), symbol='t')
        self.plot_price.addItem(self.buy_scatter)
        self.plot_price.addItem(self.sell_scatter)

        self.horizon_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', style=Qt.DashLine))
        self.sl_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', style=Qt.DashLine))
        self.tp_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('g', style=Qt.DashLine))

        for l in [self.horizon_line, self.sl_line, self.tp_line]:
            l.hide()
            self.plot_price.addItem(l)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        layout.addWidget(left_widget, stretch=1)

        self.fig = plt.figure()
        self.fig.patch.set_facecolor('#000')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas, stretch=1)


class TopoAlphaEngine(QMainWindow):
    def __init__(self, symbol='BTC/USDT', timeframe='1m'):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.tau = 5
        self.dim = 3
        self.alpha_stress_threshold = 1.5

        self.feeder = RobustDataFeeder(symbol, timeframe)
        self.tda = TDAAnalyzer()
        self.ml = TopoBooster()
        self.trader = PaperTrader(initial_balance=10000.0, horizon=5)

        self.timestamps = []
        self.prices = []
        self.stress_history = []
        self.absolute_candle_index = 0
        self.last_processed_time = 0

        self.ui = DashboardUI(self.symbol, self.alpha_stress_threshold)
        self.setCentralWidget(self.ui)
        self.setWindowTitle(f"TopoAlpha Engine - {self.symbol} | Quant Fund Mode")
        self.resize(1600, 900)

        self._preload_data()

        self.worker = MarketStreamer(self.feeder, self.tda, self.ml, self.tau, self.dim, self.prices, self.timestamps,
                                     self.stress_history)
        self.worker.data_ready.connect(self._update_interface)
        self.worker.error_occurred.connect(lambda err: logger.error(f"Streamer Error: {err}"))
        self.worker.start()

    def _preload_data(self):
        logger.info("Preloading 500 historical candles...")
        ohlcv = self.feeder.fetch_initial(limit=500)

        if not ohlcv:
            logger.error("Failed to fetch initial data.")
            return

        for candle in ohlcv:
            self.timestamps.append(candle[0])
            self.prices.append(candle[4])
            self.stress_history.append(0.0)

        self.last_processed_time = self.timestamps[-1]
        self.absolute_candle_index = len(self.timestamps)

        logger.info("Backfilling topological stress (this takes a few seconds)...")
        for i in range(50, len(self.prices)):
            window = self.prices[i - 50:i]
            data = np.array(window)
            data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
            embedded = np.vstack([data[:-(2 * self.tau)], data[self.tau:-self.tau], data[2 * self.tau:]]).T
            if len(embedded) > 0:
                self.stress_history[i] = self.tda.get_topological_stress(embedded)

        logger.info("Training ML model on backfilled data...")
        self.ml.train(self.prices, self.stress_history)
        logger.info("Engine initialization complete.")

    def _update_interface(self, timestamps, prices, stress_history, prob_up, embedded_data, current_stress):
        times_sec = [t / 1000.0 for t in timestamps]
        self.ui.price_curve.setData(times_sec, prices)
        self.ui.stress_curve.setData(times_sec, stress_history)

        curr_t = times_sec[-1]
        curr_p = prices[-1]
        curr_ms = timestamps[-1]

        if curr_ms > self.last_processed_time:
            self.absolute_candle_index += 1
            self.last_processed_time = curr_ms

        res = self.trader.update(curr_p, self.absolute_candle_index)
        if res:
            logger.info(
                f"TRADE CLOSED: {res['reason']} | PnL: {res['pnl_pct']:.2f}% | Profit: ${res['profit_usd']:.2f} | New Balance: ${self.trader.balance:.2f}")
            for l in [self.ui.horizon_line, self.ui.sl_line, self.ui.tp_line]:
                l.hide()

        pnl = self.trader.get_unrealized_pnl(curr_p)
        color = "#00FF00" if pnl >= 0 else "#FF0000"
        self.ui.portfolio_label.setText(
            f"BALANCE: ${self.trader.balance:.2f} | PNL: <font color='{color}'>{pnl:.3f}%</font> | POS: {self.trader.position or 'NONE'}"
        )

        if prob_up >= 0.6 and current_stress >= self.alpha_stress_threshold and not self.trader.position:
            if self.trader.execute_trade('LONG', curr_p, self.absolute_candle_index):
                logger.info(f"EXECUTING LONG at {curr_p} (Prob: {prob_up:.2f}, Stress: {current_stress:.2f})")
                self.ui.buy_markers_x.append(curr_t)
                self.ui.buy_markers_y.append(curr_p * 0.9995)
                self.ui.horizon_line.setPos(curr_t + 300)
                self.ui.horizon_line.show()
                self.ui.tp_line.setPos(curr_p * (1 + self.trader.tp_pct))
                self.ui.sl_line.setPos(curr_p * (1 - self.trader.sl_pct))
                self.ui.tp_line.show()
                self.ui.sl_line.show()

        elif prob_up <= 0.4 and current_stress >= self.alpha_stress_threshold and not self.trader.position:
            if self.trader.execute_trade('SHORT', curr_p, self.absolute_candle_index):
                logger.info(f"EXECUTING SHORT at {curr_p} (Prob: {prob_up:.2f}, Stress: {current_stress:.2f})")
                self.ui.sell_markers_x.append(curr_t)
                self.ui.sell_markers_y.append(curr_p * 1.0005)
                self.ui.horizon_line.setPos(curr_t + 300)
                self.ui.horizon_line.show()
                self.ui.tp_line.setPos(curr_p * (1 - self.trader.tp_pct))
                self.ui.sl_line.setPos(curr_p * (1 + self.trader.sl_pct))
                self.ui.tp_line.show()
                self.ui.sl_line.show()

        self.ui.buy_scatter.setData(self.ui.buy_markers_x, self.ui.buy_markers_y)
        self.ui.sell_scatter.setData(self.ui.sell_markers_x, self.ui.sell_markers_y)

        self.ui.ax.clear()
        self.ui.ax.set_facecolor('#000')
        if len(embedded_data) > 0:
            c = np.linspace(0, 1, len(embedded_data))
            self.ui.ax.scatter(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], c=c, cmap='cool', s=10)
        self.ui.canvas.draw_idle()

    def closeEvent(self, e):
        logger.info("Shutting down TopoAlpha Engine...")
        self.worker.stop()
        e.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    engine = TopoAlphaEngine()
    engine.show()
    sys.exit(app.exec_())