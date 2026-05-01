import sys
import logging
import numpy as np
import pyqtgraph as pg
import matplotlib
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use('Qt5Agg')
plt.style.use('dark_background')

from data_feeder import RobustDataFeeder
from tda_core import TDAAnalyzer
from ml_model import TopoBooster
from paper_trader import PaperTrader
from binance_executor import BinanceDemoExecutor
from notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s[%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TopoAlpha")


class MarketStreamer(QThread):
    data_ready = pyqtSignal(list, list, list, list, dict, np.ndarray, float)
    error_occurred = pyqtSignal(str)

    def __init__(
            self, feeder, tda, ml, tau, dim,
            prices, timestamps, stress_history, obi_history,
            prices_htf, timestamps_htf
    ):
        super().__init__()
        self.feeder = feeder
        self.tda = tda
        self.ml = ml
        self.tau = tau
        self.dim = dim

        self.prices = prices
        self.timestamps = timestamps
        self.stress_history = stress_history
        self.obi_history = obi_history
        self.prices_htf = prices_htf
        self.timestamps_htf = timestamps_htf

        self.running = True
        self.tick_counter = 0

    def run(self):
        while self.running:
            try:
                ohlcv = self.feeder.fetch_updates()
                obi = self.feeder.fetch_order_book_imbalance(depth=20)

                ohlcv_htf = self.feeder.fetch_updates_htf()
                if ohlcv_htf:
                    for candle in ohlcv_htf:
                        t, _, _, _, c, _ = candle
                        if not self.timestamps_htf or t > self.timestamps_htf[-1]:
                            self.timestamps_htf.append(t)
                            self.prices_htf.append(c)
                        elif t == self.timestamps_htf[-1]:
                            self.prices_htf[-1] = c

                    if len(self.prices_htf) > 200:
                        del self.timestamps_htf[:-200]
                        del self.prices_htf[:-200]

                if ohlcv:
                    for candle in ohlcv:
                        t, o, h, l, c, v = candle
                        if not self.timestamps or t > self.timestamps[-1]:
                            self.timestamps.append(t)
                            self.prices.append(c)
                            self.stress_history.append(
                                self.stress_history[-1] if self.stress_history else 0.0
                            )
                            self.obi_history.append(obi)
                        elif t == self.timestamps[-1]:
                            self.prices[-1] = c
                            self.obi_history[-1] = obi

                    if len(self.prices) > 500:
                        del self.timestamps[:-500]
                        del self.prices[:-500]
                        del self.stress_history[:-500]
                        del self.obi_history[:-500]

                    embedded_data = self._get_embedding()
                    current_stress = 0.0
                    probs = {'flat': 1.0, 'up': 0.0, 'down': 0.0}

                    if len(embedded_data) > 0:
                        current_stress = self.tda.get_topological_stress(embedded_data[-50:])
                        self.stress_history[-1] = current_stress

                        if self.ml.is_trained:
                            probs = self.ml.predict(
                                self.prices,
                                self.stress_history,
                                self.obi_history,
                                prices_htf=self.prices_htf
                            )

                    self.data_ready.emit(
                        list(self.timestamps),
                        list(self.prices),
                        list(self.stress_history),
                        list(self.obi_history),
                        probs,
                        embedded_data,
                        current_stress
                    )

                self.tick_counter += 1
                if self.tick_counter % 150 == 0 and len(self.prices) > 100:
                    logger.info("Dynamic Retraining ML (multi-timeframe)...")
                    self.ml.train(
                        self.prices, self.stress_history, self.obi_history,
                        prices_htf=self.prices_htf
                    )

            except Exception as e:
                self.error_occurred.emit(str(e))

            self.msleep(2000)

    def _get_embedding(self):
        n = len(self.prices)
        if n < (self.dim - 1) * self.tau + 1:
            return np.zeros((1, 3))
        data = np.array(self.prices)
        data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
        return np.vstack([
            data[:-(2 * self.tau)],
            data[self.tau:-self.tau],
            data[2 * self.tau:]
        ]).T

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

        self.portfolio_label = QLabel("BALANCE: $10000.00 | NET PNL: $0.00 | POS: NONE")
        self.portfolio_label.setStyleSheet(
            "color:#00FFFF; font-size:18px; font-weight:bold; background:#001122; padding:5px;"
        )
        left_layout.addWidget(self.portfolio_label)

        self.signal_label = QLabel("WAITING FOR DATA...")
        self.signal_label.setStyleSheet(
            "color:#FFFF00; font-size:20px; font-weight:bold; background:#111; padding:5px;"
        )
        left_layout.addWidget(self.signal_label)

        control_layout = QHBoxLayout()
        self.stress_label = QLabel(f"STRESS THRESHOLD: {self.stress_threshold:.1f}")
        self.stress_label.setStyleSheet(
            "color:#FF00FF; font-size:16px; font-weight:bold; padding:5px;"
        )
        self.stress_slider = QSlider(Qt.Horizontal)
        self.stress_slider.setMinimum(0)
        self.stress_slider.setMaximum(100)
        self.stress_slider.setValue(int(self.stress_threshold * 10))
        self.stress_slider.setTickPosition(QSlider.TicksBelow)
        self.stress_slider.setTickInterval(10)
        control_layout.addWidget(self.stress_label)
        control_layout.addWidget(self.stress_slider)
        left_layout.addLayout(control_layout)

        self.plot_price = pg.PlotWidget(
            title=f"Live Price ({self.symbol})",
            axisItems={'bottom': pg.DateAxisItem()}
        )
        self.plot_price.showGrid(x=True, y=True)
        self.price_curve = self.plot_price.plot(pen=pg.mkPen('g', width=2))
        left_layout.addWidget(self.plot_price, stretch=3)

        self.plot_stress = pg.PlotWidget(
            title="Topological Stress (LTF 1m)",
            axisItems={'bottom': pg.DateAxisItem()}
        )
        self.plot_stress.showGrid(x=True, y=True)
        self.stress_curve = self.plot_stress.plot(
            pen=pg.mkPen('r', width=2), fillLevel=0, brush=(255, 0, 0, 50)
        )
        self.plot_stress.setXLink(self.plot_price)
        left_layout.addWidget(self.plot_stress, stretch=1)

        self.stress_threshold_line = pg.InfiniteLine(
            pos=self.stress_threshold, angle=0,
            pen=pg.mkPen('m', width=2, style=Qt.DashLine)
        )
        self.plot_stress.addItem(self.stress_threshold_line)

        self.plot_obi = pg.PlotWidget(
            title="Order Book Imbalance (-1 Bears / +1 Bulls)",
            axisItems={'bottom': pg.DateAxisItem()}
        )
        self.plot_obi.showGrid(x=True, y=True)
        self.obi_curve = self.plot_obi.plot(
            pen=pg.mkPen('c', width=2), fillLevel=0, brush=(0, 255, 255, 50)
        )
        self.plot_obi.setXLink(self.plot_price)
        self.obi_zero_line = pg.InfiniteLine(
            pos=0.0, angle=0, pen=pg.mkPen('w', width=1, style=Qt.DashLine)
        )
        self.plot_obi.addItem(self.obi_zero_line)
        left_layout.addWidget(self.plot_obi, stretch=1)

        self.buy_scatter = pg.ScatterPlotItem(
            size=14, brush=pg.mkBrush(0, 255, 0), symbol='t1'
        )
        self.sell_scatter = pg.ScatterPlotItem(
            size=14, brush=pg.mkBrush(255, 0, 0), symbol='t'
        )
        self.plot_price.addItem(self.buy_scatter)
        self.plot_price.addItem(self.sell_scatter)

        self.horizon_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', style=Qt.DashLine))
        self.sl_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', style=Qt.DashLine))
        self.tp_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('g', style=Qt.DashLine))
        for line in [self.horizon_line, self.sl_line, self.tp_line]:
            line.hide()
            self.plot_price.addItem(line)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        layout.addWidget(left_widget, stretch=1)

        self.fig = plt.figure()
        self.fig.patch.set_facecolor('#000')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas, stretch=1)


class TopoAlphaEngine(QMainWindow):
    def __init__(self, symbol='BTC/USDT', timeframe='1m', htf='15m'):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.htf = htf
        self.tau = 5
        self.dim = 3
        self.alpha_stress_threshold = 1.5

        self.feeder = RobustDataFeeder(symbol, timeframe, htf=htf)
        self.tda = TDAAnalyzer()
        self.ml = TopoBooster()
        self.trader = PaperTrader(
            initial_balance=10000.0,
            margin_usdt=50.0,
            leverage=10,
            horizon=10,
            sl_pct=0.004,
            tp_pct=0.008
        )
        self.api_executor = BinanceDemoExecutor(symbol=self.symbol, leverage=10, margin_usdt=50.0)
        self.notifier = TelegramNotifier()

        self.timestamps = []
        self.prices = []
        self.stress_history = []
        self.obi_history = []

        self.timestamps_htf = []
        self.prices_htf = []

        self.last_processed_time = 0

        self.ui = DashboardUI(self.symbol, self.alpha_stress_threshold)
        self.setCentralWidget(self.ui)
        self.setWindowTitle(f"TopoAlpha Engine — {self.symbol}  [HTF: {htf}]")
        self.resize(1600, 950)

        self.ui.stress_slider.valueChanged.connect(self._on_threshold_changed)

        self._preload_data()
        self._restore_ui_state()

        self.worker = MarketStreamer(
            self.feeder, self.tda, self.ml,
            self.tau, self.dim,
            self.prices, self.timestamps, self.stress_history, self.obi_history,
            self.prices_htf, self.timestamps_htf
        )
        self.worker.data_ready.connect(self._update_interface)
        self.worker.error_occurred.connect(lambda msg: logger.error(f"STREAMER: {msg}"))
        self.worker.start()

        self.notifier.send_message(
            f"🟢 <b>TopoAlpha Engine Started</b>\n"
            f"Symbol: {self.symbol}\n"
            f"Balance: ${self.trader.balance:.2f}\n"
            f"HTF context: {self.htf} ✅"
        )

    def _on_threshold_changed(self, value):
        self.alpha_stress_threshold = value / 10.0
        self.ui.stress_label.setText(f"STRESS THRESHOLD: {self.alpha_stress_threshold:.1f}")
        self.ui.stress_threshold_line.setPos(self.alpha_stress_threshold)

    def _preload_data(self):
        ohlcv = self.feeder.fetch_initial(limit=500)
        if not ohlcv:
            return

        for candle in ohlcv:
            self.timestamps.append(candle[0])
            self.prices.append(candle[4])
            self.stress_history.append(0.0)
            self.obi_history.append(0.0)

        self.last_processed_time = self.timestamps[-1]

        ohlcv_htf = self.feeder.fetch_initial_htf(limit=150)
        for candle in ohlcv_htf:
            self.timestamps_htf.append(candle[0])
            self.prices_htf.append(candle[4])

        for i in range(50, len(self.prices)):
            window = self.prices[i - 50:i]
            data = np.array(window)
            data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
            embedded = np.vstack([
                data[:-(2 * self.tau)],
                data[self.tau:-self.tau],
                data[2 * self.tau:]
            ]).T
            if len(embedded) > 0:
                self.stress_history[i] = self.tda.get_topological_stress(embedded)

        self.ml.train(
            self.prices, self.stress_history, self.obi_history,
            prices_htf=self.prices_htf
        )

    def _restore_ui_state(self):
        if not self.trader.position:
            return

        logger.info(f"Restoring active {self.trader.position} position from database...")
        curr_p = self.trader.entry_price
        curr_t_sec = self.trader.entry_time / 1000.0

        if self.trader.position == 'LONG':
            self.ui.buy_markers_x.append(curr_t_sec)
            self.ui.buy_markers_y.append(curr_p)
            self.ui.tp_line.setPos(curr_p * (1 + self.trader.tp_pct))
            self.ui.sl_line.setPos(curr_p * (1 - self.trader.sl_pct))
        else:
            self.ui.sell_markers_x.append(curr_t_sec)
            self.ui.sell_markers_y.append(curr_p)
            self.ui.tp_line.setPos(curr_p * (1 - self.trader.tp_pct))
            self.ui.sl_line.setPos(curr_p * (1 + self.trader.sl_pct))

        self.ui.horizon_line.setPos(curr_t_sec + self.trader.horizon_ms / 1000.0)
        for line in [self.ui.horizon_line, self.ui.tp_line, self.ui.sl_line]:
            line.show()

        self.ui.buy_scatter.setData(self.ui.buy_markers_x, self.ui.buy_markers_y)
        self.ui.sell_scatter.setData(self.ui.sell_markers_x, self.ui.sell_markers_y)

    def _update_interface(
            self, timestamps, prices, stress_history,
            obi_history, probs, embedded_data, current_stress
    ):
        times_sec = [t / 1000.0 for t in timestamps]

        self.ui.price_curve.setData(times_sec, prices)
        self.ui.stress_curve.setData(times_sec, stress_history)
        self.ui.obi_curve.setData(times_sec, obi_history)

        curr_t = times_sec[-1]
        curr_p = prices[-1]
        curr_ms = timestamps[-1]

        htf_dir = ""
        if len(self.prices_htf) >= 21:
            import pandas as pd
            s = pd.Series(self.prices_htf)
            ema8 = s.ewm(span=8).mean().iloc[-1]
            ema21 = s.ewm(span=21).mean().iloc[-1]
            htf_dir = "📈HTF↑" if ema8 > ema21 else "📉HTF↓"

        prob_text = (
            f"UP: {probs['up']:.1%} | FLAT: {probs['flat']:.1%} | "
            f"DOWN: {probs['down']:.1%}  |  STRESS: {current_stress:.2f}  {htf_dir}"
        )
        self.ui.signal_label.setText(prob_text)

        if curr_ms > self.last_processed_time:
            self.last_processed_time = curr_ms

        res = self.trader.update(curr_p, curr_ms)
        if res:
            logger.info(
                f"TRADE CLOSED ({res['reason']}): "
                f"Net Profit: ${res['net_profit_usd']:.2f} | "
                f"Bal: ${self.trader.balance:.2f}"
            )
            self.api_executor.close_all_positions_and_orders()
            for line in [self.ui.horizon_line, self.ui.sl_line, self.ui.tp_line]:
                line.hide()
            icon = "✅" if res['net_profit_usd'] > 0 else "🛑"
            self.notifier.send_message(
                f"{icon} <b>TRADE CLOSED ({res['reason']})</b>\n"
                f"Net Profit: ${res['net_profit_usd']:.2f}\n"
                f"Balance: ${self.trader.balance:.2f}"
            )

        net_pnl = self.trader.get_unrealized_pnl(curr_p)
        color = "#00FF00" if net_pnl >= 0 else "#FF0000"
        self.ui.portfolio_label.setText(
            f"BALANCE: ${self.trader.balance:.2f} | "
            f"NET PNL: <font color='{color}'>${net_pnl:.2f}</font> | "
            f"POS: {self.trader.position or 'NONE'}"
        )

        if current_stress >= self.alpha_stress_threshold and not self.trader.position:
            if probs['up'] >= 0.60:
                if self.trader.execute_trade('LONG', curr_p, curr_ms):
                    self.api_executor.execute_trade(
                        'LONG', curr_p, self.trader.sl_pct, self.trader.tp_pct
                    )
                    logger.info(f"EXECUTING LONG at {curr_p}")
                    self.ui.buy_markers_x.append(curr_t)
                    self.ui.buy_markers_y.append(curr_p)
                    self.ui.horizon_line.setPos(curr_t + self.trader.horizon_ms / 1000.0)
                    self.ui.tp_line.setPos(curr_p * (1 + self.trader.tp_pct))
                    self.ui.sl_line.setPos(curr_p * (1 - self.trader.sl_pct))
                    for line in [self.ui.horizon_line, self.ui.tp_line, self.ui.sl_line]:
                        line.show()
                    self.notifier.send_message(
                        f"🚀 <b>EXECUTING LONG</b>\n"
                        f"Price: {curr_p}\n"
                        f"Stress: {current_stress:.2f}\n"
                        f"Prob UP: {probs['up']:.1%}"
                    )

            elif probs['down'] >= 0.60:
                if self.trader.execute_trade('SHORT', curr_p, curr_ms):
                    self.api_executor.execute_trade(
                        'SHORT', curr_p, self.trader.sl_pct, self.trader.tp_pct
                    )
                    logger.info(f"EXECUTING SHORT at {curr_p}")
                    self.ui.sell_markers_x.append(curr_t)
                    self.ui.sell_markers_y.append(curr_p)
                    self.ui.horizon_line.setPos(curr_t + self.trader.horizon_ms / 1000.0)
                    self.ui.tp_line.setPos(curr_p * (1 - self.trader.tp_pct))
                    self.ui.sl_line.setPos(curr_p * (1 + self.trader.sl_pct))
                    for line in [self.ui.horizon_line, self.ui.tp_line, self.ui.sl_line]:
                        line.show()
                    self.notifier.send_message(
                        f"🩸 <b>EXECUTING SHORT</b>\n"
                        f"Price: {curr_p}\n"
                        f"Stress: {current_stress:.2f}\n"
                        f"Prob DOWN: {probs['down']:.1%}"
                    )

        self.ui.buy_scatter.setData(self.ui.buy_markers_x, self.ui.buy_markers_y)
        self.ui.sell_scatter.setData(self.ui.sell_markers_x, self.ui.sell_markers_y)

        self.ui.ax.clear()
        self.ui.ax.set_facecolor('#000')
        if len(embedded_data) > 0:
            c = np.linspace(0, 1, len(embedded_data))
            self.ui.ax.scatter(
                embedded_data[:, 0],
                embedded_data[:, 1],
                embedded_data[:, 2],
                c=c, cmap='cool', s=10
            )
        self.ui.canvas.draw_idle()

    def closeEvent(self, e):
        self.worker.stop()
        self.notifier.send_message("🔴 <b>TopoAlpha Engine Stopped</b>")
        e.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    engine = TopoAlphaEngine()
    engine.show()
    sys.exit(app.exec_())
