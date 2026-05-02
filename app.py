import sys
import logging
import threading
import numpy as np
import pandas as pd
import pyqtgraph as pg
import matplotlib
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QSlider,
    QProgressDialog, QPushButton,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use("Qt5Agg")
plt.style.use("dark_background")

from data_feeder import RobustDataFeeder
from tda_core import TDAAnalyzer
from ml_model import TopoBooster
from paper_trader import PaperTrader
from notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TopoAlpha")

_3D_REDRAW_EVERY = 5


class PreloadWorker(QThread):
    """Runs historical fetch + TDA backfill + initial ML train off the main thread."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool)

    def __init__(self, feeder, tda, ml, tau, dim,
                 prices, timestamps, stress_history, obi_history,
                 prices_htf, timestamps_htf):
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

    def _build_embedding(self, price_window, tau, dim) -> np.ndarray:
        data = np.array(price_window, dtype=float)
        if len(data) < (dim - 1) * tau + 1:
            return np.zeros((1, 3))
        data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
        return np.vstack([data[:-(2 * tau)], data[tau:-tau], data[2 * tau:]]).T

    def run(self) -> None:
        try:
            self.progress.emit(5, "Fetching LTF candles…")
            ohlcv = self.feeder.fetch_initial(limit=500)
            if not ohlcv:
                self.finished.emit(False)
                return

            for c in ohlcv:
                self.timestamps.append(c[0])
                self.prices.append(c[4])
                self.stress_history.append(0.0)
                self.obi_history.append(0.0)

            self.progress.emit(20, "Fetching HTF candles…")
            for c in self.feeder.fetch_initial_htf(limit=150):
                self.timestamps_htf.append(c[0])
                self.prices_htf.append(c[4])

            self.progress.emit(30, "Back-filling topological stress…")
            n = len(self.prices)
            for i in range(50, n):
                embedded = self._build_embedding(self.prices[i - 50:i], self.tau, self.dim)
                if len(embedded) > 0:
                    self.stress_history[i] = self.tda.get_topological_stress(embedded)
                if (i - 50) % 50 == 0:
                    pct = 30 + int(50 * (i - 50) / max(n - 50, 1))
                    self.progress.emit(pct, f"TDA backfill {i}/{n}…")

            self.progress.emit(85, "Training initial ML model…")
            self.ml.train(self.prices, self.stress_history, self.obi_history,
                          prices_htf=self.prices_htf)

            self.progress.emit(100, "Ready.")
            self.finished.emit(True)
        except Exception as exc:
            logger.error(f"[PreloadWorker] {exc}")
            self.finished.emit(False)


class MarketStreamer(QThread):
    data_ready = pyqtSignal(list, list, list, list, dict, np.ndarray, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, feeder, tda, ml, tau, dim,
                 prices, timestamps, stress_history, obi_history,
                 prices_htf, timestamps_htf):
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
        self._retrain_lock = threading.Lock()

    def _get_embedding(self) -> np.ndarray:
        data = np.array(self.prices, dtype=float)
        if len(data) < (self.dim - 1) * self.tau + 1:
            return np.zeros((1, 3))
        data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
        return np.vstack([
            data[:-(2 * self.tau)],
            data[self.tau:-self.tau],
            data[2 * self.tau:],
        ]).T

    def _schedule_retrain(self) -> None:
        prices_snap = list(self.prices)
        stress_snap = list(self.stress_history)
        obi_snap = list(self.obi_history)
        htf_snap = list(self.prices_htf)

        threading.Thread(
            target=self._retrain_background,
            args=(prices_snap, stress_snap, obi_snap, htf_snap),
            daemon=True,
            name="ml-retrain",
        ).start()

    def _retrain_background(self, prices, stress, obi, prices_htf) -> None:
        if not self._retrain_lock.acquire(blocking=False):
            logger.info("Retrain already in progress — skipping.")
            return
        try:
            logger.info("Retraining ML in background…")
            self.ml.train(prices, stress, obi, prices_htf=prices_htf)
            logger.info("Retrain complete.")
        except Exception as exc:
            logger.error(f"Retrain failed: {exc}")
        finally:
            self._retrain_lock.release()

    def run(self) -> None:
        while self.running:
            try:
                obi = self.feeder.fetch_order_book_imbalance(depth=20)

                ohlcv_htf = self.feeder.fetch_updates_htf()
                if ohlcv_htf:
                    for t, _, _, _, c, _ in ohlcv_htf:
                        if not self.timestamps_htf or t > self.timestamps_htf[-1]:
                            self.timestamps_htf.append(t)
                            self.prices_htf.append(c)
                        elif t == self.timestamps_htf[-1]:
                            self.prices_htf[-1] = c
                    if len(self.prices_htf) > 200:
                        del self.timestamps_htf[:-200]
                        del self.prices_htf[:-200]

                ohlcv = self.feeder.fetch_updates()
                if ohlcv:
                    for t, _o, _h, _l, c, _v in ohlcv:
                        if not self.timestamps or t > self.timestamps[-1]:
                            self.timestamps.append(t)
                            self.prices.append(c)
                            self.stress_history.append(0.0)
                            self.obi_history.append(obi)
                        elif t == self.timestamps[-1]:
                            self.prices[-1] = c
                            self.obi_history[-1] = obi

                    if len(self.prices) > 500:
                        del self.timestamps[:-500]
                        del self.prices[:-500]
                        del self.stress_history[:-500]
                        del self.obi_history[:-500]

                    embedded = self._get_embedding()
                    current_stress = 0.0
                    probs = {"flat": 1.0, "up": 0.0, "down": 0.0}

                    if len(embedded) > 0:
                        current_stress = self.tda.get_topological_stress(embedded[-50:])
                        self.stress_history[-1] = current_stress
                        if self.ml.is_trained:
                            probs = self.ml.predict(
                                self.prices, self.stress_history, self.obi_history,
                                prices_htf=self.prices_htf,
                            )

                    self.data_ready.emit(
                        list(self.timestamps), list(self.prices),
                        list(self.stress_history), list(self.obi_history),
                        probs, embedded, current_stress,
                    )

                self.tick_counter += 1
                if self.tick_counter % 150 == 0 and len(self.prices) > 100:
                    self._schedule_retrain()

            except Exception as exc:
                self.error_occurred.emit(str(exc))

            self.msleep(2_000)

    def stop(self) -> None:
        self.running = False
        self.wait()


class DashboardUI(QWidget):
    def __init__(self, symbol: str, stress_threshold: float):
        super().__init__()
        self.symbol = symbol
        self.stress_threshold = stress_threshold
        self.buy_markers_x: list = []
        self.buy_markers_y: list = []
        self.sell_markers_x: list = []
        self.sell_markers_y: list = []
        self._scatter3d = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()

        self.portfolio_label = QLabel("BALANCE: $10,000.00  |  NET PNL: $0.00  |  POS: NONE")
        self.portfolio_label.setStyleSheet(
            "color:#00FFFF; font-size:18px; font-weight:bold; background:#001122; padding:5px;"
        )
        left_layout.addWidget(self.portfolio_label)

        self.signal_label = QLabel("WAITING FOR DATA…")
        self.signal_label.setStyleSheet(
            "color:#FFFF00; font-size:20px; font-weight:bold; background:#0d0d0d; padding:5px;"
        )
        left_layout.addWidget(self.signal_label)

        ctrl = QHBoxLayout()
        self.stress_label = QLabel(f"STRESS THRESHOLD: {self.stress_threshold:.1f}")
        self.stress_label.setStyleSheet("color:#FF00FF; font-size:16px; font-weight:bold; padding:5px;")
        self.stress_slider = QSlider(Qt.Horizontal)
        self.stress_slider.setMinimum(0)
        self.stress_slider.setMaximum(100)
        self.stress_slider.setValue(int(self.stress_threshold * 10))
        self.stress_slider.setTickPosition(QSlider.TicksBelow)
        self.stress_slider.setTickInterval(10)
        ctrl.addWidget(self.stress_label)
        ctrl.addWidget(self.stress_slider)

        self.calc_btn = QPushButton("⚡ Trade Calculator")
        self.calc_btn.setFixedHeight(28)
        self.calc_btn.setStyleSheet(
            "QPushButton { background:#0d0d22; color:#00FFFF;"
            "  border:1px solid #00FFFF; border-radius:5px; font-size:13px;"
            "  font-weight:bold; padding:2px 12px; }"
            "QPushButton:hover { background:#003355; }"
        )
        ctrl.addWidget(self.calc_btn)
        left_layout.addLayout(ctrl)

        self.plot_price = pg.PlotWidget(
            title=f"Live Price ({self.symbol})",
            axisItems={"bottom": pg.DateAxisItem()},
        )
        self.plot_price.showGrid(x=True, y=True)
        self.price_curve = self.plot_price.plot(pen=pg.mkPen("g", width=2))
        self.buy_scatter = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(0, 255, 0), symbol="t1")
        self.sell_scatter = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(255, 0, 0), symbol="t")
        self.plot_price.addItem(self.buy_scatter)
        self.plot_price.addItem(self.sell_scatter)

        self.horizon_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("y", style=Qt.DashLine))
        self.sl_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("r", style=Qt.DashLine))
        self.tp_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("g", style=Qt.DashLine))
        for line in [self.horizon_line, self.sl_line, self.tp_line]:
            line.hide()
            self.plot_price.addItem(line)
        left_layout.addWidget(self.plot_price, stretch=3)

        self.plot_stress = pg.PlotWidget(
            title="Topological Stress (H₁ persistence)",
            axisItems={"bottom": pg.DateAxisItem()},
        )
        self.plot_stress.showGrid(x=True, y=True)
        self.stress_curve = self.plot_stress.plot(pen=pg.mkPen("r", width=2), fillLevel=0, brush=(255, 0, 0, 50))
        self.plot_stress.setXLink(self.plot_price)
        self.stress_threshold_line = pg.InfiniteLine(
            pos=self.stress_threshold, angle=0,
            pen=pg.mkPen("m", width=2, style=Qt.DashLine),
        )
        self.plot_stress.addItem(self.stress_threshold_line)
        left_layout.addWidget(self.plot_stress, stretch=1)

        self.plot_obi = pg.PlotWidget(
            title="Order Book Imbalance  (−1 bears / +1 bulls)",
            axisItems={"bottom": pg.DateAxisItem()},
        )
        self.plot_obi.showGrid(x=True, y=True)
        self.obi_curve = self.plot_obi.plot(pen=pg.mkPen("c", width=2), fillLevel=0, brush=(0, 255, 255, 50))
        self.plot_obi.setXLink(self.plot_price)
        self.plot_obi.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen("w", width=1, style=Qt.DashLine)))
        left_layout.addWidget(self.plot_obi, stretch=1)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        layout.addWidget(left_widget, stretch=1)

        self.fig = plt.figure()
        self.fig.patch.set_facecolor("#000")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        layout.addWidget(self.canvas, stretch=1)


class TopoAlphaEngine(QMainWindow):
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "5m", htf: str = "1h"):
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
            initial_balance=10_000.0,
            margin_usdt=50.0,
            leverage=10,
            horizon_bars=6,
            timeframe_mins=5,
            sl_pct=0.005,
            tp_pct=0.010,
        )
        self.notifier = TelegramNotifier()

        self.timestamps: list = []
        self.prices: list = []
        self.stress_history: list = []
        self.obi_history: list = []
        self.timestamps_htf: list = []
        self.prices_htf: list = []
        self.last_processed_time: int = 0
        self._scatter_tick: int = 0

        self.worker: MarketStreamer | None = None
        self._calc_window = None

        self.ui = DashboardUI(self.symbol, self.alpha_stress_threshold)
        self.setCentralWidget(self.ui)
        self.setWindowTitle(f"TopoAlpha — {self.symbol}  [{htf} macro]")
        self.resize(1600, 950)
        self.ui.stress_slider.valueChanged.connect(self._on_threshold_changed)
        self.ui.calc_btn.clicked.connect(self._open_calculator)

        self._start_preload()

    def _start_preload(self) -> None:
        self._progress_dlg = QProgressDialog("Loading historical data…", None, 0, 100, self)
        self._progress_dlg.setWindowTitle("TopoAlpha — Initialising")
        self._progress_dlg.setWindowModality(Qt.WindowModal)
        self._progress_dlg.setMinimumDuration(0)
        self._progress_dlg.setValue(0)
        self._progress_dlg.show()

        self._preloader = PreloadWorker(
            self.feeder, self.tda, self.ml, self.tau, self.dim,
            self.prices, self.timestamps, self.stress_history, self.obi_history,
            self.prices_htf, self.timestamps_htf,
        )
        self._preloader.progress.connect(self._on_preload_progress)
        self._preloader.finished.connect(self._on_preload_finished)
        self._preloader.start()

    def _on_preload_progress(self, pct: int, msg: str) -> None:
        self._progress_dlg.setValue(pct)
        self._progress_dlg.setLabelText(msg)

    def _on_preload_finished(self, ok: bool) -> None:
        self._progress_dlg.close()
        if not ok:
            logger.error("Preload failed — check exchange connection.")
            return
        self._restore_ui_state()
        self._start_worker()

    def _restore_ui_state(self) -> None:
        if not self.trader.position:
            return
        curr_p = self.trader.entry_price
        curr_t = self.trader.entry_time / 1_000.0
        if self.trader.position == "LONG":
            self.ui.buy_markers_x.append(curr_t)
            self.ui.buy_markers_y.append(curr_p)
            self.ui.tp_line.setPos(curr_p * (1 + self.trader.tp_pct))
            self.ui.sl_line.setPos(curr_p * (1 - self.trader.sl_pct))
        else:
            self.ui.sell_markers_x.append(curr_t)
            self.ui.sell_markers_y.append(curr_p)
            self.ui.tp_line.setPos(curr_p * (1 - self.trader.tp_pct))
            self.ui.sl_line.setPos(curr_p * (1 + self.trader.sl_pct))
        self.ui.horizon_line.setPos(curr_t + self.trader.horizon_ms / 1_000.0)
        for line in [self.ui.horizon_line, self.ui.tp_line, self.ui.sl_line]:
            line.show()
        self.ui.buy_scatter.setData(self.ui.buy_markers_x, self.ui.buy_markers_y)
        self.ui.sell_scatter.setData(self.ui.sell_markers_x, self.ui.sell_markers_y)

    def _start_worker(self) -> None:
        self.worker = MarketStreamer(
            self.feeder, self.tda, self.ml, self.tau, self.dim,
            self.prices, self.timestamps, self.stress_history, self.obi_history,
            self.prices_htf, self.timestamps_htf,
        )
        self.worker.data_ready.connect(self._update_interface)
        self.worker.error_occurred.connect(lambda msg: logger.error(f"STREAMER: {msg}"))
        self.worker.start()

        self.notifier.send_message(
            f"🟢 <b>TopoAlpha Started</b>\n"
            f"Symbol: {self.symbol}  |  HTF: {self.htf}\n"
            f"Balance: ${self.trader.balance:.2f}"
        )

    def _open_calculator(self) -> None:
        from trade_calculator import TradeCalculatorWindow
        if self._calc_window is None:
            self._calc_window = TradeCalculatorWindow(parent=None)
        self._calc_window.feed(
            self.prices, self.stress_history, self.obi_history,
            self.prices_htf, self.ml, self.trader,
        )
        self._calc_window.show()
        self._calc_window.raise_()

    def _on_threshold_changed(self, value: int) -> None:
        self.alpha_stress_threshold = value / 10.0
        self.ui.stress_label.setText(f"STRESS THRESHOLD: {self.alpha_stress_threshold:.1f}")
        self.ui.stress_threshold_line.setPos(self.alpha_stress_threshold)

    def _update_interface(
            self,
            timestamps: list,
            prices: list,
            stress_history: list,
            obi_history: list,
            probs: dict,
            embedded_data: np.ndarray,
            current_stress: float,
    ) -> None:
        try:
            times_sec = [t / 1_000.0 for t in timestamps]
            curr_t = times_sec[-1]
            curr_p = prices[-1]
            curr_ms = timestamps[-1]

            self.ui.price_curve.setData(times_sec, prices)
            self.ui.stress_curve.setData(times_sec, stress_history)
            self.ui.obi_curve.setData(times_sec, obi_history)

            htf_dir = ""
            if len(self.prices_htf) >= 21:
                s = pd.Series(self.prices_htf)
                htf_dir = "📈HTF↑" if s.ewm(span=8).mean().iloc[-1] > s.ewm(span=21).mean().iloc[-1] else "📉HTF↓"

            self.ui.signal_label.setText(
                f"UP: {probs['up']:.1%}  |  FLAT: {probs['flat']:.1%}  |  "
                f"DOWN: {probs['down']:.1%}   STRESS: {current_stress:.2f}  {htf_dir}"
            )

            if curr_ms > self.last_processed_time:
                self.last_processed_time = curr_ms

            res = self.trader.update(curr_p, curr_ms)
            if res:
                logger.info(
                    f"CLOSED ({res['reason']})  P&L: ${res['net_profit_usd']:.2f}  Bal: ${self.trader.balance:.2f}")
                for line in [self.ui.horizon_line, self.ui.sl_line, self.ui.tp_line]:
                    line.hide()
                icon = "✅" if res["net_profit_usd"] > 0 else "🛑"
                self.notifier.send_message(
                    f"{icon} <b>TRADE CLOSED ({res['reason']})</b>\n"
                    f"Net P&L: ${res['net_profit_usd']:.2f}\n"
                    f"Balance: ${self.trader.balance:.2f}"
                )

            net_pnl = self.trader.get_unrealized_pnl(curr_p)
            color = "#00FF00" if net_pnl >= 0 else "#FF0000"
            self.ui.portfolio_label.setText(
                f"BALANCE: ${self.trader.balance:.2f}  |  "
                f"NET PNL: <font color='{color}'>${net_pnl:.2f}</font>  |  "
                f"POS: {self.trader.position or 'NONE'}"
            )

            if current_stress >= self.alpha_stress_threshold and not self.trader.position:
                if probs["up"] >= 0.60:
                    self._open_trade("LONG", curr_p, curr_ms, curr_t, current_stress, probs["up"])
                elif probs["down"] >= 0.60:
                    self._open_trade("SHORT", curr_p, curr_ms, curr_t, current_stress, probs["down"])

            self.ui.buy_scatter.setData(self.ui.buy_markers_x, self.ui.buy_markers_y)
            self.ui.sell_scatter.setData(self.ui.sell_markers_x, self.ui.sell_markers_y)

            self._scatter_tick += 1
            if self._scatter_tick % _3D_REDRAW_EVERY == 1 and len(embedded_data) > 1:
                c = np.linspace(0, 1, len(embedded_data))
                if self.ui._scatter3d is None:
                    self.ui.ax.set_facecolor("#000")
                    self.ui._scatter3d = self.ui.ax.scatter(
                        embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2],
                        c=c, cmap="cool", s=10,
                    )
                else:
                    self.ui._scatter3d._offsets3d = (
                        embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2]
                    )
                    self.ui._scatter3d.set_array(c)
                self.ui.canvas.draw_idle()

        except Exception as exc:
            logger.error(f"UI update error: {exc}")

    def _open_trade(self, side, price, curr_ms, curr_t, stress, prob) -> None:
        if not self.trader.execute_trade(side, price, curr_ms):
            return

        if side == "LONG":
            self.ui.buy_markers_x.append(curr_t)
            self.ui.buy_markers_y.append(price)
            self.ui.tp_line.setPos(price * (1 + self.trader.tp_pct))
            self.ui.sl_line.setPos(price * (1 - self.trader.sl_pct))
            icon, prob_label = "🚀", "UP"
        else:
            self.ui.sell_markers_x.append(curr_t)
            self.ui.sell_markers_y.append(price)
            self.ui.tp_line.setPos(price * (1 - self.trader.tp_pct))
            self.ui.sl_line.setPos(price * (1 + self.trader.sl_pct))
            icon, prob_label = "🩸", "DOWN"

        self.ui.horizon_line.setPos(curr_t + self.trader.horizon_ms / 1_000.0)
        for line in [self.ui.horizon_line, self.ui.tp_line, self.ui.sl_line]:
            line.show()

        logger.info(f"OPEN {side} @ {price}  Stress: {stress:.2f}  P({prob_label}): {prob:.2f}")
        self.notifier.send_message(
            f"{icon} <b>OPEN {side}</b>\n"
            f"Price: {price}\n"
            f"Stress: {stress:.2f}\n"
            f"P({prob_label}): {prob:.1%}"
        )

    def closeEvent(self, e) -> None:
        if self.worker:
            self.worker.stop()
        self.notifier.send_message("🔴 <b>TopoAlpha Stopped</b>")
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    engine = TopoAlphaEngine()
    engine.show()
    sys.exit(app.exec_())
