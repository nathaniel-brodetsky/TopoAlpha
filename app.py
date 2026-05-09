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
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QPicture, QPainter, QPen, QBrush, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use("Qt5Agg")
plt.style.use("dark_background")

from data_feeder import RobustDataFeeder
from tda_core import TDAAnalyzer
from ml_model import TopoBooster, detect_phase_transition
from paper_trader import PaperTrader
from notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TopoAlpha")

_3D_REDRAW_EVERY = 5
TDA_EMBED_WINDOW = 50


class CandlestickItem(pg.GraphicsObject):
    _BULL_COLOR = QColor(0, 210, 80, 160)
    _BEAR_COLOR = QColor(220, 50, 50, 160)
    _WICK_COLOR = QColor(180, 180, 180, 200)

    def __init__(self):
        super().__init__()
        self._candles: list = []
        self._picture: QPicture | None = None

    def setData(self, candles: list) -> None:
        self._candles = candles
        self._picture = None
        self.prepareGeometryChange()
        self.update()

    def _build_picture(self) -> None:
        self._picture = QPicture()
        p = QPainter(self._picture)
        p.setRenderHint(QPainter.Antialiasing, False)

        data = self._candles
        if len(data) < 2:
            p.end()
            return

        bar_w = (data[-1][0] - data[0][0]) / max(len(data) - 1, 1) * 0.38

        for c in data:
            t, o, h, l, cl = c[0], c[1], c[2], c[3], c[4]
            bull = cl >= o
            body_col = self._BULL_COLOR if bull else self._BEAR_COLOR

            p.setPen(QPen(self._WICK_COLOR, 1))
            p.drawLine(
                pg.QtCore.QPointF(t, l),
                pg.QtCore.QPointF(t, h),
            )

            body_top = max(o, cl)
            body_bot = min(o, cl)
            body_h = max(body_top - body_bot, 1e-8)
            p.setPen(QPen(body_col.darker(130), 1))
            p.setBrush(QBrush(body_col))
            p.drawRect(QRectF(t - bar_w, body_bot, bar_w * 2, body_h))

        p.end()

    def paint(self, p, *args) -> None:
        if not self._candles:
            return
        if self._picture is None:
            self._build_picture()
        self._picture.play(p)

    def boundingRect(self) -> QRectF:
        if not self._candles:
            return QRectF()
        ts = [c[0] for c in self._candles]
        ls = [c[3] for c in self._candles]
        hs = [c[2] for c in self._candles]
        w = (max(ts) - min(ts)) or 1
        h = (max(hs) - min(ls)) or 1
        return QRectF(min(ts), min(ls), w, h)


class PreloadWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool)

    def __init__(self, feeder, tda, ml, tau, dim,
                 candles, stress_history, obi_history,
                 candles_htf):
        super().__init__()
        self.feeder = feeder
        self.tda = tda
        self.ml = ml
        self.tau = tau
        self.dim = dim
        self.candles = candles
        self.stress_history = stress_history
        self.obi_history = obi_history
        self.candles_htf = candles_htf

    def _build_embedding(self, prices, tau, dim) -> np.ndarray:
        data = np.array(prices, dtype=float)
        if len(data) < (dim - 1) * tau + 1:
            return np.zeros((1, 3))
        data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
        return np.vstack([data[:-(2 * tau)], data[tau:-tau], data[2 * tau:]]).T

    def run(self) -> None:
        try:
            self.progress.emit(5, "Fetching LTF candles…")
            ohlcv = self.feeder.fetch_initial(limit=600)
            if not ohlcv:
                self.finished.emit(False)
                return

            for c in ohlcv:
                self.candles.append(c)
                self.stress_history.append(0.0)
                self.obi_history.append(0.0)

            self.progress.emit(20, "Fetching HTF candles…")
            for c in self.feeder.fetch_initial_htf(limit=200):
                self.candles_htf.append(c)

            self.progress.emit(30, "Back-filling topological stress…")
            prices = [c[4] for c in self.candles]
            n = len(prices)
            for i in range(TDA_EMBED_WINDOW, n):
                embedded = self._build_embedding(prices[i - TDA_EMBED_WINDOW:i], self.tau, self.dim)
                if len(embedded) > 0:
                    self.stress_history[i] = self.tda.get_topological_stress(embedded)
                if (i - TDA_EMBED_WINDOW) % 50 == 0:
                    pct = 30 + int(50 * (i - TDA_EMBED_WINDOW) / max(n - TDA_EMBED_WINDOW, 1))
                    self.progress.emit(pct, f"TDA backfill {i}/{n}…")

            self.progress.emit(85, "Training initial ML model…")
            self.ml.train(
                self.candles, self.stress_history, self.obi_history,
                candles_htf=self.candles_htf,
            )
            self.progress.emit(100, "Ready.")
            self.finished.emit(True)
        except Exception as exc:
            logger.error(f"[PreloadWorker] {exc}")
            self.finished.emit(False)


class MarketStreamer(QThread):
    data_ready = pyqtSignal(list, list, list, list, dict, np.ndarray, float)
    new_candle = pyqtSignal(list)

    def __init__(self, feeder, tda, ml, tau, dim,
                 candles, stress_history, obi_history, candles_htf):
        super().__init__()
        self.feeder = feeder
        self.tda = tda
        self.ml = ml
        self.tau = tau
        self.dim = dim
        self.candles = candles
        self.stress_history = stress_history
        self.obi_history = obi_history
        self.candles_htf = candles_htf
        self.running = True
        self.tick_counter = 0
        self._retrain_lock = threading.Lock()
        self._last_phase_retrain = -999

    def _get_prices(self) -> list:
        return [c[4] for c in self.candles]

    def _get_embedding(self) -> np.ndarray:
        prices = self._get_prices()
        data = np.array(prices, dtype=float)
        if len(data) < (self.dim - 1) * self.tau + 1:
            return np.zeros((1, 3))
        data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
        return np.vstack([
            data[:-(2 * self.tau)],
            data[self.tau:-self.tau],
            data[2 * self.tau:],
        ]).T

    def _schedule_retrain(self, reason: str = "") -> None:
        candles_snap = list(self.candles)
        stress_snap = list(self.stress_history)
        obi_snap = list(self.obi_history)
        htf_snap = list(self.candles_htf)
        threading.Thread(
            target=self._retrain_background,
            args=(candles_snap, stress_snap, obi_snap, htf_snap, reason),
            daemon=True, name="ml-retrain",
        ).start()

    def _retrain_background(self, candles, stress, obi, candles_htf, reason="") -> None:
        if not self._retrain_lock.acquire(blocking=False):
            return
        try:
            logger.info(f"Retraining ML… ({reason})")
            self.ml.train(candles, stress, obi, candles_htf=candles_htf)
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
                    for c in ohlcv_htf:
                        t = c[0]
                        if not self.candles_htf or t > self.candles_htf[-1][0]:
                            self.candles_htf.append(c)
                        elif t == self.candles_htf[-1][0]:
                            self.candles_htf[-1] = c
                    if len(self.candles_htf) > 250:
                        del self.candles_htf[:-250]

                ohlcv = self.feeder.fetch_updates()
                if ohlcv:
                    for c in ohlcv:
                        t = c[0]
                        if not self.candles or t > self.candles[-1][0]:
                            self.new_candle.emit(list(c))
                            self.candles.append(list(c))
                            self.stress_history.append(0.0)
                            self.obi_history.append(obi)
                        elif t == self.candles[-1][0]:
                            self.candles[-1] = list(c)
                            self.obi_history[-1] = obi

                    if len(self.candles) > 600:
                        del self.candles[:-600]
                        del self.stress_history[:-600]
                        del self.obi_history[:-600]

                    embedded = self._get_embedding()
                    current_stress = 0.0
                    probs = {"flat": 1.0, "up": 0.0, "down": 0.0}

                    if len(embedded) > 0:
                        current_stress = self.tda.get_topological_stress(embedded[-TDA_EMBED_WINDOW:])
                        self.stress_history[-1] = current_stress
                        if self.ml.is_trained:
                            probs = self.ml.predict(
                                self.candles, self.stress_history, self.obi_history,
                                candles_htf=self.candles_htf,
                            )

                    timestamps = [c[0] for c in self.candles]
                    prices = [c[4] for c in self.candles]
                    self.data_ready.emit(
                        timestamps, prices,
                        list(self.stress_history), list(self.obi_history),
                        probs, embedded, current_stress,
                    )

                if (self.tick_counter - self._last_phase_retrain > 80
                        and detect_phase_transition(self.candles)):
                    self._last_phase_retrain = self.tick_counter
                    logger.info("⚡ Phase transition → retrain")
                    self._schedule_retrain("phase-transition")

                self.tick_counter += 1
                if self.tick_counter % 200 == 0 and len(self.candles) > 100:
                    self._schedule_retrain("scheduled")

            except Exception as exc:
                logger.error(f"STREAMER: {exc}")

            self.msleep(1_000)

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
            title=f"Live Price — Candlesticks ({self.symbol})",
            axisItems={"bottom": pg.DateAxisItem()},
        )
        self.plot_price.showGrid(x=True, y=True)
        self.plot_price.setBackground("#080810")

        self.candle_item = CandlestickItem()
        self.plot_price.addItem(self.candle_item)

        self.buy_scatter = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(0, 255, 0), symbol="t1")
        self.sell_scatter = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(255, 50, 50), symbol="t")
        self.plot_price.addItem(self.buy_scatter)
        self.plot_price.addItem(self.sell_scatter)

        self.horizon_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("y", style=Qt.DashLine))
        self.sl_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(pg.mkColor(255, 80, 80), width=2, style=Qt.DashLine))
        self.tp_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(pg.mkColor(80, 255, 80), width=2, style=Qt.DashLine))
        for line in [self.horizon_line, self.sl_line, self.tp_line]:
            line.hide()
            self.plot_price.addItem(line)

        self.pending_label = QLabel("")
        self.pending_label.setStyleSheet(
            "color:#FF8800; font-size:15px; font-weight:bold; padding:2px;"
        )
        left_layout.addWidget(self.pending_label)
        left_layout.addWidget(self.plot_price, stretch=3)

        self.plot_stress = pg.PlotWidget(
            title="Topological Stress (H₁ persistence)",
            axisItems={"bottom": pg.DateAxisItem()},
        )
        self.plot_stress.showGrid(x=True, y=True)
        self.plot_stress.setBackground("#080810")
        self.stress_curve = self.plot_stress.plot(
            pen=pg.mkPen("r", width=2), fillLevel=0, brush=(255, 0, 0, 50)
        )
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
        self.plot_obi.setBackground("#080810")
        self.obi_curve = self.plot_obi.plot(
            pen=pg.mkPen("c", width=2), fillLevel=0, brush=(0, 255, 255, 50)
        )
        self.plot_obi.setXLink(self.plot_price)
        self.plot_obi.addItem(
            pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen("w", width=1, style=Qt.DashLine))
        )
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
    def __init__(
            self,
            symbol: str = "BTC/USDT",
            timeframe: str = "1m",
            htf: str = "15m",
    ):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.htf = htf
        self.tau = 5
        self.dim = 3
        self.alpha_stress_threshold = 1.5
        self._ENTRY_PROB_THRESHOLD = 0.62

        self.feeder = RobustDataFeeder(symbol, timeframe, htf=htf)
        self.tda = TDAAnalyzer()
        self.ml = TopoBooster()
        self.trader = PaperTrader(
            initial_balance=10_000.0,
            margin_usdt=50.0,
            leverage=10,
            horizon_bars=12,
            timeframe_mins=1,
            sl_atr_mult=1.5,
            tp_atr_mult=3.0,
        )
        self.notifier = TelegramNotifier()

        self.candles: list = []
        self.candles_htf: list = []
        self.stress_history: list = []
        self.obi_history: list = []

        self._last_ts: int = 0
        self._scatter_tick: int = 0
        self.worker: MarketStreamer | None = None
        self._calc_window = None

        self.ui = DashboardUI(self.symbol, self.alpha_stress_threshold)
        self.setCentralWidget(self.ui)
        self.setWindowTitle(f"TopoAlpha v2 — {self.symbol}  [{htf} macro]  LTF:{timeframe}")
        self.resize(1680, 980)
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
            self.candles, self.stress_history, self.obi_history,
            self.candles_htf,
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
        p = self.trader.entry_price
        t = self.trader.entry_time / 1_000.0
        if self.trader.position == "LONG":
            self.ui.buy_markers_x.append(t)
            self.ui.buy_markers_y.append(p)
            self.ui.tp_line.setPos(self.trader.current_tp_price or p * 1.01)
            self.ui.sl_line.setPos(self.trader.current_sl_price or p * 0.99)
        else:
            self.ui.sell_markers_x.append(t)
            self.ui.sell_markers_y.append(p)
            self.ui.tp_line.setPos(self.trader.current_tp_price or p * 0.99)
            self.ui.sl_line.setPos(self.trader.current_sl_price or p * 1.01)
        horizon_s = (self.trader.entry_time + self.trader.horizon_ms) / 1_000.0
        self.ui.horizon_line.setPos(horizon_s)
        for line in [self.ui.horizon_line, self.ui.tp_line, self.ui.sl_line]:
            line.show()

    def _start_worker(self) -> None:
        self.worker = MarketStreamer(
            self.feeder, self.tda, self.ml, self.tau, self.dim,
            self.candles, self.stress_history, self.obi_history,
            self.candles_htf,
        )
        self.worker.data_ready.connect(self._update_interface)
        self.worker.new_candle.connect(self._on_new_candle)
        self.worker.start()

        self.notifier.send_message(
            f"🟢 <b>TopoAlpha v2 Started</b>\n"
            f"Symbol: {self.symbol}  TF: {self.timeframe}  HTF: {self.htf}\n"
            f"Balance: ${self.trader.balance:.2f}"
        )

    def _on_new_candle(self, candle: list) -> None:
        open_price = candle[1]
        curr_ms = candle[0]
        executed = self.trader.execute_at_open(open_price, curr_ms)

        if executed:
            side = self.trader.position
            curr_t = curr_ms / 1_000.0
            if side == "LONG":
                self.ui.buy_markers_x.append(curr_t)
                self.ui.buy_markers_y.append(open_price)
                self.ui.tp_line.setPos(self.trader.current_tp_price or open_price * 1.01)
                self.ui.sl_line.setPos(self.trader.current_sl_price or open_price * 0.99)
                icon, prob_label = "🚀", "LONG"
            else:
                self.ui.sell_markers_x.append(curr_t)
                self.ui.sell_markers_y.append(open_price)
                self.ui.tp_line.setPos(self.trader.current_tp_price or open_price * 0.99)
                self.ui.sl_line.setPos(self.trader.current_sl_price or open_price * 1.01)
                icon, prob_label = "🩸", "SHORT"

            horizon_s = (curr_ms + self.trader.horizon_ms) / 1_000.0
            self.ui.horizon_line.setPos(horizon_s)
            for line in [self.ui.horizon_line, self.ui.tp_line, self.ui.sl_line]:
                line.show()

            logger.info(f"{icon} ENTERED {prob_label} @ {open_price:.2f} (candle open)")
            self.notifier.send_message(
                f"{icon} <b>ENTERED {prob_label} @ {open_price:.2f}</b>\n"
                f"SL: {self.trader.sl_pct * 100:.2f}%  TP: {self.trader.tp_pct * 100:.2f}%"
            )
            self.ui.pending_label.setText("")
        else:
            if self.trader.pending_signal:
                sig = self.trader.pending_signal
                self.ui.pending_label.setText(
                    f"⏳ PENDING {sig} — will execute at next candle open"
                )

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

            candle_display = [
                [c[0] / 1_000.0, c[1], c[2], c[3], c[4]]
                for c in self.candles[-200:]
            ]
            self.ui.candle_item.setData(candle_display)

            self.ui.stress_curve.setData(times_sec, stress_history)
            self.ui.obi_curve.setData(times_sec, obi_history)

            htf_prices = [c[4] for c in self.candles_htf]
            htf_dir = ""
            if len(htf_prices) >= 21:
                s = pd.Series(htf_prices)
                ema8 = s.ewm(span=8, adjust=False).mean().iloc[-1]
                ema21 = s.ewm(span=21, adjust=False).mean().iloc[-1]
                htf_dir = "📈HTF↑" if ema8 > ema21 else "📉HTF↓"

            pending_txt = f"  ⏳{self.trader.pending_signal}" if self.trader.pending_signal else ""
            self.ui.signal_label.setText(
                f"UP: {probs['up']:.1%}  |  FLAT: {probs['flat']:.1%}  |  "
                f"DOWN: {probs['down']:.1%}   STRESS: {current_stress:.2f}  "
                f"{htf_dir}{pending_txt}"
            )

            res = self.trader.update(curr_p, curr_ms)
            if res:
                logger.info(
                    f"CLOSED ({res['reason']})  P&L: ${res['net_profit_usd']:.2f}  "
                    f"Bal: ${self.trader.balance:.2f}"
                )
                for line in [self.ui.horizon_line, self.ui.sl_line, self.ui.tp_line]:
                    line.hide()
                icon = "✅" if res["net_profit_usd"] > 0 else "🛑"
                self.notifier.send_message(
                    f"{icon} <b>TRADE CLOSED ({res['reason']})</b>\n"
                    f"Net P&L: ${res['net_profit_usd']:.2f}\n"
                    f"Balance: ${self.trader.balance:.2f}"
                )

            if (current_stress >= self.alpha_stress_threshold
                    and self.trader.position is None
                    and self.trader.pending_signal is None):

                htf_prices_list = [c[4] for c in self.candles_htf]
                htf_bull = False;
                htf_bear = False
                if len(htf_prices_list) >= 21:
                    s = pd.Series(htf_prices_list)
                    gap = (s.ewm(span=8).mean().iloc[-1] - s.ewm(span=21).mean().iloc[-1])
                    gap_pct = gap / (s.iloc[-1] + 1e-10)
                    htf_bull = gap_pct > 0.001
                    htf_bear = gap_pct < -0.001

                atr_pct = self._current_atr_pct()
                if probs["up"] >= self._ENTRY_PROB_THRESHOLD and htf_bull:
                    self.trader.set_pending("LONG", atr_pct)
                    self.ui.pending_label.setText(
                        f"⏳ PENDING LONG — will execute at next candle open"
                    )
                elif probs["down"] >= self._ENTRY_PROB_THRESHOLD and htf_bear:
                    self.trader.set_pending("SHORT", atr_pct)
                    self.ui.pending_label.setText(
                        f"⏳ PENDING SHORT — will execute at next candle open"
                    )

            net_pnl = self.trader.get_unrealized_pnl(curr_p)
            color = "#00FF00" if net_pnl >= 0 else "#FF0000"
            self.ui.portfolio_label.setText(
                f"BALANCE: ${self.trader.balance:.2f}  |  "
                f"NET PNL: <font color='{color}'>${net_pnl:.2f}</font>  |  "
                f"POS: {self.trader.position or 'NONE'}"
            )

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

    def _current_atr_pct(self, period: int = 14) -> float:
        if len(self.candles) < period + 1:
            return 0.003
        prices = [c[4] for c in self.candles[-(period + 1):]]
        arr = np.array(prices, dtype=float)
        atr = float(np.abs(np.diff(arr)).mean())
        return atr / (prices[-1] + 1e-10)

    def _on_threshold_changed(self, value: int) -> None:
        self.alpha_stress_threshold = value / 10.0
        self.ui.stress_label.setText(f"STRESS THRESHOLD: {self.alpha_stress_threshold:.1f}")
        self.ui.stress_threshold_line.setPos(self.alpha_stress_threshold)

    def _open_calculator(self) -> None:
        from trade_calculator import TradeCalculatorWindow
        prices = [c[4] for c in self.candles]
        htf_px = [c[4] for c in self.candles_htf]
        if self._calc_window is None:
            self._calc_window = TradeCalculatorWindow(parent=None)
        self._calc_window.feed(prices, self.stress_history, self.obi_history,
                               htf_px, self.ml, self.trader)
        self._calc_window.show()
        self._calc_window.raise_()

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
