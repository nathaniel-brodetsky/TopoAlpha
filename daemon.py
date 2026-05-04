import time
import logging
import threading
import numpy as np

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
logger = logging.getLogger("TopoAlpha.Daemon")

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
HTF = "1h"
TIMEFRAME_MINS = 5
HORIZON_BARS = 6
RETRAIN_EVERY = 300
POLL_INTERVAL_S = 2

ENTRY_PROB_THRESHOLD = 0.65

HTF_MIN_CANDLES = 21


class TopoAlphaDaemon:
    """Headless trading daemon — runs without a GUI, suitable for servers.

    Start with:
        python daemon.py

    All trades are paper-simulated and persisted in trading_state.db.
    Telegram alerts fire on every entry/exit when credentials are set in .env.
    """

    def __init__(
            self,
            symbol: str = SYMBOL,
            timeframe: str = TIMEFRAME,
            htf: str = HTF,
    ):
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
            horizon_bars=HORIZON_BARS,
            timeframe_mins=TIMEFRAME_MINS,
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
        self.tick_counter: int = 0
        self._retrain_lock = threading.Lock()

    # ── HTF trend filter ────────────────────────────────────────────────── #

    def _htf_trend(self) -> str:
        """Return 'bull', 'bear' or 'flat' based on EMA8 vs EMA21 on HTF closes.

        Uses simple arithmetic means over the last N closes as a lightweight
        proxy for EMA — accurate enough given we only need direction.
        Falls back to 'flat' (= no trade) when data is insufficient.
        """
        if len(self.prices_htf) < HTF_MIN_CANDLES:
            return "flat"

        ema8 = float(np.mean(self.prices_htf[-8:]))
        ema21 = float(np.mean(self.prices_htf[-21:]))

        gap_pct = (ema8 - ema21) / (ema21 + 1e-10)
        if gap_pct > 0.001:
            return "bull"
        elif gap_pct < -0.001:
            return "bear"
        return "flat"

    # ── Lifecycle ───────────────────────────────────────────────────────── #

    def preload_data(self) -> bool:
        logger.info("Preloading historical data…")

        ohlcv = self.feeder.fetch_initial(limit=500)
        if not ohlcv:
            logger.error("Failed to load initial LTF data.")
            return False

        for candle in ohlcv:
            self.timestamps.append(candle[0])
            self.prices.append(candle[4])
            self.stress_history.append(0.0)
            self.obi_history.append(0.0)

        self.last_processed_time = self.timestamps[-1]

        for candle in self.feeder.fetch_initial_htf(limit=150):
            self.timestamps_htf.append(candle[0])
            self.prices_htf.append(candle[4])

        logger.info(f"Loaded {len(self.prices)} LTF + {len(self.prices_htf)} HTF candles.")

        logger.info("Back-filling topological stress…")
        for i in range(50, len(self.prices)):
            embedded = self._build_embedding(self.prices[i - 50:i])
            if len(embedded) > 0:
                self.stress_history[i] = self.tda.get_topological_stress(embedded)

        logger.info("Training initial ML model…")
        self.ml.train(self.prices, self.stress_history, self.obi_history, prices_htf=self.prices_htf)

        if self.trader.position:
            logger.info(f"Restored active {self.trader.position} position from database.")

        htf_trend = self._htf_trend()
        self.notifier.send_message(
            f"🤖 <b>TopoAlpha Started</b>\n"
            f"Symbol: {self.symbol}\n"
            f"HTF trend: {htf_trend}\n"
            f"Balance: ${self.trader.balance:.2f}"
        )
        return True

    def _build_embedding(self, price_window: list) -> np.ndarray:
        data = np.array(price_window, dtype=float)
        if len(data) < (self.dim - 1) * self.tau + 1:
            return np.zeros((1, 3))
        data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
        return np.vstack([
            data[:-(2 * self.tau)],
            data[self.tau:-self.tau],
            data[2 * self.tau:],
        ]).T

    def _trim(self, limit: int = 500) -> None:
        if len(self.prices) > limit:
            self.timestamps = self.timestamps[-limit:]
            self.prices = self.prices[-limit:]
            self.stress_history = self.stress_history[-limit:]
            self.obi_history = self.obi_history[-limit:]
        if len(self.prices_htf) > 200:
            self.timestamps_htf = self.timestamps_htf[-200:]
            self.prices_htf = self.prices_htf[-200:]

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
        if not self.preload_data():
            return
        logger.info("Running in headless mode. Press Ctrl+C to stop.")
        while True:
            try:
                self._tick()
            except Exception as exc:
                logger.error(f"Tick error: {exc}")
                time.sleep(5)
            else:
                time.sleep(POLL_INTERVAL_S)

    # ── Core tick ───────────────────────────────────────────────────────── #

    def _tick(self) -> None:
        obi = self.feeder.fetch_order_book_imbalance(depth=20)

        ohlcv_htf = self.feeder.fetch_updates_htf()
        if ohlcv_htf:
            for t, _, _, _, c, _ in ohlcv_htf:
                if not self.timestamps_htf or t > self.timestamps_htf[-1]:
                    self.timestamps_htf.append(t)
                    self.prices_htf.append(c)
                elif t == self.timestamps_htf[-1]:
                    self.prices_htf[-1] = c

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

            embedded = self._build_embedding(self.prices)
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

            curr_p = self.prices[-1]
            curr_ms = self.timestamps[-1]

            res = self.trader.update(curr_p, curr_ms)
            if res:
                icon = "✅" if res["net_profit_usd"] > 0 else "🛑"
                logger.info(
                    f"CLOSED ({res['reason']})  P&L: ${res['net_profit_usd']:.2f}  "
                    f"Balance: ${self.trader.balance:.2f}"
                )
                self.notifier.send_message(
                    f"{icon} <b>TRADE CLOSED ({res['reason']})</b>\n"
                    f"Net P&L: ${res['net_profit_usd']:.2f}\n"
                    f"Balance: ${self.trader.balance:.2f}"
                )

            if current_stress >= self.alpha_stress_threshold and not self.trader.position:
                htf_trend = self._htf_trend()

                if probs["up"] >= ENTRY_PROB_THRESHOLD and htf_trend == "bull":
                    self._open_trade("LONG", curr_p, curr_ms, current_stress, probs["up"], htf_trend)
                elif probs["down"] >= ENTRY_PROB_THRESHOLD and htf_trend == "bear":
                    self._open_trade("SHORT", curr_p, curr_ms, current_stress, probs["down"], htf_trend)
                else:
                    dominant = "up" if probs["up"] > probs["down"] else "down"
                    logger.debug(
                        f"Signal suppressed — HTF: {htf_trend}  "
                        f"P(up): {probs['up']:.2f}  P(down): {probs['down']:.2f}"
                    )

        self._trim()

        self.tick_counter += 1
        if self.tick_counter % RETRAIN_EVERY == 0 and len(self.prices) > 100:
            self._schedule_retrain()

    def _open_trade(
            self,
            side: str,
            price: float,
            timestamp: int,
            stress: float,
            prob: float,
            htf_trend: str,
    ) -> None:
        if not self.trader.execute_trade(side, price, timestamp):
            return

        icon = "🚀" if side == "LONG" else "🩸"
        prob_label = "UP" if side == "LONG" else "DOWN"
        logger.info(
            f"OPEN {side} @ {price}  Stress: {stress:.2f}  "
            f"P({prob_label}): {prob:.2f}  HTF: {htf_trend}"
        )
        self.notifier.send_message(
            f"{icon} <b>OPEN {side}</b>\n"
            f"Price: {price}\n"
            f"Stress: {stress:.2f}\n"
            f"P({prob_label}): {prob:.1%}\n"
            f"HTF trend: {htf_trend}"
        )


if __name__ == "__main__":
    TopoAlphaDaemon().run()