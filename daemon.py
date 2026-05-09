import time
import logging
import threading
import numpy as np

from data_feeder import RobustDataFeeder
from tda_core import TDAAnalyzer
from ml_model import TopoBooster, detect_phase_transition
from paper_trader import PaperTrader
from notifier import TelegramNotifier
from risk_manager import RiskManager
from wfo import WalkForwardOptimizer, run_pre_live_wfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TopoAlpha.Daemon")
SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
HTF = "1h"
TIMEFRAME_MINS = 15
HORIZON_BARS = 48

ENTRY_PROB_THRESHOLD = 0.45
OBI_CONFIRM_MIN = 0.04
RETRAIN_EVERY_TICKS = 200
PHASE_RETRAIN_COOLDOWN = 80

WFO_ENABLED = True
WFO_FOLDS = 5
WFO_MODE = "rolling"
WFO_OPTIMIZE = False
WFO_MIN_SHARPE = 0.30
WFO_MIN_PF = 1.10
POLL_INTERVAL_S = 1

HTF_MIN_CANDLES = 21
TDA_EMBED_WINDOW = 50

_BACKOFF_BASE_S = 2.0
_BACKOFF_MAX_S = 60.0
_BACKOFF_RETRIES = 5


def _with_backoff(fn, *args, label: str = "", **kwargs):
    delay = _BACKOFF_BASE_S
    for attempt in range(_BACKOFF_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == _BACKOFF_RETRIES - 1:
                logger.error(f"[Backoff] {label} failed after {_BACKOFF_RETRIES} tries: {exc}")
                return None
            logger.warning(f"[Backoff] {label} attempt {attempt + 1} failed ({exc}). "
                           f"Retrying in {delay:.0f}s…")
            time.sleep(delay)
            delay = min(delay * 2, _BACKOFF_MAX_S)
    return None


class TopoAlphaDaemon:

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
        self._base_stress_threshold = 0.3

        self.feeder = RobustDataFeeder(symbol, timeframe, htf=htf)
        self.tda = TDAAnalyzer()
        self.ml = TopoBooster()
        self.trader = PaperTrader(
            initial_balance=10_000.0,
            margin_usdt=50.0,
            leverage=10,
            horizon_bars=HORIZON_BARS,
            timeframe_mins=TIMEFRAME_MINS,
            sl_atr_mult=2.0,
            tp_atr_mult=3.5,
        )
        self.notifier = TelegramNotifier()
        self.risk = RiskManager(initial_balance=10_000.0)

        self.candles: list = []
        self.candles_htf: list = []
        self.stress_history: list = []
        self.obi_history: list = []
        self.prices: list = []
        self.prices_htf: list = []
        self.timestamps: list = []

        self._last_candle_ts: int = 0
        self.tick_counter: int = 0
        self._retrain_lock = threading.Lock()
        self._last_phase_retrain: int = -PHASE_RETRAIN_COOLDOWN
        self._wfo_passed: bool = not WFO_ENABLED

        self._pending_signal_context: dict = {}

    def _htf_trend(self) -> str:
        if len(self.prices_htf) < HTF_MIN_CANDLES:
            return "flat"
        arr = np.array(self.prices_htf, dtype=float)
        alpha8 = 2.0 / 9
        alpha21 = 2.0 / 22
        ema8 = arr[0]
        ema21 = arr[0]
        for p in arr[1:]:
            ema8 = alpha8 * p + (1 - alpha8) * ema8
            ema21 = alpha21 * p + (1 - alpha21) * ema21
        gap = (ema8 - ema21) / (ema21 + 1e-10)
        if gap > 0.001: return "bull"
        if gap < -0.001: return "bear"
        return "flat"

    def _mean_reversion_ok(self, side: str, period_rsi: int = 14, period_bb: int = 20) -> bool:
        """Return True when the current bar shows an overextended move in `side` direction.
        SHORT: price has rallied to overbought territory (RSI>55 or above BB mid).
        LONG:  price has dropped to oversold territory (RSI<45 or below BB mid)."""
        closes = [c[4] for c in self.candles]
        n = len(closes)
        if n < max(period_rsi, period_bb) + 2:
            return True

        arr = np.array(closes, dtype=float)
        delta = np.diff(arr)
        gains = np.where(delta > 0, delta, 0.0)[-period_rsi:]
        losses = np.where(delta < 0, -delta, 0.0)[-period_rsi:]
        avg_g = gains.mean()
        avg_l = losses.mean()
        rsi = 100.0 - 100.0 / (1.0 + avg_g / (avg_l + 1e-8))

        window = arr[-period_bb:]
        mu = window.mean()
        sig = window.std()
        bb_pos = (arr[-1] - mu) / (2.0 * sig + 1e-8)

        if side == "SHORT":
            return rsi > 55 or bb_pos > 0.15
        else:  # LONG
            return rsi < 45 or bb_pos < -0.15

    def _current_atr_pct(self, period: int = 14) -> float:
        if len(self.candles) < period + 1:
            return 0.003
        recent = self.candles[-(period + 1):]
        tr_vals = []
        for i in range(1, len(recent)):
            h = recent[i][2]
            l = recent[i][3]
            prev_c = recent[i - 1][4]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            tr_vals.append(tr)
        atr_abs = float(np.mean(tr_vals))
        return atr_abs / (self.prices[-1] + 1e-10)

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

    def _trim(self, limit: int = 600) -> None:
        if len(self.candles) > limit:
            self.candles = self.candles[-limit:]
            self.prices = self.prices[-limit:]
            self.timestamps = self.timestamps[-limit:]
            self.stress_history = self.stress_history[-limit:]
            self.obi_history = self.obi_history[-limit:]
        if len(self.candles_htf) > 250:
            self.candles_htf = self.candles_htf[-250:]
            self.prices_htf = self.prices_htf[-250:]

    def _schedule_retrain(self) -> None:
        candles_snap = list(self.candles)
        stress_snap = list(self.stress_history)
        obi_snap = list(self.obi_history)
        htf_snap = list(self.candles_htf)
        threading.Thread(
            target=self._retrain_background,
            args=(candles_snap, stress_snap, obi_snap, htf_snap),
            daemon=True, name="ml-retrain",
        ).start()

    def _retrain_background(self, candles, stress, obi, candles_htf) -> None:
        if not self._retrain_lock.acquire(blocking=False):
            logger.info("Retrain already in progress — skipping.")
            return
        try:
            logger.info("Retraining ML in background…")
            ok = self.ml.train(candles, stress, obi, candles_htf=candles_htf)
            logger.info(f"Retrain {'complete' if ok else 'skipped (insufficient data)'}.")
        except Exception as exc:
            logger.error(f"Retrain failed: {exc}")
        finally:
            self._retrain_lock.release()

    def _maybe_phase_retrain(self) -> None:
        if (self.tick_counter - self._last_phase_retrain) <= PHASE_RETRAIN_COOLDOWN:
            return
        if detect_phase_transition(self.candles):
            self._last_phase_retrain = self.tick_counter
            logger.info("⚡ Phase transition detected → immediate retrain")
            self._schedule_retrain()

    def preload_data(self) -> bool:
        logger.info("Preloading historical data…")

        ohlcv = _with_backoff(self.feeder.fetch_initial, limit=600, label="fetch_initial")
        if not ohlcv:
            logger.error("Failed to load initial LTF data.")
            return False

        for c in ohlcv:
            self.candles.append(c)
            self.timestamps.append(c[0])
            self.prices.append(c[4])
            self.stress_history.append(0.0)
            self.obi_history.append(0.0)

        self._last_candle_ts = self.timestamps[-1]

        htf_candles = _with_backoff(
            self.feeder.fetch_initial_htf, limit=200, label="fetch_initial_htf"
        ) or []
        for c in htf_candles:
            self.candles_htf.append(c)
            self.prices_htf.append(c[4])

        logger.info(f"Loaded {len(self.prices)} LTF + {len(self.prices_htf)} HTF candles.")

        logger.info("Back-filling topological stress…")
        for i in range(TDA_EMBED_WINDOW, len(self.prices)):
            embedded = self._build_embedding(self.prices[i - TDA_EMBED_WINDOW:i])
            if len(embedded) > 0:
                s = self.tda.get_topological_stress(embedded)
                self.stress_history[i] = s
                self.risk.update_stress(s)

        logger.info("Training initial ML model…")
        self.ml.train(
            self.candles, self.stress_history, self.obi_history,
            candles_htf=self.candles_htf,
        )

        if WFO_ENABLED:
            logger.info("Running Walk-Forward Validation…")
            passed = run_pre_live_wfo(
                self.candles, self.stress_history, self.obi_history,
                candles_htf=self.candles_htf,
                n_folds=WFO_FOLDS,
                mode=WFO_MODE,
                optimize=WFO_OPTIMIZE,
                min_sharpe=WFO_MIN_SHARPE,
                min_pf=WFO_MIN_PF,
            )
            if not passed:
                self.notifier.send_message(
                    "🔴 <b>WFO gates not met</b> — strategy not validated.\n"
                    "Daemon running in MONITOR-ONLY mode (no new trades)."
                )
                self._wfo_passed = False
            else:
                self._wfo_passed = True
        else:
            self._wfo_passed = True

        if self.trader.position:
            logger.info(f"Restored active {self.trader.position} from DB.")
        if self.trader.pending_signal:
            logger.info(f"Restored pending {self.trader.pending_signal} from DB.")

        self.notifier.send_message(
            f"🤖 <b>TopoAlpha v3 Started</b>\n"
            f"Symbol: {self.symbol}  TF: {self.timeframe}\n"
            f"HTF trend: {self._htf_trend()}\n"
            f"Balance: ${self.trader.balance:.2f}"
        )
        return True

    def run(self) -> None:
        if not self.preload_data():
            return
        logger.info("Headless daemon running. Ctrl+C to stop.")
        while True:
            try:
                self._tick()
            except Exception as exc:
                logger.error(f"Tick error: {exc}")
                time.sleep(5)
            else:
                time.sleep(POLL_INTERVAL_S)

    def _tick(self) -> None:
        obi = self.feeder.fetch_order_book_imbalance(depth=20)

        ohlcv_htf = _with_backoff(self.feeder.fetch_updates_htf, label="htf_update")
        if ohlcv_htf:
            for c in ohlcv_htf:
                t = c[0]
                if not self.candles_htf or t > self.candles_htf[-1][0]:
                    self.candles_htf.append(c)
                    self.prices_htf.append(c[4])
                elif t == self.candles_htf[-1][0]:
                    self.candles_htf[-1] = c
                    self.prices_htf[-1] = c[4]

        ohlcv = _with_backoff(self.feeder.fetch_updates, label="ltf_update")
        if not ohlcv:
            return

        new_candle_opened = False
        prev_candle: list | None = None

        for c in ohlcv:
            t = c[0]
            if not self.candles or t > self.candles[-1][0]:
                if self.candles:
                    prev_candle = self.candles[-1]
                new_candle_opened = True
                self.candles.append(c)
                self.timestamps.append(t)
                self.prices.append(c[4])
                self.stress_history.append(0.0)
                self.obi_history.append(obi)
            elif t == self.candles[-1][0]:
                self.candles[-1] = c
                self.prices[-1] = c[4]
                self.obi_history[-1] = obi

        curr_p = self.prices[-1]
        curr_ms = self.timestamps[-1]

        if new_candle_opened and self.candles:
            new_open = self.candles[-1][1]
            executed = self.trader.execute_at_open(new_open, curr_ms)
            if executed:
                side = self.trader.position
                icon = "🚀" if side == "LONG" else "🩸"
                logger.info(
                    f"{icon} EXECUTED {side} @ OPEN {new_open:.2f}  "
                    f"SL: {self.trader.sl_pct * 100:.2f}%  "
                    f"TP: {self.trader.tp_pct * 100:.2f}%"
                )
                self.notifier.send_message(
                    f"{icon} <b>ENTERED {side}</b>\n"
                    f"Price: {new_open:.2f}\n"
                    f"SL: {self.trader.sl_pct * 100:.2f}%  "
                    f"TP: {self.trader.tp_pct * 100:.2f}%\n"
                    f"{self.risk.status_line(self.trader.balance)}"
                )

        embedded = self._build_embedding(self.prices[-TDA_EMBED_WINDOW:])
        current_stress = 0.0
        probs = {"flat": 1.0, "up": 0.0, "down": 0.0}

        if len(embedded) > 0:
            current_stress = self.tda.get_topological_stress(embedded)
            self.stress_history[-1] = current_stress
            self.risk.update_stress(current_stress)

            if self.ml.is_trained:
                probs = self.ml.predict(
                    self.candles, self.stress_history, self.obi_history,
                    candles_htf=self.candles_htf,
                )

        res = self.trader.update(curr_p, curr_ms)
        if res:
            icon = "✅" if res["net_profit_usd"] > 0 else "🛑"
            logger.info(
                f"CLOSED ({res['reason']})  "
                f"P&L: ${res['net_profit_usd']:.2f}  "
                f"Balance: ${self.trader.balance:.2f}"
            )
            self.notifier.send_message(
                f"{icon} <b>TRADE CLOSED ({res['reason']})</b>\n"
                f"Net P&L: ${res['net_profit_usd']:.2f}\n"
                f"Balance: ${self.trader.balance:.2f}\n"
                f"{self.risk.status_line(self.trader.balance)}"
            )
            self.risk.update_after_trade(res, self.trader.balance)
            self.risk.log_trade(res, self.trader.balance, self._pending_signal_context)
            self._pending_signal_context = {}

        if new_candle_opened and prev_candle is not None:
            htf_trend = self._htf_trend()
            atr_pct = self._current_atr_pct()
            no_pos = self.trader.position is None
            no_pending = self.trader.pending_signal is None

            threshold = self.risk.adaptive_stress_threshold(
                base=self._base_stress_threshold
            )
            stress_ok = current_stress >= threshold

            if stress_ok and no_pos and no_pending and self._wfo_passed and not self.risk.is_halted():
                ml_up = probs["up"] >= ENTRY_PROB_THRESHOLD
                ml_down = probs["down"] >= ENTRY_PROB_THRESHOLD

                if ml_up and obi <= -OBI_CONFIRM_MIN:
                    if self._mean_reversion_ok("SHORT"):
                        ctx = {
                            "stress": current_stress,
                            "prob_up": probs["up"],
                            "prob_down": probs["down"],
                            "htf_trend": htf_trend,
                            "atr_pct": atr_pct,
                            "obi": obi,
                        }
                        self.trader.set_pending("SHORT", atr_pct)
                        self._pending_signal_context = ctx
                        logger.info(
                            f"↓ SIGNAL SHORT (fade rally, pending)  "
                            f"Stress: {current_stress:.2f}/{threshold:.2f}  "
                            f"P(up): {probs['up']:.2f}  "
                            f"HTF: {htf_trend}  OBI: {obi:.3f}  "
                            f"ATR: {atr_pct * 100:.3f}%"
                        )

                elif ml_down and obi >= OBI_CONFIRM_MIN:
                    if self._mean_reversion_ok("LONG"):
                        ctx = {
                            "stress": current_stress,
                            "prob_up": probs["up"],
                            "prob_down": probs["down"],
                            "htf_trend": htf_trend,
                            "atr_pct": atr_pct,
                            "obi": obi,
                        }
                        self.trader.set_pending("LONG", atr_pct)
                        self._pending_signal_context = ctx
                        logger.info(
                            f"↑ SIGNAL LONG (buy dip, pending)  "
                            f"Stress: {current_stress:.2f}/{threshold:.2f}  "
                            f"P(dn): {probs['down']:.2f}  "
                            f"HTF: {htf_trend}  OBI: {obi:.3f}  "
                            f"ATR: {atr_pct * 100:.3f}%"
                        )
            elif stress_ok and self.risk.is_halted():
                logger.info("🔴 Signal suppressed — circuit breaker active.")

        self._trim()
        self._maybe_phase_retrain()

        self.tick_counter += 1
        if self.tick_counter % RETRAIN_EVERY_TICKS == 0 and len(self.candles) > 100:
            self._schedule_retrain()


if __name__ == "__main__":
    TopoAlphaDaemon().run()
