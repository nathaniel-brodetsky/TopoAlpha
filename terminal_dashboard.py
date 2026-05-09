#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TopoAlpha  ─  Terminal Dashboard  v3.0                           ║
║                                                                              ║
║  Полноценный инструмент ручной торговли на терминале.                       ║
║  Отображает все механизмы теории: TDA, ML Ensemble, HTF Trend,             ║
║  индикаторы, сигналы и риск-менеджмент.                                    ║
║  Без привязки к Binance API — только публичные данные (ccxt).               ║
║                                                                              ║
║  Зависимости:  pip install textual plotext rich ccxt ripser scikit-learn    ║
║                 pandas numpy python-dotenv requests                          ║
║                                                                              ║
║  Запуск:       python terminal_dashboard.py                                 ║
║                                                                              ║
║  Горячие клавиши:                                                            ║
║    L  ──  Открыть LONG       S  ──  Открыть SHORT                          ║
║    C  ──  Закрыть позицию    R  ──  Сбросить сигнал                        ║
║    T  ──  Сменить таймфрейм  Q  ──  Выход                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import csv
import logging
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotext as plx
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Label, Static

try:
    from data_feeder  import RobustDataFeeder
    from tda_core     import TDAAnalyzer
    from ml_model     import TopoBooster, detect_phase_transition
    from paper_trader import PaperTrader
    from risk_manager import RiskManager
    from notifier     import TelegramNotifier
except ImportError as e:
    print(f"\n[ОШИБКА] Не найден модуль проекта: {e}")
    print("Убедитесь, что terminal_dashboard.py находится в одной папке с остальными файлами.\n")
    sys.exit(1)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("TopoAlpha.Dashboard")

SYMBOL               = "BTC/USDT"
TIMEFRAMES           = ["1m", "5m", "15m", "1h"]
DEFAULT_TF_IDX       = 2          # 15m
HTF                  = "1h"
TIMEFRAME_MINS_MAP   = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
HORIZON_BARS         = 48
ENTRY_PROB_THRESHOLD = 0.45
OBI_CONFIRM_MIN      = 0.04
RETRAIN_EVERY_TICKS  = 200
PHASE_RETRAIN_CD     = 80
TDA_EMBED_WINDOW     = 50
POLL_INTERVAL_S      = 3.0
CHART_BARS           = 70

INITIAL_BALANCE      = 10_000.0
MARGIN_USDT          = 50.0
LEVERAGE             = 10

JOURNAL_PATH         = "dashboard_journal.csv"
JOURNAL_COLS = [
    "unix_ts", "datetime", "side",
    "entry_usd", "exit_usd", "reason",
    "net_usd", "balance_after",
    "stress", "prob_up", "prob_dn",
    "htf_trend", "atr_pct", "obi",
]

_BLOCKS = " ▁▂▃▄▅▆▇█"

def _sparkline(values: list, width: int = 40, low: float | None = None,
               high: float | None = None) -> str:
    if not values:
        return "─" * width
    arr = np.array(values[-width:], dtype=float)
    if len(arr) < width:
        arr = np.pad(arr, (width - len(arr), 0), mode="edge")
    mn = low  if low  is not None else arr.min()
    mx = high if high is not None else arr.max()
    if mx <= mn:
        return "─" * width
    norm = (arr - mn) / (mx - mn)
    return "".join(_BLOCKS[min(int(v * 8), 8)] for v in norm)


def _bar(value: float, total: float = 1.0, width: int = 18,
         fill: str = "█", empty: str = "░") -> str:
    filled = max(0, min(int((value / (total + 1e-10)) * width), width))
    return fill * filled + empty * (width - filled)


def _obi_bar(obi: float, width: int = 20) -> str:
    half = width // 2
    filled = min(int(abs(obi) * half), half)
    if obi >= 0:
        return "░" * half + "█" * filled + "░" * (half - filled)
    else:
        return "░" * (half - filled) + "█" * filled + "░" * half


def _stress_bar(stress: float, thresh: float, width: int = 20) -> str:
    top = max(stress, thresh * 2, 3.0)
    filled = min(int(stress / top * width), width)
    thresh_pos = min(int(thresh / top * width), width - 1)
    result = list("░" * width)
    for i in range(filled):
        result[i] = "█"
    if 0 <= thresh_pos < width:
        result[thresh_pos] = "▎"
    return "".join(result)


def _prob_row(label: str, prob: float, width: int = 16,
              color_on: str = "", color_off: str = "") -> str:
    bar = _bar(prob, 1.0, width)
    return f"{label:<8} {bar}  {prob:.1%}"


def _rsi_label(rsi: float) -> str:
    if rsi > 70: return "OB"
    if rsi < 30: return "OS"
    if rsi > 60: return "↑↑"
    if rsi < 40: return "↓↓"
    return "●"


def _price_change_arrow(p_now: float, p_prev: float) -> str:
    if p_now > p_prev:  return "▲"
    if p_now < p_prev:  return "▼"
    return "─"


def _kelly(win_prob: float, rr: float) -> float:
    if rr <= 0: return 0.0
    q = 1.0 - win_prob
    k = (win_prob * rr - q) / rr
    return max(0.0, min(k * 0.5, 0.25))


def _journal_init() -> None:
    p = Path(JOURNAL_PATH)
    if not p.exists():
        with p.open("w", newline="") as f:
            csv.writer(f).writerow(JOURNAL_COLS)


def _journal_write(row: list) -> None:
    try:
        with open(JOURNAL_PATH, "a", newline="") as f:
            csv.writer(f).writerow(row)
    except Exception:
        pass


class StrategyCore:
    """
    Инкапсулирует весь пайплайн:
      DataFeeder → TDA → ML → RiskManager → PaperTrader
    Работает в фоновом потоке; UI читает атрибуты без блокировок
    (все запись через append/replace, не замена ссылки).
    """

    def __init__(self, timeframe: str = TIMEFRAMES[DEFAULT_TF_IDX]) -> None:
        self.timeframe  = timeframe
        self.tf_mins    = TIMEFRAME_MINS_MAP.get(timeframe, 15)
        self.tau        = 5
        self.dim        = 3
        self._base_thresh = 1.5

        self.feeder  = RobustDataFeeder(SYMBOL, timeframe, htf=HTF)
        self.tda     = TDAAnalyzer()
        self.ml      = TopoBooster()
        self.trader  = PaperTrader(
            initial_balance = INITIAL_BALANCE,
            margin_usdt     = MARGIN_USDT,
            leverage        = LEVERAGE,
            horizon_bars    = HORIZON_BARS,
            timeframe_mins  = self.tf_mins,
            sl_atr_mult     = 2.0,
            tp_atr_mult     = 3.5,
        )
        self.risk     = RiskManager(initial_balance=INITIAL_BALANCE)
        self.notifier = TelegramNotifier()

        self.candles:        list = []
        self.candles_htf:    list = []
        self.stress_history: list = []
        self.obi_history:    list = []
        self.prices:         list = []
        self.prices_htf:     list = []
        self.timestamps:     list = []

        self.stress_buf: deque = deque(maxlen=CHART_BARS)
        self.obi_buf:    deque = deque(maxlen=CHART_BARS)

        self.current_stress: float = 0.0
        self.current_obi:    float = 0.0
        self.probs:   dict = {"flat": 1.0, "up": 0.0, "down": 0.0}
        self.signal:  str  = "WAIT"          # LONG / SHORT / WAIT / HALTED
        self.signal_ctx: dict = {}
        self.htf_trend:  str  = "flat"
        self.htf_ema8:  float = 0.0
        self.htf_ema21: float = 0.0
        self.phase_transition: bool = False

        self.tick_counter:  int  = 0
        self.status_msg:    str  = "Инициализация…"
        self.loaded:        bool = False
        self.trade_log:     list = []        # закрытые сделки, новые в начале

        self._retrain_lock        = threading.Lock()
        self._last_phase_retrain  = -PHASE_RETRAIN_CD
        self._pending_ctx:  dict  = {}

        _journal_init()

    def _embed(self, prices: list) -> np.ndarray:
        data = np.array(prices, dtype=float)
        if len(data) < (self.dim - 1) * self.tau + 1:
            return np.zeros((1, 3))
        data = (data - data.mean()) / (data.std() + 1e-8) * 10
        return np.vstack([data[:-(2*self.tau)],
                          data[self.tau:-self.tau],
                          data[2*self.tau:]]).T

    def _calc_htf_trend(self) -> None:
        if len(self.prices_htf) < 21:
            self.htf_trend = "flat"
            self.htf_ema8  = 0.0
            self.htf_ema21 = 0.0
            return
        arr    = np.array(self.prices_htf, dtype=float)
        a8, a21 = 2/9, 2/22
        e8, e21 = arr[0], arr[0]
        for p in arr[1:]:
            e8  = a8 * p + (1-a8) * e8
            e21 = a21 * p + (1-a21) * e21
        gap = (e8 - e21) / (e21 + 1e-10)
        self.htf_ema8  = e8
        self.htf_ema21 = e21
        if gap >  0.001: self.htf_trend = "bull"
        elif gap < -0.001: self.htf_trend = "bear"
        else: self.htf_trend = "flat"

    def _atr_pct(self, period: int = 14) -> float:
        if len(self.candles) < period + 1:
            return 0.003
        recent = self.candles[-(period+1):]
        trs = [
            max(c[2]-c[3], abs(c[2]-recent[i][4]), abs(c[3]-recent[i][4]))
            for i, c in enumerate(recent[1:], 0)
        ]
        return float(np.mean(trs)) / (self.prices[-1] + 1e-10)

    def _rsi(self, period: int = 14) -> float:
        if len(self.prices) < period + 2:
            return 50.0
        closes = np.array(self.prices[-(period+2):])
        delta  = np.diff(closes)
        ag = np.where(delta > 0, delta, 0.0).mean()
        al = np.where(delta < 0, -delta, 0.0).mean()
        return 100.0 - 100.0 / (1.0 + ag / (al + 1e-8))

    def _macd_hist(self) -> float:
        if len(self.prices) < 26:
            return 0.0
        s     = pd.Series(self.prices)
        ml    = s.ewm(span=12, adjust=False).mean()
        sl    = ml.ewm(span=9,  adjust=False).mean()
        hist  = (ml - sl).iloc[-1]
        return float(hist / (self.prices[-1] + 1e-8))

    def _stoch(self) -> float:
        if len(self.candles) < 14:
            return 0.5
        h14 = max(c[2] for c in self.candles[-14:])
        l14 = min(c[3] for c in self.candles[-14:])
        return (self.prices[-1] - l14) / (h14 - l14 + 1e-8)

    def _bb_pos(self) -> float:
        if len(self.prices) < 20:
            return 0.0
        s = pd.Series(self.prices[-20:])
        return float((self.prices[-1] - s.mean()) / (2*s.std() + 1e-8))

    def _mean_rev_ok(self, side: str) -> bool:
        rsi = self._rsi()
        bb  = self._bb_pos()
        if side == "SHORT":
            return rsi > 55 or bb > 0.15
        return rsi < 45 or bb < -0.15

    def preload(self) -> bool:
        try:
            self.status_msg = "📡 Загрузка LTF свечей…"
            ohlcv = self.feeder.fetch_initial(limit=600)
            if not ohlcv:
                self.status_msg = "❌ Нет данных от биржи"
                return False
            for c in ohlcv:
                self.candles.append(c)
                self.prices.append(c[4])
                self.timestamps.append(c[0])
                self.stress_history.append(0.0)
                self.obi_history.append(0.0)

            self.status_msg = "📡 Загрузка HTF свечей…"
            for c in self.feeder.fetch_initial_htf(limit=200):
                self.candles_htf.append(c)
                self.prices_htf.append(c[4])

            self.status_msg = "🔬 Расчёт топологического стресса…"
            n = len(self.prices)
            for i in range(TDA_EMBED_WINDOW, n):
                emb = self._embed(self.prices[i-TDA_EMBED_WINDOW:i])
                if len(emb) > 0:
                    self.stress_history[i] = self.tda.get_topological_stress(emb)
                if (i - TDA_EMBED_WINDOW) % 100 == 0:
                    self.status_msg = f"🔬 TDA backfill {i}/{n}…"

            self.status_msg = "🤖 Обучение ML модели…"
            self.ml.train(
                self.candles, self.stress_history, self.obi_history,
                candles_htf=self.candles_htf,
            )

            self._calc_htf_trend()
            self.stress_buf.extend(self.stress_history[-CHART_BARS:])
            self.obi_buf.extend(self.obi_history[-CHART_BARS:])
            self.loaded     = True
            self.status_msg = "✅ Готов"
            return True

        except Exception as exc:
            self.status_msg = f"❌ {exc}"
            logger.exception("preload error")
            return False

    def tick(self) -> None:
        try:
            self.current_obi = self.feeder.fetch_order_book_imbalance(depth=20)
        except Exception:
            pass

        try:
            upd_htf = self.feeder.fetch_updates_htf()
            if upd_htf:
                for c in upd_htf:
                    t = c[0]
                    if not self.candles_htf or t > self.candles_htf[-1][0]:
                        self.candles_htf.append(c)
                        self.prices_htf.append(c[4])
                    elif t == self.candles_htf[-1][0]:
                        self.candles_htf[-1] = c
                        self.prices_htf[-1]  = c[4]
        except Exception:
            pass

        try:
            ohlcv = self.feeder.fetch_updates()
        except Exception:
            return
        if not ohlcv:
            return

        new_candle = False
        for c in ohlcv:
            t = c[0]
            if not self.candles or t > self.candles[-1][0]:
                new_candle = True
                self.candles.append(c)
                self.prices.append(c[4])
                self.timestamps.append(t)
                self.stress_history.append(0.0)
                self.obi_history.append(self.current_obi)
            elif t == self.candles[-1][0]:
                self.candles[-1]     = c
                self.prices[-1]      = c[4]
                self.obi_history[-1] = self.current_obi

        curr_p  = self.prices[-1]
        curr_ms = self.timestamps[-1]

        if new_candle:
            new_open = self.candles[-1][1]
            if self.trader.execute_at_open(new_open, curr_ms):
                side = self.trader.position
                icon = "🚀" if side == "LONG" else "🩸"
                self.status_msg = (
                    f"{icon} ENTERED {side} @ {new_open:,.2f}"
                )
                self.notifier.send_message(
                    f"{icon} <b>ENTERED {side}</b>\n"
                    f"Price: {new_open:.2f}\n"
                    f"SL: {self.trader.sl_pct*100:.2f}%  "
                    f"TP: {self.trader.tp_pct*100:.2f}%"
                )

        emb = self._embed(self.prices[-TDA_EMBED_WINDOW:])
        if len(emb) > 1:
            self.current_stress = self.tda.get_topological_stress(emb)
            self.stress_history[-1] = self.current_stress
            self.risk.update_stress(self.current_stress)

        self.stress_buf.append(self.current_stress)
        self.obi_buf.append(self.current_obi)

        self.phase_transition = detect_phase_transition(self.candles)

        if self.ml.is_trained:
            self.probs = self.ml.predict(
                self.candles, self.stress_history, self.obi_history,
                candles_htf=self.candles_htf,
            )

        res = self.trader.update(curr_p, curr_ms)
        if res:
            icon = "✅" if res["net_profit_usd"] > 0 else "🛑"
            self.status_msg = (
                f"{icon} CLOSED ({res['reason']}) "
                f"P&L: ${res['net_profit_usd']:+.2f}  "
                f"Bal: ${self.trader.balance:,.2f}"
            )
            self.trade_log.insert(0, {**res, "ts": time.time()})
            if len(self.trade_log) > 30:
                self.trade_log.pop()

            self.risk.update_after_trade(res, self.trader.balance)

            ctx = self._pending_ctx
            _journal_write([
                int(time.time()),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                res.get("type",""),
                round(res.get("entry",0),4),
                round(res.get("exit",0),4),
                res.get("reason",""),
                round(res.get("net_profit_usd",0),4),
                round(self.trader.balance,4),
                round(ctx.get("stress",0),4),
                round(ctx.get("prob_up",0),4),
                round(ctx.get("prob_dn",0),4),
                ctx.get("htf_trend",""),
                round(ctx.get("atr_pct",0),6),
                round(ctx.get("obi",0),4),
            ])
            self._pending_ctx = {}

            self.notifier.send_message(
                f"{icon} <b>TRADE CLOSED ({res['reason']})</b>\n"
                f"Net P&L: ${res['net_profit_usd']:.2f}\n"
                f"Balance: ${self.trader.balance:.2f}"
            )

        self._calc_htf_trend()

        if new_candle:
            threshold = self.risk.adaptive_stress_threshold(self._base_thresh)
            stress_ok = self.current_stress >= threshold
            no_pos    = self.trader.position is None
            no_pend   = self.trader.pending_signal is None

            if stress_ok and no_pos and no_pend and not self.risk.is_halted():
                atr_pct = self._atr_pct()

                # Контр-трендовая стратегия: ML предсказывает UP → Short (fade rally)
                #                             ML предсказывает DOWN → Long (buy dip)
                if (self.probs["up"] >= ENTRY_PROB_THRESHOLD
                        and self.current_obi <= -OBI_CONFIRM_MIN
                        and self._mean_rev_ok("SHORT")):
                    ctx = {
                        "stress":   self.current_stress,
                        "prob_up":  self.probs["up"],
                        "prob_dn":  self.probs["down"],
                        "htf_trend": self.htf_trend,
                        "atr_pct":  atr_pct,
                        "obi":      self.current_obi,
                    }
                    self.trader.set_pending("SHORT", atr_pct)
                    self._pending_ctx = ctx
                    self.signal = "SHORT"
                    self.signal_ctx = ctx
                    self.status_msg = "↓ СИГНАЛ SHORT (fade rally)"

                elif (self.probs["down"] >= ENTRY_PROB_THRESHOLD
                        and self.current_obi >= OBI_CONFIRM_MIN
                        and self._mean_rev_ok("LONG")):
                    ctx = {
                        "stress":   self.current_stress,
                        "prob_up":  self.probs["up"],
                        "prob_dn":  self.probs["down"],
                        "htf_trend": self.htf_trend,
                        "atr_pct":  atr_pct,
                        "obi":      self.current_obi,
                    }
                    self.trader.set_pending("LONG", atr_pct)
                    self._pending_ctx = ctx
                    self.signal = "LONG"
                    self.signal_ctx = ctx
                    self.status_msg = "↑ СИГНАЛ LONG (buy dip)"

                else:
                    if not self.trader.position and not self.trader.pending_signal:
                        self.signal = "WAIT"

            elif self.risk.is_halted():
                self.signal = "HALTED"

        limit = 600
        if len(self.candles) > limit:
            self.candles        = self.candles[-limit:]
            self.prices         = self.prices[-limit:]
            self.timestamps     = self.timestamps[-limit:]
            self.stress_history = self.stress_history[-limit:]
            self.obi_history    = self.obi_history[-limit:]
        if len(self.candles_htf) > 250:
            self.candles_htf = self.candles_htf[-250:]
            self.prices_htf  = self.prices_htf[-250:]

        self.tick_counter += 1
        if self.tick_counter % RETRAIN_EVERY_TICKS == 0:
            self._schedule_retrain()
        if (self.phase_transition
                and (self.tick_counter - self._last_phase_retrain) > PHASE_RETRAIN_CD):
            self._last_phase_retrain = self.tick_counter
            self._schedule_retrain()

    def _schedule_retrain(self) -> None:
        snap = (list(self.candles), list(self.stress_history),
                list(self.obi_history), list(self.candles_htf))
        threading.Thread(
            target=self._retrain_bg, args=snap,
            daemon=True, name="ml-retrain",
        ).start()

    def _retrain_bg(self, candles, stress, obi, htf) -> None:
        if not self._retrain_lock.acquire(blocking=False):
            return
        try:
            prev = self.status_msg
            self.status_msg = "⚙ Переобучение ML…"
            ok = self.ml.train(candles, stress, obi, candles_htf=htf)
            self.status_msg = (
                f"✓ ML переобучен ({len(candles)} баров)"
                if ok else prev
            )
        finally:
            self._retrain_lock.release()

    def manual_open(self, side: str) -> bool:
        """Открыть позицию вручную по текущей цене."""
        if self.trader.position or self.trader.pending_signal:
            return False
        if not self.prices:
            return False
        price   = self.prices[-1]
        curr_ms = self.timestamps[-1] if self.timestamps else int(time.time()*1000)
        atr_pct = self._atr_pct()
        self.trader.sl_pct = max(atr_pct * 2.0, 0.002)
        self.trader.tp_pct = max(atr_pct * 3.5, 0.004)
        self.trader.position    = side
        self.trader.entry_price = price
        self.trader.entry_time  = curr_ms
        self.trader._save_state()
        self.signal = side
        icon = "🚀" if side == "LONG" else "🩸"
        self.status_msg = f"{icon} РУЧНОЙ ВХОД {side} @ {price:,.2f}"
        self.notifier.send_message(
            f"{icon} <b>РУЧНОЙ ВХОД {side}</b>\nPrice: {price:.2f}"
        )
        return True

    def manual_close(self) -> bool:
        """Закрыть позицию вручную."""
        if not self.trader.position:
            return False
        if not self.prices:
            return False
        price   = self.prices[-1]
        curr_ms = self.timestamps[-1] if self.timestamps else int(time.time()*1000)
        res = self.trader.update(price, curr_ms - 1)  # форсируем закрытие
        if not res:
            pos  = self.trader.position
            entry = self.trader.entry_price
            pnl_pct = (
                (price - entry)/entry if pos == "LONG"
                else (entry - price)/entry
            )
            pos_usd = MARGIN_USDT * LEVERAGE
            net = pos_usd * pnl_pct - pos_usd * 0.0004 * 2
            self.trader.balance += net
            res = {
                "type": pos, "entry": entry, "exit": price,
                "net_profit_usd": net, "reason": "MANUAL",
                "sl_pct": self.trader.sl_pct, "tp_pct": self.trader.tp_pct,
            }
            self.trader.position    = None
            self.trader.entry_price = 0.0
            self.trader.entry_time  = 0
            self.trader._save_state()

        icon = "✅" if res.get("net_profit_usd", 0) > 0 else "🛑"
        self.status_msg = (
            f"{icon} ЗАКРЫТО (MANUAL) "
            f"P&L: ${res.get('net_profit_usd',0):+.2f}  "
            f"Bal: ${self.trader.balance:,.2f}"
        )
        self.trade_log.insert(0, {**res, "ts": time.time()})
        self.signal = "WAIT"
        self.notifier.send_message(
            f"{icon} <b>ЗАКРЫТО (MANUAL)</b>\n"
            f"Net P&L: ${res.get('net_profit_usd',0):.2f}\n"
            f"Balance: ${self.trader.balance:.2f}"
        )
        return True

    def reset_signal(self) -> None:
        """Сбросить отложенный сигнал."""
        self.trader.cancel_pending()
        self.signal = "WAIT"
        self.status_msg = "Сигнал сброшен"


class ChartWidget(Static):
    """OHLCV-чарт через plotext или ASCII fallback."""

    DEFAULT_CSS = """
    ChartWidget {
        border: solid #1a3050;
        height: 22;
    }
    """

    def update_chart(self, core: StrategyCore) -> None:
        if not core.candles or len(core.candles) < 5:
            self.update("📡 Загрузка данных…")
            return

        candles = core.candles[-CHART_BARS:]
        closes  = [c[4] for c in candles]
        highs   = [c[2] for c in candles]
        lows    = [c[3] for c in candles]
        opens   = [c[1] for c in candles]
        n = len(candles)

        if _HAS_PLT:
            try:
                w = self.size.width  - 4
                h = self.size.height - 5
                if w < 20 or h < 5:
                    raise ValueError("too small")

                plx.clear_figure()
                plx.theme("dark")
                plx.plot_size(w, h)

                xs = list(range(n))
                plx.candlestick(opens, highs, lows, closes)

                stress_vals = list(core.stress_buf)
                if stress_vals:
                    sl = _sparkline(stress_vals, width=min(n, CHART_BARS))
                    obi_sl = _sparkline(
                        [v + 1 for v in list(core.obi_buf)],
                        width=min(n, CHART_BARS)
                    )
                    chart_str = plx.build()
                    chart_str += f"\n  Stress [{core.timeframe}]:  {sl[-50:]}"
                    chart_str += f"\n  OBI       :  {obi_sl[-50:]}"
                else:
                    chart_str = plx.build()

                self.update(chart_str)
                return
            except Exception:
                pass

        height = max(self.size.height - 6, 8)
        width  = max(self.size.width  - 14, 40)
        grid   = [[" "] * n for _ in range(height)]

        p_min = min(lows)
        p_max = max(highs)
        p_rng = p_max - p_min
        if p_rng < 1e-8:
            self.update("─ Нет данных ─")
            return

        def row(price: float) -> int:
            return int((price - p_min) / p_rng * (height - 1))

        for i, c in enumerate(candles):
            o, h, l, cl = c[1], c[2], c[3], c[4]
            bull = cl >= o
            hi_r, lo_r = row(h), row(l)
            o_r,  c_r  = row(o), row(cl)
            body_top = max(o_r, c_r)
            body_bot = min(o_r, c_r)
            for r in range(lo_r, hi_r + 1):
                if body_bot <= r <= body_top:
                    grid[r][i] = "█" if bull else "░"
                else:
                    grid[r][i] = "│"

        lines_out = []
        for r in range(height - 1, -1, -1):
            price_at_r = p_min + r / (height - 1) * p_rng
            axis = (f"{price_at_r:>10,.1f} "
                    if r in (height-1, height//2, 0) else "           ")
            row_str = ""
            for j, ch in enumerate(grid[r]):
                if j < n:
                    c_data = candles[j]
                    bull   = c_data[4] >= c_data[1]
                    if ch == "█":
                        row_str += "▲" if bull else "▼"
                    elif ch == "░":
                        row_str += "▼"
                    else:
                        row_str += ch
                else:
                    row_str += ch
            lines_out.append(axis + row_str[:width])

        stress_sl = _sparkline(list(core.stress_buf), width=width)
        obi_sl    = _sparkline([v+1 for v in list(core.obi_buf)], width=width)

        lines_out.append(f"  Stress: {stress_sl[:width]}")
        lines_out.append(f"     OBI: {obi_sl[:width]}")

        self.update("\n".join(lines_out))


class TDAWidget(Static):
    """Панель топологического анализа."""

    DEFAULT_CSS = """
    TDAWidget {
        border: solid #1a3050;
        height: 14;
    }
    """

    def update_panel(self, core: StrategyCore) -> None:
        threshold  = core.risk.adaptive_stress_threshold(core._base_thresh)
        stress_ok  = core.current_stress >= threshold
        status_sym = "[bold green]✓ ACTIVE[/]" if stress_ok else "[dim red]✗ BELOW[/]"
        s_col      = "green" if stress_ok else "yellow"
        s_bar      = _stress_bar(core.current_stress, threshold, 22)
        t_bar      = "▎" + "─" * int(threshold / max(threshold * 2, 3.0) * 22)

        phase = "⚡ [bold yellow]DETECTED[/]" if core.phase_transition else "[dim]Stable[/]"

        obi_bar  = _obi_bar(core.current_obi, width=22)
        obi_sign = "BID" if core.current_obi > 0 else "ASK"
        obi_col  = "green" if core.current_obi > 0 else "red"

        buf = list(core.stress_buf)
        if len(buf) > 5:
            avg_s  = np.mean(buf)
            max_s  = np.max(buf)
            pct75  = np.percentile(buf, 75)
        else:
            avg_s = max_s = pct75 = core.current_stress

        lines = [
            f"  [bold cyan]Топологический стресс[/bold cyan]",
            f"",
            f"  Current  [{s_col}]{s_bar}[/] [{s_col}]{core.current_stress:.3f}[/]",
            f"  Thresh   {threshold:.3f}  {status_sym}",
            f"  Avg/Max  {avg_s:.3f} / {max_s:.3f}",
            f"  P75      {pct75:.3f}  (адаптивный порог)",
            f"",
            f"  Phase    {phase}",
            f"",
            f"  [bold cyan]Order Book Imbalance[/bold cyan]",
            f"  [{obi_col}]{obi_bar}[/]  [{obi_col}]{core.current_obi:+.3f} {obi_sign}[/]",
        ]
        self.update("\n".join(lines))


class MLWidget(Static):
    """Панель ML ансамбля — вероятности и Top-Features."""

    DEFAULT_CSS = """
    MLWidget {
        border: solid #1a3050;
        height: 16;
    }
    """

    def update_panel(self, core: StrategyCore) -> None:
        p_up   = core.probs["up"]
        p_dn   = core.probs["down"]
        p_fl   = core.probs["flat"]
        score  = p_up - p_dn

        up_bar  = _bar(p_up,  1.0, 18)
        dn_bar  = _bar(p_dn,  1.0, 18)
        fl_bar  = _bar(p_fl,  1.0, 18)
        sc_col  = "green" if score > 0.05 else ("red" if score < -0.05 else "yellow")

        half = 9
        filled = min(int(abs(score) * half), half)
        if score >= 0:
            comp = "░" * (half - filled) + "▓" * filled + "│" + "░" * half
        else:
            comp = "░" * half + "│" + "▓" * filled + "░" * (half - filled)

        top3 = list(core.ml.feature_importance.items())[:3] if core.ml.feature_importance else []
        feat_str = "  ".join(f"{k}" for k, _ in top3) if top3 else "—"

        ml_status = "[green]✓ Обучена[/]" if core.ml.is_trained else "[yellow]⏳ Обучение…[/]"

        # Логика входа: контр-трендовая
        logic_up = ("[bold red]→SHORT FADE[/]"
                    if core.probs["up"] >= ENTRY_PROB_THRESHOLD
                       and core.current_obi <= -OBI_CONFIRM_MIN
                    else "[dim]─[/]")
        logic_dn = ("[bold green]→LONG BUY DIP[/]"
                    if core.probs["down"] >= ENTRY_PROB_THRESHOLD
                       and core.current_obi >= OBI_CONFIRM_MIN
                    else "[dim]─[/]")

        lines = [
            f"  [bold cyan]TopoBooster Ensemble[/bold cyan]  {ml_status}",
            f"",
            f"  📈 UP   [green]{up_bar}[/]  [bold]{p_up:.1%}[/]  {logic_dn}",
            f"  📉 DOWN [red]{dn_bar}[/]  [bold]{p_dn:.1%}[/]  {logic_up}",
            f"  ─ FLAT  [yellow]{fl_bar}[/]  [bold]{p_fl:.1%}[/]",
            f"",
            f"  Composite [{sc_col}]{comp}[/] [{sc_col}]{score:+.3f}[/]",
            f"",
            f"  [dim]Контр-трендовая логика (mean-reversion)[/dim]",
            f"  [dim]ML UP → SHORT fade  |  ML DOWN → LONG dip[/dim]",
            f"",
            f"  Top features: [cyan]{feat_str}[/cyan]",
        ]
        self.update("\n".join(lines))


class SignalWidget(Static):
    """Панель текущего сигнала и параметров сделки."""

    DEFAULT_CSS = """
    SignalWidget {
        border: solid #1a3050;
        height: 20;
    }
    """

    def update_panel(self, core: StrategyCore) -> None:
        curr_p  = core.prices[-1] if core.prices else 0.0
        pos     = core.trader.position
        pend    = core.trader.pending_signal

        atr_pct = core._atr_pct()
        sl_pct  = core.trader.sl_pct if pos else max(atr_pct * 2.0, 0.002)
        tp_pct  = core.trader.tp_pct if pos else max(atr_pct * 3.5, 0.004)
        rr      = tp_pct / (sl_pct + 1e-10)

        win_p   = max(core.probs["up"], core.probs["down"])
        kelly   = _kelly(win_p, rr)

        if pos == "LONG":
            entry = core.trader.entry_price
            sl    = entry * (1 - sl_pct)
            tp    = entry * (1 + tp_pct)
            sig_line = "[bold green on black]  🚀 LONG (в позиции)  [/]"
        elif pos == "SHORT":
            entry = core.trader.entry_price
            sl    = entry * (1 + sl_pct)
            tp    = entry * (1 - tp_pct)
            sig_line = "[bold red on black]  🩸 SHORT (в позиции)  [/]"
        elif pend == "LONG":
            entry = curr_p
            sl    = curr_p * (1 - sl_pct)
            tp    = curr_p * (1 + tp_pct)
            sig_line = "[bold yellow]  ⏳ PENDING LONG → след. свеча  [/]"
        elif pend == "SHORT":
            entry = curr_p
            sl    = curr_p * (1 + sl_pct)
            tp    = curr_p * (1 - tp_pct)
            sig_line = "[bold yellow]  ⏳ PENDING SHORT → след. свеча  [/]"
        elif core.signal == "LONG":
            entry = curr_p
            sl    = curr_p * (1 - sl_pct)
            tp    = curr_p * (1 + tp_pct)
            sig_line = "[bold green]  ↑ СИГНАЛ LONG (ожидание)  [/]"
        elif core.signal == "SHORT":
            entry = curr_p
            sl    = curr_p * (1 + sl_pct)
            tp    = curr_p * (1 - tp_pct)
            sig_line = "[bold red]  ↓ СИГНАЛ SHORT (ожидание)  [/]"
        elif core.signal == "HALTED":
            entry = curr_p
            sl    = curr_p * (1 - sl_pct)
            tp    = curr_p * (1 + tp_pct)
            sig_line = "[bold red]  🔴 CIRCUIT BREAKER HALTED  [/]"
        else:
            entry = curr_p
            sl    = curr_p * (1 - sl_pct)
            tp    = curr_p * (1 + tp_pct)
            sig_line = "[dim]  ⏸  ОЖИДАНИЕ СИГНАЛА  [/dim]"

        sl_dist = abs(entry - sl) / (entry + 1e-8) * 100
        tp_dist = abs(tp - entry) / (entry + 1e-8) * 100

        net_pnl = core.trader.get_unrealized_pnl(curr_p)
        pnl_col = "green" if net_pnl >= 0 else "red"

        htf_icons = {
            "bull": ("[bold green]📈 BULL ↑[/]"),
            "bear": ("[bold red]📉 BEAR ↓[/]"),
            "flat": ("[yellow]➡ FLAT[/]"),
        }

        lines = [
            sig_line,
            f"",
            f"  Entry  [bold white]${entry:>13,.2f}[/]",
            f"  SL     [red]${sl:>13,.2f}[/]   [red]{sl_dist:.2f}%[/]",
            f"  TP     [green]${tp:>13,.2f}[/]   [green]{tp_dist:.2f}%[/]",
            f"  R/R    [bold]{rr:.2f}×[/]",
            f"",
            f"  Kelly  [cyan]{kelly:.1%}[/]  (half-Kelly sizing)",
            f"  ATR    [cyan]{atr_pct*100:.3f}%[/]",
            f"",
            f"  Unrealized  [{pnl_col}]${net_pnl:+.2f}[/{pnl_col}]",
            f"",
            f"  HTF Trend  {htf_icons.get(core.htf_trend,'─')}",
            f"  [dim]EMA8: ${core.htf_ema8:,.1f}   EMA21: ${core.htf_ema21:,.1f}[/dim]",
        ]
        self.update("\n".join(lines))


class IndicatorsWidget(Static):
    """Строка технических индикаторов."""

    DEFAULT_CSS = """
    IndicatorsWidget {
        border: solid #1a3050;
        height: 7;
    }
    """

    def update_panel(self, core: StrategyCore) -> None:
        rsi   = core._rsi()
        mh    = core._macd_hist()
        atr   = core._atr_pct()
        sk    = core._stoch()
        bb    = core._bb_pos()
        curr_p = core.prices[-1] if core.prices else 0.0

        rsi_col = "red" if rsi > 70 else ("green" if rsi < 30 else
                  "yellow" if (rsi > 60 or rsi < 40) else "white")
        mh_col  = "green" if mh > 0 else "red"
        sk_col  = "red" if sk > 0.8 else ("green" if sk < 0.2 else "white")
        bb_col  = "red" if bb > 0.5 else ("green" if bb < -0.5 else "white")

        rsi_bar = _bar(rsi, 100, 14)
        sk_bar  = _bar(sk,  1.0, 14)
        mh_bar  = _bar(abs(mh)*100, 0.3, 14)
        bb_bar  = _bar((bb+1)/2, 1.0, 14)

        lines = [
            f"  [bold cyan]Индикаторы[/bold cyan]",
            (f"  RSI(14) [{rsi_col}]{rsi_bar}[/] [{rsi_col}]{rsi:5.1f}[/]  [{rsi_col}]{_rsi_label(rsi)}[/]"
             f"   │  Stoch K [{sk_col}]{sk_bar}[/] [{sk_col}]{sk:.2f}[/]  "
             f"[{sk_col}]{'OB' if sk>0.8 else 'OS' if sk<0.2 else '●'}[/]"),
            (f"  MACD hist [{mh_col}]{mh_bar}[/] [{mh_col}]{mh:+.5f}[/]  [{mh_col}]"
             f"{'↑' if mh>0 else '↓'}[/]"
             f"   │  BB Pos  [{bb_col}]{bb_bar}[/] [{bb_col}]{bb:+.3f}[/]  "
             f"[{bb_col}]{'Upper' if bb>0 else 'Lower'}[/]"),
            (f"  ATR(14)  [cyan]{atr*100:.4f}%[/]"
             f"  │  Price [bold]${curr_p:,.2f}[/]"
             f"  │  TF [bold]{core.timeframe}[/] / HTF [bold]{HTF}[/]"
             f"  │  Tick [dim]{core.tick_counter}[/]"),
        ]
        self.update("\n".join(lines))


class PortfolioWidget(Static):
    """Строка портфеля и риска."""

    DEFAULT_CSS = """
    PortfolioWidget {
        border: solid #1a3050;
        height: 5;
    }
    """

    def update_panel(self, core: StrategyCore) -> None:
        bal    = core.trader.balance
        curr_p = core.prices[-1] if core.prices else 0.0
        pnl    = core.trader.get_unrealized_pnl(curr_p)
        pos    = core.trader.position or "─"
        halted = core.risk.is_halted()
        cl     = core.risk.consecutive_losses
        peak   = core.risk.peak_balance
        dd     = (peak - bal) / (peak + 1e-10) * 100

        bal_col = "green" if bal >= INITIAL_BALANCE else "yellow"
        pnl_col = "green" if pnl >= 0 else "red"
        risk_col = "red" if halted else "green"
        dd_col   = "red" if dd > 10 else ("yellow" if dd > 5 else "green")
        ml_st    = "[green]✓[/]" if core.ml.is_trained else "[yellow]⏳[/]"

        pend_str = (f"  PENDING: [bold yellow]{core.trader.pending_signal}[/]"
                    if core.trader.pending_signal else "")

        lines = [
            f"  [bold]БАЛАНС[/]  [{bal_col}]${bal:>11,.2f}[/]"
            f"   │  Unrealized [{pnl_col}]${pnl:+.2f}[/]"
            f"   │  Позиция [bold]{pos}[/]"
            f"{pend_str}",

            f"  [bold]РИСК[/]  [{risk_col}]{'🔴 HALTED' if halted else '✅ OK'}[/]"
            f"   │  ConsecLoss [{('red' if cl > 2 else 'white')}]{cl}/4[/]"
            f"   │  Drawdown [{dd_col}]{dd:.1f}%[/] (max 15%)"
            f"   │  ML {ml_st}",
        ]
        self.update("\n".join(lines))


class TopoAlphaDashboard(App):
    """
    TopoAlpha Terminal Dashboard — главное окно.
    """

    TITLE = "⬡ TopoAlpha  Terminal Dashboard"
    SUB_TITLE = f"{SYMBOL}  {TIMEFRAMES[DEFAULT_TF_IDX]}/{HTF}"

    CSS = """
    Screen {
        background: #04080f;
        color: #a0c8e8;
    }

    Header {
        background: #06101e;
        color: #3a80c8;
        text-style: bold;
    }

    Footer {
        background: #06101e;
        color: #2a5080;
    }

    #top_row {
        height: 1fr;
        min-height: 22;
    }

    #chart_col {
        width: 3fr;
    }

    #right_col {
        width: 2fr;
    }

    #ml_tda_row {
        width: 1fr;
    }

    ChartWidget {
        height: 22;
        border: solid #0a2040;
    }

    TDAWidget {
        height: 13;
        border: solid #0a2040;
    }

    MLWidget {
        height: 15;
        border: solid #0a2040;
    }

    SignalWidget {
        height: 20;
        border: solid #1a3a60;
    }

    IndicatorsWidget {
        height: 7;
        border: solid #0a2040;
    }

    PortfolioWidget {
        height: 5;
        border: solid #0a2040;
    }

    DataTable {
        height: 9;
        border: solid #0a2040;
        background: #04080f;
        color: #8ab0d0;
    }

    DataTable > .datatable--header {
        background: #08101e;
        color: #3a80c8;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: #0a2040;
        color: #e0f0ff;
    }

    #status_bar {
        height: 1;
        background: #06101e;
        color: #3a80c8;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("l",       "open_long",  "LONG",    show=True),
        Binding("s",       "open_short", "SHORT",   show=True),
        Binding("c",       "close_pos",  "CLOSE",   show=True),
        Binding("r",       "reset_sig",  "RESET",   show=True),
        Binding("t",       "next_tf",    "TIMEFRAME",show=True),
        Binding("q",       "quit",       "Quit",    show=True),
    ]

    tf_idx: reactive[int] = reactive(DEFAULT_TF_IDX)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="top_row"):
            with Vertical(id="chart_col"):
                yield ChartWidget(id="chart")
                yield IndicatorsWidget(id="indicators")
                yield PortfolioWidget(id="portfolio")

            with Horizontal(id="right_col"):
                with Vertical(id="ml_tda_row"):
                    yield TDAWidget(id="tda")
                    yield MLWidget(id="ml")
                yield SignalWidget(id="signal")

        yield DataTable(id="trade_log")
        yield Static("", id="status_bar")
        yield Footer()

    def on_mount(self) -> None:
        self.core       = StrategyCore(timeframe=TIMEFRAMES[self.tf_idx])
        self._n_trades  = 0
        self._setup_table()
        self._do_preload()

    def _setup_table(self) -> None:
        tbl = self.query_one("#trade_log", DataTable)
        tbl.add_columns(
            "#", "Время", "Сторона", "Вход", "Выход",
            "Причина", "P&L", "Баланс",
        )
        tbl.cursor_type = "row"

    @work(thread=True, name="preload")
    def _do_preload(self) -> None:
        ok = self.core.preload()
        self.call_from_thread(self._on_loaded, ok)

    def _on_loaded(self, ok: bool) -> None:
        if ok:
            self._update_all()
            self.set_interval(POLL_INTERVAL_S, self._do_tick)
        else:
            self.query_one("#status_bar", Static).update(
                f"  ❌ Ошибка загрузки: {self.core.status_msg}"
            )

    @work(thread=True, name="tick")
    def _do_tick(self) -> None:
        try:
            self.core.tick()
        except Exception as exc:
            self.core.status_msg = f"⚠ Tick error: {exc}"
        self.call_from_thread(self._update_all)

    def _update_all(self) -> None:
        core = self.core

        for res in core.trade_log[:]:
            if res.get("_logged"):
                break
            res["_logged"] = True
            self._push_trade_row(res)

        self.query_one("#chart",       ChartWidget).update_chart(core)
        self.query_one("#tda",         TDAWidget).update_panel(core)
        self.query_one("#ml",          MLWidget).update_panel(core)
        self.query_one("#signal",      SignalWidget).update_panel(core)
        self.query_one("#indicators",  IndicatorsWidget).update_panel(core)
        self.query_one("#portfolio",   PortfolioWidget).update_panel(core)

        p    = core.prices[-1] if core.prices else 0
        prev = core.prices[-2] if len(core.prices) > 1 else p
        chg  = (p - prev) / (prev + 1e-8) * 100
        arr  = _price_change_arrow(p, prev)
        self.query_one("#status_bar", Static).update(
            f"  {SYMBOL}  ${p:,.2f} {arr} {chg:+.2f}%"
            f"   │  TF: {core.timeframe}/{HTF}"
            f"   │  {core.status_msg}"
            f"   │  {datetime.now().strftime('%H:%M:%S')}"
        )

    def _push_trade_row(self, res: dict) -> None:
        self._n_trades += 1
        tbl = self.query_one("#trade_log", DataTable)
        net = res.get("net_profit_usd", 0)
        pc  = "green" if net >= 0 else "red"
        sc  = "green" if res.get("type") == "LONG" else "red"
        ts  = datetime.fromtimestamp(res.get("ts", time.time())).strftime("%H:%M:%S")
        tbl.add_row(
            str(self._n_trades),
            ts,
            f"[bold {sc}]{res.get('type','─')}[/]",
            f"${res.get('entry',0):,.2f}",
            f"${res.get('exit',0):,.2f}",
            res.get("reason", "─"),
            f"[{pc}]${net:+.2f}[/]",
            f"${self.core.trader.balance:,.2f}",
        )

    def action_open_long(self) -> None:
        if self.core.loaded:
            self.core.manual_open("LONG")
            self._update_all()

    def action_open_short(self) -> None:
        if self.core.loaded:
            self.core.manual_open("SHORT")
            self._update_all()

    def action_close_pos(self) -> None:
        if self.core.loaded:
            self.core.manual_close()
            self._update_all()

    def action_reset_sig(self) -> None:
        if self.core.loaded:
            self.core.reset_signal()
            self._update_all()

    def action_next_tf(self) -> None:
        """Циклически переключает таймфрейм и перезапускает ядро."""
        new_idx = (self.tf_idx + 1) % len(TIMEFRAMES)
        self.tf_idx  = new_idx
        new_tf       = TIMEFRAMES[new_idx]
        self.core    = StrategyCore(timeframe=new_tf)
        self.SUB_TITLE = f"{SYMBOL}  {new_tf}/{HTF}"
        self._do_preload()

    def action_quit(self) -> None:
        self.core.trader.close()
        self.exit()


if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("  ⬡  TopoAlpha  Terminal Dashboard")
    print("═" * 70)
    print(f"  Symbol: {SYMBOL}   TF: {TIMEFRAMES[DEFAULT_TF_IDX]} / {HTF}")
    print(f"  Зависимости: textual rich plotext ccxt ripser scikit-learn")
    print(f"  Журнал сделок: {JOURNAL_PATH}")
    print()
    print("  Загрузка данных и обучение ML занимает ~30 секунд…")
    print("═" * 70 + "\n")

    TopoAlphaDashboard().run()