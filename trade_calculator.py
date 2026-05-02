"""
trade_calculator.py — TopoAlpha Trade Calculator
=================================================
A standalone analysis panel that computes a full trade breakdown:
entry, SL, TP, position size, expected value and Kelly criterion.

Run standalone:
    python trade_calculator.py

Or embed inside the main engine (app.py calls this automatically via the ⚡ button):
    from trade_calculator import TradeCalculatorWindow
    calc = TradeCalculatorWindow()
    calc.feed(prices, stress_history, obi_history, prices_htf, ml, trader)
    calc.show()
"""

from __future__ import annotations

import sys
import math
import logging
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QComboBox, QDoubleSpinBox,
    QGroupBox, QFormLayout,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextCursor

logger = logging.getLogger("TopoAlpha.Calculator")


def _atr(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 0.0
    arr = np.array(prices[-period - 1:], dtype=float)
    return float(np.abs(np.diff(arr)).mean())


def _rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    arr = np.array(prices[-(period + 1):], dtype=float)
    delta = np.diff(arr)
    gain = delta.clip(min=0).mean()
    loss = (-delta.clip(max=0)).mean()
    return 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)


def _htf_trend(prices_htf: list) -> tuple[str, float]:
    """EMA8 vs EMA21 on HTF.  Returns ('BULL'|'BEAR'|'NEUTRAL', strength_pct)."""
    if len(prices_htf) < 21:
        return "NEUTRAL", 0.0
    s = pd.Series(prices_htf, dtype=float)
    ema8 = s.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = s.ewm(span=21, adjust=False).mean().iloc[-1]
    diff = (ema8 - ema21) / (ema21 + 1e-10) * 100
    if diff > 0.05: return "BULL", diff
    if diff < -0.05: return "BEAR", abs(diff)
    return "NEUTRAL", abs(diff)


def _vol_rank(prices: list, window: int = 50) -> float:
    """Percentile rank of current 10-bar volatility over the last *window* bars."""
    if len(prices) < window + 1:
        return 0.5
    rets = pd.Series(prices, dtype=float).pct_change().dropna()
    vol_roll = rets.rolling(10).std().dropna()
    if len(vol_roll) < 2:
        return 0.5
    return float((vol_roll <= vol_roll.iloc[-1]).mean())


def _bar(value: float, width: int = 20, char: str = "█") -> str:
    filled = round(value * width)
    return char * filled + "░" * (width - filled)


class TradeCalculator:

    def __init__(
            self,
            atr_sl_mult: float = 1.5,
            atr_tp_mult: float = 3.0,
            min_sl_pct: float = 0.003,
            max_sl_pct: float = 0.015,
            prob_threshold: float = 0.55,
    ):
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.min_sl_pct = min_sl_pct
        self.max_sl_pct = max_sl_pct
        self.prob_threshold = prob_threshold

    def calculate(
            self,
            prices: list,
            stress_history: list,
            obi_history: list,
            prices_htf: list,
            ml_probs: dict | None = None,
            margin_usdt: float = 50.0,
            leverage: int = 10,
    ) -> dict:
        if len(prices) < 30:
            return {"error": "Not enough data (need ≥ 30 bars)"}

        W = 52
        sep = "─" * W
        dsep = "═" * W
        lines: list[str] = []

        def h(title: str) -> None:
            lines.append(f"\n{dsep}")
            lines.append(f"  {title}")
            lines.append(sep)

        curr_price = prices[-1]
        curr_stress = stress_history[-1] if stress_history else 0.0
        curr_obi = obi_history[-1] if obi_history else 0.0

        h("STEP 1 — MARKET CONTEXT")
        lines.append(f"  Price            : {curr_price:>12,.2f} USDT")
        lines.append(f"  Bars loaded      : {len(prices)}")

        atr_val = _atr(prices, 14)
        atr_pct = atr_val / curr_price * 100
        rsi_val = _rsi(prices, 14)
        mom5 = (prices[-1] / prices[-6] - 1) * 100 if len(prices) >= 6 else 0.0
        mom20 = (prices[-1] / prices[-21] - 1) * 100 if len(prices) >= 21 else 0.0
        vol_r = _vol_rank(prices, 50)

        h("STEP 2 — TECHNICAL INDICATORS")
        lines.append(f"  ATR(14)          : {atr_val:.2f}  ({atr_pct:.3f}% of price)")
        lines.append(
            f"  RSI(14)          : {rsi_val:.1f}  {'⚡ oversold' if rsi_val < 35 else ('⚡ overbought' if rsi_val > 65 else 'neutral')}")
        lines.append(f"  Momentum  5-bar  : {mom5:+.3f}%")
        lines.append(f"  Momentum 20-bar  : {mom20:+.3f}%")
        lines.append(f"  Vol rank (50-bar): {vol_r:.0%}  {_bar(vol_r, 15)}")

        htf_dir, htf_str = _htf_trend(prices_htf)
        htf_icon = {"BULL": "📈", "BEAR": "📉", "NEUTRAL": "➡️"}[htf_dir]

        h("STEP 3 — HTF MACRO TREND  (EMA8 vs EMA21)")
        lines.append(f"  Direction        : {htf_icon} {htf_dir}")
        lines.append(f"  EMA separation   : {htf_str:.4f}%")

        h("STEP 4 — TOPOLOGICAL STRESS + ORDER BOOK")
        lines.append(f"  Stress (H1 pers.): {curr_stress:.4f}")
        stress_bar = _bar(min(curr_stress / 3.0, 1.0), 15)
        lines.append(
            f"  Stress level     : {stress_bar}  {'✅ active' if curr_stress >= 1.5 else '⚠️  below threshold'}")
        obi_label = "bullish pressure" if curr_obi > 0.1 else ("bearish pressure" if curr_obi < -0.1 else "neutral")
        lines.append(f"  OBI              : {curr_obi:+.4f}  {obi_label}")

        p_up = ml_probs.get("up", 0.0) if ml_probs else 0.0
        p_down = ml_probs.get("down", 0.0) if ml_probs else 0.0
        p_flat = ml_probs.get("flat", 1.0) if ml_probs else 1.0

        h("STEP 5 — ML PROBABILITIES  (TopoBooster)")
        lines.append(f"  P(UP)   {_bar(p_up, 12)}  {p_up:.1%}")
        lines.append(f"  P(DOWN) {_bar(p_down, 12)}  {p_down:.1%}")
        lines.append(f"  P(FLAT) {_bar(p_flat, 12)}  {p_flat:.1%}")
        lines.append(f"  Signal threshold : ≥ {self.prob_threshold:.0%}")

        h("STEP 6 — COMPOSITE SCORE")

        score_long = p_up * 40
        score_short = p_down * 40
        lines.append(f"  ML → LONG  +{score_long:.1f} / SHORT +{score_short:.1f}  (weight 40)")

        if htf_dir == "BULL":
            score_long += 20
            lines.append("  HTF BULL → LONG  +20")
        elif htf_dir == "BEAR":
            score_short += 20
            lines.append("  HTF BEAR → SHORT +20")
        else:
            lines.append("  HTF NEUTRAL — no bonus")

        if rsi_val < 35:
            score_long += 15
            lines.append(f"  RSI {rsi_val:.1f} oversold  → LONG  +15")
        elif rsi_val > 65:
            score_short += 15
            lines.append(f"  RSI {rsi_val:.1f} overbought → SHORT +15")
        else:
            lines.append(f"  RSI {rsi_val:.1f} neutral — no bonus")

        if mom5 > 0 and mom20 > 0:
            score_long += 10
            lines.append(f"  Momentum ↑↑ ({mom5:+.2f}% / {mom20:+.2f}%) → LONG  +10")
        elif mom5 < 0 and mom20 < 0:
            score_short += 10
            lines.append(f"  Momentum ↓↓ ({mom5:+.2f}% / {mom20:+.2f}%) → SHORT +10")
        else:
            lines.append(f"  Momentum mixed — no bonus")

        if curr_obi > 0.15:
            score_long += 15
            lines.append(f"  OBI {curr_obi:+.3f} (buy pressure) → LONG  +15")
        elif curr_obi < -0.15:
            score_short += 15
            lines.append(f"  OBI {curr_obi:+.3f} (sell pressure) → SHORT +15")
        else:
            lines.append(f"  OBI {curr_obi:+.3f} neutral — no bonus")

        lines.append(sep)
        lines.append(f"  LONG  {_bar(score_long / 100, 20)}  {score_long:.1f} pts")
        lines.append(f"  SHORT {_bar(score_short / 100, 20)}  {score_short:.1f} pts")

        h("STEP 7 — DIRECTION")

        if score_long >= score_short and score_long >= 30:
            direction = "LONG"
        elif score_short > score_long and score_short >= 30:
            direction = "SHORT"
        else:
            direction = "FLAT"

        dir_icon = {"LONG": "🚀", "SHORT": "🩸", "FLAT": "⏸️"}[direction]
        lines.append(f"  Rule: max(LONG, SHORT) if ≥ 30 pts")
        lines.append(f"  Decision: {dir_icon}  {direction}")

        entry = curr_price
        sl_price = tp_price = rr = None

        if direction != "FLAT":
            h("STEP 8 — SL / TP LEVELS")

            raw_sl_pct = (atr_val * self.atr_sl_mult) / curr_price
            sl_pct = max(self.min_sl_pct, min(raw_sl_pct, self.max_sl_pct))
            tp_pct = sl_pct * (self.atr_tp_mult / self.atr_sl_mult)
            rr = tp_pct / sl_pct

            lines.append(f"  SL formula  : ATR({atr_val:.2f}) × {self.atr_sl_mult} = {atr_val * self.atr_sl_mult:.2f}")
            lines.append(f"  SL% (raw)   : {raw_sl_pct:.4%}")
            lines.append(f"  SL% (clamped [{self.min_sl_pct:.2%}…{self.max_sl_pct:.2%}]) = {sl_pct:.4%}")
            lines.append(f"  TP% = SL% × {self.atr_tp_mult / self.atr_sl_mult:.1f} = {tp_pct:.4%}")
            lines.append(f"  Risk/Reward : 1 : {rr:.2f}")

            if direction == "LONG":
                sl_price = round(entry * (1 - sl_pct), 2)
                tp_price = round(entry * (1 + tp_pct), 2)
            else:
                sl_price = round(entry * (1 + sl_pct), 2)
                tp_price = round(entry * (1 - tp_pct), 2)

            lines.append(f"  Entry        : {entry:,.2f}")
            lines.append(f"  Stop-loss    : {sl_price:,.2f}")
            lines.append(f"  Take-profit  : {tp_price:,.2f}")

            h("STEP 9 — POSITION SIZE & P&L")

            pos_usd = margin_usdt * leverage
            amount_coins = math.floor((pos_usd / entry) * 1_000) / 1_000.0
            profit_usd = pos_usd * tp_pct
            loss_usd = pos_usd * sl_pct
            fees_usd = pos_usd * 0.0004 * 2
            net_profit = profit_usd - fees_usd
            net_loss = loss_usd + fees_usd

            lines.append(f"  Margin       : {margin_usdt:.0f} USDT")
            lines.append(f"  Leverage     : {leverage}×")
            lines.append(f"  Position     : {pos_usd:.0f} USDT  ({amount_coins:.3f} BTC)")
            lines.append(f"  TP profit    : +{net_profit:.2f} USDT  ({net_profit / margin_usdt:.1%} on margin)")
            lines.append(f"  SL loss      : −{net_loss:.2f} USDT  ({net_loss / margin_usdt:.1%} on margin)")
            lines.append(f"  Fees (×2)    : −{fees_usd:.2f} USDT")

            h("STEP 10 — EXPECTED VALUE & KELLY CRITERION")

            win_prob = p_up if direction == "LONG" else p_down
            ev = win_prob * net_profit - (1 - win_prob) * net_loss
            kelly = (win_prob * rr - (1 - win_prob)) / rr if rr > 0 else 0.0
            kelly_f = max(0.0, min(kelly, 0.25))

            ev_icon = "✅" if ev > 0 else "🚫"
            lines.append(f"  Win probability  : {win_prob:.1%}")
            lines.append(f"  Expected value   : {ev_icon}  {ev:+.2f} USDT per trade")
            lines.append(f"  Full Kelly f     : {kelly:.1%}")
            lines.append(f"  Suggested f (¼K) : {kelly_f:.1%}  → {kelly_f * margin_usdt:.1f} USDT margin")
            if ev <= 0:
                lines.append("  ⚠️  Negative EV — consider skipping this signal.")

        else:
            h("STEP 8 — SL / TP LEVELS")
            lines.append("  Insufficient signal strength — no trade recommended.")

        lines.append(f"\n{'╔' + '═' * W + '╗'}")
        lines.append(f"  {dir_icon}  RECOMMENDATION: {direction}")
        if direction != "FLAT":
            lines.append(f"  📍 ENTRY      : {entry:,.2f} USDT")
            lines.append(f"  🛑 STOP-LOSS  : {sl_price:,.2f} USDT")
            lines.append(f"  🎯 TAKE-PROFIT: {tp_price:,.2f} USDT")
            lines.append(f"  ⚖️  R/R         : 1 : {rr:.2f}")
        lines.append(f"{'╚' + '═' * W + '╝'}")

        return {
            "direction": direction,
            "entry": entry,
            "sl": sl_price,
            "tp": tp_price,
            "rr": rr,
            "score_long": score_long,
            "score_short": score_short,
            "steps": lines,
        }


class TradeCalculatorWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚡ TopoAlpha — Trade Calculator")
        self.resize(800, 920)
        self.setStyleSheet("background:#09090f; color:#e0e0e0;")

        self._prices: list | None = None
        self._stress_history: list | None = None
        self._obi_history: list | None = None
        self._prices_htf: list | None = None
        self._ml = None
        self._trader = None
        self._standalone_feeder = None

        self._calc = TradeCalculator()
        self._build_ui()

    def feed(self, prices, stress_history, obi_history, prices_htf, ml=None, trader=None) -> None:
        self._prices = prices
        self._stress_history = stress_history
        self._obi_history = obi_history
        self._prices_htf = prices_htf
        self._ml = ml
        self._trader = trader
        self._status.setText("🟢  Connected to live engine")
        self._status.setStyleSheet("color:#00FF88; font-size:13px;")

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        title = QLabel("⚡  TRADE CALCULATOR")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Consolas", 18, QFont.Bold))
        title.setStyleSheet("color:#00FFFF; padding:4px;")
        layout.addWidget(title)

        self._status = QLabel("⚠️  No data — run the main engine or wait for standalone load")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet("color:#FFAA00; font-size:13px;")
        layout.addWidget(self._status)

        params = QGroupBox("Parameters")
        params.setStyleSheet(
            "QGroupBox { color:#888; font-size:13px; border:1px solid #2a2a3a;"
            "  border-radius:6px; margin-top:6px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:10px; }"
        )
        form = QFormLayout(params)
        form.setSpacing(6)

        def spin(lo, hi, val, dec, step):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setStyleSheet(
                "QDoubleSpinBox { background:#111; color:#ddd;"
                "  border:1px solid #333; border-radius:4px; padding:3px 6px; }"
            )
            return s

        self._spin_margin = spin(10, 10_000, 50, 0, 10)
        self._spin_leverage = spin(1, 125, 10, 0, 1)
        self._spin_sl_mult = spin(0.5, 5.0, 1.5, 1, 0.5)
        self._spin_tp_mult = spin(0.5, 10.0, 3.0, 1, 0.5)
        self._spin_prob_thr = spin(0.40, 0.90, 0.55, 2, 0.05)

        form.addRow("Margin (USDT):", self._spin_margin)
        form.addRow("Leverage:", self._spin_leverage)
        form.addRow("ATR × SL:", self._spin_sl_mult)
        form.addRow("ATR × TP:", self._spin_tp_mult)
        form.addRow("Min P(ML):", self._spin_prob_thr)
        layout.addWidget(params)

        btn_row = QHBoxLayout()

        self._calc_btn = QPushButton("🔍  CALCULATE")
        self._calc_btn.setFixedHeight(44)
        self._calc_btn.setFont(QFont("Consolas", 13, QFont.Bold))
        self._calc_btn.setStyleSheet(
            "QPushButton { background:#002233; color:#00FFFF;"
            "  border:2px solid #00FFFF; border-radius:8px; }"
            "QPushButton:hover   { background:#004466; }"
            "QPushButton:pressed { background:#006688; }"
        )
        self._calc_btn.clicked.connect(self._on_calculate)

        self._clear_btn = QPushButton("✕  Clear")
        self._clear_btn.setFixedHeight(44)
        self._clear_btn.setStyleSheet(
            "QPushButton { background:#1a0000; color:#FF5555;"
            "  border:2px solid #662222; border-radius:8px; padding:0 16px; }"
            "QPushButton:hover { background:#330000; }"
        )
        self._clear_btn.clicked.connect(self._on_clear)

        btn_row.addWidget(self._calc_btn, stretch=4)
        btn_row.addWidget(self._clear_btn, stretch=1)
        layout.addLayout(btn_row)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(QFont("Consolas", 11))
        self._output.setStyleSheet(
            "QTextEdit { background:#060608; color:#c8d0d8;"
            "  border:1px solid #1e1e2a; border-radius:6px; padding:8px; }"
        )
        layout.addWidget(self._output, stretch=1)

        self._result_bar = QLabel("— no calculation —")
        self._result_bar.setAlignment(Qt.AlignCenter)
        self._result_bar.setFixedHeight(54)
        self._result_bar.setFont(QFont("Consolas", 14, QFont.Bold))
        self._result_bar.setStyleSheet(
            "background:#0d0d1a; color:#555; border:2px solid #222; border-radius:8px;"
        )
        layout.addWidget(self._result_bar)

    def _on_calculate(self) -> None:
        self._calc_btn.setEnabled(False)
        self._calc_btn.setText("⏳  Calculating…")
        QApplication.processEvents()
        try:
            self._run_calculation()
        finally:
            self._calc_btn.setEnabled(True)
            self._calc_btn.setText("🔍  CALCULATE")

    def _run_calculation(self) -> None:
        if not self._prices:
            self._load_standalone()
            if not self._prices:
                self._output.setPlainText("❌  No data. Start the main engine or check your connection.")
                return

        self._calc.atr_sl_mult = self._spin_sl_mult.value()
        self._calc.atr_tp_mult = self._spin_tp_mult.value()
        self._calc.prob_threshold = self._spin_prob_thr.value()

        ml_probs = None
        if self._ml and self._ml.is_trained:
            try:
                ml_probs = self._ml.predict(
                    self._prices,
                    self._stress_history or [0.0] * len(self._prices),
                    self._obi_history or [0.0] * len(self._prices),
                    prices_htf=self._prices_htf,
                )
            except Exception as exc:
                logger.warning(f"ML predict: {exc}")

        res = self._calc.calculate(
            prices=self._prices,
            stress_history=self._stress_history or [0.0] * len(self._prices),
            obi_history=self._obi_history or [0.0] * len(self._prices),
            prices_htf=self._prices_htf or [],
            ml_probs=ml_probs,
            margin_usdt=self._spin_margin.value(),
            leverage=int(self._spin_leverage.value()),
        )

        if "error" in res:
            self._output.setPlainText(f"❌  {res['error']}")
            return

        self._output.setPlainText("\n".join(res["steps"]))
        self._output.moveCursor(QTextCursor.End)

        d = res["direction"]
        if d == "LONG":
            bar_text = f"🚀 LONG   Entry {res['entry']:,.2f}  │  SL {res['sl']:,.2f}  │  TP {res['tp']:,.2f}  │  R/R 1:{res['rr']:.1f}"
            bar_style = "background:#002200; color:#00FF88; border:2px solid #00FF88; border-radius:8px;"
        elif d == "SHORT":
            bar_text = f"🩸 SHORT  Entry {res['entry']:,.2f}  │  SL {res['sl']:,.2f}  │  TP {res['tp']:,.2f}  │  R/R 1:{res['rr']:.1f}"
            bar_style = "background:#220000; color:#FF5555; border:2px solid #FF5555; border-radius:8px;"
        else:
            bar_text = "⏸️   FLAT — signal strength below threshold"
            bar_style = "background:#111100; color:#BBBB00; border:2px solid #888800; border-radius:8px;"

        self._result_bar.setText(bar_text)
        self._result_bar.setStyleSheet(f"QLabel {{ {bar_style} font-size:14px; font-weight:bold; }}")

    def _on_clear(self) -> None:
        self._output.clear()
        self._result_bar.setText("— no calculation —")
        self._result_bar.setStyleSheet(
            "background:#0d0d1a; color:#555; border:2px solid #222; border-radius:8px;"
            " font-size:14px; font-weight:bold;"
        )

    def _load_standalone(self) -> None:
        try:
            from data_feeder import RobustDataFeeder
            if self._standalone_feeder is None:
                self._standalone_feeder = RobustDataFeeder("BTC/USDT", "5m", htf="1h")
            self._status.setText("🔄  Loading from Binance…")
            QApplication.processEvents()

            ohlcv = self._standalone_feeder.fetch_initial(limit=300)
            ohlcv_htf = self._standalone_feeder.fetch_initial_htf(limit=50)

            self._prices = [c[4] for c in ohlcv]
            self._stress_history = [0.0] * len(self._prices)
            self._obi_history = [self._standalone_feeder.fetch_order_book_imbalance(20)] * len(self._prices)
            self._prices_htf = [c[4] for c in ohlcv_htf]

            self._status.setText(f"🟡  Standalone mode — {len(self._prices)} bars loaded")
            self._status.setStyleSheet("color:#FFCC00; font-size:13px;")
        except Exception as exc:
            logger.error(f"Standalone load: {exc}")
            self._status.setText(f"❌  Load failed: {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = TradeCalculatorWindow()
    win.show()
    sys.exit(app.exec_())
