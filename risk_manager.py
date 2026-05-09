from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("TopoAlpha.RiskManager")

_MAX_CONSECUTIVE_LOSSES: int   = 4
_MAX_DRAWDOWN_PCT:       float = 0.15
_COOLDOWN_MINUTES:       int   = 30
_JOURNAL_PATH:           str   = "trade_journal.csv"

_JOURNAL_HEADERS = [
    "unix_ts", "side", "entry_price", "exit_price", "reason",
    "net_profit_usd", "balance_after",
    "sl_pct", "tp_pct",
    "stress_at_signal", "prob_up", "prob_down",
    "htf_trend", "atr_pct", "obi",
    "consecutive_losses_before", "drawdown_pct_before",
]

class RiskManager:

    def __init__(
        self,
        initial_balance:       float = 10_000.0,
        max_consecutive_losses: int  = _MAX_CONSECUTIVE_LOSSES,
        max_drawdown_pct:      float = _MAX_DRAWDOWN_PCT,
        cooldown_minutes:      int   = _COOLDOWN_MINUTES,
        journal_path:          str   = _JOURNAL_PATH,
        stress_percentile:     float = 75.0,
        stress_buffer_len:     int   = 200,
    ) -> None:
        self._peak_balance           = initial_balance
        self._consecutive_losses     = 0
        self._max_consecutive_losses = max_consecutive_losses
        self._max_drawdown_pct       = max_drawdown_pct
        self._cooldown_s             = cooldown_minutes * 60
        self._circuit_open_time:     float | None = None

        self._stress_buf:     list  = []
        self._stress_buf_max: int   = stress_buffer_len
        self._stress_pct:     float = stress_percentile

        self._journal_path = journal_path
        self._init_journal()

    def _init_journal(self) -> None:
        p = Path(self._journal_path)
        if not p.exists():
            with p.open("w", newline="") as f:
                csv.writer(f).writerow(_JOURNAL_HEADERS)
            logger.info(f"[Journal] Created {self._journal_path}")

    def log_trade(self, result: dict, balance: float, context: dict) -> None:
        current_dd = (self._peak_balance - balance) / (self._peak_balance + 1e-10)
        row = [
            int(time.time()),
            result.get("type",           ""),
            round(result.get("entry",    0.0), 6),
            round(result.get("exit",     0.0), 6),
            result.get("reason",         ""),
            round(result.get("net_profit_usd", 0.0), 4),
            round(balance,               4),
            round(result.get("sl_pct",   0.0), 6),
            round(result.get("tp_pct",   0.0), 6),
            round(context.get("stress",  0.0), 4),
            round(context.get("prob_up", 0.0), 4),
            round(context.get("prob_down", 0.0), 4),
            context.get("htf_trend",    ""),
            round(context.get("atr_pct", 0.0), 6),
            round(context.get("obi",     0.0), 4),
            self._consecutive_losses,
            round(current_dd,            4),
        ]
        try:
            with open(self._journal_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as exc:
            logger.warning(f"[Journal] Write error: {exc}")

    def update_after_trade(self, result: dict, current_balance: float) -> None:
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance

        profit = result.get("net_profit_usd", 0.0)
        if profit < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        drawdown = (self._peak_balance - current_balance) / (self._peak_balance + 1e-10)

        if self._consecutive_losses >= self._max_consecutive_losses:
            self._trip(
                f"{self._consecutive_losses} consecutive losses "
                f"(limit {self._max_consecutive_losses})"
            )
        elif drawdown >= self._max_drawdown_pct:
            self._trip(
                f"drawdown {drawdown * 100:.1f}% ≥ "
                f"limit {self._max_drawdown_pct * 100:.0f}%"
            )

    def _trip(self, reason: str) -> None:
        if self._circuit_open_time is None:
            self._circuit_open_time = time.time()
            logger.warning(
                f"🔴 CIRCUIT BREAKER TRIPPED — {reason}. "
                f"Cooldown: {self._cooldown_s // 60} min."
            )

    def is_halted(self) -> bool:
        if self._circuit_open_time is None:
            return False
        elapsed = time.time() - self._circuit_open_time
        if elapsed >= self._cooldown_s:
            self._circuit_open_time  = None
            self._consecutive_losses = 0
            logger.info(
                f"✅ Circuit breaker reset after {self._cooldown_s // 60} min cooldown. "
                "Trading resumed."
            )
            return False
        remaining_min = int(self._cooldown_s - elapsed) // 60
        logger.debug(f"[RiskManager] Circuit open — {remaining_min} min remaining.")
        return True

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def drawdown_pct(self) -> float:
        return 0.0

    @property
    def peak_balance(self) -> float:
        return self._peak_balance

    def update_stress(self, stress: float) -> None:
        if stress > 0:
            self._stress_buf.append(stress)
        if len(self._stress_buf) > self._stress_buf_max:
            del self._stress_buf[0]

    def adaptive_stress_threshold(self, base: float = 1.5) -> float:
        if len(self._stress_buf) < 30:
            return base
        p75 = float(np.percentile(self._stress_buf, self._stress_pct))
        threshold = max(base, p75)
        return threshold

    def status_line(self, current_balance: float) -> str:
        dd = (self._peak_balance - current_balance) / (self._peak_balance + 1e-10)
        halted = "HALTED" if self.is_halted() else "OK"
        return (
            f"RiskManager [{halted}] | "
            f"ConsecLoss={self._consecutive_losses}/{self._max_consecutive_losses} | "
            f"DD={dd * 100:.1f}% / limit {self._max_drawdown_pct * 100:.0f}% | "
            f"StressBuf={len(self._stress_buf)}"
        )