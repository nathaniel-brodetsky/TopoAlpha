from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd

from ml_model import TopoBooster

logger = logging.getLogger("TopoAlpha.WFO")


# ── In-memory trade simulator (mirrors PaperTrader without SQLite) ────────── #

@dataclass
class _Trade:
    side: str
    entry: float
    entry_idx: int
    sl_pct: float
    tp_pct: float
    horizon_bars: int
    margin_usdt: float = 50.0
    leverage: int = 10
    fee_pct: float = 0.0004

    def evaluate(self, candles: list, from_idx: int) -> dict | None:
        pos_usd = self.margin_usdt * self.leverage
        end_idx = min(from_idx + self.horizon_bars, len(candles) - 1)

        for i in range(from_idx, end_idx + 1):
            c = candles[i]
            hi, lo = c[2], c[3]
            close = c[4]

            if self.side == "LONG":
                tp_price = self.entry * (1 + self.tp_pct)
                sl_price = self.entry * (1 - self.sl_pct)
                hit_tp = hi >= tp_price
                hit_sl = lo <= sl_price
            else:
                tp_price = self.entry * (1 - self.tp_pct)
                sl_price = self.entry * (1 + self.sl_pct)
                hit_tp = lo <= tp_price
                hit_sl = hi >= sl_price

            if hit_tp and not hit_sl:
                exit_p = tp_price
                reason = "TP"
            elif hit_sl and not hit_tp:
                exit_p = sl_price
                reason = "SL"
            elif hit_tp and hit_sl:
                exit_p = sl_price
                reason = "SL"
            elif i == end_idx:
                exit_p = close
                reason = "TIME"
            else:
                continue

            pnl_pct = (
                (exit_p - self.entry) / self.entry
                if self.side == "LONG"
                else (self.entry - exit_p) / self.entry
            )
            net_pnl = pos_usd * pnl_pct - pos_usd * self.fee_pct * 2
            return {
                "side": self.side,
                "entry": self.entry,
                "exit": exit_p,
                "entry_idx": self.entry_idx,
                "exit_idx": i,
                "reason": reason,
                "pnl_pct": pnl_pct,
                "net_pnl_usd": net_pnl,
                "sl_pct": self.sl_pct,
                "tp_pct": self.tp_pct,
            }
        return None


# ── Per-fold metrics ──────────────────────────────────────────────────────── #

@dataclass
class FoldResult:
    fold: int
    train_bars: int
    oos_bars: int
    trades: list = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t["net_pnl_usd"] > 0)
        return wins / len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t["net_pnl_usd"] for t in self.trades)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t["net_pnl_usd"] for t in self.trades if t["net_pnl_usd"] > 0)
        gross_loss = abs(sum(t["net_pnl_usd"] for t in self.trades if t["net_pnl_usd"] < 0))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")

    @property
    def expectancy(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t["net_pnl_usd"] for t in self.trades) / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        equity = np.concatenate([[0.0], np.cumsum([t["net_pnl_usd"] for t in self.trades])])
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        return float(dd.max())

    @property
    def sharpe(self) -> float:
        if len(self.trades) < 5:
            return 0.0
        pnls = np.array([t["net_pnl_usd"] for t in self.trades])
        std = pnls.std()
        if std < 1e-6:  # guard: near-identical trades → meaningless ratio
            return 0.0
        return float(pnls.mean() / std * np.sqrt(len(pnls)))

    def summary(self) -> dict:
        return {
            "fold": self.fold,
            "train_bars": self.train_bars,
            "oos_bars": self.oos_bars,
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "profit_factor": round(self.profit_factor, 3),
            "expectancy": round(self.expectancy, 2),
            "max_dd": round(self.max_drawdown, 2),
            "sharpe": round(self.sharpe, 3),
        }


# ── Parallel fold worker (module-level = picklable) ───────────────────────── #

def _wfo_fold_worker(k, ts, te, os_, oe, candles, stress_history, obi_history,
                     htf_trend_list, candles_htf, cfg):
    import time as _t
    t0 = _t.time()
    model = TopoBooster()
    if not model.train(candles[ts:te], stress_history[ts:te], obi_history[ts:te],
                       candles_htf=candles_htf):
        return None
    wfo = WalkForwardOptimizer(**cfg)
    probs_fold = wfo._precalc_probs(
        model, candles[ts:oe], stress_history[ts:oe], obi_history[ts:oe], candles_htf
    )
    is_len = te - ts
    probs_is = probs_fold[:is_len]
    probs_oos = probs_fold[is_len:]
    trend_is = htf_trend_list[ts:te]
    trend_oos = htf_trend_list[os_:oe]

    if cfg["optimize_params"]:
        p = wfo._best_params_on_is(candles[ts:te], stress_history[ts:te],
                                   obi_history[ts:te], probs_is, trend_is)
        st_used = p["stress_threshold"]
        pt_used = p["prob_threshold"]
    else:
        st_used = cfg["stress_threshold"]
        pt_used = cfg["prob_threshold"]

    trades = wfo._simulate_oos(candles[os_:oe], stress_history[os_:oe],
                               obi_history[os_:oe], probs_oos, trend_oos,
                               st_used, pt_used)
    fr = FoldResult(fold=k + 1, train_bars=te - ts, oos_bars=oe - os_, trades=trades)
    return fr, _t.time() - t0, st_used, pt_used


def _wfo_fold_worker_best_is(k, is_candidates, te, os_, oe, candles, stress_history,
                             obi_history, htf_trend_list, candles_htf, cfg):
    """
    Best-IS selection: for a given OOS window, try every candidate IS start,
    score each on IS validation, train the final model on the winner.
    Candidates are historic IS windows that don't overlap with OOS.
    """
    import time as _t
    import math
    t0 = _t.time()

    wfo = WalkForwardOptimizer(**cfg)
    best_score = -math.inf
    best_ts = is_candidates[0]
    best_st = cfg["stress_threshold"]
    best_pt = cfg["prob_threshold"]
    best_model = None

    for ts in is_candidates:
        model_try = TopoBooster()
        if not model_try.train(candles[ts:te], stress_history[ts:te],
                               obi_history[ts:te], candles_htf=candles_htf):
            continue

        probs_is = wfo._precalc_probs(
            model_try, candles[ts:te], stress_history[ts:te],
            obi_history[ts:te], candles_htf
        )
        trend_is = htf_trend_list[ts:te]

        if cfg["optimize_params"]:
            p = wfo._best_params_on_is(candles[ts:te], stress_history[ts:te],
                                       obi_history[ts:te], probs_is, trend_is)
            st = p["stress_threshold"]
            pt = p["prob_threshold"]
        else:
            st = cfg["stress_threshold"]
            pt = cfg["prob_threshold"]

        trades_is = wfo._simulate_oos(
            candles[ts:te], stress_history[ts:te], obi_history[ts:te],
            probs_is, trend_is, st, pt
        )
        n = len(trades_is)
        if n < cfg["is_min_trades"]:
            continue

        pnls = [t["net_pnl_usd"] for t in trades_is]
        gross_win = sum(p for p in pnls if p > 0)
        gross_los = abs(sum(p for p in pnls if p < 0))
        pf = min(gross_win / gross_los, 3.0) if gross_los > 0 else 2.0
        pnls_arr = np.array(pnls)
        std = float(pnls_arr.std()) if len(pnls_arr) > 1 else 1e-8
        sharpe_bonus = max(0.0, float(pnls_arr.mean() / (std + 1e-8)) * math.sqrt(n)) / 2.0
        score = pf * math.log1p(n) * (1.0 + sharpe_bonus)

        if score > best_score:
            best_score = score
            best_ts = ts
            best_st = st
            best_pt = pt
            best_model = model_try

    if best_model is None:
        # Fallback: use most recent IS window without optimization
        best_model = TopoBooster()
        if not best_model.train(candles[is_candidates[-1]:te],
                                stress_history[is_candidates[-1]:te],
                                obi_history[is_candidates[-1]:te],
                                candles_htf=candles_htf):
            return None
        best_ts = is_candidates[-1]

    # Run OOS simulation with the winning IS params
    probs_oos = wfo._precalc_probs(
        best_model, candles[os_:oe], stress_history[os_:oe],
        obi_history[os_:oe], candles_htf
    )
    trend_oos = htf_trend_list[os_:oe]
    trades = wfo._simulate_oos(
        candles[os_:oe], stress_history[os_:oe], obi_history[os_:oe],
        probs_oos, trend_oos, best_st, best_pt
    )

    fr = FoldResult(fold=k + 1, train_bars=te - best_ts, oos_bars=oe - os_, trades=trades)
    return fr, _t.time() - t0, best_st, best_pt, best_ts


# ── Walk-Forward Optimizer ────────────────────────────────────────────────── #

class WalkForwardOptimizer:
    # stress_threshold=0.0 means "pass everything" — optimizer will find the right cut.
    # For 1h LTF, TDA stress values are lower than 15m; starting at 0.0 is correct.
    DEFAULT_PARAM_GRID = {
        "stress_threshold": [0.0, 0.1, 0.2, 0.3],
        "prob_threshold": [0.45, 0.48, 0.50, 0.52, 0.55],  # Подняли пороги
    }

    def __init__(
            self,
            n_folds: int = 5,
            train_pct: float = 0.70,
            mode: Literal["anchored", "rolling", "best_is"] = "rolling",
            stress_threshold: float = 0.0,  # 0.0 = pass-all; optimizer picks the real cut
            prob_threshold: float = 0.47,  # back to calibrated baseline
            # Synced with TopoBooster training labels
            horizon_bars: int = TopoBooster.HORIZON,  # 16
            sl_atr_mult: float = TopoBooster.SL_MULT,  # 1.5
            tp_atr_mult: float = TopoBooster.TP_MULT,  # 2.5
            margin_usdt: float = 50.0,
            leverage: int = 10,
            fee_pct: float = 0.0004,
            min_atr_pct: float = 0.002,
            optimize_params: bool = False,
            param_grid: dict | None = None,
            min_train_bars: int = 300,
            is_min_trades: int = 3,  # <--- CHANGED FROM 8
            use_trend_filter: bool = False,
            use_trend_confirm: bool = True,  # was False — default to trend-follow for HTF profitability
    ):

        self.n_folds = n_folds
        self.train_pct = train_pct
        self.mode = mode
        self.stress_threshold = stress_threshold
        self.prob_threshold = prob_threshold
        self.horizon_bars = horizon_bars
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.margin_usdt = margin_usdt
        self.leverage = leverage
        self.fee_pct = fee_pct
        self.min_atr_pct = min_atr_pct
        self.optimize_params = optimize_params
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.min_train_bars = min_train_bars
        self.is_min_trades = is_min_trades
        self.use_trend_filter = use_trend_filter
        self.use_trend_confirm = use_trend_confirm

    def _make_windows(self, n: int) -> list[tuple[int, int, int, int]]:
        fold_size = n // (self.n_folds + 1)
        windows = []

        for k in range(self.n_folds):
            oos_start = fold_size * (k + 1)
            oos_end = fold_size * (k + 2)

            if self.mode == "anchored":
                train_start = 0
            elif self.mode == "best_is":
                # IS start same as rolling — candidate generation happens in worker
                train_size = int(fold_size / (1 - self.train_pct) * self.train_pct)
                train_start = max(0, oos_start - train_size)
            else:
                train_size = int(fold_size / (1 - self.train_pct) * self.train_pct)
                train_start = max(0, oos_start - train_size)

            train_end = oos_start
            if train_end - train_start < self.min_train_bars:
                continue
            windows.append((train_start, train_end, oos_start, oos_end))

        return windows

    def _precalc_trend(self, candles: list) -> list[str]:
        closes = np.array([c[4] for c in candles], dtype=float)
        trend = ["flat"] * len(candles)
        if len(closes) < 84:
            return trend

        s = pd.Series(closes)
        # Daemon uses EMA8 + EMA21 on real 1h candles.
        # WFO works on 15m LTF → multiply by 4: EMA32 + EMA84.
        # Matches daemon ~8h / ~21h responsiveness exactly.
        ema8_span = getattr(TopoBooster, "_HTF_EMA8_SPAN", 32)  # was 120 (30h)
        ema21_span = getattr(TopoBooster, "_HTF_EMA21_SPAN", 84)  # was 315 (78h)
        ema8 = s.ewm(span=ema8_span, adjust=False).mean()
        ema21 = s.ewm(span=ema21_span, adjust=False).mean()
        gap = (ema8 - ema21) / (ema21 + 1e-10)

        for i in range(len(candles)):
            g = gap.iloc[i]
            if g > 0.0005:  # was 0.002 — 1h EMAs move slowly; 0.0005 gives realistic bull/bear freq
                trend[i] = "bull"
            elif g < -0.0005:
                trend[i] = "bear"
        return trend

    def _best_params_on_is(self, candles_is: list, stress_is: list, obi_is: list, probs_is: list,
                           trend_is: list) -> dict:
        best_score = -np.inf
        best_params = {
            "stress_threshold": self.stress_threshold,
            "prob_threshold": self.prob_threshold,
        }
        for st, pt in product(self.param_grid["stress_threshold"],
                              self.param_grid["prob_threshold"]):
            trades = self._simulate_oos(candles_is, stress_is, obi_is, probs_is, trend_is, st, pt)
            n = len(trades)
            if n < self.is_min_trades:
                continue
            pnls = np.array([t["net_pnl_usd"] for t in trades])
            gross_win = pnls[pnls > 0].sum()
            gross_los = abs(pnls[pnls < 0].sum())
            pf = min(gross_win / gross_los, 3.0) if gross_los > 0 else 2.0
            # Sharpe bonus: reward consistent outperformance, not just high PF
            std = pnls.std()
            sharpe_bonus = max(0.0, float(pnls.mean() / (std + 1e-8)) * np.sqrt(n)) / 2.0
            score = pf * np.log1p(n) * (1.0 + sharpe_bonus)  # quality × freq × consistency
            if score > best_score:
                best_score = score
                best_params = {"stress_threshold": st, "prob_threshold": pt}
        return best_params

    def _compute_atr_pct(self, candles: list, idx: int, period: int = 14) -> float:
        if idx < period + 1:
            return self.min_atr_pct
        closes = np.array([c[4] for c in candles[idx - period - 1: idx + 1]], dtype=float)
        atr = float(np.abs(np.diff(closes)).mean())
        return max(atr / (closes[-1] + 1e-10), self.min_atr_pct)

    def _simulate_oos(
            self,
            candles: list,
            stress_history: list,
            obi_history: list,
            probs_list: list,
            trend_list: list,
            stress_threshold: float,
            prob_threshold: float,
    ) -> list[dict]:
        trades: list[dict] = []
        in_trade = False
        n = len(candles)
        warmup = max(TopoBooster.HORIZON + TopoBooster.LOOKBACK + 30, 60)

        i = warmup
        while i < n - self.horizon_bars - 1:
            if not in_trade:
                stress = stress_history[i] if i < len(stress_history) else 0.0

                if stress >= stress_threshold:
                    probs = probs_list[i]
                    atr_pct = self._compute_atr_pct(candles, i)
                    sl_pct = max(atr_pct * self.sl_atr_mult, 0.002)
                    tp_pct = max(atr_pct * self.tp_atr_mult, 0.004)

                    # Flat class acts as veto: signal direction must be argmax
                    argmax = max(probs, key=probs.get)
                    side = None

                    if self.use_trend_filter:
                        # Pure counter-trend: invert ML signal, no HTF direction gate.
                        # HTF direction gate caused starvation in trending markets
                        # (e.g. sustained bull: HTF=bear never occurs → 0 trades).
                        # htf_atr_real stays in features as a volatility measure.
                        if argmax == "up" and probs["up"] >= prob_threshold:
                            side = "SHORT"
                        elif argmax == "down" and probs["down"] >= prob_threshold:
                            side = "LONG"
                    elif self.use_trend_confirm:
                        # Trend-confirm gate: ML direction must agree with HTF trend
                        trend = trend_list[i]
                        if argmax == "up" and probs["up"] >= prob_threshold and trend == "bull":
                            side = "LONG"
                        elif argmax == "down" and probs["down"] >= prob_threshold and trend == "bear":
                            side = "SHORT"
                    else:
                        # ML-only: trade in the direction the model predicts
                        if argmax == "up" and probs["up"] >= prob_threshold:
                            side = "LONG"
                        elif argmax == "down" and probs["down"] >= prob_threshold:
                            side = "SHORT"

                    if side:
                        entry_price = candles[i + 1][1]
                        trade = _Trade(
                            side=side,
                            entry=entry_price,
                            entry_idx=i + 1,
                            sl_pct=sl_pct,
                            tp_pct=tp_pct,
                            horizon_bars=self.horizon_bars,
                            margin_usdt=self.margin_usdt,
                            leverage=self.leverage,
                            fee_pct=self.fee_pct,
                        )
                        result = trade.evaluate(candles, i + 1)
                        if result:
                            trades.append(result)
                            i = result["exit_idx"] + 1
                            continue
            i += 1
        return trades

    def _precalc_probs(
            self,
            model: TopoBooster,
            candles: list,
            stress: list,
            obi: list,
            candles_htf: list | None = None
    ) -> list[dict]:
        import numpy as np
        from collections import deque

        n = len(candles)
        probs_list = [{"flat": 1.0, "up": 0.0, "down": 0.0} for _ in range(n)]

        if not model.is_trained:
            return probs_list

        df = model._build_features(candles, stress, obi, candles_htf)
        drop_cols = ["open", "high", "low", "close", "volume", "htf_ema8", "htf_ema21", "atr"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        for col in model._feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[model._feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X_scaled = model._scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        all_preds = []
        classes_ref = None

        for base_model in model._models:
            preds = base_model.predict_proba(X_scaled)
            all_preds.append(preds)
            if classes_ref is None:
                classes_ref = list(base_model.classes_)

        if not all_preds:
            return probs_list

        raw_proba = np.mean(all_preds, axis=0)
        mapped_proba = np.zeros((n, 3))

        for i, cls in enumerate(classes_ref):
            if int(cls) < 3:
                mapped_proba[:, int(cls)] = raw_proba[:, i]

        sums = mapped_proba.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        mapped_proba /= sums

        buf = deque(maxlen=model.BUFFER_SIZE)
        smooth = model._NEUTRAL_PROBS.copy()

        for i in range(n):
            buf.append(mapped_proba[i])
            buf_mean = np.mean(list(buf), axis=0)
            smooth = model.SMOOTH_ALPHA * buf_mean + (1.0 - model.SMOOTH_ALPHA) * smooth
            probs_list[i] = {
                "flat": float(smooth[0]),
                "up": float(smooth[1]),
                "down": float(smooth[2])
            }

        return probs_list

    def run(
            self,
            candles: list,
            stress_history: list,
            obi_history: list,
            candles_htf: list | None = None,
            n_jobs: int = -1,
    ) -> list[FoldResult]:
        n = len(candles)
        windows = self._make_windows(n)
        n_folds = len(windows)
        cpu = os.cpu_count() or 1
        workers = cpu if n_jobs == -1 else min(abs(n_jobs), cpu)
        workers = min(workers, n_folds)

        htf_trend_list = self._precalc_trend(candles)
        _trend_mode = ("counter" if self.use_trend_filter
                       else "confirm" if self.use_trend_confirm
        else "ml-only")
        logger.info(
            f"WFO | mode={self.mode} folds={n_folds} bars={n} "
            f"opt={self.optimize_params} workers={workers} "
            f"sl={self.sl_atr_mult}×ATR tp={self.tp_atr_mult}×ATR "
            f"trend={_trend_mode}"
        )

        cfg = dict(
            n_folds=self.n_folds, train_pct=self.train_pct, mode=self.mode,
            stress_threshold=self.stress_threshold, prob_threshold=self.prob_threshold,
            horizon_bars=self.horizon_bars, sl_atr_mult=self.sl_atr_mult,
            tp_atr_mult=self.tp_atr_mult, margin_usdt=self.margin_usdt,
            leverage=self.leverage, fee_pct=self.fee_pct, min_atr_pct=self.min_atr_pct,
            optimize_params=self.optimize_params, param_grid=self.param_grid,
            min_train_bars=self.min_train_bars, is_min_trades=self.is_min_trades,
            use_trend_filter=self.use_trend_filter,
            use_trend_confirm=self.use_trend_confirm,
        )

        ordered: list[FoldResult | None] = [None] * n_folds
        futures: dict = {}

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, (ts, te, os_, oe) in enumerate(windows):
                if self.mode == "best_is":
                    # Generate IS candidates: all historic start points in fold_size steps
                    fold_size_local = n // (self.n_folds + 1)
                    candidates = []
                    step = max(self.min_train_bars // 2, fold_size_local)
                    cur = 0
                    while cur + self.min_train_bars <= te:
                        candidates.append(cur)
                        cur += step
                    if not candidates or candidates[-1] != ts:
                        candidates.append(ts)  # always include the natural IS start
                    candidates = sorted(set(candidates))
                    logger.info(
                        f"Submitting fold {k + 1}/{n_folds} — "
                        f"OOS [{os_}:{oe}]  IS candidates: {len(candidates)}"
                    )
                    fut = pool.submit(
                        _wfo_fold_worker_best_is, k, candidates, te, os_, oe,
                        candles, stress_history, obi_history,
                        htf_trend_list, candles_htf, cfg,
                    )
                else:
                    logger.info(f"Submitting fold {k + 1}/{n_folds} — IS [{ts}:{te}] OOS [{os_}:{oe}]")
                    fut = pool.submit(
                        _wfo_fold_worker, k, ts, te, os_, oe,
                        candles, stress_history, obi_history,
                        htf_trend_list, candles_htf, cfg,
                    )
                futures[fut] = k

            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    logger.error(f"Fold {k + 1} raised: {exc}")
                    continue
                if res is None:
                    logger.warning(f"Fold {k + 1}: training failed.")
                    continue
                if self.mode == "best_is":
                    fr, elapsed, st_used, pt_used, best_ts = res
                    logger.info(
                        f"Fold {k + 1} done in {elapsed:.1f}s | "
                        f"best_IS_start={best_ts} prob={pt_used} | "
                        f"trades={fr.n_trades} wr={fr.win_rate:.1%} pnl=${fr.total_pnl:.2f}"
                    )
                else:
                    fr, elapsed, st_used, pt_used = res
                ordered[k] = fr
                logger.info(
                    f"Fold {k + 1} done in {elapsed:.1f}s | "
                    f"prob={pt_used} | "
                    f"trades={fr.n_trades} wr={fr.win_rate:.1%} pnl=${fr.total_pnl:.2f}"
                )

        return [fr for fr in ordered if fr is not None]

    @staticmethod
    def aggregate(results: list[FoldResult]) -> dict:
        all_trades = [t for fr in results for t in fr.trades]
        if not all_trades:
            return {"error": "no trades across all folds"}

        pnls = np.array([t["net_pnl_usd"] for t in all_trades])
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        gross_win = pnls[pnls > 0].sum()
        gross_loss = abs(pnls[pnls < 0].sum())
        win_rate = (pnls > 0).mean()
        avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0.0
        avg_loss = pnls[pnls < 0].mean() if (pnls < 0).any() else 0.0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        kelly = win_rate - (1 - win_rate) / rr if rr > 0 else 0.0

        return {
            "folds": len(results),
            "total_trades": len(all_trades),
            "total_pnl_usd": round(float(pnls.sum()), 2),
            "win_rate": round(float(win_rate), 4),
            "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
            "expectancy_usd": round(float(pnls.mean()), 2),
            "avg_win_usd": round(float(avg_win), 2),
            "avg_loss_usd": round(float(avg_loss), 2),
            "rr_ratio": round(rr, 2),
            "kelly_fraction": round(float(np.clip(kelly, 0, 0.25)), 4),
            "max_dd_usd": round(float(drawdown.max()), 2),
            "sharpe": round(float(pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(len(pnls))), 3),
            "oos_bars_total": sum(fr.oos_bars for fr in results),
        }

    @staticmethod
    def print_report(results: list[FoldResult]) -> None:
        print("\n" + "═" * 72)
        print("  TopoAlpha — Walk-Forward Validation Report")
        print("═" * 72)

        headers = ["Fold", "IS bars", "OOS bars", "Trades", "Win%", "PnL $", "PF", "E[$]", "MaxDD", "Sharpe"]
        col_w = [5, 8, 9, 7, 7, 9, 6, 7, 8, 7]
        header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
        print(f"\n  {header_row}")
        print("  " + "─" * 72)

        for fr in results:
            s = fr.summary()
            row_vals = [
                str(s["fold"]),
                str(s["train_bars"]),
                str(s["oos_bars"]),
                str(s["n_trades"]),
                f"{s['win_rate']:.1%}",
                f"{s['total_pnl']:+.2f}",
                f"{s['profit_factor']:.2f}",
                f"{s['expectancy']:+.2f}",
                f"{s['max_dd']:.2f}",
                f"{s['sharpe']:+.2f}",
            ]
            print("  " + "  ".join(v.ljust(w) for v, w in zip(row_vals, col_w)))

        print("\n" + "─" * 72)
        agg = WalkForwardOptimizer.aggregate(results)
        print("  AGGREGATE (out-of-sample)")
        print("─" * 72)
        for k, v in agg.items():
            print(f"  {k:<22} {v}")
        print("═" * 72 + "\n")

    @staticmethod
    def plot(results: list[FoldResult], save_path: str | None = None) -> None:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            logger.error("matplotlib not installed — cannot plot.")
            return

        all_trades = [t for fr in results for t in fr.trades]
        if not all_trades:
            print("No trades to plot.")
            return

        pnls = np.array([t["net_pnl_usd"] for t in all_trades])
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak

        win_pnls = pnls[pnls > 0]
        loss_pnls = pnls[pnls < 0]

        fig = plt.figure(figsize=(15, 9))
        fig.patch.set_facecolor("#0d0d1a")
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_eq = fig.add_subplot(gs[0, :])
        ax_dd = fig.add_subplot(gs[1, 0])
        ax_bar = fig.add_subplot(gs[1, 1])
        ax_dis = fig.add_subplot(gs[2, 0])
        ax_wr = fig.add_subplot(gs[2, 1])

        for ax in [ax_eq, ax_dd, ax_bar, ax_dis, ax_wr]:
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        palette = plt.cm.plasma(np.linspace(0.15, 0.85, len(results)))
        idx = 0
        for fr, colour in zip(results, palette):
            n = len(fr.trades)
            xs = range(idx, idx + n)
            ys = equity[idx: idx + n]
            ax_eq.plot(xs, ys, color=colour, lw=2, label=f"Fold {fr.fold}")
            if idx > 0:
                ax_eq.axvline(idx, color="#444", lw=1, ls="--")
            idx += n

        ax_eq.fill_between(range(len(equity)), equity, 0,
                           where=(equity >= 0), color="#00ff88", alpha=0.08)
        ax_eq.fill_between(range(len(equity)), equity, 0,
                           where=(equity < 0), color="#ff4444", alpha=0.08)
        ax_eq.axhline(0, color="#555", lw=1)
        ax_eq.set_title("OOS Equity Curve (cumulative P&L, USD)", color="#ccc")
        ax_eq.set_ylabel("$", color="#aaa")
        ax_eq.legend(loc="upper left", fontsize=8,
                     facecolor="#111", labelcolor="#ccc", framealpha=0.5)

        ax_dd.fill_between(range(len(drawdown)), drawdown, 0, color="#ff4444", alpha=0.6)
        ax_dd.set_title("Drawdown (USD)", color="#ccc")
        ax_dd.set_ylabel("$", color="#aaa")

        fold_pnls = [fr.total_pnl for fr in results]
        fold_labels = [f"F{fr.fold}" for fr in results]
        colours = ["#00ff88" if p > 0 else "#ff4444" for p in fold_pnls]
        ax_bar.bar(fold_labels, fold_pnls, color=colours, edgecolor="#333")
        ax_bar.axhline(0, color="#555", lw=1)
        ax_bar.set_title("PnL per Fold (USD)", color="#ccc")

        ax_dis.hist(pnls, bins=25, color="#7b68ee", edgecolor="#0d0d1a", alpha=0.85)
        ax_dis.axvline(0, color="#aaa", lw=1, ls="--")
        ax_dis.set_title("Trade P&L Distribution", color="#ccc")
        ax_dis.set_xlabel("USD", color="#aaa")

        fold_wr = [fr.win_rate for fr in results]
        ax_wr.bar(fold_labels, fold_wr, color="#00bfff", edgecolor="#333")
        ax_wr.axhline(0.5, color="#aaa", lw=1, ls="--")
        ax_wr.set_ylim(0, 1)
        ax_wr.set_title("Win Rate per Fold", color="#ccc")
        ax_wr.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

        agg = WalkForwardOptimizer.aggregate(results)
        suptitle = (
            f"TopoAlpha WFO  |  Trades: {agg['total_trades']}  "
            f"WR: {agg['win_rate']:.1%}  PF: {agg['profit_factor']:.2f}  "
            f"Sharpe: {agg['sharpe']:.2f}  Kelly: {agg['kelly_fraction']:.1%}"
        )
        fig.suptitle(suptitle, color="#eee", fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
            print(f"Chart saved → {save_path}")
        else:
            plt.tight_layout()
            plt.show()


# ── daemon.py integration helper ─────────────────────────────────────────── #

def run_pre_live_wfo(
        candles: list,
        stress_history: list,
        obi_history: list,
        candles_htf: list | None = None,
        n_folds: int = 5,
        mode: str = "rolling",
        optimize: bool = False,
        min_sharpe: float = 0.3,
        min_pf: float = 1.1,
) -> bool:
    wfo = WalkForwardOptimizer(
        n_folds=n_folds, mode=mode, optimize_params=optimize,
        use_trend_filter=False,
        use_trend_confirm=True,  # trend-confirm is required for positive HTF expectancy
    )
    results = wfo.run(candles, stress_history, obi_history, candles_htf)
    wfo.print_report(results)

    if not results:
        logger.error("[WFO] No fold results — cannot validate.")
        return False

    agg = WalkForwardOptimizer.aggregate(results)
    logger.info(f"[WFO] Aggregate: {agg}")

    sharpe_ok = agg.get("sharpe", 0) >= min_sharpe
    pf_ok = agg.get("profit_factor", 0) >= min_pf
    trades_ok = agg.get("total_trades", 0) >= 10

    passed = sharpe_ok and pf_ok and trades_ok
    if passed:
        logger.info("✅ WFO gates passed — proceeding to live trading.")
    else:
        logger.warning(
            f"🔴 WFO gates FAILED | "
            f"Sharpe {agg.get('sharpe'):.2f} (need ≥{min_sharpe}) | "
            f"PF {agg.get('profit_factor'):.2f} (need ≥{min_pf}) | "
            f"Trades {agg.get('total_trades')} (need ≥10)"
        )
    return passed


# ── Standalone CLI ────────────────────────────────────────────────────────── #

def _fetch_data(symbol: str, timeframe: str, limit: int, htf: str = "1h"):
    from data_feeder import RobustDataFeeder
    from tda_core import TDAAnalyzer
    import numpy as np

    feeder = RobustDataFeeder(symbol, timeframe, htf=htf)
    tda = TDAAnalyzer()

    print(f"Fetching {limit} {timeframe} candles for {symbol}…")
    candles = feeder.fetch_initial(limit=limit)
    obi = [0.0] * len(candles)

    # ── Fetch real HTF candles ─────────────────────────────────────────────── #
    htf_limit = max(200, limit // 10)
    print(f"Fetching {htf_limit} {htf} HTF candles…")
    candles_htf = feeder.fetch_initial_htf(limit=htf_limit)
    print(f"Got {len(candles_htf)} HTF candles.")

    print("Back-filling topological stress (this takes ~10-15 seconds)...")
    stress = [0.0] * len(candles)
    prices = [c[4] for c in candles]

    tau, dim = 5, 3
    tda_window = 50

    for i in range(tda_window, len(prices)):
        window = prices[i - tda_window:i]
        data = np.array(window, dtype=float)

        if len(data) >= (dim - 1) * tau + 1:
            data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
            embedded = np.vstack([
                data[:-(2 * tau)],
                data[tau:-tau],
                data[2 * tau:]
            ]).T

            if len(embedded) > 0:
                stress[i] = tda.get_topological_stress(embedded)

        if i > 0 and i % 1500 == 0:
            print(f"  ... {i}/{len(prices)} bars processed")

    print(f"Got {len(candles)} LTF candles.")
    return candles, stress, obi, candles_htf


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="TopoAlpha Walk-Forward Optimizer")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--htf", default="4h",
                    help="Higher timeframe for real HTF features (default: 4h)")
    ap.add_argument("--bars", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--mode", default="rolling", choices=["rolling", "anchored", "best_is"])
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--trend-filter", action="store_true",
                    help="Counter-trend gate: ML-up+HTF-bear→SHORT, ML-down+HTF-bull→LONG")
    ap.add_argument("--trend-confirm", action="store_true",
                    help="Trend-confirm gate: ML-up+HTF-bull→LONG, ML-down+HTF-bear→SHORT")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save-chart", default=None)
    ap.add_argument("--jobs", type=int, default=-1,
                    help="Parallel workers: -1=all cores, 1=sequential")
    args = ap.parse_args()

    candles, stress, obi, candles_htf = _fetch_data(
        args.symbol, args.timeframe, args.bars, htf=args.htf
    )

    wfo = WalkForwardOptimizer(
        n_folds=args.folds,
        mode=args.mode,
        optimize_params=args.optimize,
        use_trend_filter=args.trend_filter,
        use_trend_confirm=args.trend_confirm,
    )
    results = wfo.run(candles, stress, obi, candles_htf=candles_htf, n_jobs=args.jobs)
    wfo.print_report(results)

    if args.plot or args.save_chart:
        wfo.plot(results, save_path=args.save_chart)
