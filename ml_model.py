from __future__ import annotations

import logging
import warnings
from collections import deque

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
logger = logging.getLogger("TopoAlpha.MLModel")


def _ema_smooth(old: np.ndarray, new: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    return alpha * new + (1.0 - alpha) * old


def _true_ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def detect_phase_transition(candles: list, lookback: int = 80) -> bool:
    if len(candles) < lookback:
        return False

    closes = np.array([c[4] for c in candles[-lookback:]], dtype=float)
    rets = np.diff(closes) / (closes[:-1] + 1e-10)

    vol_fast = float(np.std(rets[-10:])) + 1e-10
    vol_slow = float(np.std(rets[-40:])) + 1e-10
    vol_ratio = vol_fast / vol_slow
    vol_transition = vol_ratio > 2.2 or vol_ratio < 0.35

    if len(closes) < 24:
        return vol_transition

    ema8 = _true_ema(closes, 8)
    ema21 = _true_ema(closes, 21)

    cross_now = ema8[-1] > ema21[-1]
    cross_prev = ema8[-4] > ema21[-4]
    trend_transition = cross_now != cross_prev

    return vol_transition and trend_transition


class TopoBooster:
    MIN_SIGNAL_PROB: float = 0.40
    SMOOTH_ALPHA: float = 0.40
    BUFFER_SIZE: int = 5
    HORIZON: int = 48
    LOOKBACK: int = 20
    _PREDICT_WINDOW: int = 400
    _NEUTRAL_PROBS = np.array([1 / 3, 1 / 3, 1 / 3])

    TP_MULT: float = 3.5
    SL_MULT: float = 2.0

    _HTF_EMA8_SPAN: int = 32
    _HTF_EMA21_SPAN: int = 84

    def __init__(self) -> None:
        self.is_trained: bool = False
        self._feature_cols: list | None = None
        self._scaler = StandardScaler()
        self._models: list = []
        self._build_ensemble()
        self._prob_buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._smooth_probs = self._NEUTRAL_PROBS.copy()
        self.feature_importance: dict = {}

    def _build_ensemble(self) -> None:
        gbm = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            subsample=0.75, min_samples_leaf=15, random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=7, min_samples_leaf=15,
            max_features="sqrt", class_weight="balanced",
            random_state=7, n_jobs=-1,
        )
        et = ExtraTreesClassifier(
            n_estimators=300, max_depth=7, min_samples_leaf=15,
            class_weight="balanced", random_state=13, n_jobs=-1,
        )
        self._models = [gbm, rf, et]

    def _build_features(
            self,
            candles: list,
            stress_history: list,
            obi_history: list,
            candles_htf: list | None = None,
    ) -> pd.DataFrame:
        n = len(candles)

        opens = [c[1] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]
        volumes = [c[5] for c in candles]

        stress = list(stress_history)[:n]
        obi = list(obi_history)[:n]
        while len(stress) < n: stress.append(0.0)
        while len(obi) < n:    obi.append(0.0)

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
            "stress": stress, "obi": obi,
        })

        df["body"] = (df["close"] - df["open"]) / (df["open"].abs() + 1e-8)
        body_top = df[["open", "close"]].max(axis=1)
        body_bot = df[["open", "close"]].min(axis=1)
        df["upper_wick"] = (df["high"] - body_top) / (df["open"].abs() + 1e-8)
        df["lower_wick"] = (body_bot - df["low"]) / (df["open"].abs() + 1e-8)
        df["hl_range"] = (df["high"] - df["low"]) / (df["open"].abs() + 1e-8)
        df["body_ratio"] = df["body"].abs() / (df["hl_range"] + 1e-8)

        df["body_sign"] = np.sign(df["body"])
        df["run_3"] = df["body_sign"].rolling(3).sum() / 3.0
        df["run_5"] = df["body_sign"].rolling(5).sum() / 5.0

        def _entropy8(s):
            p = (s > 0).sum() / max(len(s), 1)
            if p in (0.0, 1.0):
                return 0.0
            return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

        df["body_entropy8"] = df["body_sign"].rolling(8).apply(_entropy8, raw=True)

        vol_ma = df["volume"].rolling(20).mean().replace(0, np.nan)
        df["vol_ratio"] = df["volume"] / (vol_ma + 1e-8)
        df["vol_spike"] = (df["vol_ratio"] > 1.5).astype(float)
        df["vol_body"] = df["vol_ratio"] * df["body"].abs()

        df["returns"] = df["close"].pct_change()
        df["vol_5"] = df["returns"].rolling(5).std()
        df["vol_10"] = df["returns"].rolling(10).std()
        df["vol_30"] = df["returns"].rolling(30).std()
        df["vol_ratio_5_30"] = df["vol_5"] / (df["vol_30"] + 1e-8)
        df["vol_rank"] = df["vol_10"].rolling(50).rank(pct=True)

        df["atr"] = df["hl_range"].rolling(14).mean()
        df["atr_pct"] = df["returns"].abs().rolling(14).mean()

        df["mom_5"] = df["close"] / df["close"].shift(5) - 1
        df["mom_10"] = df["close"] / df["close"].shift(10) - 1
        df["mom_20"] = df["close"] / df["close"].shift(20) - 1

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-8))
        df["rsi_norm"] = (df["rsi_14"] - 50.0) / 50.0

        bb_mean = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_pos"] = (df["close"] - bb_mean) / (bb_std * 2.0 + 1e-8)

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = (ema12 - ema26) / (df["close"].abs() + 1e-8)
        df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_sig"]
        df["macd_cross"] = (np.sign(df["macd_hist"]) - np.sign(df["macd_hist"].shift(1)))

        low14 = df["low"].rolling(14).min()
        high14 = df["high"].rolling(14).max()
        df["stoch_k"] = (df["close"] - low14) / (high14 - low14 + 1e-8)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        df["stoch_norm"] = df["stoch_k"] - 0.5
        df["stoch_divergence"] = df["stoch_k"] - df["stoch_d"]

        cum_pv = (df["close"] * df["volume"]).rolling(20).sum()
        cum_v = df["volume"].rolling(20).sum()
        vwap = cum_pv / (cum_v + 1e-8)
        df["vwap_dev"] = (df["close"] - vwap) / (vwap + 1e-8)

        df["stress_delta"] = df["stress"].diff()
        df["stress_accel"] = df["stress_delta"].diff()
        df["stress_ma5"] = df["stress"].rolling(5).mean()
        df["stress_zscore"] = (
                (df["stress"] - df["stress"].rolling(30).mean())
                / (df["stress"].rolling(30).std() + 1e-8)
        )

        df["obi_ma5"] = df["obi"].rolling(5).mean()
        df["obi_delta"] = df["obi"].diff()
        df["obi_cumsum5"] = df["obi"].rolling(5).sum()

        for i in range(1, self.LOOKBACK + 1):
            df[f"ret_lag_{i}"] = df["returns"].shift(i)
            df[f"stress_lag_{i}"] = df["stress"].shift(i)
            df[f"obi_lag_{i}"] = df["obi"].shift(i)
            df[f"body_lag_{i}"] = df["body"].shift(i)

        df["htf_ema8"] = df["close"].ewm(span=120, adjust=False).mean()
        df["htf_ema21"] = df["close"].ewm(span=315, adjust=False).mean()
        df["htf_trend"] = (df["htf_ema8"] - df["htf_ema21"]) / (df["htf_ema21"] + 1e-8)
        df["htf_mom5"] = df["close"] / df["close"].shift(90) - 1
        df["htf_regime"] = np.sign(df["htf_trend"])

        if candles_htf and len(candles_htf) >= 22:
            htf_ts = np.array([c[0] for c in candles_htf], dtype=np.int64)
            htf_c = np.array([c[4] for c in candles_htf], dtype=float)
            htf_h = np.array([c[2] for c in candles_htf], dtype=float)
            htf_l = np.array([c[3] for c in candles_htf], dtype=float)

            htf_e8 = pd.Series(htf_c).ewm(span=8, adjust=False).mean().values
            htf_e21 = pd.Series(htf_c).ewm(span=21, adjust=False).mean().values
            htf_trend_r = (htf_e8 - htf_e21) / (np.abs(htf_e21) + 1e-10)

            htf_s = pd.Series(htf_c)
            htf_d = htf_s.diff()
            htf_g = htf_d.clip(lower=0).rolling(14, min_periods=1).mean()
            htf_ls = (-htf_d.clip(upper=0)).rolling(14, min_periods=1).mean()
            htf_rsi_n = ((100 - 100 / (1 + htf_g / (htf_ls + 1e-8))) - 50.0) / 50.0
            htf_rsi_n = htf_rsi_n.values

            htf_mom_r = pd.Series(htf_c).pct_change(3).fillna(0.0).values

            htf_hl = (htf_h - htf_l) / (htf_c + 1e-10)
            htf_atr_r = pd.Series(htf_hl).rolling(14, min_periods=1).mean().values

            ltf_ts = np.array([c[0] for c in candles], dtype=np.int64)
            idx = np.searchsorted(htf_ts, ltf_ts, side="right") - 1
            idx = np.clip(idx, 0, len(htf_ts) - 1)

            df["htf_trend_real"] = htf_trend_r[idx]
            df["htf_rsi_norm"] = np.nan_to_num(htf_rsi_n[idx])
            df["htf_mom_real"] = np.nan_to_num(htf_mom_r[idx])
            df["htf_atr_real"] = np.nan_to_num(htf_atr_r[idx], nan=0.003)
            df["htf_regime_real"] = np.sign(df["htf_trend_real"])
        else:
            df["htf_trend_real"] = df["htf_trend"]
            df["htf_rsi_norm"] = 0.0
            df["htf_mom_real"] = df["htf_mom5"]
            df["htf_atr_real"] = 0.003
            df["htf_regime_real"] = df["htf_regime"]

        if "htf_atr_real" in df.columns:
            htf_atr_med = df["htf_atr_real"].rolling(30, min_periods=5).median()
            df["htf_vol_regime"] = (df["htf_atr_real"] > htf_atr_med).astype(float)
        else:
            df["htf_vol_regime"] = 1.0

        return df

    def _make_xy(self, candles, stress_history, obi_history, candles_htf=None):
        df = self._build_features(candles, stress_history, obi_history, candles_htf)
        n = len(df)

        atr = df["atr_pct"].values
        threshold_tp = np.clip(atr * self.TP_MULT, 0.003, 0.015)
        threshold_sl = np.clip(atr * self.SL_MULT, 0.0015, 0.0075)

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        labels = np.zeros(n, dtype=int)
        for i in range(n - self.HORIZON - 1):
            entry = closes[i]
            tp_dist = threshold_tp[i]
            sl_dist = threshold_sl[i]

            long_tp = entry * (1 + tp_dist)
            long_sl = entry * (1 - sl_dist)
            short_tp = entry * (1 - tp_dist)
            short_sl = entry * (1 + sl_dist)

            long_status = 0
            short_status = 0

            for j in range(i + 1, i + self.HORIZON + 1):
                h = highs[j]
                l = lows[j]

                if long_status == 0:
                    if l <= long_sl:
                        long_status = -1
                    elif h >= long_tp:
                        long_status = 1

                if short_status == 0:
                    if h >= short_sl:
                        short_status = -1
                    elif l <= short_tp:
                        short_status = 1

                if long_status != 0 and short_status != 0:
                    break

            if long_status == 1 and short_status != 1:
                labels[i] = 1
            elif short_status == 1 and long_status != 1:
                labels[i] = 2
            else:
                labels[i] = 0

        df["target"] = labels
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        drop_cols = ["open", "high", "low", "close", "volume", "target",
                     "htf_ema8", "htf_ema21", "atr"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return X, df["target"]

    def _update_importance(self, feature_names: list) -> None:
        importances: dict[str, list] = {}
        for base in self._models:
            try:
                fi = base.feature_importances_
            except Exception:
                continue
            if len(fi) != len(feature_names):
                continue
            for name, val in zip(feature_names, fi):
                importances.setdefault(name, []).append(float(val))

        if importances:
            self.feature_importance = {
                k: round(float(np.mean(v)), 6)
                for k, v in sorted(importances.items(), key=lambda x: -np.mean(x[1]))
            }
            top5 = list(self.feature_importance.items())[:5]
            logger.info(f"[TopoBooster] Top-5 features: {top5}")

    def train(
            self,
            candles,
            stress_history,
            obi_history,
            candles_htf=None,
    ) -> bool:
        if len(candles) < 250:
            return False

        X, y = self._make_xy(candles, stress_history, obi_history, candles_htf)
        if len(X) < 60 or y.nunique() < 2:
            return False

        self._feature_cols = list(X.columns)
        X_scaled = self._scaler.fit_transform(X)

        n_samples = len(X)
        recency = np.exp(np.linspace(-1.0, 0.0, n_samples))
        sample_w = compute_sample_weight("balanced", y) * recency
        first_fit = not self.is_trained

        for model in self._models:
            try:
                model.fit(X_scaled, y, sample_weight=sample_w)
            except TypeError:
                model.fit(X_scaled, y)

        self.is_trained = True
        self._update_importance(self._feature_cols)

        if first_fit:
            self._prob_buffer.clear()
            self._smooth_probs = self._NEUTRAL_PROBS.copy()

        logger.info(
            f"[TopoBooster] Trained on {len(X)} rows | "
            f"label dist: flat={int((y == 0).sum())} "
            f"up={int((y == 1).sum())} dn={int((y == 2).sum())}"
        )
        return True

    def predict(
            self,
            candles,
            current_stress,
            current_obi,
            candles_htf=None,
    ) -> dict:
        _flat = {"flat": 1.0, "up": 0.0, "down": 0.0}
        if not self.is_trained or self._feature_cols is None:
            return _flat

        tail = slice(-self._PREDICT_WINDOW, None)
        candles_t = candles[tail] if len(candles) > self._PREDICT_WINDOW else candles

        df = self._build_features(
            candles_t,
            current_stress[tail] if hasattr(current_stress, "__getitem__") else current_stress,
            current_obi[tail] if hasattr(current_obi, "__getitem__") else current_obi,
            candles_htf,
        )
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        if len(df) == 0:
            return _flat

        _HTF_DIR = []
        drop_cols = ["open", "high", "low", "close", "volume",
                     "htf_ema8", "htf_ema21", "atr"] + _HTF_DIR
        last_X = df.drop(columns=[c for c in drop_cols if c in df.columns]).iloc[-1:]

        for col in self._feature_cols:
            if col not in last_X.columns:
                last_X[col] = 0.0
        last_X = last_X[self._feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        last_X_scaled = self._scaler.transform(last_X)

        if not np.all(np.isfinite(last_X_scaled)):
            last_X_scaled = np.nan_to_num(last_X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        all_probas: list = []
        classes_ref = None
        for model in self._models:
            try:
                p = model.predict_proba(last_X_scaled)[0]
                all_probas.append(p)
                if classes_ref is None:
                    classes_ref = list(model.classes_)
            except Exception:
                continue

        if not all_probas:
            return _flat

        raw_proba = np.mean(all_probas, axis=0)
        if not np.all(np.isfinite(raw_proba)):
            return {
                "flat": float(self._smooth_probs[0]),
                "up": float(self._smooth_probs[1]),
                "down": float(self._smooth_probs[2]),
            }

        p_vec = np.zeros(3)
        for i, cls in enumerate(classes_ref):
            if int(cls) < 3:
                p_vec[int(cls)] = raw_proba[i]
        total = p_vec.sum()
        p_vec = p_vec / total if total > 0 else self._NEUTRAL_PROBS.copy()

        if not np.all(np.isfinite(p_vec)):
            p_vec = self._NEUTRAL_PROBS.copy()

        self._prob_buffer.append(p_vec.copy())
        buf_mean = np.mean(list(self._prob_buffer), axis=0)
        new_smooth = _ema_smooth(self._smooth_probs, buf_mean, self.SMOOTH_ALPHA)

        if not np.all(np.isfinite(new_smooth)):
            self._smooth_probs = self._NEUTRAL_PROBS.copy()
            self._prob_buffer.clear()
        else:
            self._smooth_probs = new_smooth

        p_flat, p_up, p_down = self._smooth_probs
        return {"flat": float(p_flat), "up": float(p_up), "down": float(p_down)}
