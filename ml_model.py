import numpy as np
import pandas as pd
import warnings
from collections import deque

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

def _ema_smooth(old: np.ndarray, new: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    return alpha * new + (1.0 - alpha) * old

class TopoBooster:
    MIN_SIGNAL_PROB: float = 0.42
    MIN_GAP: float = 0.05
    SMOOTH_ALPHA: float = 0.25
    BUFFER_SIZE: int = 5
    HORIZON: int = 6  
    LOOKBACK: int = 20

    def __init__(self):
        self.is_trained: bool = False
        self._feature_cols = None
        self._scaler = StandardScaler()
        self._models =[]
        self._build_ensemble()
        self._prob_buffer = deque(maxlen=self.BUFFER_SIZE)
        self._smooth_probs = np.array([1.0, 0.0, 0.0])

    def _build_ensemble(self):
        gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.04, max_depth=4, subsample=0.8, min_samples_leaf=20, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20, max_features="sqrt", class_weight="balanced", random_state=7, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=13, n_jobs=-1)
        self._models =[
            CalibratedClassifierCV(gbm, method="isotonic", cv=3),
            CalibratedClassifierCV(rf, method="isotonic", cv=3),
            CalibratedClassifierCV(et, method="isotonic", cv=3),
        ]

    def _build_features(self, prices: list, stress_history: list, obi_history: list, prices_htf=None) -> pd.DataFrame:
        df = pd.DataFrame({"close": prices, "stress": stress_history, "obi": obi_history})
        df["returns"] = df["close"].pct_change()
        df["vol_5"] = df["returns"].rolling(5).std()
        df["vol_10"] = df["returns"].rolling(10).std()
        df["vol_30"] = df["returns"].rolling(30).std()
        df["vol_ratio"] = df["vol_5"] / (df["vol_30"] + 1e-10)
        df["vol_rank"] = df["vol_10"].rolling(50).rank(pct=True)
        df["mom_5"] = df["close"] / df["close"].shift(5) - 1
        df["mom_20"] = df["close"] / df["close"].shift(20) - 1
        
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        df["stress_delta"] = df["stress"].diff()
        df["stress_accel"] = df["stress_delta"].diff()
        df["stress_ma5"] = df["stress"].rolling(5).mean()
        df["stress_zscore"] = ((df["stress"] - df["stress"].rolling(30).mean()) / (df["stress"].rolling(30).std() + 1e-10))

        df["obi_ma5"] = df["obi"].rolling(5).mean()
        df["obi_delta"] = df["obi"].diff()
        df["obi_cumsum5"] = df["obi"].rolling(5).sum()

        for i in range(1, self.LOOKBACK + 1):
            df[f"ret_lag_{i}"] = df["returns"].shift(i)
            df[f"stress_lag_{i}"] = df["stress"].shift(i)
            df[f"obi_lag_{i}"] = df["obi"].shift(i)

        df["htf_ema8"] = df["close"].ewm(span=96, adjust=False).mean()
        df["htf_ema21"] = df["close"].ewm(span=252, adjust=False).mean()
        df["htf_trend"] = (df["htf_ema8"] - df["htf_ema21"]) / (df["htf_ema21"] + 1e-10)
        df["htf_mom5"] = df["close"] / df["close"].shift(60) - 1
        df["htf_vol"] = df["returns"].rolling(240).std()

        return df

    def _make_xy(self, prices: list, stress_history: list, obi_history: list, prices_htf=None):
        df = self._build_features(prices, stress_history, obi_history, prices_htf)
        df["atr_pct"] = df["returns"].abs().rolling(14).mean()
        df["threshold"] = (df["atr_pct"] * 1.5).clip(0.0015, 0.005)
        df["future_ret"] = df["close"].shift(-self.HORIZON) / df["close"] - 1
        conditions = [df["future_ret"] >= df["threshold"], df["future_ret"] <= -df["threshold"]]
        df["target"] = np.select(conditions, [1, 2], default=0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        drop_cols =["close", "target", "future_ret", "atr_pct", "threshold", "htf_ema8", "htf_ema21"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df["target"]
        return X, y

    def train(self, prices: list, stress_history: list, obi_history: list, prices_htf=None) -> bool:
        if len(prices) < 350: return False
        X, y = self._make_xy(prices, stress_history, obi_history, prices_htf)
        if len(X) < 60 or y.nunique() < 2: return False
        self._feature_cols = list(X.columns)
        X_scaled = self._scaler.fit_transform(X)
        sample_w = compute_sample_weight("balanced", y)
        for model in self._models:
            try: model.fit(X_scaled, y, sample_weight=sample_w)
            except TypeError: model.fit(X_scaled, y)
        self.is_trained = True
        self._prob_buffer.clear()
        self._smooth_probs = np.array([1.0, 0.0, 0.0])
        return True

    def predict(self, current_prices: list, current_stress: list, current_obi: list, prices_htf=None) -> dict:
        _flat = {"flat": 1.0, "up": 0.0, "down": 0.0}
        if not self.is_trained or self._feature_cols is None: return _flat
        df = self._build_features(current_prices, current_stress, current_obi, prices_htf)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        if len(df) == 0: return _flat
        last_X = df.drop(columns=["close"]).iloc[-1:]
        for col in self._feature_cols:
            if col not in last_X.columns: last_X[col] = 0.0
        last_X = last_X[self._feature_cols]
        last_X_scaled = self._scaler.transform(last_X)
        all_probas =[]
        classes_ref = None
        for model in self._models:
            try:
                p = model.predict_proba(last_X_scaled)[0]
                all_probas.append(p)
                if classes_ref is None: classes_ref = list(model.classes_)
            except Exception: continue
        if not all_probas: return _flat
        raw_proba = np.mean(all_probas, axis=0)
        p_vec = np.zeros(3)
        for i, cls in enumerate(classes_ref):
            if cls < 3: p_vec[cls] = raw_proba[i]
        p_vec /= p_vec.sum() + 1e-10
        self._prob_buffer.append(p_vec.copy())
        buf_mean = np.mean(list(self._prob_buffer), axis=0)
        self._smooth_probs = _ema_smooth(self._smooth_probs, buf_mean, self.SMOOTH_ALPHA)
        p_flat, p_up, p_down = self._smooth_probs
        sorted_p = np.sort(self._smooth_probs)[::-1]
        if sorted_p[0] < self.MIN_SIGNAL_PROB or (sorted_p[0] - sorted_p[1]) < self.MIN_GAP:
            return {"flat": float(p_flat + p_up + p_down), "up": 0.0, "down": 0.0}
        return {"flat": float(p_flat), "up": float(p_up), "down": float(p_down)}