import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class TopoBooster:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1
        )
        self.is_trained = False
        self.lookback = 10
        self.threshold = 0.0005

    def prepare_features(self, prices, stress_history, obi_history):
        df = pd.DataFrame({
            'close': prices,
            'stress': stress_history,
            'obi': obi_history
        })

        df['returns'] = df['close'].pct_change()

        for i in range(1, self.lookback + 1):
            df[f'return_lag_{i}'] = df['returns'].shift(i)
            df[f'stress_lag_{i}'] = df['stress'].shift(i)
            df[f'obi_lag_{i}'] = df['obi'].shift(i)

        horizon = 5

        df[f'future_return_{horizon}'] = df['close'].shift(-horizon) / df['close'] - 1

        conditions = [
            df[f'future_return_{horizon}'] >= self.threshold,
            df[f'future_return_{horizon}'] <= -self.threshold
        ]
        choices = [1, 2]
        df['target'] = np.select(conditions, choices, default=0)

        df.dropna(inplace=True)
        return df

    def train(self, prices, stress_history, obi_history):
        if len(prices) < 100:
            return False

        df = self.prepare_features(prices, stress_history, obi_history)

        if df['target'].nunique() < 2:
            return False

        X = df.drop(['close', 'target', f'future_return_5'], axis=1, errors='ignore')
        y = df['target']

        self.model.fit(X, y)
        self.is_trained = True
        return True

    def predict(self, current_prices, current_stress, current_obi):
        if not self.is_trained:
            return {'flat': 1.0, 'up': 0.0, 'down': 0.0}

        df = self.prepare_features(current_prices, current_stress, current_obi)
        if len(df) == 0:
            return {'flat': 1.0, 'up': 0.0, 'down': 0.0}

        last_X = df.drop(['close', 'target', f'future_return_5'], axis=1, errors='ignore').iloc[-1:]

        probas = self.model.predict_proba(last_X)[0]
        classes = list(self.model.classes_)

        p_flat = probas[classes.index(0)] if 0 in classes else 0.0
        p_long = probas[classes.index(1)] if 1 in classes else 0.0
        p_short = probas[classes.index(2)] if 2 in classes else 0.0

        return {'flat': float(p_flat), 'up': float(p_long), 'down': float(p_short)}