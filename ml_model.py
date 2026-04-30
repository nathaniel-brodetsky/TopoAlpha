import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

class TopoBooster:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1
        )
        self.is_trained = False
        self.lookback = 10

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
        df['target'] = (df[f'future_return_{horizon}'] > 0.0005).astype(int)

        df.dropna(inplace=True)
        return df

    def train(self, prices, stress_history, obi_history):
        if len(prices) < 100:
            return False

        df = self.prepare_features(prices, stress_history, obi_history)

        X = df.drop(['close', 'target', f'future_return_5'], axis=1, errors='ignore')
        y = df['target']

        self.model.fit(X, y)
        self.is_trained = True
        return True

    def predict(self, current_prices, current_stress, current_obi):
        if not self.is_trained:
            return 0.5

        df = self.prepare_features(current_prices, current_stress, current_obi)
        if len(df) == 0:
            return 0.5

        last_X = df.drop(['close', 'target', f'future_return_5'], axis=1, errors='ignore').iloc[-1:]
        prob_up = self.model.predict_proba(last_X)[0][1]
        return prob_up