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

    def prepare_features(self, prices, stress_history):
        """
        Собираем датасет из временного ряда цены и топологического стресса
        """
        df = pd.DataFrame({
            'close': prices,
            'stress': stress_history
        })

        df['returns'] = df['close'].pct_change()

        for i in range(1, self.lookback + 1):
            df[f'return_lag_{i}'] = df['returns'].shift(i)
            df[f'stress_lag_{i}'] = df['stress'].shift(i)

        df['target'] = (df['returns'].shift(-1) > 0).astype(int)

        df.dropna(inplace=True)
        return df

    def train(self, prices, stress_history):
        if len(prices) < 100:
            print("Not enough data to train. Need at least 100 points.")
            return False

        df = self.prepare_features(prices, stress_history)

        X = df.drop(['close', 'target'], axis=1)
        y = df['target']

        self.model.fit(X, y)
        self.is_trained = True
        print(f"Model trained on {len(X)} samples. Ready for inference.")
        return True

    def predict(self, current_prices, current_stress):
        if not self.is_trained:
            return 0.5

        df = self.prepare_features(current_prices, current_stress)
        if len(df) == 0:
            return 0.5

        last_X = df.drop(['close', 'target'], axis=1).iloc[-1:]

        prob_up = self.model.predict_proba(last_X)[0][1]
        return prob_up