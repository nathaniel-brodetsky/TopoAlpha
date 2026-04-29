import ccxt
import time

class RobustDataFeeder:
    def __init__(self, symbol='BTC/USDT', timeframe='1m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.last_timestamp = None

    def fetch_initial(self, limit=500):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            if ohlcv:
                self.last_timestamp = ohlcv[-1][0]
            return ohlcv
        except Exception as e:
            print(f"[FEEDER ERROR] Initial fetch failed: {e}. Retrying in 5s...")
            time.sleep(5)
            return self.fetch_initial(limit)

    def fetch_updates(self):
        """
        Умное обновление. Запрашивает данные начиная с последней известной точки.
        Если интернет падал на 10 минут, он скачает ровно 10 пропущенных свечей.
        """
        try:
            if not self.last_timestamp:
                return self.fetch_initial(limit=2)

            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=self.last_timestamp)
            if ohlcv:
                self.last_timestamp = ohlcv[-1][0]
            return ohlcv
        except Exception as e:
            print(f"[NETWORK DROP] Connection lost or rate limit. Holding tight... ({e})")
            return None