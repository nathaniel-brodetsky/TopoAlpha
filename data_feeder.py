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
            time.sleep(5)
            return self.fetch_initial(limit)

    def fetch_updates(self):
        try:
            if not self.last_timestamp:
                return self.fetch_initial(limit=2)
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=self.last_timestamp)
            if ohlcv:
                self.last_timestamp = ohlcv[-1][0]
            return ohlcv
        except Exception:
            return None

    def fetch_order_book_imbalance(self, depth=20):
        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=depth)
            bids = ob['bids']
            asks = ob['asks']
            bid_vol = sum([b[1] for b in bids])
            ask_vol = sum([a[1] for a in asks])
            if bid_vol + ask_vol == 0:
                return 0.0
            return (bid_vol - ask_vol) / (bid_vol + ask_vol)
        except Exception:
            return 0.0