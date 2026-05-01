import ccxt
import time
import logging

logger = logging.getLogger("TopoAlpha.DataFeeder")


class RobustDataFeeder:
    def __init__(self, symbol='BTC/USDT', timeframe='1m', htf='15m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.htf = htf
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.last_timestamp = None
        self.last_timestamp_htf = None

    def fetch_initial(self, limit: int = 5000):
        all_ohlcv = []
        timeframe_mins = self.exchange.parse_timeframe(self.timeframe)
        since = self.exchange.milliseconds() - (limit * timeframe_mins * 60 * 1000)

        while len(all_ohlcv) < limit:
            chunk = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=1000)
            if not chunk:
                break

            all_ohlcv.extend(chunk)
            since = chunk[-1][0] + 1

            import time
            time.sleep(self.exchange.rateLimit / 1000)

        return all_ohlcv[-limit:]

    def fetch_updates(self):
        """Fetch new LTF candles since last known timestamp."""
        try:
            if not self.last_timestamp:
                return self.fetch_initial(limit=2)
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, self.timeframe, since=self.last_timestamp
            )
            if ohlcv:
                self.last_timestamp = ohlcv[-1][0]
            return ohlcv
        except Exception as e:
            logger.warning(f"[DataFeeder] fetch_updates error: {e}")
            return None

    def fetch_initial_htf(self, limit=150, retries=5):
        """Fetch initial HTF (15m) candles for macro-trend context."""
        for attempt in range(retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.htf, limit=limit)
                if ohlcv:
                    self.last_timestamp_htf = ohlcv[-1][0]
                return ohlcv
            except Exception as e:
                logger.warning(f"[DataFeeder] fetch_initial_htf attempt {attempt + 1}/{retries} failed: {e}")
                time.sleep(5)
        logger.error("[DataFeeder] fetch_initial_htf: all retries exhausted.")
        return []

    def fetch_updates_htf(self):
        try:
            if not self.last_timestamp_htf:
                return self.fetch_initial_htf(limit=5)
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, self.htf, since=self.last_timestamp_htf
            )
            if ohlcv:
                self.last_timestamp_htf = ohlcv[-1][0]
            return ohlcv
        except Exception as e:
            logger.warning(f"[DataFeeder] fetch_updates_htf error: {e}")
            return None

    def fetch_order_book_imbalance(self, depth=20):
        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=depth)
            bid_vol = sum(b[1] for b in ob['bids'])
            ask_vol = sum(a[1] for a in ob['asks'])
            if bid_vol + ask_vol == 0:
                return 0.0
            return (bid_vol - ask_vol) / (bid_vol + ask_vol)
        except Exception:
            return 0.0
