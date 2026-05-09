import time
import logging

import ccxt

logger = logging.getLogger("TopoAlpha.DataFeeder")

_EXCHANGE_TIMEOUT_MS = 10_000

class RobustDataFeeder:
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1m", htf: str = "15m"):
        self.symbol    = symbol
        self.timeframe = timeframe
        self.htf       = htf

        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "timeout": _EXCHANGE_TIMEOUT_MS,
            "options": {"defaultType": "future"},
        })
        self.last_timestamp:     int | None = None
        self.last_timestamp_htf: int | None = None

    def fetch_initial(self, limit: int = 600):
        all_ohlcv: list = []
        tf_secs = self.exchange.parse_timeframe(self.timeframe)
        since   = self.exchange.milliseconds() - limit * tf_secs * 1_000

        while len(all_ohlcv) < limit:
            chunk = self.exchange.fetch_ohlcv(
                self.symbol, self.timeframe, since=since, limit=1_000
            )
            if not chunk:
                break
            all_ohlcv.extend(chunk)
            since = chunk[-1][0] + 1
            time.sleep(self.exchange.rateLimit / 1_000)

        data = all_ohlcv[-limit:]
        if data:
            self.last_timestamp = data[-1][0]
        return data

    def fetch_updates(self):
        try:
            if not self.last_timestamp:
                return self.fetch_initial(limit=2)
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, self.timeframe, since=self.last_timestamp
            )
            if ohlcv:
                self.last_timestamp = ohlcv[-1][0]
            return ohlcv
        except Exception as exc:
            logger.warning(f"[DataFeeder] fetch_updates: {exc}")
            return None

    def fetch_initial_htf(self, limit: int = 200, retries: int = 5):
        for attempt in range(retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.htf, limit=limit)
                if ohlcv:
                    self.last_timestamp_htf = ohlcv[-1][0]
                return ohlcv
            except Exception as exc:
                logger.warning(f"[DataFeeder] fetch_initial_htf attempt {attempt+1}: {exc}")
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
        except Exception as exc:
            logger.warning(f"[DataFeeder] fetch_updates_htf: {exc}")
            return None

    def fetch_order_book_imbalance(self, depth: int = 20) -> float:
        try:
            ob      = self.exchange.fetch_order_book(self.symbol, limit=depth)
            bid_vol = sum(b[1] for b in ob["bids"])
            ask_vol = sum(a[1] for a in ob["asks"])
            total   = bid_vol + ask_vol
            return (bid_vol - ask_vol) / total if total else 0.0
        except Exception:
            return 0.0