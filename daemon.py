import time
import logging
import numpy as np

from data_feeder import RobustDataFeeder
from tda_core import TDAAnalyzer
from ml_model import TopoBooster
from paper_trader import PaperTrader
from binance_executor import BinanceDemoExecutor
from notifier import TelegramNotifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s[%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("TopoAlpha.Daemon")


class TopoAlphaDaemon:
    def __init__(self, symbol='BTC/USDT', timeframe='1m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.tau = 5
        self.dim = 3
        self.alpha_stress_threshold = 1.5

        self.feeder = RobustDataFeeder(symbol, timeframe)
        self.tda = TDAAnalyzer()
        self.ml = TopoBooster()
        self.trader = PaperTrader(initial_balance=10000.0, margin_usdt=50.0, leverage=10, horizon=5)
        self.api_executor = BinanceDemoExecutor(symbol=self.symbol, leverage=10, margin_usdt=50.0)
        self.notifier = TelegramNotifier()

        self.timestamps = []
        self.prices = []
        self.stress_history = []
        self.obi_history = []

        self.last_processed_time = 0
        self.tick_counter = 0

    def preload_data(self):
        logger.info("DAEMON INIT: Preloading historical data...")
        ohlcv = self.feeder.fetch_initial(limit=500)
        if not ohlcv:
            logger.error("Failed to load initial data. Exiting.")
            return False

        for candle in ohlcv:
            self.timestamps.append(candle[0])
            self.prices.append(candle[4])
            self.stress_history.append(0.0)
            self.obi_history.append(0.0)

        self.last_processed_time = self.timestamps[-1]

        logger.info("DAEMON INIT: Backfilling topological stress...")
        for i in range(50, len(self.prices)):
            window = self.prices[i - 50:i]
            data = np.array(window)
            data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
            embedded = np.vstack([data[:-(2 * self.tau)], data[self.tau:-self.tau], data[2 * self.tau:]]).T
            if len(embedded) > 0:
                self.stress_history[i] = self.tda.get_topological_stress(embedded)

        logger.info("DAEMON INIT: Training initial ML model...")
        self.ml.train(self.prices, self.stress_history, self.obi_history)

        if self.trader.position:
            logger.info(f"DAEMON INIT: Restored active {self.trader.position} position from SQLite database.")

        self.notifier.send_message(
            f"🤖 <b>TopoAlpha Daemon Started</b>\nSymbol: {self.symbol}\nBalance: ${self.trader.balance:.2f}")
        return True

    def get_embedding(self):
        n = len(self.prices)
        if n < (self.dim - 1) * self.tau + 1:
            return np.zeros((1, 3))
        data = np.array(self.prices)
        data = (data - np.mean(data)) / (np.std(data) + 1e-8) * 10
        return np.vstack([data[:-(2 * self.tau)], data[self.tau:-self.tau], data[2 * self.tau:]]).T

    def run(self):
        if not self.preload_data():
            return

        logger.info("DAEMON LOOP: Running in headless mode. Press Ctrl+C to stop.")

        while True:
            try:
                ohlcv = self.feeder.fetch_updates()
                obi = self.feeder.fetch_order_book_imbalance(depth=20)

                if ohlcv:
                    for candle in ohlcv:
                        t, o, h, l, c, v = candle
                        if len(self.timestamps) == 0 or t > self.timestamps[-1]:
                            self.timestamps.append(t)
                            self.prices.append(c)
                            self.stress_history.append(self.stress_history[-1] if self.stress_history else 0.0)
                            self.obi_history.append(obi)
                        elif t == self.timestamps[-1]:
                            self.prices[-1] = c
                            self.obi_history[-1] = obi

                    if len(self.prices) > 500:
                        self.timestamps = self.timestamps[-500:]
                        self.prices = self.prices[-500:]
                        self.stress_history = self.stress_history[-500:]
                        self.obi_history = self.obi_history[-500:]

                    embedded_data = self.get_embedding()
                    current_stress = 0.0
                    probs = {'flat': 1.0, 'up': 0.0, 'down': 0.0}

                    if len(embedded_data) > 0:
                        current_stress = self.tda.get_topological_stress(embedded_data[-50:])
                        self.stress_history[-1] = current_stress

                        if self.ml.is_trained:
                            probs = self.ml.predict(self.prices, self.stress_history, self.obi_history)

                    curr_p = self.prices[-1]
                    curr_ms = self.timestamps[-1]

                    if curr_ms > self.last_processed_time:
                        self.last_processed_time = curr_ms

                    res = self.trader.update(curr_p, curr_ms)
                    if res:
                        logger.info(
                            f"TRADE CLOSED ({res['reason']}): Net Profit: ${res['net_profit_usd']:.2f} | Bal: ${self.trader.balance:.2f}")
                        self.api_executor.close_all_positions_and_orders()
                        icon = "✅" if res['net_profit_usd'] > 0 else "🛑"
                        msg = f"{icon} <b>TRADE CLOSED ({res['reason']})</b>\nNet Profit: ${res['net_profit_usd']:.2f}\nBalance: ${self.trader.balance:.2f}"
                        self.notifier.send_message(msg)

                    if current_stress >= self.alpha_stress_threshold and not self.trader.position:
                        if probs['up'] >= 0.60:
                            if self.trader.execute_trade('LONG', curr_p, curr_ms):
                                self.api_executor.execute_trade('LONG', curr_p, self.trader.sl_pct, self.trader.tp_pct)
                                logger.info(
                                    f"EXECUTING LONG at {curr_p} (Stress: {current_stress:.2f}, Prob UP: {probs['up']:.2f})")
                                msg = f"🚀 <b>EXECUTING LONG</b>\nPrice: {curr_p}\nStress: {current_stress:.2f}\nProb UP: {probs['up']:.1%}"
                                self.notifier.send_message(msg)

                        elif probs['down'] >= 0.60:
                            if self.trader.execute_trade('SHORT', curr_p, curr_ms):
                                self.api_executor.execute_trade('SHORT', curr_p, self.trader.sl_pct, self.trader.tp_pct)
                                logger.info(
                                    f"EXECUTING SHORT at {curr_p} (Stress: {current_stress:.2f}, Prob DOWN: {probs['down']:.2f})")
                                msg = f"🩸 <b>EXECUTING SHORT</b>\nPrice: {curr_p}\nStress: {current_stress:.2f}\nProb DOWN: {probs['down']:.1%}"
                                self.notifier.send_message(msg)

                self.tick_counter += 1
                if self.tick_counter % 150 == 0 and len(self.prices) > 100:
                    logger.info("Dynamic Retraining ML with recent Order Book Imbalance data...")
                    self.ml.train(self.prices, self.stress_history, self.obi_history)

            except Exception as e:
                logger.error(f"DAEMON ERROR: {e}")

            time.sleep(2)


if __name__ == '__main__':
    daemon = TopoAlphaDaemon()
    daemon.run()