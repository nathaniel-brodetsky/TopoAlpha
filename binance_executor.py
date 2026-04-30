import os
import math
import logging
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()
logger = logging.getLogger("TopoAlpha.Executor")


class BinanceDemoExecutor:
    def __init__(self, symbol='BTC/USDT', leverage=10, margin_usdt=50.0):
        self.symbol = symbol.replace('/', '')
        self.leverage = leverage
        self.margin_usdt = margin_usdt

        api_key = os.getenv("BINANCE_DEMO_API_KEY")
        api_secret = os.getenv("BINANCE_DEMO_SECRET_KEY")

        if not api_key or not api_secret:
            logger.error("[BINANCE API] Keys missing in .env file!")
            return

        self.client = Client(api_key, api_secret)
        self.client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            logger.info(f"[BINANCE API] Connected to Testnet. Leverage: {self.leverage}x")
        except Exception as e:
            logger.error(f"[BINANCE API] Init Error: {e}")

    def execute_trade(self, side, current_price, sl_pct=0.002, tp_pct=0.004):
        try:
            order_side = 'BUY' if side == 'LONG' else 'SELL'
            close_side = 'SELL' if side == 'LONG' else 'BUY'

            pos_value_usd = self.margin_usdt * self.leverage
            amount_coins = pos_value_usd / current_price
            amount_coins = math.floor(amount_coins * 1000) / 1000.0

            if amount_coins <= 0.001:
                logger.error("[BINANCE API] Trade size too small.")
                return False

            logger.info(f"[BINANCE API] Sending MARKET {order_side} order: {amount_coins} {self.symbol}")

            self.client.futures_create_order(
                symbol=self.symbol,
                side=order_side,
                type='MARKET',
                quantity=amount_coins
            )

            if side == 'LONG':
                sl_price = current_price * (1 - sl_pct)
                tp_price = current_price * (1 + tp_pct)
            else:
                sl_price = current_price * (1 + sl_pct)
                tp_price = current_price * (1 - tp_pct)

            sl_price = round(sl_price, 1)
            tp_price = round(tp_price, 1)

            self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type='STOP_MARKET',
                stopPrice=sl_price,
                closePosition=True
            )

            self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=tp_price,
                closePosition=True
            )

            logger.info(f"[BINANCE API] SL set at {sl_price} | TP set at {tp_price}")
            return True

        except Exception as e:
            logger.error(f"[BINANCE API] Execution failed: {e}")
            return False