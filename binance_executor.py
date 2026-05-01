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

        # Увеличиваем таймаут запросов до 20 секунд для лагучего Testnet
        self.client = Client(api_key, api_secret, requests_params={'timeout': 20})
        self.client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage, recvWindow=60000)
            logger.info(f"[BINANCE API] Connected to Testnet. Leverage: {self.leverage}x")
        except Exception as e:
            logger.error(f"[BINANCE API] Init Error: {e}")

    def execute_trade(self, side, current_price, sl_pct=0.002, tp_pct=0.004):
        try:
            # ЖЕСТКАЯ ОЧИСТКА: Удаляем все старые висящие стопы перед новой сделкой
            self.client.futures_cancel_all_open_orders(symbol=self.symbol, recvWindow=60000)

            order_side = 'BUY' if side == 'LONG' else 'SELL'
            close_side = 'SELL' if side == 'LONG' else 'BUY'

            pos_value_usd = self.margin_usdt * self.leverage
            amount_coins = pos_value_usd / current_price
            amount_coins = math.floor(amount_coins * 1000) / 1000.0

            if amount_coins <= 0.001:
                logger.error("[BINANCE API] Trade size too small.")
                return False

            self.client.futures_create_order(
                symbol=self.symbol,
                side=order_side,
                type='MARKET',
                quantity=amount_coins,
                recvWindow=60000
            )
            logger.info(f"[BINANCE API] MARKET {order_side} FILLED: {amount_coins} {self.symbol}")

            sl_price = round(current_price * (1 - sl_pct) if side == 'LONG' else current_price * (1 + sl_pct), 1)
            tp_price = round(current_price * (1 + tp_pct) if side == 'LONG' else current_price * (1 - tp_pct), 1)

            self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type='STOP_MARKET',
                stopPrice=sl_price,
                closePosition=True,
                recvWindow=60000
            )

            self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=tp_price,
                closePosition=True,
                recvWindow=60000
            )
            logger.info(f"[BINANCE API] SL: {sl_price} | TP: {tp_price}")
            return True

        except Exception as e:
            logger.error(f"[BINANCE API] Execution failed: {e}")
            return False

    def close_all_positions_and_orders(self):
        try:
            self.client.futures_cancel_all_open_orders(symbol=self.symbol, recvWindow=60000)

            positions = self.client.futures_position_information(symbol=self.symbol, recvWindow=60000)
            for pos in positions:
                amt = float(pos['positionAmt'])
                if amt != 0:
                    side = 'SELL' if amt > 0 else 'BUY'
                    self.client.futures_create_order(
                        symbol=self.symbol,
                        side=side,
                        type='MARKET',
                        quantity=abs(amt),
                        recvWindow=60000
                    )
                    logger.info(f"[BINANCE API] Position closed via MARKET {side}: {abs(amt)} {self.symbol}")
        except Exception as e:
            logger.error(f"[BINANCE API] Close position error: {e}")