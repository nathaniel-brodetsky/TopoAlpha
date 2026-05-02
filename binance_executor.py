# import os
# import math
# import time
# import logging
#
# from binance.client import Client
# from dotenv import load_dotenv
#
# load_dotenv()
# logger = logging.getLogger("TopoAlpha.Executor")
#
# _MIN_SL_TP_BUFFER_PCT = 0.0015
#
#
# class BinanceDemoExecutor:
#     def __init__(
#             self,
#             symbol: str = "BTC/USDT",
#             leverage: int = 10,
#             margin_usdt: float = 50.0,
#     ):
#         self.symbol = symbol.replace("/", "")
#         self.leverage = leverage
#         self.margin_usdt = margin_usdt
#         self.client: Client | None = None
#
#         api_key = os.getenv("BINANCE_DEMO_API_KEY")
#         api_secret = os.getenv("BINANCE_DEMO_SECRET_KEY")
#
#         if not api_key or not api_secret:
#             logger.error("[BINANCE API] Keys missing in .env file!")
#             return
#
#         self.client = Client(api_key, api_secret, requests_params={"timeout": 20})
#         self.client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
#
#         try:
#             self.client.futures_change_leverage(
#                 symbol=self.symbol, leverage=self.leverage, recvWindow=60_000
#             )
#             logger.info(f"[BINANCE API] Connected to Testnet. Leverage: {self.leverage}x")
#         except Exception as exc:
#             logger.error(f"[BINANCE API] Init error: {exc}")
#
#     def _get_actual_fill_price(self) -> float | None:
#         """Return the average entry price of the current open position."""
#         try:
#             positions = self.client.futures_position_information(
#                 symbol=self.symbol, recvWindow=60_000
#             )
#             for pos in positions:
#                 if abs(float(pos["positionAmt"])) > 0:
#                     entry = float(pos["entryPrice"])
#                     if entry > 0:
#                         return entry
#         except Exception as exc:
#             logger.warning(f"[BINANCE API] Could not fetch fill price: {exc}")
#         return None
#
#     def _safe_sl_price(self, side: str, fill_price: float, sl_pct: float) -> float:
#         pct = max(sl_pct, _MIN_SL_TP_BUFFER_PCT)
#         return round(fill_price * (1 - pct if side == "LONG" else 1 + pct), 1)
#
#     def _safe_tp_price(self, side: str, fill_price: float, tp_pct: float) -> float:
#         pct = max(tp_pct, _MIN_SL_TP_BUFFER_PCT)
#         return round(fill_price * (1 + pct if side == "LONG" else 1 - pct), 1)
#
#     def execute_trade(
#             self,
#             side: str,
#             current_price: float,
#             sl_pct: float = 0.004,
#             tp_pct: float = 0.008,
#     ) -> bool:
#         if not self.client:
#             logger.error("[BINANCE API] Client not initialised.")
#             return False
#
#         try:
#             self.client.futures_cancel_all_open_orders(symbol=self.symbol, recvWindow=60_000)
#
#             order_side = "BUY" if side == "LONG" else "SELL"
#             close_side = "SELL" if side == "LONG" else "BUY"
#
#             pos_value_usd = self.margin_usdt * self.leverage
#             amount_coins = math.floor((pos_value_usd / current_price) * 1_000) / 1_000.0
#
#             if amount_coins <= 0.001:
#                 logger.error("[BINANCE API] Trade size too small.")
#                 return False
#
#             self.client.futures_create_order(
#                 symbol=self.symbol,
#                 side=order_side,
#                 type="MARKET",
#                 quantity=amount_coins,
#                 recvWindow=60_000,
#             )
#             logger.info(f"[BINANCE API] MARKET {order_side} FILLED: {amount_coins} {self.symbol}")
#
#             time.sleep(0.5)
#             fill_price = self._get_actual_fill_price()
#             if fill_price is None:
#                 logger.warning("[BINANCE API] Fill price unknown; falling back to signal price.")
#                 fill_price = current_price
#
#             sl_price = self._safe_sl_price(side, fill_price, sl_pct)
#             tp_price = self._safe_tp_price(side, fill_price, tp_pct)
#             logger.info(f"[BINANCE API] Fill: {fill_price} | SL: {sl_price} | TP: {tp_price}")
#
#             try:
#                 self.client.futures_create_order(
#                     symbol=self.symbol,
#                     side=close_side,
#                     type="STOP_MARKET",
#                     stopPrice=sl_price,
#                     closePosition=True,
#                     recvWindow=60_000,
#                 )
#             except Exception as exc:
#                 logger.error(f"[BINANCE API] SL order failed: {exc}")
#
#             try:
#                 self.client.futures_create_order(
#                     symbol=self.symbol,
#                     side=close_side,
#                     type="TAKE_PROFIT_MARKET",
#                     stopPrice=tp_price,
#                     closePosition=True,
#                     recvWindow=60_000,
#                 )
#             except Exception as exc:
#                 logger.error(f"[BINANCE API] TP order failed: {exc}")
#
#             return True
#
#         except Exception as exc:
#             logger.error(f"[BINANCE API] Execution failed: {exc}")
#             return False
#
#     def close_all_positions_and_orders(self) -> None:
#         if not self.client:
#             return
#         try:
#             self.client.futures_cancel_all_open_orders(symbol=self.symbol, recvWindow=60_000)
#
#             positions = self.client.futures_position_information(
#                 symbol=self.symbol, recvWindow=60_000
#             )
#             for pos in positions:
#                 amt = float(pos["positionAmt"])
#                 if amt == 0:
#                     continue
#                 close_side = "SELL" if amt > 0 else "BUY"
#                 self.client.futures_create_order(
#                     symbol=self.symbol,
#                     side=close_side,
#                     type="MARKET",
#                     quantity=abs(amt),
#                     recvWindow=60_000,
#                 )
#                 logger.info(
#                     f"[BINANCE API] Position closed via MARKET {close_side}: "
#                     f"{abs(amt)} {self.symbol}"
#                 )
#         except Exception as exc:
#             logger.error(f"[BINANCE API] Close position error: {exc}")
