class PaperTrader:
    def __init__(self, initial_balance=10000.0, margin_usdt=50.0, leverage=10, horizon=5, sl_pct=0.002, tp_pct=0.004,
                 fee_pct=0.0004):
        self.balance = initial_balance
        self.margin_usdt = margin_usdt
        self.leverage = leverage
        self.horizon = horizon
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.fee_pct = fee_pct

        self.position = None
        self.entry_price = 0.0
        self.entry_x = 0
        self.trade_history = []

    def execute_trade(self, signal, price, current_x):
        if self.position is not None:
            return False
        self.position = signal
        self.entry_price = price
        self.entry_x = current_x
        return True

    def update(self, current_price, current_x):
        if self.position is None:
            return None

        if self.position == 'LONG':
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        hit_sl = pnl_pct <= -self.sl_pct
        hit_tp = pnl_pct >= self.tp_pct
        hit_time = current_x >= (self.entry_x + self.horizon)

        if hit_sl or hit_tp or hit_time:
            pos_size_usd = self.margin_usdt * self.leverage
            gross_profit = pos_size_usd * pnl_pct
            total_fees = pos_size_usd * self.fee_pct * 2
            net_profit = gross_profit - total_fees

            self.balance += net_profit

            reason = "TIME"
            if hit_sl:
                reason = "SL"
            elif hit_tp:
                reason = "TP"

            trade_result = {
                'type': self.position,
                'entry': self.entry_price,
                'exit': current_price,
                'net_profit_usd': net_profit,
                'reason': reason
            }
            self.trade_history.append(trade_result)

            self.position = None
            self.entry_price = 0.0
            self.entry_x = 0
            return trade_result

        return None

    def get_unrealized_pnl(self, current_price):
        if self.position is None:
            return 0.0

        if self.position == 'LONG':
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        pos_size_usd = self.margin_usdt * self.leverage
        gross_profit = pos_size_usd * pnl_pct
        total_fees = pos_size_usd * self.fee_pct * 2
        net_profit = gross_profit - total_fees
        return net_profit