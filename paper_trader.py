class PaperTrader:
    def __init__(self, initial_balance=10000.0, horizon=5, sl_pct=0.002, tp_pct=0.004):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position = None
        self.entry_price = 0.0
        self.entry_x = 0
        self.horizon = horizon
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.trade_history = list()

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
            pnl = (current_price - self.entry_price) / self.entry_price
        else:
            pnl = (self.entry_price - current_price) / self.entry_price

        hit_sl = pnl <= -self.sl_pct
        hit_tp = pnl >= self.tp_pct
        hit_time = current_x >= (self.entry_x + self.horizon)

        if hit_sl or hit_tp or hit_time:
            profit_usd = self.balance * pnl
            self.balance += profit_usd

            reason = "TIME"
            if hit_sl:
                reason = "SL"
            elif hit_tp:
                reason = "TP"

            trade_result = {
                'type': self.position,
                'entry': self.entry_price,
                'exit': current_price,
                'pnl_pct': pnl * 100,
                'profit_usd': profit_usd,
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
            return (current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - current_price) / self.entry_price * 100