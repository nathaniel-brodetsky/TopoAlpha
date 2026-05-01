import sqlite3


class PaperTrader:
    def __init__(self, db_path='trading_state.db', initial_balance=10000.0, margin_usdt=50.0, leverage=10, horizon=10,
                 sl_pct=0.004, tp_pct=0.008, fee_pct=0.0004):
        self.margin_usdt = margin_usdt
        self.leverage = leverage
        self.horizon_ms = horizon * 60 * 1000
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.fee_pct = fee_pct

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db(initial_balance)
        self._load_state()

    def _init_db(self, initial_balance):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS state
                          (
                              id
                              INTEGER
                              PRIMARY
                              KEY,
                              balance
                              REAL,
                              pos_type
                              TEXT,
                              entry_price
                              REAL,
                              entry_time
                              REAL
                          )''')
        cursor.execute('SELECT balance FROM state WHERE id = 1')
        if not cursor.fetchone():
            cursor.execute(
                'INSERT INTO state (id, balance, pos_type, entry_price, entry_time) VALUES (1, ?, NULL, 0.0, 0)',
                (initial_balance,))
        self.conn.commit()

    def _load_state(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT balance, pos_type, entry_price, entry_time FROM state WHERE id = 1')
        row = cursor.fetchone()
        self.balance = row[0]
        self.position = row[1]
        self.entry_price = row[2]
        self.entry_time = row[3]

    def _save_state(self):
        cursor = self.conn.cursor()
        cursor.execute('UPDATE state SET balance = ?, pos_type = ?, entry_price = ?, entry_time = ? WHERE id = 1',
                       (self.balance, self.position, self.entry_price, self.entry_time))
        self.conn.commit()

    def execute_trade(self, signal, price, current_time):
        if self.position is not None:
            return False
        self.position = signal
        self.entry_price = price
        self.entry_time = current_time
        self._save_state()
        return True

    def update(self, current_price, current_time):
        if self.position is None:
            return None

        if self.position == 'LONG':
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        hit_sl = pnl_pct <= -self.sl_pct
        hit_tp = pnl_pct >= self.tp_pct
        hit_time = current_time >= (self.entry_time + self.horizon_ms)

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

            self.position = None
            self.entry_price = 0.0
            self.entry_time = 0
            self._save_state()
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
        return gross_profit - total_fees
