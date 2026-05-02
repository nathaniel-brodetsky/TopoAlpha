import sqlite3


class PaperTrader:
    """Simulated futures trader backed by a local SQLite database.

    Survives process restarts: position, balance and entry metadata are
    persisted on every state change and restored on construction.
    """

    def __init__(
            self,
            db_path: str = "trading_state.db",
            initial_balance: float = 10_000.0,
            margin_usdt: float = 50.0,
            leverage: int = 10,
            horizon_bars: int = 6,
            timeframe_mins: int = 5,
            sl_pct: float = 0.005,
            tp_pct: float = 0.010,
            fee_pct: float = 0.0004,
    ):
        self.margin_usdt = margin_usdt
        self.leverage = leverage
        self.horizon_ms = horizon_bars * timeframe_mins * 60 * 1_000
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.fee_pct = fee_pct

        self._db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db(initial_balance)
        self._load_state()

    # ── Resource management ─────────────────────────────────────────────── #

    def close(self) -> None:
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            finally:
                self.conn = None  # type: ignore[assignment]

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "PaperTrader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Persistence ─────────────────────────────────────────────────────── #

    def _init_db(self, initial_balance: float) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS state
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
            )
            """
        )
        cur.execute("SELECT balance FROM state WHERE id = 1")
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO state (id, balance, pos_type, entry_price, entry_time) "
                "VALUES (1, ?, NULL, 0.0, 0)",
                (initial_balance,),
            )
        self.conn.commit()

    def _load_state(self) -> None:
        row = self.conn.cursor().execute(
            "SELECT balance, pos_type, entry_price, entry_time FROM state WHERE id = 1"
        ).fetchone()
        self.balance: float = row[0]
        self.position: str | None = row[1]
        self.entry_price: float = row[2]
        self.entry_time: float = row[3]

    def _save_state(self) -> None:
        self.conn.cursor().execute(
            "UPDATE state SET balance=?, pos_type=?, entry_price=?, entry_time=? WHERE id=1",
            (self.balance, self.position, self.entry_price, self.entry_time),
        )
        self.conn.commit()

    # ── Trading interface ───────────────────────────────────────────────── #

    def execute_trade(self, signal: str, price: float, current_time: float) -> bool:
        if self.position is not None:
            return False
        self.position = signal
        self.entry_price = price
        self.entry_time = current_time
        self._save_state()
        return True

    def update(self, current_price: float, current_time: float) -> dict | None:
        if self.position is None:
            return None

        pnl_pct = (
            (current_price - self.entry_price) / self.entry_price
            if self.position == "LONG"
            else (self.entry_price - current_price) / self.entry_price
        )

        hit_sl = pnl_pct <= -self.sl_pct
        hit_tp = pnl_pct >= self.tp_pct
        hit_time = current_time >= (self.entry_time + self.horizon_ms)

        if not (hit_sl or hit_tp or hit_time):
            return None

        pos_size_usd = self.margin_usdt * self.leverage
        net_profit = pos_size_usd * pnl_pct - pos_size_usd * self.fee_pct * 2
        self.balance += net_profit

        reason = "SL" if hit_sl else ("TP" if hit_tp else "TIME")
        result = {
            "type": self.position,
            "entry": self.entry_price,
            "exit": current_price,
            "net_profit_usd": net_profit,
            "reason": reason,
        }

        self.position = None
        self.entry_price = 0.0
        self.entry_time = 0
        self._save_state()
        return result

    def get_unrealized_pnl(self, current_price: float) -> float:
        if self.position is None:
            return 0.0
        pnl_pct = (
            (current_price - self.entry_price) / self.entry_price
            if self.position == "LONG"
            else (self.entry_price - current_price) / self.entry_price
        )
        pos_size_usd = self.margin_usdt * self.leverage
        return pos_size_usd * pnl_pct - pos_size_usd * self.fee_pct * 2
