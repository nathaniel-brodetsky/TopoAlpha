import sqlite3

class PaperTrader :
    def __init__ (
    self ,
    db_path :str ="trading_state.db",
    initial_balance :float =10_000.0 ,
    margin_usdt :float =50.0 ,
    leverage :int =10 ,
    horizon_bars :int =12 ,
    timeframe_mins :int =1 ,
    sl_atr_mult :float =1.5 ,
    tp_atr_mult :float =3.0 ,
    fee_pct :float =0.0004 ,
    ):
        self .margin_usdt =margin_usdt
        self .leverage =leverage
        self .horizon_ms =horizon_bars *timeframe_mins *60 *1_000
        self .sl_atr_mult =sl_atr_mult
        self .tp_atr_mult =tp_atr_mult
        self .fee_pct =fee_pct

        self ._default_sl_pct =0.004
        self ._default_tp_pct =0.008

        self ._db_path =db_path
        self .conn =sqlite3 .connect (db_path ,check_same_thread =False )
        self ._init_db (initial_balance )
        self ._load_state ()

    def close (self )->None :
        if self .conn :
            try :
                self .conn .close ()
            except Exception :
                pass
            finally :
                self .conn =None

    def __del__ (self )->None :
        self .close ()

    def __enter__ (self )->"PaperTrader":
        return self

    def __exit__ (self ,*_ )->None :
        self .close ()

    def _init_db (self ,initial_balance :float )->None :
        cur =self .conn .cursor ()
        cur .execute ("""
            CREATE TABLE IF NOT EXISTS state (
                id          INTEGER PRIMARY KEY,
                balance     REAL,
                pos_type    TEXT,
                entry_price REAL,
                entry_time  REAL,
                sl_pct      REAL,
                tp_pct      REAL,
                pending_sig TEXT
            )
        """)
        cur .execute ("SELECT balance FROM state WHERE id = 1")
        if not cur .fetchone ():
            cur .execute (
            "INSERT INTO state (id, balance, pos_type, entry_price, entry_time, "
            "sl_pct, tp_pct, pending_sig) VALUES (1, ?, NULL, 0.0, 0, ?, ?, NULL)",
            (initial_balance ,self ._default_sl_pct ,self ._default_tp_pct ),
            )
        else :

            cols =[r [1 ]for r in cur .execute ("PRAGMA table_info(state)").fetchall ()]
            for col ,default in [("sl_pct",self ._default_sl_pct ),
            ("tp_pct",self ._default_tp_pct ),
            ("pending_sig","NULL")]:
                if col not in cols :
                    if col =="pending_sig":
                        cur .execute (f"ALTER TABLE state ADD COLUMN {col } TEXT DEFAULT NULL")
                    else :
                        cur .execute (f"ALTER TABLE state ADD COLUMN {col } REAL DEFAULT {default }")
        self .conn .commit ()

    def _load_state (self )->None :
        row =self .conn .cursor ().execute (
        "SELECT balance, pos_type, entry_price, entry_time, sl_pct, tp_pct, pending_sig "
        "FROM state WHERE id = 1"
        ).fetchone ()
        self .balance :float =row [0 ]
        self .position :str |None =row [1 ]
        self .entry_price :float =row [2 ]
        self .entry_time :float =row [3 ]
        self .sl_pct :float =row [4 ]if row [4 ]else self ._default_sl_pct
        self .tp_pct :float =row [5 ]if row [5 ]else self ._default_tp_pct
        self .pending_signal :str |None =row [6 ]

    def _save_state (self )->None :
        self .conn .cursor ().execute (
        "UPDATE state SET balance=?, pos_type=?, entry_price=?, entry_time=?, "
        "sl_pct=?, tp_pct=?, pending_sig=? WHERE id=1",
        (self .balance ,self .position ,self .entry_price ,self .entry_time ,
        self .sl_pct ,self .tp_pct ,self .pending_signal ),
        )
        self .conn .commit ()

    def set_pending (self ,signal :str ,atr_pct :float |None =None )->bool :

        if self .position is not None or self .pending_signal is not None :
            return False
        self .pending_signal =signal

        if atr_pct and atr_pct >0 :
            self .sl_pct =max (atr_pct *self .sl_atr_mult ,0.002 )
            self .tp_pct =max (atr_pct *self .tp_atr_mult ,0.004 )
        else :
            self .sl_pct =self ._default_sl_pct
            self .tp_pct =self ._default_tp_pct
        self ._save_state ()
        return True

    def cancel_pending (self )->None :
        if self .pending_signal is not None :
            self .pending_signal =None
            self ._save_state ()

    def execute_at_open (self ,open_price :float ,current_time :float )->bool :

        if self .pending_signal is None or self .position is not None :
            self .pending_signal =None
            return False
        signal =self .pending_signal
        self .pending_signal =None
        self .position =signal
        self .entry_price =open_price
        self .entry_time =current_time
        self ._save_state ()
        return True

    def execute_trade (self ,signal :str ,price :float ,current_time :float )->bool :
        if self .position is not None :
            return False
        self .position =signal
        self .entry_price =price
        self .entry_time =current_time
        self ._save_state ()
        return True

    def update (self ,current_price :float ,current_time :float )->dict |None :

        if self .position is None :
            return None

        pnl_pct =(
        (current_price -self .entry_price )/self .entry_price
        if self .position =="LONG"
        else (self .entry_price -current_price )/self .entry_price
        )

        hit_sl =pnl_pct <=-self .sl_pct
        hit_tp =pnl_pct >=self .tp_pct
        hit_time =current_time >=(self .entry_time +self .horizon_ms )

        if not (hit_sl or hit_tp or hit_time ):
            return None

        pos_size_usd =self .margin_usdt *self .leverage
        net_profit =pos_size_usd *pnl_pct -pos_size_usd *self .fee_pct *2
        self .balance +=net_profit

        reason ="SL"if hit_sl else ("TP"if hit_tp else "TIME")
        result ={
        "type":self .position ,
        "entry":self .entry_price ,
        "exit":current_price ,
        "net_profit_usd":net_profit ,
        "reason":reason ,
        "sl_pct":self .sl_pct ,
        "tp_pct":self .tp_pct ,
        }

        self .position =None
        self .entry_price =0.0
        self .entry_time =0
        self ._save_state ()
        return result

    def get_unrealized_pnl (self ,current_price :float )->float :
        if self .position is None :
            return 0.0
        pnl_pct =(
        (current_price -self .entry_price )/self .entry_price
        if self .position =="LONG"
        else (self .entry_price -current_price )/self .entry_price
        )
        pos_size_usd =self .margin_usdt *self .leverage
        return pos_size_usd *pnl_pct -pos_size_usd *self .fee_pct *2

    @property
    def current_sl_price (self )->float |None :
        if not self .position :
            return None
        if self .position =="LONG":
            return self .entry_price *(1 -self .sl_pct )
        return self .entry_price *(1 +self .sl_pct )

    @property
    def current_tp_price (self )->float |None :
        if not self .position :
            return None
        if self .position =="LONG":
            return self .entry_price *(1 +self .tp_pct )
        return self .entry_price *(1 -self .tp_pct )