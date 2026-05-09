#!/usr/bin/env python3
"""
scalper.py  ─  TopoAlpha Terminal Scalper
══════════════════════════════════════════════════════════════════════════════
Paper mode  |  Leverage 1×  |  Position = 10 % of balance per trade

  L  ──  Open LONG        S  ──  Open SHORT
  C  ──  Close position   Q  ──  Quit
══════════════════════════════════════════════════════════════════════════════

Install extra deps (once):
    pip install textual plotext
"""
from __future__ import annotations

import csv
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import ccxt
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static

try:
    import plotext as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


SYMBOL          = "BTC/USDT"
TIMEFRAME       = "1m"
LEVERAGE        = 1            # fixed: paper scalper runs unlevered
POSITION_PCT    = 0.10         # 10 % of balance per trade
FEE_PCT         = 0.0004       # 0.04 % taker, both legs
CHART_BARS      = 80           # 1-minute candles on chart
POLL_SECS       = 2.0          # price refresh interval
DB_PATH         = "scalper_state.db"
JOURNAL_PATH    = "scalper_journal.csv"
INITIAL_BALANCE = 10_000.0

_JOURNAL_COLS = [
    "unix_ts", "datetime", "side",
    "entry_usd", "exit_usd", "notional_usd",
    "gross_usd", "net_usd", "pnl_pct",
]


class ScalperBook:
    """
    Minimal SQLite-backed paper-trading ledger.

    State is persisted immediately on every transition so a
    crash between keystrokes loses nothing.
    """

    def __init__(
        self,
        db      : str   = DB_PATH,
        initial : float = INITIAL_BALANCE,
    ) -> None:
        self.conn = sqlite3.connect(db, check_same_thread=False)
        self._ensure_schema(initial)
        self._load()


    def _ensure_schema(self, initial: float) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scalper_book (
                id          INTEGER PRIMARY KEY,
                balance     REAL    NOT NULL,
                pos_side    TEXT,
                entry_price REAL    NOT NULL DEFAULT 0,
                pos_size    REAL    NOT NULL DEFAULT 0,
                entry_ts    REAL    NOT NULL DEFAULT 0
            )
        """)
        if not self.conn.execute(
            "SELECT id FROM scalper_book WHERE id = 1"
        ).fetchone():
            self.conn.execute(
                "INSERT INTO scalper_book VALUES (1, ?, NULL, 0, 0, 0)",
                (initial,),
            )
        self.conn.commit()

    def _load(self) -> None:
        row = self.conn.execute(
            "SELECT balance, pos_side, entry_price, pos_size, entry_ts "
            "FROM scalper_book WHERE id = 1"
        ).fetchone()
        self.balance : float      = row[0]
        self.side    : str | None = row[1]
        self.entry   : float      = row[2]
        self.size    : float      = row[3]   # notional USD
        self.ts_open : float      = row[4]

    def _flush(self) -> None:
        self.conn.execute(
            "UPDATE scalper_book "
            "SET balance=?, pos_side=?, entry_price=?, pos_size=?, entry_ts=? "
            "WHERE id=1",
            (self.balance, self.side,
             self.entry or 0.0, self.size or 0.0, self.ts_open or 0.0),
        )
        self.conn.commit()


    def open(self, side: str, price: float) -> bool:
        """Open a position. Returns False if one is already open."""
        if self.side is not None or price <= 0:
            return False
        margin        = self.balance * POSITION_PCT
        notional      = margin * LEVERAGE
        entry_fee     = notional * FEE_PCT
        self.side     = side
        self.entry    = price
        self.size     = notional
        self.ts_open  = time.time()
        self.balance -= entry_fee
        self._flush()
        return True

    def close(self, price: float) -> dict | None:
        """Close the open position. Returns a result dict or None."""
        if self.side is None or price <= 0:
            return None
        pnl_pct = (
            (price - self.entry) / self.entry
            if self.side == "LONG"
            else (self.entry - price) / self.entry
        )
        gross        = self.size * pnl_pct
        exit_fee     = self.size * FEE_PCT
        net          = gross - exit_fee
        self.balance += net
        result = dict(
            side    = self.side,
            entry   = self.entry,
            exit    = price,
            size    = self.size,
            gross   = gross,
            net     = net,
            pnl_pct = pnl_pct,
            ts      = time.time(),
        )
        self.side    = None
        self.entry   = 0.0
        self.size    = 0.0
        self.ts_open = 0.0
        self._flush()
        return result


    def unrealized(self, price: float) -> float:
        if self.side is None or self.entry == 0:
            return 0.0
        pct = (
            (price - self.entry) / self.entry
            if self.side == "LONG"
            else (self.entry - price) / self.entry
        )
        return self.size * pct - self.size * FEE_PCT   # net of exit fee

    def pnl_pct(self, price: float) -> float:
        if self.side is None or self.entry == 0:
            return 0.0
        return (
            (price - self.entry) / self.entry
            if self.side == "LONG"
            else (self.entry - price) / self.entry
        )

    def close_db(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


class Feed:
    """Thin ccxt wrapper — Binance Futures public endpoints only."""

    def __init__(self) -> None:
        self.ex = ccxt.binance({
            "enableRateLimit": True,
            "timeout"        : 10_000,
            "options"        : {"defaultType": "future"},
        })

    def ticker(self) -> dict:
        return self.ex.fetch_ticker(SYMBOL)

    def ohlcv(self, limit: int = CHART_BARS) -> list:
        return self.ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)


def _ascii_line(
    values : list[float],
    entry  : float | None,
    side   : str | None,
    width  : int,
    height : int,
) -> str:
    """Fallback Unicode line chart — no external deps."""
    if not values:
        return "  no data"
    lo, hi = min(values), max(values)
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    span   = hi - lo
    cols   = min(len(values), width)
    data   = values[-cols:]
    rows   = []
    for r in range(height - 1, -1, -1):
        lo_r = lo + r * span / height
        hi_r = lo + (r + 1) * span / height
        line : list[str] = []
        for v in data:
            if lo_r <= v < hi_r:
                line.append("▪")
            elif entry is not None and lo_r <= entry < hi_r:
                line.append("─")
            else:
                line.append(" ")
        if entry is not None and lo_r <= entry < hi_r:
            line = [("◆" if ch == "▪" else "─") for ch in line]
        # price axis on the right
        price_lbl = f" {(lo_r + hi_r) / 2:,.1f}" if r % max(1, height // 4) == 0 else ""
        rows.append("".join(line) + price_lbl)
    rows.append("─" * cols)
    return "\n".join(rows)


def build_chart(
    ohlcv  : list,
    entry  : float | None,
    side   : str | None,
    width  : int,
    height : int,
) -> str | Text:
    """Return a Rich-renderable price chart string."""
    if not ohlcv:
        return "  Fetching market data…"

    closes = [c[4] for c in ohlcv]
    times  = [
        datetime.fromtimestamp(c[0] / 1_000).strftime("%H:%M")
        for c in ohlcv
    ]

    if not _HAS_PLT:
        return _ascii_line(closes, entry, side, width, height)

    try:
        plt.clf()
        plt.theme("dark")
        plt.plot_size(width, height)
        plt.plot(closes, color="cyan", label="price")

        if entry is not None and side:
            entry_line = [entry] * len(closes)
            col = "green" if side == "LONG" else "red"
            plt.plot(entry_line, color=col, label=f"entry {entry:,.2f}")

        tick_step = max(1, len(times) // 8)
        plt.xticks(
            list(range(0, len(times), tick_step)),
            [times[i] for i in range(0, len(times), tick_step)],
        )
        plt.xlabel(f"  {SYMBOL}  ·  {TIMEFRAME}  ·  {times[0]} → {times[-1]}  ")
        raw = plt.build()
        return Text.from_ansi(raw)
    except Exception as exc:
        return f"  chart error: {exc}"


def init_journal() -> None:
    p = Path(JOURNAL_PATH)
    if not p.exists():
        with p.open("w", newline="") as f:
            csv.writer(f).writerow(_JOURNAL_COLS)


def log_trade(r: dict) -> None:
    row = [
        int(r["ts"]),
        datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M:%S"),
        r["side"],
        round(r["entry"],   2),
        round(r["exit"],    2),
        round(r["size"],    2),
        round(r["gross"],   4),
        round(r["net"],     4),
        f"{r['pnl_pct'] * 100:+.4f}",
    ]
    with open(JOURNAL_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _fp(v: float, d: int = 2) -> str:
    return f"${v:,.{d}f}"

def _sign(v: float) -> str:
    return "+" if v >= 0 else ""

def _pc(v: float) -> str:
    return "green" if v >= 0 else "red"

def _sc(side: str | None) -> str:
    return "green" if side == "LONG" else "red"


class ScalperApp(App):

    TITLE     = "TopoAlpha  ·  Terminal Scalper"
    SUB_TITLE = (
        f"{SYMBOL}  |  Paper  |  "
        f"Lev {LEVERAGE}×  |  "
        f"{int(POSITION_PCT * 100)} % per trade"
    )

    CSS = """
    /* ── root ───────────────────────────────────────────────────────── */
    Screen {
        background: #06091a;
    }

    /* ── main row (chart + sidebar) ─────────────────────────────────── */
    #main_row {
        height: 1fr;
    }

    #chart_pane {
        width:      1fr;
        height:     1fr;
        border:     solid #152040;
        padding:    0 0;
        background: #06091a;
        overflow-y: hidden;
    }

    #sidebar {
        width:      32;
        height:     1fr;
        background: #06091a;
    }

    /* ── sidebar cards ─────────────────────────────────────────────── */
    #price_box {
        height:     6;
        border:     solid #152040;
        padding:    0 1;
        background: #0a1020;
    }

    #pos_box {
        height:     10;
        border:     solid #152040;
        padding:    0 1;
        background: #0a1020;
    }

    #act_box {
        height:     1fr;
        border:     solid #152040;
        padding:    1 2;
        background: #0a1020;
        content-align: left middle;
    }

    /* ── history table ──────────────────────────────────────────────── */
    #hist_pane {
        height:     11;
        border:     solid #152040;
        background: #06091a;
    }

    DataTable {
        background: #06091a;
        color:      #8ab4d4;
        height:     1fr;
    }

    DataTable > .datatable--header {
        background:  #0a1020;
        color:       #3a70b8;
        text-style:  bold;
    }

    DataTable > .datatable--cursor {
        background: #152040;
        color:      #e8f0ff;
    }

    DataTable > .datatable--fixed {
        background: #0a1020;
    }

    /* ── header / footer ────────────────────────────────────────────── */
    Header {
        background: #08101e;
        color:      #4a82c8;
        text-style: bold;
    }

    Footer {
        background: #08101e;
        color:      #3a5880;
    }
    """

    BINDINGS = [
        Binding("l", "open_long",  "LONG",  show=True),
        Binding("s", "open_short", "SHORT", show=True),
        Binding("c", "close_pos",  "CLOSE", show=True),
        Binding("q", "quit",       "Quit",  show=True),
    ]


    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_row"):
            yield Static("", id="chart_pane")
            with Vertical(id="sidebar"):
                yield Static("", id="price_box")
                yield Static("", id="pos_box")
                yield Static("", id="act_box")
        yield DataTable(id="hist_pane", show_cursor=True)
        yield Footer()


    def on_mount(self) -> None:
        self.book        = ScalperBook()
        self.feed        = Feed()
        self._price      : float = 0.0
        self._chg24h     : float = 0.0
        self._ohlcv      : list  = []
        self._n_trades   : int   = 0
        self._fetch_lock         = threading.Lock()

        init_journal()
        self._setup_table()
        self._render_action_box()
        self._do_fetch()                              # immediate first fetch
        self.set_interval(POLL_SECS, self._do_fetch)


    def _setup_table(self) -> None:
        tbl: DataTable = self.query_one("#hist_pane", DataTable)
        tbl.add_columns(
            "#", "Time", "Side",
            "Entry", "Exit", "Size",
            "Gross", "Net P&L", "P&L %",
        )
        tbl.cursor_type = "row"

    def _push_row(self, r: dict) -> None:
        self._n_trades += 1
        tbl: DataTable = self.query_one("#hist_pane", DataTable)
        net  = r["net"]
        sc   = _sc(r["side"])
        pc   = _pc(net)
        ts_s = datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S")
        tbl.add_row(
            str(self._n_trades),
            ts_s,
            f"[bold {sc}]{r['side']}[/]",
            _fp(r["entry"]),
            _fp(r["exit"]),
            f"${r['size']:,.0f}",
            f"[{_pc(r['gross'])}]{_sign(r['gross'])}{_fp(r['gross'])}[/]",
            f"[{pc}]{_sign(net)}{_fp(net)}[/]",
            f"[{pc}]{r['pnl_pct'] * 100:+.3f}%[/]",
        )


    def _render_price_box(self) -> str:
        p   = self._price
        chg = self._chg24h
        cc  = _pc(chg)
        bal = self.book.balance
        eq  = bal + self.book.unrealized(p)
        return (
            f"\n"
            f" [bold white]{_fp(p)}[/]  [{cc}]{_sign(chg)}{chg:.2f} %[/{cc}]\n"
            f"\n"
            f" Balance  [cyan]{_fp(bal)}[/cyan]\n"
            f" Equity   [white]{_fp(eq)}[/white]"
        )


    def _render_pos_box(self) -> str:
        b = self.book
        p = self._price
        if b.side is None:
            return (
                " [dim]─── No Open Position ──[/dim]\n"
                "\n"
                " Side    [dim]—[/dim]\n"
                " Entry   [dim]—[/dim]\n"
                " Size    [dim]—[/dim]\n"
                " Unreal  [dim]—[/dim]\n"
                " P&L %   [dim]—[/dim]\n"
                " Open    [dim]—[/dim]"
            )
        unreal = b.unrealized(p)
        pct    = b.pnl_pct(p) * 100
        pc     = _pc(unreal)
        sc     = _sc(b.side)
        dur    = int(time.time() - b.ts_open)
        dur_s  = f"{dur // 60}m {dur % 60:02d}s"
        return (
            f" [dim]────── Position ────[/dim]\n"
            f"\n"
            f" Side    [bold {sc}]{b.side}[/]\n"
            f" Entry   [white]{_fp(b.entry)}[/white]\n"
            f" Size    [white]${b.size:,.0f}[/white]\n"
            f" Unreal  [{pc}]{_sign(unreal)}{_fp(unreal)}[/]\n"
            f" P&L %   [{pc}]{pct:+.3f} %[/]\n"
            f" Open    [dim]{dur_s}[/dim]"
        )


    def _render_action_box(self) -> None:
        has = self.book.side is not None
        txt = (
            "[dim] ▲   L   ──────[/dim]\n"
            "[dim] ▼   S   ──────[/dim]\n"
            "\n"
            "[bold yellow] ✕   C   CLOSE [/bold yellow]"
        ) if has else (
            "[bold green] ▲   L   LONG  [/bold green]\n"
            "[bold red]   ▼   S   SHORT [/bold red]\n"
            "\n"
            "[dim] ✕   C   ──────[/dim]"
        )
        self.query_one("#act_box", Static).update(txt)


    def _render_chart(self) -> None:
        widget = self.query_one("#chart_pane", Static)
        w = max(widget.size.width  - 2, 40)
        h = max(widget.size.height - 2, 10)
        b = self.book
        widget.update(
            build_chart(
                self._ohlcv,
                b.entry if b.side else None,
                b.side,
                w, h,
            )
        )


    def _refresh_all(self) -> None:
        self._render_chart()
        self.query_one("#price_box", Static).update(self._render_price_box())
        self.query_one("#pos_box",   Static).update(self._render_pos_box())
        self._render_action_box()


    @work(thread=True)
    def _do_fetch(self) -> None:
        """Fetch ticker + OHLCV in a worker thread; never blocks the UI."""
        if not self._fetch_lock.acquire(blocking=False):
            return                                    # previous fetch in flight
        try:
            ticker       = self.feed.ticker()
            self._price  = float(ticker["last"])
            self._chg24h = float(ticker.get("percentage") or 0.0)
            self._ohlcv  = self.feed.ohlcv(limit=CHART_BARS)
        except Exception:
            pass                                      # keep stale values
        finally:
            self._fetch_lock.release()
        self.call_from_thread(self._refresh_all)


    def action_open_long(self) -> None:
        if self._price <= 0:
            return
        if self.book.open("LONG", self._price):
            self._refresh_all()

    def action_open_short(self) -> None:
        if self._price <= 0:
            return
        if self.book.open("SHORT", self._price):
            self._refresh_all()

    def action_close_pos(self) -> None:
        if self._price <= 0 or self.book.side is None:
            return
        result = self.book.close(self._price)
        if result:
            log_trade(result)
            self._push_row(result)
            self._refresh_all()

    def action_quit(self) -> None:
        self.book.close_db()
        self.exit()


if __name__ == "__main__":
    ScalperApp().run()