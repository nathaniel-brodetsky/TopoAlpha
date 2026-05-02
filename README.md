# TopoAlpha

> Topological Data Analysis meets Machine Learning for intraday crypto trading.

TopoAlpha detects geometric structure in price trajectories using **persistent homology** (TDA), feeds the signal into a calibrated three-model ML ensemble, and executes paper trades with full risk management. No black boxes — every number the system produces can be inspected in the built-in Trade Calculator.

---

## How it works

```
Market data (Binance Futures)
        │
        ▼
  Delay Embedding          prices → 3-D phase-space point cloud
        │
        ▼
  Ripser (TDA)             persistent homology → H₁ stress score
        │
        ├── stress ≥ threshold?
        │
        ▼
  TopoBooster (ML)         GBM + RF + ExtraTrees (calibrated, isotonic)
        │
        ├── P(UP) or P(DOWN) ≥ 0.60?
        │
        ▼
  PaperTrader              ATR-based SL/TP, fee simulation, SQLite state
        │
        ▼
  Telegram notifier        entry / exit alerts
```

### Topological Stress

A **delay embedding** converts the last 50 close prices into a 3-D point cloud in phase space. Ripser computes its **Vietoris–Rips filtration** and extracts the maximum persistence of H₁ (loop) features. High persistence means the price trajectory has strong cyclical structure — a precondition for a directional signal.

### TopoBooster

A three-way classifier (flat / up / down) built from a calibrated ensemble:

| Model | Role |
|---|---|
| GradientBoostingClassifier | Strong on ordinal / monotone features |
| RandomForestClassifier | Decorrelated bagging, handles noise |
| ExtraTreesClassifier | Max randomisation, fast |

All three use `CalibratedClassifierCV(method="isotonic", cv=3)` so probabilities are well-calibrated and directly comparable. Predictions are smoothed through a rolling EMA buffer to suppress tick noise.

Features include ATR percentiles, RSI, momentum, topological stress lags, order-book imbalance (OBI), and synthetic HTF trend features derived from long-span EMAs on the LTF series.

---

## Project structure

```
topoalpha/
├── app.py               GUI dashboard (PyQt5 + pyqtgraph + matplotlib 3-D)
├── daemon.py            Headless server mode
├── trade_calculator.py  Standalone analysis panel (runs independently)
├── data_feeder.py       Binance OHLCV + order-book feed (ccxt)
├── tda_core.py          Topological stress via ripser
├── ml_model.py          TopoBooster ensemble
├── paper_trader.py      Simulated trade execution + SQLite persistence
├── notifier.py          Telegram alerts
└── .env                 API credentials (not committed)
```

---

## Requirements

```
Python 3.11+
```

```
pip install pyqt5 pyqtgraph matplotlib numpy pandas scikit-learn \
            ccxt python-dotenv ripser requests python-binance
```

---

## Configuration

Create a `.env` file in the project root:

```env
# Optional — Telegram alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

No exchange API keys are required for paper trading. The data feed uses Binance's public OHLCV and order-book endpoints via ccxt (no authentication needed).

---

## Running

### GUI dashboard

```bash
python app.py
```

Shows four panes side by side:

- **Live price chart** with entry markers (▲ long / ▼ short), SL/TP lines, and horizon cutoff
- **Topological stress** with adjustable threshold slider (drag to tune sensitivity)
- **Order Book Imbalance** (bid vs ask volume balance, ±1)
- **3-D phase-space** scatter plot of the delay-embedded price trajectory

### Headless daemon (for servers / VPS)

```bash
python daemon.py
```

Identical logic, no GUI dependency. Suitable for `screen` / `tmux` / `systemd`.

### Trade Calculator (standalone)

```bash
python trade_calculator.py
```

Loads 300 bars from Binance automatically. Produces a step-by-step breakdown covering:

1. Market context (price, bar count)
2. Technical indicators (ATR, RSI, momentum, volatility rank)
3. HTF macro trend (EMA8 vs EMA21)
4. Topological stress + order book
5. ML probabilities with visual bars
6. Composite directional score
7. Direction decision
8. ATR-based SL / TP levels
9. Position size and net P&L
10. **Expected Value** and **Kelly Criterion** sizing

---

## Parameters

All key parameters live at the top of `daemon.py` / `app.py`:

| Parameter | Default | Description |
|---|---|---|
| `SYMBOL` | `BTC/USDT` | Trading pair |
| `TIMEFRAME` | `5m` | Entry timeframe |
| `HTF` | `1h` | Macro context timeframe |
| `HORIZON_BARS` | `6` | Max bars per trade (6 × 5m = 30 min) |
| `RETRAIN_EVERY` | `150` | Ticks between ML re-fits |
| `alpha_stress_threshold` | `1.5` | Min topological stress to consider a trade |
| `sl_pct` | `0.5%` | Stop-loss distance |
| `tp_pct` | `1.0%` | Take-profit distance |
| `margin_usdt` | `50` | Margin per trade (USDT) |
| `leverage` | `10×` | Futures leverage |

The stress threshold is also adjustable live from the GUI slider without restart.

---

## Design decisions

**Why paper trading only?**  
Live execution adds exchange API complexity without changing the signal logic. The paper engine is identical in logic to what a live system would do — the only missing layer is the HTTP order. Adding a live executor is a one-file addition when you are ready.

**Why SQLite?**  
Position state survives restarts. If the daemon crashes mid-trade, the position, entry price and entry time are restored from the database on the next launch.

**Why background threads for ML retrain?**  
Three `CalibratedClassifierCV(cv=3)` models on 500 rows takes several seconds. Running it on the main thread freezes the UI or the poll loop. All retrains run in a daemon thread; a lock prevents double-running.

**Why a 4-second timeout for Ripser?**  
On degenerate or near-constant point clouds, the Vietoris–Rips filtration can hang indefinitely. The single-worker `ThreadPoolExecutor` caps the call at 4 seconds and returns `0.0` on timeout, keeping the main loop responsive.

---

## Signals at a glance

A trade opens when **all three gates** pass simultaneously:

```
Gate 1  topological stress  ≥  threshold (default 1.5)
Gate 2  ML probability      ≥  0.60  (UP for long, DOWN for short)
Gate 3  no position currently open
```

A trade closes on the first of:

```
• Price hits stop-loss     (−0.5% from entry)
• Price hits take-profit   (+1.0% from entry)
• Trade horizon expires    (30 minutes at default settings)
```

---

## Telegram alerts

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.  
Create a bot via [@BotFather](https://t.me/BotFather), get your chat ID from [@userinfobot](https://t.me/userinfobot).

You will receive alerts for:
- 🟢 Engine started
- 🚀 Long entry
- 🩸 Short entry  
- ✅ Trade closed in profit
- 🛑 Trade closed at stop-loss
- 🔴 Engine stopped

---

## License

MIT
