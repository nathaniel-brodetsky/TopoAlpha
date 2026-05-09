# TopoAlpha

> Persistent homology meets calibrated machine learning for intraday crypto futures trading.

---

## Mathematical Foundation

### 1 · Delay Embedding

Raw price series $\{p_t\}$ are lifted into a $d$-dimensional **phase space** via Takens' delay embedding theorem. For embedding dimension $d = 3$ and delay $\tau$:

$$\mathbf{x}_t = \bigl(p_t,\; p_{t-\tau},\; p_{t-2\tau}\bigr) \in \mathbb{R}^3$$

A sliding window of $W = 50$ bars produces a point cloud

$$\mathcal{P} = \{\mathbf{x}_{t-W+1}, \ldots, \mathbf{x}_t\} \subset \mathbb{R}^3$$

that encodes the **geometric topology of the price trajectory** rather than its level.

---

### 2 · Persistent Homology (TDA Core)

Given $\mathcal{P}$, the **Vietoris–Rips filtration** grows simplicial complexes $\mathcal{R}(\varepsilon)$ as the radius $\varepsilon$ increases from $0$ to $\infty$. The $k$-th **Betti number** $\beta_k$ counts independent $k$-dimensional holes:

| $k$ | Feature | Interpretation |
|-----|---------|---------------|
| $0$ | Connected components | Cluster structure |
| $1$ | Loops ($H_1$) | **Cyclical price structure** |

For each $H_1$ generator born at $\varepsilon_b$ and dying at $\varepsilon_d$, its **persistence** is:

$$\text{pers}(b, d) = \varepsilon_d - \varepsilon_b \geq 0$$

The **topological stress score** is the maximum finite persistence:

$$\sigma = \max_{(b,d)\,\in\, H_1,\; d < \infty} \bigl(\varepsilon_d - \varepsilon_b\bigr)$$

High $\sigma$ signals that the price trajectory has durable loop structure in phase space — a geometric precondition for a directional signal.  
Gate 1 opens when $\sigma \geq \sigma^*$, where $\sigma^*$ is adapted dynamically as the 75th percentile of a 200-sample rolling buffer.

---

### 3 · Feature Engineering

The ML ensemble receives a 20-dimensional feature vector $\mathbf{f}_t$ at each bar:

$$\mathbf{f}_t = \bigl(\underbrace{\text{ATR}_{14}^{(\%)}}_{\text{volatility}},\; \underbrace{\text{RSI}_{14}}_{\text{momentum}},\; \underbrace{r_{5},\, r_{10},\, r_{20}}_{\text{return lags}},\; \underbrace{\sigma_t,\, \sigma_{t-1},\, \sigma_{t-2}}_{\text{stress lags}},\; \underbrace{\text{OBI}}_{\text{order book}},\; \underbrace{\widetilde{\mu}_8,\, \widetilde{\mu}_{21}}_{\text{HTF EMAs}},\; \ldots\bigr)$$

HTF trend is synthesised directly on the LTF series via EMA spans scaled by the timeframe ratio, eliminating the need for a separate high-frequency data stream at prediction time.

---

### 4 · TopoBooster Ensemble

Three base classifiers predict $\hat{y} \in \{\text{flat},\,\text{up},\,\text{down}\}$:

$$\mathcal{M} = \bigl\{\underbrace{f_{\text{GBM}}}_{\text{ordinal structure}},\; \underbrace{f_{\text{RF}}}_{\text{decorrelated bagging}},\; \underbrace{f_{\text{ET}}}_{\text{max randomisation}}\bigr\}$$

Each base model is wrapped in isotonic probability calibration:

$$\hat{P}_k = \text{CalibratedClassifierCV}\bigl(f_k,\; \text{method}=\text{"isotonic"},\; cv=3\bigr)$$

The ensemble probability vector is the arithmetic mean of calibrated outputs:

$$\hat{p}(\text{class}) = \frac{1}{|\mathcal{M}|} \sum_{k} \hat{P}_k(\text{class} \mid \mathbf{f}_t)$$

Predictions are smoothed through a rolling EMA buffer of depth 5 to suppress intra-bar noise:

$$\tilde{p}_t = \alpha\, \hat{p}_t + (1 - \alpha)\, \tilde{p}_{t-1}, \qquad \alpha = 0.40$$

Gate 2 opens when $\tilde{p}(\text{up}) \geq 0.60$ (long) or $\tilde{p}(\text{down}) \geq 0.60$ (short).

---

### 5 · Risk & Execution Model

Stop-loss and take-profit distances are ATR-proportional:

$$\text{SL} = \max\!\bigl(\lambda_{\text{SL}} \cdot \text{ATR}_t^{(\%)},\; 0.2\%\bigr), \qquad \text{TP} = \max\!\bigl(\lambda_{\text{TP}} \cdot \text{ATR}_t^{(\%)},\; 0.4\%\bigr)$$

with defaults $\lambda_{\text{SL}} = 1.5$, $\lambda_{\text{TP}} = 3.0$ (target $\text{RR} = 2$).

Net P&L per trade (both fees counted):

$$\text{PnL} = M \cdot L \cdot r_{\text{exit}} - 2 M \cdot L \cdot f$$

where $M$ is margin (USDT), $L$ is leverage, $r_{\text{exit}}$ is return to exit, and $f = 0.04\%$ is the fee rate.

The circuit breaker trips on either of:

$$\text{consecutive losses} \geq 4 \qquad \text{or} \qquad \text{drawdown} = \frac{B_{\text{peak}} - B_t}{B_{\text{peak}}} \geq 15\%$$

with a 30-minute cooling-off period before trading resumes.

---

### 6 · Expected Value & Kelly Sizing

The Trade Calculator reports per-trade EV and the fractional Kelly criterion:

$$\text{EV} = p_w \cdot \text{TP}_{\$} - p_l \cdot \text{SL}_{\$}$$

$$f^* = \frac{p_w}{|\text{SL}|} - \frac{p_l}{|\text{TP}|} \quad\text{(full Kelly)}$$

where $p_w = \hat{p}(\text{signal direction})$ and $p_l = 1 - p_w$.

---

### 7 · Walk-Forward Optimisation

`wfo.py` performs **anchored walk-forward validation** over a grid of hyperparameters $\Theta$:

$$\Theta = \bigl\{\sigma^*,\; \lambda_{\text{SL}},\; \lambda_{\text{TP}},\; H,\; p_{\min}\bigr\}$$

Each fold trains on $[\,t_0,\, t_0 + T_{\text{train}}]$ and evaluates on $[\,t_0 + T_{\text{train}},\, t_0 + T_{\text{train}} + T_{\text{test}}]$, rolling forward by $T_{\text{step}}$. The selection criterion is **Calmar ratio** (annualised return / max drawdown).

---

## Signal Gate Summary

A position opens only when all three gates pass simultaneously:

```
Gate 1 · Topological   σ ≥ σ*   (adaptive 75th-pct threshold)
Gate 2 · ML            p̃ ≥ 0.60 (calibrated ensemble probability)
Gate 3 · Risk          no open position  ∧  circuit breaker inactive
```

A position closes on whichever arrives first:

```
· Price crosses stop-loss    (−λ_SL · ATR from entry)
· Price crosses take-profit  (+λ_TP · ATR from entry)
· Horizon expires            (H bars after entry)
```

---

## Architecture

```
Market data  ──►  RobustDataFeeder  (ccxt / Binance Futures, LTF + HTF)
                        │
                        ▼
              Delay Embedding  ──►  TDAAnalyzer  ──►  σ (H₁ stress)
                        │                                    │
                        │           FeatureBuilder  ◄────────┘
                        │                │
                        ▼                ▼
                  TopoBooster  (GBM + RF + ET, isotonic calibration)
                        │
                   [optional]
                        │
               TopoDynPredictor  (Hurst · Lyapunov · RQA · EWS · MFDFA)
                        │
                        ▼
                  PaperTrader  (ATR SL/TP, SQLite persistence)
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
         RiskManager          TelegramNotifier
     (circuit breaker)       (entry / exit alerts)
              │
    ┌─────────┴──────────┐
    ▼                    ▼
app.py (PyQt5 GUI)   daemon.py (headless)
         │
    terminal_dashboard.py  (Textual TUI)
    scalper.py             (keyboard scalper)
```

---

## Project Structure

```
topoalpha/
├── app.py                    GUI dashboard  (PyQt5 · pyqtgraph · matplotlib 3-D)
├── daemon.py                 Headless server mode
├── terminal_dashboard.py     Full-featured Textual TUI  (manual + auto trading)
├── scalper.py                Keyboard-driven paper scalper  (Textual · 1× leverage)
├── trade_calculator.py       Standalone step-by-step analysis panel  (PyQt5)
├── topo_dyn_predictor.py     Advanced signal engine  (TDA + Dynamical Systems Theory)
├── wfo.py                    Walk-forward optimisation over parameter grid
├── data_feeder.py            Binance OHLCV + order-book feed  (ccxt)
├── tda_core.py               Topological stress via Ripser
├── ml_model.py               TopoBooster ensemble + phase-transition detector
├── paper_trader.py           Simulated execution + SQLite state persistence
├── binance_executor.py       Live Binance Futures executor  (stub — disabled)
├── risk_manager.py           Circuit breaker + trade journal
├── notifier.py               Telegram alerts
└── .env                      API credentials  (not committed)
```

---

## Requirements

```
Python 3.11+
```

```bash
pip install pyqt5 pyqtgraph matplotlib numpy pandas scikit-learn \
            ccxt python-dotenv ripser requests python-binance
```

---

## Configuration

Create `.env` in the project root:

```env
# Telegram alerts (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Binance Testnet (only required for live execution)
BINANCE_DEMO_API_KEY=your_key
BINANCE_DEMO_SECRET_KEY=your_secret
```

No exchange keys are required for paper trading — the data feed uses Binance's public endpoints via ccxt.

---

## Running

### GUI dashboard
```bash
python app.py
```
Four panes rendered side-by-side:

- **Live candlestick chart** — entry markers (▲ long / ▼ short), SL/TP lines, horizon cutoff
- **Topological stress** — live $\sigma$ series with adjustable threshold slider
- **Order Book Imbalance** — bid/ask volume ratio in $[-1, +1]$
- **3-D phase space** — delay-embedded point cloud coloured by time

### Headless daemon
```bash
python daemon.py
```
Identical signal logic, no GUI dependency. Suitable for `screen` / `tmux` / `systemd`.

### Rich terminal dashboard
```bash
python terminal_dashboard.py
```
Full-width live dashboard in any modern terminal using the Rich library.

### Trade Calculator
```bash
python trade_calculator.py
```
Loads 300 bars automatically and prints a numbered step-by-step breakdown:
market context → indicators → HTF trend → TDA → ML → composite score → direction → SL/TP → sizing → EV → Kelly.

### Terminal Scalper
```bash
python scalper.py
```
Keyboard-driven paper trading in any terminal. No ML, no automation — pure manual execution with SQLite persistence and a live plotext chart. Bindings: `L` long · `S` short · `C` close · `Q` quit.

### Advanced Signal Engine (standalone)
```bash
python -c "
from topo_dyn_predictor import TopoDynPredictor
import numpy as np
p = TopoDynPredictor()
prices = np.random.randn(300).cumsum() + 50000
print(p.signal(prices))
"
```
`TopoDynPredictor` can be imported into any script. It returns a `TopoDynSignal` dataclass with 14 indicators: Hurst (R/S + DFA), Lyapunov exponent, permutation entropy, sample entropy, correlation dimension, H₁ stress + persistence entropy + landscape L¹, RQA determinism + entropy, EWS score, and multifractal Δh.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SYMBOL` | `BTC/USDT` | Trading pair |
| `TIMEFRAME` | `5m` | Entry timeframe |
| `HTF` | `1h` | Macro context timeframe |
| `HORIZON_BARS` | `6` | Max bars per trade |
| `RETRAIN_EVERY` | `150` | Ticks between ML re-fits |
| `alpha_stress_threshold` | `1.5` | Base topological stress gate |
| `sl_atr_mult` $(\lambda_{\text{SL}})$ | `1.5` | SL distance in ATR units |
| `tp_atr_mult` $(\lambda_{\text{TP}})$ | `3.0` | TP distance in ATR units |
| `margin_usdt` | `50` | Margin per trade (USDT) |
| `leverage` | `10×` | Futures leverage |
| `ENTRY_PROB_THRESHOLD` | `0.60` | Minimum ML gate probability |

The stress threshold is also adjustable live from the GUI slider without restart.

---

## Design Decisions

**Why paper trading by default?**  
Live execution adds exchange-API complexity without changing the signal logic. The paper engine is structurally identical to what a live system would do — the only absent layer is the HTTP order. `binance_executor.py` provides a drop-in live executor when you are ready.

**Why SQLite?**  
Position state survives restarts. If the daemon crashes mid-trade, position, entry price and entry timestamp are restored from the database on the next launch.

**Why background threads for ML retrain?**  
Three `CalibratedClassifierCV(cv=3)` models on 500 rows takes several seconds. Retraining on the main thread freezes the UI or the poll loop. All retrains run in a daemon thread protected by a re-entrancy lock.

**Why a 4-second timeout for Ripser?**  
On degenerate or near-constant point clouds the Vietoris–Rips filtration can hang indefinitely. A single-worker `ThreadPoolExecutor` caps the computation at $4\,\text{s}$ and returns $\sigma = 0$ on timeout, keeping the main loop responsive.

**Why isotonic calibration?**  
Gradient boosting and tree ensembles produce poorly calibrated probabilities by default. Isotonic regression on held-out folds corrects the probability scale so that $\hat{p} = 0.60$ actually means the model is right ~60% of the time — critical for honest Kelly sizing.

---

## Telegram Alerts

Create a bot via [@BotFather](https://t.me/BotFather) and retrieve your chat ID from [@userinfobot](https://t.me/userinfobot). You will receive:

| Event | Emoji |
|-------|-------|
| Engine started | 🟢 |
| Long entry | 🚀 |
| Short entry | 🩸 |
| Trade closed in profit | ✅ |
| Trade closed at stop-loss | 🛑 |
| Circuit breaker tripped | 🔴 |
| Engine stopped | 🔴 |

---

## License

MIT