# Regime-Based Trading Framework (NIFTY 50)

A market-regime detection and adaptive-positioning system built on top of the NIFTY 50 index and its constituents. Instead of trying to predict tomorrow's price — a notoriously hard problem — this project tries to answer a more tractable question:

> **What *kind* of market are we in right now, and how much risk should I be taking because of it?**

The whole framework is built around that idea. It reads the "mood" of the market from breadth and volatility, labels each day with a regime (bull/bear × calm/turbulent), trains a separate model to recognise each regime, and then sizes exposure up or down depending on which regime the models think we're in. The payoff isn't beating the market on raw return — it's getting *similar* returns with meaningfully smaller drawdowns.

---

## Why regimes instead of price prediction?

Markets don't behave the same way all the time. A trend-following idea that prints money in a calm bull market gets shredded in a choppy, high-volatility bear market. The signals that matter also change: in a quiet uptrend, broad participation (lots of stocks above their moving averages) is what carries you; in a panic, sudden volatility spikes and clustered extreme moves are what you need to react to.

So rather than build one model and hope it generalises across every environment, this project does something closer to how a discretionary trader actually thinks:

1. **Classify the environment** (the regime).
2. **Use a specialist for each environment** (one model per regime).
3. **Adjust position size to the environment's risk** (the strategy layer).

This is the core thesis the notebook sets out to demonstrate: *different regimes are governed by different signals, and an adaptive strategy should respect that.*

---

## The data: where it comes from and how it's built

Everything downstream depends on two raw inputs, both pulled live from Yahoo Finance via `yfinance`.

### 1. The index itself (`^NSEI`)
The NIFTY 50 index OHLC series. This is the price series the strategy actually trades and the series regimes are derived from. It's downloaded with `auto_adjust=True` so corporate actions are already baked in.

### 2. The 50 constituents (market breadth)
This is the more interesting half. The list of 50 constituent tickers is fetched directly from the **official NSE archive** (`ind_nifty50list.csv`), with a local-file fallback in case the live link is unreachable. Each symbol gets a `.NS` suffix so Yahoo recognises it (e.g. `RELIANCE.NS`).

The framework then scans every one of those 50 stocks, day by day, and asks a handful of yes/no questions about each:

- Is it trading **above its 20-day moving average?** Above its **50-day**?
- Did it move **±4.5% today** (a notable single-day move)?
- Did it move **±20% over the last 5 days** (an extreme multi-day move)?

Aggregating those per-stock answers across all 50 names for each date produces the **daily market breadth summary** — the real heart of the dataset. The columns are:

| Column | Meaning |
|---|---|
| `Total_Stocks_Scanned` | How many names had valid data that day (≈50) |
| `Stocks_Above_MA20` / `Stocks_Below_MA20` | Short-term trend participation |
| `Stocks_Above_MA50` / `Stocks_Below_MA50` | Medium-term trend participation |
| `Stocks_Up_4.5pct_Today` / `Stocks_Down_4.5pct_Today` | Daily momentum extremes |
| `Stocks_Up_20pct_5Days` / `Stocks_Down_20pct_5Days` | Multi-day momentum extremes |

**Why breadth matters:** the index level alone can lie. A handful of mega-cap names can drag the index up while the *majority* of stocks quietly roll over — a fragile rally. Breadth catches that. When the index is rising *and* 40 of 50 stocks are above their 20-day MA, that's a healthy, broad-based trend. When the index is rising but only 15 stocks are participating, that's the kind of narrow leadership that tends to precede trouble. The whole feature set is built to surface this "how many stocks actually agree with the index?" signal.

The scan runs across a **3-year window (2023-01-01 → 2026-01-01)**. That window length is deliberate and important — more on why below.

---

## Defining regimes (building the labels)

Regimes are derived purely from the index price series in `define_market_regimes()`. The logic:

1. **Compute two raw descriptors** over a 20-day lookback:
   - **Rolling volatility** — the standard deviation of returns (the "is it turbulent?" axis).
   - **Trend** — the 20-day percentage change (the "which direction?" axis).

2. **Normalise them with rolling z-scores.** Each descriptor is compared against its own **90-day rolling mean and standard deviation**. This is the key design choice: a z-score asks *"is volatility unusually high **relative to recent history**?"* rather than against a fixed threshold. It lets the regime definitions adapt as the market's baseline shifts over time — what counted as "high vol" in a sleepy year is different from a crisis year.

3. **Threshold the z-scores at ±0.5** to get four raw boolean regimes:
   - `High_Volatility` (vol z-score > 0.5), `Low_Volatility` (< −0.5)
   - `Bull_Market` (trend z-score > 0.5), `Bear_Market` (< −0.5)

4. **Apply a persistence filter.** Raw thresholds flicker — one noisy day can flip a label. To stop that whipsawing, the "stable" regimes (`Stable_Bull`, `Stable_Bear`, `Stable_Low_Vol`) require the condition to hold **on at least 4 of the last 5 days** before they fire. This is what turns a jumpy signal into something you'd actually be willing to trade on.

The result is a per-day set of binary regime flags. These become the **target labels** the models learn to predict.

---

## Feature engineering (building the predictors)

`create_regime_features()` turns the raw breadth + price data into **19 features** that describe the *state* of the market. They fall into a few families:

**Breadth-strength signals**
- `MA20_Strength` — fraction of stocks above their 20-day MA (broad participation).
- `Market_Momentum` — net balance: (above MA20 − below MA20) / total. Distinguishes a real trend from a fragile one.
- `Advance_Decline_Ratio` — a smoothed up-movers vs down-movers ratio (the `+1`s avoid divide-by-zero).
- `Participation_Rate` — how many stocks are *actively moving* at all, a proxy for volatility building beneath the surface.

**Extreme-move / tail signals**
- `Extreme_Move_Ratio` — share of stocks making ±20%-in-5-days moves.
- `Extreme_Momentum_Bias` — directional tilt of those extreme moves (up-extremes vs down-extremes).
- `Extreme_Volatility_Indicator` — recent (5-day) vs baseline (20-day) extreme-move activity; detects *acceleration* of tail behaviour.

**Price-based signals**
- `Price_Volatility_Ratio` — short-term vs longer-term realised volatility of the index.
- `Trend_Strength` — index price relative to its own 20-day MA.

**Derived dynamics**
- 6 **lagged (`_lag1`) copies** of the core breadth features (yesterday's reading as its own feature).
- `Momentum_Change_5d`, `Breadth_Acceleration`, `Extreme_Momentum_Change`, `Extreme_Move_Acceleration` — rates of change and accelerations, to capture *transitions* rather than just levels.

### Two things every feature respects

1. **Look-ahead safety.** The entire feature frame is `.shift(1)`-ed. A feature available for "today's" decision only ever uses data that was actually known by the *previous* close. Without this, the backtest would be quietly cheating by peeking at the future.

2. **Skew and outlier control** (in the normalisation cell): heavily skewed ratio features (`Advance_Decline_Ratio`, `Extreme_Move_Ratio`, `Extreme_Momentum_Bias` and their lags) are `log1p`-transformed to tame fat tails, and the noisy acceleration features are clipped to `[-3, 3]` so a single freak day can't dominate what the model learns.

---

## The models (one specialist per regime)

Rather than a single multi-class classifier, the framework trains **one binary Random Forest per regime** (`train_all_regime_models`). Each model answers a single focused question: *"Given today's features, is regime X active or not?"*

Key modelling choices:

- **`RandomForestClassifier`** with `n_estimators=100`, `max_depth=10`, `class_weight="balanced"`. The balanced class weighting matters because regimes are imbalanced — bear-market days are far rarer than neutral ones, and without it the model would just learn to always say "no."
- **Time-aware validation.** Splitting is done with `TimeSeriesSplit` (walk-forward CV), never a random shuffle. In time series, randomly mixing past and future into train/test leaks information; walk-forward keeps the test set strictly *after* the training set, the way real trading works.
- **A held-out final test set** (the last 30% of history) that the CV never touches, for an honest out-of-sample read.
- **Graceful degeneracy handling.** If a regime never fires inside a training fold (only one class present), that fold is skipped rather than crashing — important on shorter windows where rare regimes might be absent from a slice.

Beyond raw accuracy, the notebook reports **precision / recall / F1 per regime** (because for a rare bear signal, catching it matters more than overall accuracy), and a **feature-importance table** showing which signals each regime leans on. The recurring story there: trend and breadth-strength features dominate everywhere, while the acceleration/extreme features earn their keep specifically in high-volatility and transition regimes.

---

## The strategies (turning predictions into positions)

The model predictions feed two distinct trading strategies, both backtested with realistic transaction costs (`cost_rate=0.0005` per unit of turnover, ₹1,000,000 starting capital).

### 1. Four-Regime Adaptive Strategy
Continuously scales exposure (0 to 1) based on the combination of trend and volatility regime:

| Regime | Exposure |
|---|---|
| Low-Vol Bull (the ideal: calm uptrend) | 1.0 |
| High-Vol Bull (uptrend but jumpy) | 0.9 |
| Low-Vol Bear (orderly decline) | 0.6 |
| High-Vol Bear (the danger zone) | 0.2 |
| Neutral (default) | 0.9, but capped to 0.6 inside a bear market |

High-volatility regimes take priority when assigning the daily label, on the principle that risk management trumps everything.

Three refinements make the exposure tradeable rather than twitchy:
- **Sticky positions** — exposure is held fixed for the duration of an unbroken regime run (implemented vectorised via a block-id `groupby`, not a Python loop), so you're not re-trading on every tiny flicker.
- **Rolling smoothing** — a 5-day average further softens transitions.
- **Execution lag** — positions are `.shift(1)`-ed so you only ever act on yesterday's signal (no look-ahead), seeded with `bfill` rather than an arbitrary constant.

### 2. Crash-Hedge (Two-Regime) Strategy
A simpler tail-protection overlay: stay fully invested (1.0) in normal conditions, trim to 0.7 in any stable bear regime, and de-risk hard to **0.3** during *crash-like* conditions (confirmed bear **and** stable bear **and** high volatility firing together). Same look-ahead-safe execution lag.

---

## Evaluation (judging it honestly)

`evaluate_clean()` computes a full tear sheet, and crucially does it on **excess returns over a risk-free rate** (`rf_annual=0.06`, a sensible Indian short-rate), not raw returns — otherwise the Sharpe ratio flatters everything in a positive-rate environment. The metrics:

- **Total Return, CAGR** — headline growth.
- **Max Drawdown** — the single most important number for this project, since drawdown control is the whole point.
- **Sharpe** — excess return per unit of total volatility.
- **Sortino** — excess return per unit of *downside* volatility only (it doesn't punish you for upside swings).
- **Calmar** — CAGR ÷ worst drawdown. This is the metric that most directly rewards what the strategy optimises for: smooth, drawdown-aware growth.
- **Win Rate, Average Position, Annualised Volatility** — supporting diagnostics.

Everything is plotted against a **Buy & Hold benchmark** on an interactive Plotly tear sheet (equity curve, drawdown comparison, and exposure over time). The expected pattern: comparable-to-slightly-lower total return, but a visibly shallower drawdown profile and a better Calmar — i.e. a smoother ride for similar destination.

---

## Important nuances & gotchas (read this before you run it)

This section captures the non-obvious things that were learned building and debugging this on the NIFTY 50 universe specifically.

**1. Index alignment.** The downloaded index series and the breadth summary must share a real `DatetimeIndex`. The index download resets its index (moving `Date` to a column), so the price series is explicitly re-indexed by `Date` before being intersected with the breadth dates. Skip that and the date↔integer index mismatch produces *zero* overlapping dates and a silently empty feature set.

**2. Sparse extreme-move features on a small universe.** With only 50 stocks, a ±20%-in-5-days move is genuinely rare — `Extreme_Move_Ratio` is **0 on the large majority of days**. That makes its derived ratio (`Extreme_Volatility_Indicator`) a 0÷0 = undefined value on most rows. The framework treats "no extreme activity" as a neutral **0** for these structurally-sparse features instead of dropping the row — otherwise you'd discard almost the entire sample. (This is the difference from a NIFTY *500* version, where thousands of stocks make extreme moves common and these features are well-defined.) Relatedly, don't be alarmed that `Participation_Rate`, `Extreme_Move_Ratio`, and `Extreme_Momentum_Bias` show lots of zeros — that's real and correct for a 50-stock breadth, not a bug. The trend and breadth-strength features carry the actual signal.

**3. Infinities and `log1p`.** Ratio features can divide by zero, and `log1p(-1)` is `-inf`. Both are guarded (inf → NaN before the warm-up drop; `log1p` input clipped just above −1) so scikit-learn never sees a non-finite value at `.fit()`.

**4. Why a 3-year window.** The regime z-scores use a **90-day rolling baseline**. On a 1-year window, that baseline eats most of the sample before any regime can even be classified, leaving too little usable data — the result collapses to "all Neutral", flat metrics, and a degenerate plot. The 3-year window gives the baseline enough history to classify regimes properly, lets the walk-forward CV actually run, and produces a realistic mix of bull/bear/high-vol transitions for the models to learn from. If you want even richer regime behaviour, widen it further in the data-window cell.

---

## How to run

1. Install dependencies:
   ```bash
   pip install yfinance pandas numpy scikit-learn matplotlib seaborn plotly openpyxl
   ```
2. Open the notebook and run **top to bottom**. The first cells pull live data from Yahoo Finance and the NSE archive, so an internet connection is required. The breadth scan across 50 tickers takes a little time on the first run.
3. Generated artefacts (CSV breadth summaries, the pickled regime models, the signals file) are written to the working directory and are safe to delete — they're recreated on the next run.

## Project structure (after a run)

```
Updated_Regime_based_project.ipynb   # the full pipeline
nifty50_ohlc.csv                     # index OHLC (generated)
all_stock_data.csv                   # per-stock scan results (generated)
daily_summary.csv                    # daily breadth summary (generated)
market_regime_models.pkl             # trained per-regime models (generated)
market_regime_signals.csv            # model regime predictions (generated)
```

---

## In one sentence

It reads the market's breadth and volatility, decides what regime we're in, lets a regime-specific model confirm it, and dials risk up or down accordingly — trading a little bit of return for a much smoother drawdown profile than simply buying and holding.
