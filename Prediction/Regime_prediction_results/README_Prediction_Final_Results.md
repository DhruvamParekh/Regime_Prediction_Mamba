# Regime Prediction — Final Results Guide (Causal-Confirmed)

This folder contains the final curated regime prediction results, one
subfolder per stock, plus cross-stock summary files at the root.

**This is a genuine prediction task.** The model is trained to forecast the
regime **7 trading days ahead** (forward return window), using only
information available up to the prediction date — so a call made on date
`T` is a forecast of the regime around `T+7`, not a same-date classification.

Train / Val / Test split used everywhere below:
- **Train**: up to 2022-12-31
- **Val**: 2023-01-01 to 2023-12-31
- **Test**: 2024-01-01 onwards

---

## Per-stock files (inside each stock's subfolder)

### CSVs

**`full_predictions.csv`** — The core output. One row per prediction point
(every 7 trading days), covering the entire history, not just the test
period.
- `Date` — the date the prediction was made
- `true_idx` / `true_label` — the actual regime that materialized ~7 days later (0=Bearish, 1=Neutral, 2=Bullish)
- `pred_idx` / `pred_label` — the model's forecast for that future regime
- `p_bearish` / `p_neutral` / `p_bullish` — predicted class probabilities
- `confidence` — max probability across the three classes

**`period_accuracy.csv`** — Forecast accuracy broken down by Train/Val/Test.
- `overall_acc` — % of forecasts matching the regime that actually occurred
- `Bullish_acc`, `Neutral_acc`, `Bearish_acc` — per-class recall
- `Bullish_n`, `Neutral_n`, `Bearish_n` — sample counts per class

**`regime_duration.csv`** — How long regimes actually lasted vs. how long the
model's forecast regimes lasted, per class.
- `true_avg_days` / `pred_avg_days` — average duration of a regime "run"
- `true_count` / `pred_count` — number of separate regime runs

**`period_returns.csv`** — Return the model would have earned per period if
it only held the stock when its forecast said Bullish, vs. Buy & Hold.
- `model_return_pct`, `bnh_return_pct`, `alpha_pct`
- `days_in_market` / `days_total` / `pct_in_market`

**`regime_returns.csv`** — Daily-return stats broken down by which regime the
model forecast, per period.
- `pred_regime`, `n_days`, `avg_daily` (%), `cum_return` (%), `win_rate` (%)

**`backtest_metrics.csv`** — The main test-period backtest, and the single
most important file if you're using this stock's predictions downstream (e.g.
in the rolling portfolio). Compares three strategies: **Model** (long when
the forecast regime is Bullish, else cash), **Perfect Foresight** (long when
the *true* regime was Bullish — theoretical ceiling), and **Buy and Hold**.
- `total_ret_pct`, `annual_ret_pct`, `volatility_pct`, `sharpe`, `max_drawdown`, `win_rate_pct`, `days_in_market`

**`feature_importance.csv`** — Which input features the model relied on most
for making its forecast.
- `feature`, `importance` (raw), `rank`, `importance_pct` (normalized to 100%)

### Charts

**`{stock}_full_regime_chart.png`** — Price chart with the full forecast
regime history overlaid (train/val/test shaded), plus val/test accuracy in
the title.

**`{stock}_analysis.png`** — 2×2 panel: period accuracy, regime duration
comparison (true vs. forecast), period returns, and regime returns.

**`{stock}_confusion_matrices.png`** — Confusion matrices for Val and Test,
showing exactly where the model confuses Bullish/Neutral/Bearish forecasts.

**`{stock}_backtest.png`** — Cumulative return curves for Model vs. Perfect
Foresight vs. Buy & Hold over the test period, with a position (long/cash)
strip underneath. Visual companion to `backtest_metrics.csv`.

**`{stock}_nov2025_prediction.png`** — The model's most recent forecast:
current regime, the regime it's forecasting ~7 days out, whether a transition
is flagged, and class probabilities. This is the latest signal, genuinely
forward-looking.

---

## Cross-stock files (folder root)

**`backtest_summary.csv`** — `backtest_metrics.csv` for every stock, stacked
into one table, for comparing strategies across the whole universe.

**`cross_stock_summary.csv`** / **`.png`** — One row per stock: val/test
accuracy, backtest returns, Sharpe, max drawdown, top feature. Fastest way
to see which stocks the model forecasts well on.

**`cross_stock_feature_importance.csv`** — Feature importance averaged
across all stocks.

---

## Suggested reading order

1. `cross_stock_summary.csv` — lay of the land across all stocks
2. Pick a stock → `{stock}_full_regime_chart.png` — see what it forecast over time
3. `{stock}_backtest.png` + `backtest_metrics.csv` — did acting on the forecast make money
4. `{stock}_confusion_matrices.png` + `period_accuracy.csv` — was the forecast actually accurate, or lucky
5. `{stock}_nov2025_prediction.png` — what's it forecasting right now

---

## Where this feeds downstream

`full_predictions.csv` (per-date forecast + probabilities) and
`backtest_metrics.csv` (realized strategy returns) from every stock here are
the two files consumed by the rolling portfolio pipeline. If you're tracing
a rolling-portfolio result back to its source, start with these two.
