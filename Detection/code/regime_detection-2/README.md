# Regime Detection — Run 5 Results Guide

This folder contains the curated results for Run 5 (pseudo-labels, 19 stocks).
Each stock has its own subfolder named `{STOCK}_PSEUDO_Run5`, containing the
same 12 files. Global cross-stock summary files sit at the root.

**Note on terminology:** this is a regime *detection* system, not a forward
price-prediction system — the model is classifying which regime (Bullish /
Neutral / Bearish) a given date belongs to. The column/file names below still
say `pred_*` / `prediction` etc. because that's what's literally in the code
and CSVs, but read every instance of "predicted" as "detected" — it's the
model's regime call for that date, not a forecast of future price.

Train / Val / Test split used everywhere below:
- **Train**: up to 2022-12-31
- **Val**: 2023-01-01 to 2023-12-31
- **Test**: 2024-01-01 onwards

---

## Per-stock files (inside each `{STOCK}_PSEUDO_Run5/` folder)

### CSVs

**`full_predictions.csv`** — The core output. One row per detection point
(every 7 trading days, per `prediction_freq`), covering the entire history,
not just the test period.
- `Date` — detection date
- `true_idx` / `true_label` — the pseudo-label regime at that date (0=Bearish, 1=Neutral, 2=Bullish)
- `pred_idx` / `pred_label` — the model's detected regime (column name says "pred", but it's a same-date classification, not a future forecast)
- `p_bearish` / `p_neutral` / `p_bullish` — detected class probabilities
- `confidence` — max probability across the three classes

**`period_accuracy.csv`** — Accuracy broken down by Train/Val/Test.
- `overall_acc` — % of detections matching the true label
- `Bullish_acc`, `Neutral_acc`, `Bearish_acc` — per-class recall (accuracy only on rows where that was the true regime)
- `Bullish_n`, `Neutral_n`, `Bearish_n` — sample counts per class

**`regime_duration.csv`** — How long regimes actually lasted vs. how long the
model's detected regimes lasted, per class.
- `true_avg_days` / `pred_avg_days` — average duration of a regime "run" (true vs. detected)
- `true_count` / `pred_count` — number of separate regime runs

**`period_returns.csv`** — Return the model would have earned per period if
it only held the stock while its regime detection said Bullish, vs. Buy & Hold.
- `model_return_pct` — return while in market per model's detected calls
- `bnh_return_pct` — Buy & Hold return over the same period
- `alpha_pct` — model return minus Buy & Hold
- `days_in_market` / `days_total` / `pct_in_market`

**`regime_returns.csv`** — Daily-return stats broken down by which regime the
model detected, per period.
- `pred_regime` — the detected regime (naming holdover, not a forecast)
- `n_days`, `avg_daily` (%), `cum_return` (%), `win_rate` (%)

**`backtest_metrics.csv`** — The main test-period backtest. Compares three
strategies: **Model** (long when regime detected as Bullish, else cash),
**Perfect Pseudo Foresight** (long when the *true* label was Bullish — the
theoretical ceiling), and **Buy and Hold**.
- `total_ret_pct`, `annual_ret_pct`, `volatility_pct`, `sharpe`, `max_drawdown`, `win_rate_pct`, `days_in_market`

**`feature_importance.csv`** — Which input features the model relied on most
for detecting the regime (gradient-based importance, averaged over training batches).
- `feature`, `importance` (raw), `rank`, `importance_pct` (normalized to 100%)

### Charts

**`{stock}_full_regime_chart.png`** — Price chart with the full detected
regime history overlaid (train/val/test shaded), plus val/test accuracy in
the title. Best single chart for "what regime did the model actually detect,
over time."

**`{stock}_analysis.png`** — 2×2 panel: period accuracy, regime duration
comparison (true vs. detected), period returns, and regime returns —
visual companion to the four CSVs above.

**`{stock}_confusion_matrices.png`** — Confusion matrices for Val and Test,
showing exactly where the model confuses Bullish/Neutral/Bearish.

**`{stock}_backtest.png`** — Cumulative return curves for Model vs. Perfect
Foresight vs. Buy & Hold over the test period, with a position (long/cash)
strip underneath. Visual companion to `backtest_metrics.csv`.

**`{stock}_nov2025_prediction.png`** — Most recent regime detection: current
regime, next regime the model detects a shift toward, whether a transition is
flagged, and class probabilities. Despite the filename, this is the "latest
detected signal" chart, not a price forecast.

---

## Cross-stock files (folder root)

**`backtest_summary.csv`** — `backtest_metrics.csv` for every stock, stacked
into one table, for comparing strategies across the whole universe.

**`cross_stock_summary_PSEUDO_Run5.csv`** / **`.png`** — One row per stock:
val/test accuracy, backtest returns, Sharpe, max drawdown, top feature. The
fastest way to see which stocks the model detects regimes well on.

**`cross_stock_feature_importance_PSEUDO_Run5.csv`** — Feature importance
averaged across all stocks, to see which inputs matter for regime detection
in general vs. just for one ticker.

---

## Suggested reading order

1. `cross_stock_summary_PSEUDO_Run5.csv` — get the lay of the land across all 19 stocks
2. Pick a stock of interest → `{stock}_full_regime_chart.png` — see what regime it actually detected over time
3. `{stock}_backtest.png` + `backtest_metrics.csv` — did acting on the detected regime make money
4. `{stock}_confusion_matrices.png` + `period_accuracy.csv` — was the detection actually accurate, or lucky
5. `{stock}_nov2025_prediction.png` — what regime is it detecting right now

---

## What's *not* in this folder (intentionally excluded)

- `predictions.csv` — test-set-only detections; superseded by `full_predictions.csv`
- `training_history.csv` — epoch-by-epoch training loss/accuracy curves; useful only if debugging training itself
- `true_regime_runs.csv` / `pred_regime_runs.csv` — raw list of every individual regime run; already summarized in `regime_duration.csv`

These still exist in the original `Run5 all results` folder if needed later.
