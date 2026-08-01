# Rolling Portfolio — Final Results Guide

This folder contains the final curated results for the rolling portfolio
pipeline, split into two subfolders:

- **`meta_model_results/`** — the "trust" classifier that decides how much to
  rely on each stock's regime prediction, including every candidate model
  that was trained and compared, not just the one that got deployed.
- **`portfolio_results/`** — the actual rolling portfolio built using that
  meta-model, its equity curve, rebalancing history, and performance
  breakdowns.

The meta-model's job: for every regime prediction made (from the prediction
pipeline), estimate the probability that the prediction will turn out to be
*correct*. The rolling portfolio then uses that trust score to decide which
stocks to hold and how much weight to give each one at every rebalance.

---

## `meta_model_results/`

**`model_comparison.csv`** — Every candidate model trained (Gradient
Boosting, Random Forest, Logistic Regression), side by side.
- `model` — model name
- `train_acc` / `test_acc` — accuracy at predicting whether a regime call was correct
- `auc` — ROC-AUC on the test set (the metric used to pick the deployed model)

**`{model_name}_model.pkl`** — Each trained candidate model, saved
individually with its scaler and feature list, so any of them can be
reloaded and compared or swapped in later.
- Contains: `model`, `scaler`, `feature_cols`, `train_acc`, `test_acc`, `auc`

**`{model_name}_feature_importance.csv`** — What each model relied on to
judge whether a prediction was trustworthy (tree-based importance for
Gradient Boosting / Random Forest, absolute coefficient magnitude for
Logistic Regression).
- `feature`, `importance`

**`test_probabilities_all_models.csv`** — Every model's predicted
probability-of-correctness on the same held-out test rows, plus the actual
outcome, so models can be compared prediction-by-prediction.
- one column per model + `y_test` (1 = the regime call was actually correct)

**`meta_model.pkl`** — The deployed model: the single best-performing
candidate (highest AUC), which is what the rolling portfolio actually calls
at each rebalance.
- Contains: `model`, `scaler`, `feature_cols`, `threshold`, `model_name`

**`meta_model_trust.csv`** — Per-stock average trust score from the deployed
model, compared against how accurate that stock's predictions actually were.
- `stock`, `avg_trust`, `actual_acc`, `gap` (trust minus actual — flags
  stocks the model is over- or under-confident about)

---

## `portfolio_results/`

**`rolling_equity.csv`** — The day-by-day equity curve for each portfolio
strategy tested, indexed by date. This is the master output — every return
number elsewhere in this folder is derived from it.

**`rolling_metrics.csv`** — Summary performance metrics (return, volatility,
Sharpe, drawdown, etc.) for each strategy in `rolling_equity.csv`.

**`rolling_rotation_log.csv`** — The full rebalancing history: which stocks
were held after each review date, and why.
- `date`, `stock`, `acc_score`, `conf_score`, `acc_weight`

**`sectionA_metrics.csv`** / **`sectionB_metrics.csv`** — Performance metrics
for two portfolio construction approaches evaluated over the backtest
period, for direct comparison.

**`portfolio_apr30.csv`** / **`portfolio_may22.csv`** — Point-in-time
portfolio snapshots: exactly which stocks were selected as of that review
date, and how long that selection is valid for.
- includes `valid_from`, `review_on`

**`block_performance.csv`** — The backtest period split into consecutive
review blocks (each block = the time between two rebalances), with
performance computed separately per block. Useful for seeing whether returns
were driven by a few strong stretches or were consistent throughout.

**`rolling_portfolio.png`** — The main equity curve chart: portfolio value
over time across all strategies tested.

**`period_returns.png`** — Bar chart of returns by period, for a quicker
read than scrolling through `rolling_metrics.csv`.

---

## Suggested reading order

1. `meta_model_results/model_comparison.csv` — which trust model won, and by how much
2. `meta_model_results/meta_model_trust.csv` — which stocks the deployed model trusts most, and whether that trust is justified
3. `portfolio_results/rolling_portfolio.png` + `rolling_metrics.csv` — how the resulting portfolio actually performed
4. `portfolio_results/rolling_rotation_log.csv` — what was actually held at each point in time
5. `portfolio_results/block_performance.csv` — whether performance was consistent or concentrated in a few periods
