# Regime Prediction (Phase 2) — NSE Stock Regime Classifier

> ⚠️ **GPU REQUIRED.** This project uses **Mamba SSM** (`mamba-ssm`),
> which only runs on an **NVIDIA GPU with CUDA 12.6**. It will fail
> to import — let alone train — on CPU. Run `setup_env.sh` on a GPU
> runtime (e.g. Colab with a GPU runtime type) before doing anything
> else.

## What this project does

Predicts the **market regime** (Bullish / Neutral / Bearish) for
individual NSE-listed stocks on a weekly basis, using a deep learning
model built on **Mamba SSM** — a modern, efficient alternative to
Transformers for long time-series.

The pipeline has three phases:

1. **Label generation** — raw OHLCV price data is converted into
   regime labels using two strategies: forward-looking "pseudo"
   labels (clean ground truth, needs future data) and "causal
   confirmed" labels (zero future data, fully tradeable — this is
   the label source actually used for training).
2. **Model training** — a separate `RegimePredictionMamba` model is
   trained per stock. It takes a 60-day lookback window of 13
   technical + macro features and predicts the dominant regime for
   the next 7 days.
3. **Evaluation & inference** — trained models are evaluated on a
   held-out test set, a full prediction history is generated, and
   period accuracy, regime-duration quality, backtest returns
   (Model vs Perfect Foresight vs Buy & Hold), and specific-date
   predictions are computed. A cross-stock summary consolidates
   everything.

This repo contains both the **regime prediction pipeline** (Phase 2)
and the **rolling portfolio** (Phase 3), which reads each stock's
`full_predictions.csv` and simulates a periodically-rebalanced
portfolio on top of the regime predictions — see "Rolling portfolio"
below.

## Folder structure

```
regime_prediction/
│
├── README.md              ← this file
├── requirements.txt        ← pip-installable dependencies
├── setup_env.sh            ← CUDA 12.6 + mamba-ssm install (GPU only, run first)
│
├── config.py                ← ALL tuneable constants (paths, model/train
│                                cfg, split dates, label cfgs, regime maps)
│
├── data/
│   ├── loader.py            ← load_stock_csv, load_macro, load_nifty
│   └── features.py          ← compute_features() — the 13 model features
│
├── labels/
│   ├── pseudo.py            ← forward-looking pseudo label generation
│   ├── causal.py            ← causal-confirmed label generation (used for training)
│   └── regime_loader.py     ← label-source dispatcher + prediction-label builder
│
├── dataset.py               ← RegimePredictionDataset (PyTorch Dataset)
├── model.py                 ← MambaBlock, RegimePredictionMamba
├── train.py                 ← training loop, class weights, feature importance
├── inference.py             ← full prediction history, specific-week prediction
├── evaluate.py               ← period accuracy, regime quality, returns-by-regime
├── backtest.py               ← Model vs Perfect Foresight vs Buy & Hold backtest
├── visualise.py               ← every chart function used across the pipeline
├── summary.py                 ← cross-stock summary table + feature importance
│
└── main.py                    ← END-TO-END RUNNER — imports everything above,
                                   contains no modelling logic of its own
```

## Setup

```bash
# Step 1 — one-time GPU environment setup (installs CUDA 12.6,
# PyTorch built for cu126, causal-conv1d, mamba-ssm==2.3.1)
bash setup_env.sh

# Step 2 — restart the runtime (Colab: Runtime → Restart runtime)

# Step 3 — run the full pipeline end to end
python main.py
```

`main.py`'s internal order:

```
1. Mount Google Drive, create output folders
2. Load shared data (macro, NIFTY)
3. For each stock in config.STOCKS_TO_TRAIN:
     a. load CSV
     b. compute features
     c. generate / load labels (pseudo + configured LABEL_SOURCE)
     d. assemble + split (Train/Val/Test)
     e. build datasets
     f. train model
     g. evaluate on test set
     h. run full prediction history
     i. plot regime chart
     j. run period accuracy / regime quality / returns analysis
     k. run backtest
4. Run cross-stock summary
5. Run the specific-week prediction (config.TARGET_DATE)
6. Save all CSVs and charts
```

## How to configure

Everything tuneable lives in **`config.py`** — nowhere else. In
particular:

- `STOCKS_TO_TRAIN` — which NSE tickers to run the pipeline on
- `LABEL_SOURCE` — `'causal_confirmed'` (production), `'pseudo'`, or
  `'detected'`
- `RUN_ID` — namespaces checkpoints/results so experiments don't
  collide
- `SPLIT_DATES` — Train/Val/Test date boundaries
- `MODEL_CFG` / `TRAIN_CFG` — architecture and training hyperparameters
- `BASE` — the root Google Drive path; change this one line to
  relocate the whole project

## Where outputs land

Given `BASE` in `config.py` (defaults to
`/content/drive/MyDrive/Regime_Prediction_Final`):

- `results/Run{RUN_ID}/{STOCK}_{LABEL_SOURCE}_Run{RUN_ID}/` — one
  folder per stock, containing `full_predictions.csv`, charts,
  period accuracy, backtest metrics, feature importance, etc.
- `results/Run{RUN_ID}/backtest_summary.csv`,
  `cross_stock_summary_*.csv` — cross-stock roll-ups.
- `checkpoints/` — best model weights + fitted scaler per stock.
- `pseudo_regimes/` — cached label CSVs (pseudo + causal-confirmed),
  regenerated once then reused on subsequent runs.

## Rolling portfolio

`portfolio/` reads each stock's saved
`results/Run{RUN_ID}/{STOCK}_{LABEL_SOURCE}_Run{RUN_ID}/full_predictions.csv`
(columns: `Date, true_idx, pred_idx, true_label, pred_label,
p_bearish, p_neutral, p_bullish, confidence`) — the single clean
interface between the two halves of this repo. It never touches
model weights, training code, or label generation from Phase 2, and
`main.py` did not need to change to support it.

Every `review_days`, it scores every stock in the universe on
rolling prediction accuracy over the last `lookback_days`, using a
**hybrid, zero-lookahead label**: predictions old enough that their
30-day forward return has already resolved use that fwd-30 label;
more recent predictions fall back to a causal-lagged label computed
only from prices between T-160 and T-10 (the same 9-signal scoring
approach as `labels/causal.py`, re-anchored per prediction date). It
then picks the top `top_k` stocks and compares four position
strategies against an equal-weight buy-and-hold benchmark:

- **EqWt_LongOnly** / **EqWt_LongShort** — equal weight across the
  top-K stocks, long only vs. long/short
- **AccWt_LongOnly** / **AccWt_LongShort** — same, but weighted by
  each stock's accuracy score (softmax-sharpened)
- **BnH_EqWt** — equal-weight buy & hold across the full universe (benchmark)

### Folder structure

```
portfolio/
├── __init__.py
├── config.py             ← re-exports PORTFOLIO_CFG from the top-level
│                            config.py (sim dates, review/lookback days,
│                            top_k, hybrid-label params)
│
├── labels.py              ← hybrid, zero-lookahead labels used to score
│                             prediction accuracy at review time:
│                             compute_fwd30_label, compute_causal_lagged_label
│
├── loader.py               ← loads full_predictions.csv + raw prices per
│                              stock, attaches both hybrid labels to
│                              every prediction date
├── simulation.py            ← run_simulation() — the daily rolling
│                              rebalance loop; run_single_review() — score
│                              the universe as of any single date
├── metrics.py                ← calc_metrics, quarterly return breakdown
├── composition.py             ← rotation log, selection frequency, streak
│                                analysis, rolling accuracy table, rotation
│                                turnover, per-stock return contribution
├── visualise.py                ← equity curves + portfolio size, selection
│                                 frequency bar chart, presence heatmap,
│                                 streak charts, per-stock contribution chart
│
└── run_portfolio.py             ← ENTRY POINT — orchestration only: load →
                                    simulate → metrics/composition → plots →
                                    save → one ad-hoc "review as of today" check
```

### Configuration

Portfolio parameters live in the top-level `config.py`'s
`PORTFOLIO_CFG` dict (`sim_start`, `sim_end`, `review_days`,
`lookback_days`, `top_k`, `fwd_days`, `causal_lag`) — edit them
there, not in `portfolio/config.py`, which only re-exports them.

### Run order

```bash
# Step 1 — train models and generate predictions (unchanged)
python main.py

# Step 2 — run the rolling portfolio simulation
python portfolio/run_portfolio.py
```

### Where portfolio outputs land

Directly under `results/Run{RUN_ID}/` alongside the regime-prediction
outputs (these are cross-stock, not namespaced by the per-stock
`{STOCK}_{LABEL_SOURCE}_Run{RUN_ID}` subfolders):

- `rolling_equity_Run{RUN_ID}.csv` — daily cumulative return per strategy
- `rolling_metrics_Run{RUN_ID}.csv` — total/annualised return, Sharpe, max drawdown per strategy
- `rolling_rotation_log_Run{RUN_ID}.csv` — every stock selected at every review, with its scores/weights
- `selection_frequency_Run{RUN_ID}.csv` / `streak_analysis_Run{RUN_ID}.csv` / `stock_contribution_Run{RUN_ID}.csv` — composition analysis
- `ad_hoc_review_{date}_Run{RUN_ID}.csv` — the "review as of today" check
- `rolling_portfolio_Run{RUN_ID}.png`, `selection_frequency_Run{RUN_ID}.png`, `portfolio_heatmap_Run{RUN_ID}.png`, `streak_analysis_Run{RUN_ID}.png`, `stock_contribution_Run{RUN_ID}.png` — charts

### Ad-hoc "as of today" review

`portfolio/simulation.py::run_single_review(review_date, hold_days, ...)`
reuses the exact same scoring/ranking logic as one rebalance step
inside the full simulation, but for a single date you pass in — e.g.
"what should I hold starting today, for the next `review_days`
days". `run_portfolio.py` calls this once automatically
(`AD_HOC_REVIEW_DATE`, defaulting to `sim_end`) — change that
constant to check any other date, past or present.

### A note on the original notebook's structure

`rolling_portfolio_final.py` (3,213 lines) was not a clean "cells
1–N" notebook like the regime-prediction script — it had accreted
real duplication over time:

- An earlier, simpler draft of the simulation, which additionally
  depended on a pre-trained "meta-model trust score" and a
  return-confirmation weighting — **dropped** as a superseded
  duplicate, since the later, richer simulation (below) is a strict
  superset of its scoring logic and does **not** use the meta-model
  at all.
- A second, more complete copy of the simulation (same hybrid-label
  scoring, no meta-model dependency) plus portfolio-composition
  analytics the first copy lacked (selection frequency, streaks,
  turnover, per-stock contribution) — this became `simulation.py` +
  `composition.py`.
- Several hand-copied "PORTFOLIO REVIEW AS OF [date]" cells (April 30,
  May 15, May 22, Section A/B/C) that re-ran the identical
  scoring/ranking formula for one specific date each — generalised
  into the single `run_single_review(review_date)` function rather
  than kept as separate hardcoded scripts.
- A standalone meta-model ("trust score") training script — **not**
  ported, since the final simulation logic doesn't call it.

Every formula, threshold, and algorithm in the simulation that *is*
included here is preserved exactly as written. The one deliberate,
non-logic-affecting change: global reads (`STOCKS_TO_TRAIN`,
`RESULTS_PATH`, etc.) became explicit function parameters so each
module works standalone, the same pattern used throughout Phase 2.

## Notes

- Every function signature, algorithm, threshold, and formula in the
  regime-prediction pipeline is preserved exactly as in the original
  notebook — this restructuring only reorganises code into modules
  and swaps global-variable reads for explicit imports/parameters.
  No modelling logic changed.
- Code after the original regime-prediction notebook's Cell 18 was
  standalone experimental analysis (different `RAW_DATA_PATH`,
  different `REGIME_MAP`) and is intentionally **not** included here.
