# Regime_Prediction_Mamba

> ⚠️ **GPU REQUIRED.**
> The Prediction module uses **Mamba SSM** (`mamba-ssm`), which only runs on
> an **NVIDIA GPU with CUDA 12.6**. Detection and Rolling Portfolio are
> CPU-compatible. See the Prediction folder's README for GPU setup.

---

## What this project does

This project identifies and predicts **market regimes** — Bullish, Neutral,
and Bearish — for NSE-listed stocks, and uses those predictions to run a
**rolling portfolio simulation** that dynamically allocates capital based on
the model's forward-looking regime signals.

The work is split across three self-contained modules that build on each other
in a clear sequence:

```
Detection  →  Prediction  →  Rolling Portfolio
```

---

## Repository structure

```
Regime_Prediction_Mamba/
│
├── Detection/                   ← Phase 1: Identify historical regimes in price data
│   ├── code/                    ← Modular Python source (see Detection README)
│   └── Detection_final_results/ ← Per-stock detected regime CSVs and charts
│
├── Prediction/                  ← Phase 2: Train Mamba model to predict future regimes
│   ├── Code/regime_prediction-2/← Modular Python source (see Prediction README)
│   └── Regime_prediction_results/← Per-stock predictions, backtests, feature importance
│
└── Rolling_Portfolio_Results/   ← Phase 3: Portfolio simulation driven by predictions
    ├── meta_model_results/       ← Trained trust-score meta-model outputs
    ├── portfolio_results/        ← Equity curves, allocation history, metrics CSVs
    └── README_Rolling_Portfolio_Final_Results.md
```

Each folder has its own `README` that covers setup, configuration, and outputs
in detail. This file gives you the top-level picture and tells you how the
three phases connect.

---

## How the three phases connect

### Phase 1 — Detection
Analyses raw OHLCV price data for each stock and assigns a historical
**regime label** (Bullish / Neutral / Bearish) to every trading day using
a rule-based scoring system built from momentum, trend, volume, and drawdown
signals. No future data is used. The output — one CSV per stock — is the
ground-truth label set used to train and evaluate the prediction model.

### Phase 2 — Prediction
Trains a separate **Mamba SSM** deep learning model per stock. The model
takes a 60-day lookback window of 13 technical and macro features and
predicts the dominant regime for the coming week. Models are evaluated on a
held-out test period and their full prediction histories are saved as
`full_predictions.csv` — one file per stock.

### Phase 3 — Rolling Portfolio
Reads the `full_predictions.csv` files produced by Phase 2 and runs a
**rolling portfolio simulation**: at each rebalance date it scores every stock
using a combination of the regime prediction, a causal momentum signal, and a
trust score from a meta-model (GradientBoosting classifier trained on
prediction accuracy history), then selects the top-K stocks and allocates
capital accordingly. Four strategies are compared against a Buy-and-Hold
benchmark.

---

## Quick-start (run order)

```bash
# Phase 1 — Detection (CPU, run once per stock universe)
cd Detection/code
python main.py

# Phase 2 — Prediction (GPU required — read Prediction/README first)
cd Prediction/Code/regime_prediction-2
bash setup_env.sh          # installs CUDA 12.6 + mamba-ssm (one-time)
# restart runtime
python main.py

# Phase 3 — Rolling Portfolio (CPU)
cd Prediction/Code/regime_prediction-2
python portfolio/train_meta_model.py   # one-time, only if meta-model .pkl missing
python portfolio/run_portfolio.py
```

---

## Data requirements

All three phases read raw OHLCV CSVs from a shared data folder on Google Drive.
The expected format matches Yahoo Finance / NSE Bhav Copy downloads:
`Date, Open, High, Low, Close, Volume` — one file per stock, named
`{TICKER}.csv`. Two additional files are required:

- `macro_features.csv` — daily repo rate, CPI inflation, INR/USD
- `NIFTY50_INDEX.csv` — NIFTY 50 index OHLCV (used as market-relative feature)

The root data path is configured in `Prediction/Code/regime_prediction-2/config.py`
(`BASE` and `RAW_DATA_PATH`). Detection has its own equivalent path config.

---

## Technology

| Component | Library / Tool |
|---|---|
| Deep learning model | `mamba-ssm 2.3.1` (Mamba SSM) + PyTorch 2.6 |
| Feature engineering | `pandas`, `numpy` |
| Meta-model (trust score) | `scikit-learn` GradientBoostingClassifier |
| Visualisation | `matplotlib` |
| Environment | Google Colab (GPU for Prediction, CPU for rest) |

---

## Notes

- All modelling logic — every formula, threshold, and algorithm — is
  preserved exactly from the original research notebooks. The restructuring
  only organises code into modules for readability and reusability.
- Detection results feed into Prediction as optional external label inputs
  (`LABEL_SOURCE = 'detected'`). The default Prediction label source
  (`causal_confirmed`) is self-contained and does not require running
  Detection first.
- Rolling Portfolio reads only `full_predictions.csv` from Phase 2 — it
  never touches model weights, training code, or label generation.
