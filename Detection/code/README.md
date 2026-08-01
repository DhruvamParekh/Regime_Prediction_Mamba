# Regime Detection

A Mamba-based classifier that detects Bearish / Neutral / Bullish market
regimes for a set of stocks, trained on pseudo labels, with full evaluation
(accuracy, regime quality, backtest) and cross-stock summary reporting.

This is a reorganized, multi-file version of a single ~4,200-line notebook
export. All computational logic is unchanged from the original — this is a
restructuring for readability, not a rewrite. See "Known issues & caveats"
below for the handful of things worth knowing before you run it.

---

## Detection, not prediction

This is a **detection** exercise, not a forecasting system. The labels the
model is trained on (`labels/pseudo.py`) are built from a *forward* return
window — looking `fwd_return_days` (60) days ahead of each date, smoothing,
then thresholding into Bearish/Neutral/Bullish. Because those training
labels are themselves derived from future price data, this project verifies
that the Mamba architecture can pick up a regime signal at all, rather than
producing a genuine walk-forward prediction. See the docstring at the top
of `config.py` for the full explanation.

(At *inference* time — see `evaluation/latest_signal.py` — the model only
uses history up to the target date, no future data. It's specifically the
*training labels* that use future information.)

---

## Known issues & caveats — please read before running

**1. GPU / CUDA is mandatory, not optional.**
This model uses [`mamba-ssm`](https://github.com/state-spaces/mamba), which
compiles custom CUDA kernels at install time. You need an NVIDIA GPU with a
working CUDA toolkit (`nvcc` on your PATH) for `pip install mamba-ssm` to
succeed at all — there is no CPU fallback. On a CPU-only machine,
`pip install mamba-ssm` will most likely fail to build; if it somehow
installs, model construction/training will still fail or be unusably slow.
`main.py` prints a warning and will *attempt* to run on CPU if no GPU is
detected, but this is not a supported path — it's there so the pipeline
fails loudly and predictably rather than silently.



## File-by-file guide

### Root

**`main.py`** — The entry point and only file you run directly
(`python main.py`). Orchestrates every stage in the same order the original
notebook's cells ran: setup → train all stocks → full-history charts →
analysis → backtest → latest signal → cross-stock summary. Each stage is
its own function (`run_training`, `run_analysis`, etc.) that calls into the
modules below — read this file top to bottom to see the whole pipeline flow
without digging into implementation details.

**`config.py`** — Every constant, path, toggle, and hyperparameter lives
here and nowhere else: data paths (with `REGIME_DETECTION_BASE` env var
override), stock list, label source, pseudo-label thresholds, train/val/test
split dates, model architecture sizes, training hyperparameters, and the
regime name↔index mapping. This is the only file you should need to edit
for routine use (changing which stocks to train, tweaking hyperparameters,
pointing at a different data folder).

**`requirements.txt`** — Python dependencies, with install-order notes
(PyTorch first, matching your CUDA version, then `mamba-ssm`).

**`dataset.py`** — Turns one stock's raw data into model-ready tensors:
`assemble_stock_data()` merges engineered features with the input regime
signal and the prediction target; `make_splits()` cuts that into
Train/Val/Test by date; `RegimePredictionDataset` is the PyTorch `Dataset`
that yields sliding lookback-window sequences.

**`model.py`** — The neural network itself. `MambaBlock` wraps a single
Mamba state-space layer with pre-norm + residual connection.
`RegimePredictionMamba` stacks several of these on top of a feature
projection and a regime embedding, ending in a small classifier head. This
is the file that requires the real `mamba-ssm` package to import.

**`training.py`** — Everything needed to train one stock's model: class
weights (to counter regime imbalance), gradient-based feature importance,
one training epoch, evaluation, and the full `train_model()` loop with
checkpointing on best validation accuracy and early stopping.

### `data/`

**`data/loaders.py`** — Reads raw CSVs off disk: `load_stock_csv()` for
individual stock OHLCV files, `load_macro()` for the shared macro-economic
file (repo rate, CPI, INR/USD), `load_nifty()` for the NIFTY50 index (used
as a market-wide feature). `discover_stock_files()` scans the raw data
folder and figures out which files are stocks vs. the shared macro/NIFTY
files.

**`data/features.py`** — `compute_features()`: turns raw price/macro/NIFTY
data into the 21 engineered features the model trains on — returns,
moving averages, RSI, MACD, rolling volatility, ATR, Bollinger band width,
drawdown, Sharpe ratio, volume/volatility ratios, and macro features.

### `labels/`

**`labels/pseudo.py`** — `generate_pseudo_labels()`: builds the
forward-looking pseudo regime labels described above (this is the file
most directly responsible for why this is a "detection" project — it uses
future returns to build training labels). `load_or_generate_pseudo()`
caches these to disk per stock so they're not recomputed every run.

**`labels/targets.py`** — `load_regime_labels()`: loads the per-day input
regime signal, either from the cached pseudo labels or from a
`{stock}_detected.csv` file if `LABEL_SOURCE='detected'` (falls back to
pseudo if that file doesn't exist). `build_prediction_labels()`: turns that
day-by-day regime series into the actual training target — the *majority*
regime over the next `forward_window` (7) days.

### `evaluation/`

**`evaluation/predictions.py`** — `get_full_predictions()`: reloads a
stock's best checkpoint and runs it over its *entire* history (not just the
test period) to build a full prediction timeline. `plot_full_regime_chart()`:
the 4-panel chart (price with regime-colored dots, predicted regime
timeline, stacked probability area, confidence over time).

**`evaluation/analysis.py`** — Three analysis functions plus their plots:
`analyze_period_accuracy()` (accuracy by Train/Val/Test and by class),
`analyze_regime_quality()` (true vs. detected regime run durations, and how
quickly transitions are picked up), `analyze_returns_by_regime()` (what
you'd have earned trading on the detected regime vs. Buy & Hold).
`plot_analysis()` draws the 2x2 summary panel; `plot_confusion_matrices()`
draws Train/Val/Test confusion matrices.

**`evaluation/backtest.py`** — `run_backtest()`: a long/cash backtest over
the test period only, comparing three strategies (Model, Perfect Foresight
using the true labels, Buy & Hold) with no transaction costs or shorting.
`plot_backtest()`: cumulative return curves plus a position (long/cash)
strip underneath.

**`evaluation/latest_signal.py`** — `predict_specific_week()`: runs
inference for a single target date using only history up to that date (no
future data at inference time — see the "Detection, not prediction" note
above). `plot_prediction_context()`: shows the lookback-window price
history leading up to that call, with the detected regime and transition
status displayed clearly.

### `summary/`

**`summary/cross_stock.py`** — `build_cross_stock_summary()`: pulls
accuracy, backtest, and feature-importance results from every trained stock
into one table. `plot_cross_stock_summary()`: three comparison charts (Val
vs. Test accuracy, Model vs. Buy & Hold returns, alpha bars).
`print_cross_stock_summary()`: prints a readable report and saves the
summary table plus aggregated cross-stock feature importance to CSV.

---

## Project structure

```
regime_detection/
├── main.py
├── config.py
├── requirements.txt
├── dataset.py
├── model.py
├── training.py
├── data/
│   ├── loaders.py
│   └── features.py
├── labels/
│   ├── pseudo.py
│   └── targets.py
├── evaluation/
│   ├── predictions.py
│   ├── analysis.py
│   ├── backtest.py
│   └── latest_signal.py
└── summary/
    └── cross_stock.py
```

---

## Setup

1. Get a CUDA-capable machine (see caveat #1 above).
2. Install PyTorch matching your CUDA version first (see the note at the
   top of `requirements.txt`), then:
   ```bash
   pip install -r requirements.txt
   ```
3. Point the pipeline at your data. By default it looks for a
   `project_data/` folder next to `config.py`; override with:
   ```bash
   export REGIME_DETECTION_BASE=/path/to/your/data
   ```
   Your data folder needs this structure:
   ```
   <BASE>/
   └── data/
       └── raw/
           ├── NIFTY50_INDEX.csv
           ├── macro_features.csv
           └── {STOCK}.csv          # one file per stock, OHLCV format
   ```
   (`results/`, `checkpoints/`, and `pseudo_regimes/` are created
   automatically on first run.)
4. Edit `config.py` to set `STOCKS_TO_TRAIN`, `LABEL_SOURCE`, `RUN_ID`, and
   any hyperparameters you want to change.
5. Run:
   ```bash
   python main.py
   ```

## Output

Same structure as the original notebook — one folder per stock under
`<BASE>/results/{STOCK}_{LABEL_SOURCE}_Run{RUN_ID}/`, containing
predictions, training history, feature importance, period accuracy, regime
duration/returns, backtest metrics, and five charts. Cross-stock summary
CSVs/PNGs and `backtest_summary.csv` are written at the results root.
