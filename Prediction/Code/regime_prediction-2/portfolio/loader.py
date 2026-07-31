"""
portfolio/loader.py
====================
Reads each stock's saved `full_predictions.csv` (produced by the
regime-prediction pipeline's main.py) plus raw price CSVs, and
attaches the hybrid fwd30/causal-lagged labels needed for scoring.

This is the ONLY integration point with the regime-prediction side
of the repo: it reads files from disk, never Python objects, so the
portfolio layer never touches model weights, training code, or
label-generation code directly.
"""

from pathlib import Path

import pandas as pd

from portfolio.config import EXCLUDE_FILES, FWD_DAYS, CAUSAL_LAG
from portfolio.labels import compute_fwd30_label, compute_causal_lagged_label


def discover_raw_price_files(raw_data_path):
    """Scans raw_data_path for per-stock price CSVs, keyed by uppercased filename stem."""
    return {
        f.stem.upper(): f
        for f in Path(raw_data_path).glob("*.csv")
        if f.name.lower() not in EXCLUDE_FILES
    }


def load_price_csv(filepath):
    """
    Loads a stock's raw price CSV down to just Date/Close, with the
    daily percentage return precomputed, indexed by Date.
    """
    df = pd.read_csv(filepath, skiprows=[1])
    df.columns = [c.strip() for c in df.columns]
    df["Date"]  = pd.to_datetime(df["Date"])
    df          = df.sort_values("Date").reset_index(drop=True)
    df          = df[["Date", "Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df          = df.dropna(subset=["Close"])
    df["daily_return"] = df["Close"].pct_change()
    return df.set_index("Date").sort_index()


def load_stock_predictions_and_prices(stocks_to_train, results_path, label_source,
                                       run_id, raw_data_path):
    """
    For every stock in stocks_to_train:
      - loads results/Run{run_id}/{STOCK}_{LABEL_SOURCE}_Run{run_id}/full_predictions.csv
      - loads the matching raw price CSV
      - precomputes BOTH the fwd30 and causal-lagged hybrid labels
        for every prediction date (the simulation picks whichever
        applies at each review time — see simulation.py)

    Returns (stock_preds, stock_prices, universe) where stock_preds
    and stock_prices are dicts keyed by stock name, and universe is
    the list of stocks with both predictions and prices available.
    """
    raw_files = discover_raw_price_files(raw_data_path)

    stock_preds  = {}
    stock_prices = {}

    for stock_name in stocks_to_train:
        pred_path = (
            Path(results_path)
            / f"{stock_name}_{label_source.upper()}_Run{run_id}"
            / "full_predictions.csv"
        )
        if not pred_path.exists():
            print(f"  MISSING predictions: {pred_path}")
            continue

        df = pd.read_csv(pred_path, parse_dates=["Date"])
        df = df[["Date", "pred_label", "true_label", "confidence"]].dropna(
            subset=["pred_label"]
        )
        df = df.sort_values("Date").set_index("Date")

        price_key = raw_files.get(stock_name) or raw_files.get(stock_name + ".NS")
        if price_key is None:
            print(f"  PRICE MISSING: {stock_name}")
            continue
        price_df = load_price_csv(price_key)
        stock_prices[stock_name] = price_df
        price_series = price_df["Close"]

        # Precompute BOTH labels for every prediction date — don't pick yet
        fwd30_labels  = {
            dt: compute_fwd30_label(price_series, dt, FWD_DAYS) for dt in df.index
        }
        causal_labels = {
            dt: compute_causal_lagged_label(price_series, dt, CAUSAL_LAG) for dt in df.index
        }

        df["fwd30_label"]  = pd.Series(fwd30_labels)
        df["causal_label"] = pd.Series(causal_labels)

        stock_preds[stock_name] = df

    loaded = list(stock_preds.keys())
    priced = list(stock_prices.keys())
    universe = [s for s in loaded if s in stock_prices]
    print(f"Loaded predictions for {len(loaded)}/{len(stocks_to_train)} stocks")
    print(f"Loaded prices for {len(priced)}/{len(loaded)} stocks")
    print(f"Universe for simulation: {len(universe)} stocks")

    return stock_preds, stock_prices, universe
