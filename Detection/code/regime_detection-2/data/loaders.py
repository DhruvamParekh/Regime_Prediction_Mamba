"""
data/loaders.py
================
Raw CSV loading: individual stock price files, the shared macro-economic
file, and the NIFTY50 index file (used as a market-wide feature). Also
handles discovering which stock files are available on disk.

Logic is unchanged from the original notebook — only reorganized into
functions that live in one place instead of being scattered across cells.
"""

from pathlib import Path

import pandas as pd

from config import RAW_DATA_PATH, MACRO_FILE, NIFTY_FILE


def load_stock_csv(filepath):
    """Load one stock's OHLCV CSV, clean columns, and coerce to numeric."""
    df = pd.read_csv(filepath, skiprows=[1])
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df[["Open", "High", "Low", "Close", "Volume"]] = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])
    return df


def load_macro(filepath):
    """Load the shared macro-economic features file (repo rate, CPI, INR/USD)."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df["repo_rate"]     = pd.to_numeric(df["repo_rate"],     errors="coerce")
    df["cpi_inflation"] = pd.to_numeric(df["cpi_inflation"], errors="coerce")
    df["inr_usd"]       = pd.to_numeric(df["inr_usd"],       errors="coerce")
    df[["repo_rate", "cpi_inflation", "inr_usd"]] = df[
        ["repo_rate", "cpi_inflation", "inr_usd"]
    ].ffill()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_nifty(filepath):
    """Load the NIFTY50 index and derive a daily % change feature from it."""
    df = load_stock_csv(filepath)
    df["nifty_change"] = df["Close"].pct_change()
    df = df[["Date", "nifty_change"]].dropna()
    return df


def discover_stock_files():
    """
    Scan RAW_DATA_PATH for stock CSVs, excluding the shared macro and NIFTY
    files. Returns a dict of {stock_name: filepath} plus the macro/nifty
    paths for convenience.
    """
    exclude_files = {MACRO_FILE.lower(), NIFTY_FILE.lower()}
    all_files = sorted(Path(RAW_DATA_PATH).glob("*.csv"))
    all_stock_files = {
        f.stem: f for f in all_files if f.name.lower() not in exclude_files
    }

    macro_path = Path(RAW_DATA_PATH) / MACRO_FILE
    nifty_path = Path(RAW_DATA_PATH) / NIFTY_FILE

    return all_stock_files, macro_path, nifty_path
