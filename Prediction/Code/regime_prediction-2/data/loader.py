"""
data/loader.py
==============
Raw CSV loaders for individual stock OHLCV data, macro-economic
features (repo rate, CPI inflation, INR/USD), and the NIFTY50 index
(used to compute market-relative returns).

These are the only functions in the codebase that touch raw CSV
files directly — everything downstream works with the DataFrames
these functions return.
"""

import pandas as pd


def load_stock_csv(filepath):
    """
    Load a single stock's OHLCV CSV.

    Skips the second header row (Yahoo-Finance-style export quirk),
    strips whitespace from column names, parses dates, sorts
    chronologically, and coerces price/volume columns to numeric.
    """
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
    """
    Load macro-economic features: repo rate, CPI inflation, INR/USD.
    Forward-fills gaps since these series update less frequently
    than daily price data.
    """
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
    """
    Load the NIFTY50 index and derive its daily percentage change,
    used as a market-wide feature for every stock.
    """
    df = load_stock_csv(filepath)
    df["nifty_change"] = df["Close"].pct_change()
    df = df[["Date", "nifty_change"]].dropna()
    return df
