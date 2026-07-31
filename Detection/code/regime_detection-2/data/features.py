"""
data/features.py
=================
Feature engineering: turns raw OHLCV + macro + NIFTY data into the 21
engineered features the model actually trains on (returns, moving averages,
RSI, MACD, volatility, drawdown, Sharpe, etc.).

Logic is unchanged from the original notebook.
"""

import numpy as np
import pandas as pd


def compute_features(stock_df, macro_df, nifty_df):
    """
    Merge a stock's price data with macro and NIFTY features, then engineer
    the 21 features used as model input.

    Returns
    -------
    result : pd.DataFrame
        Columns = ['Date'] + FEATURE_COLS, with NaN/inf rows dropped.
    FEATURE_COLS : list[str]
        Names of the 21 engineered feature columns, in order.
    """
    df = stock_df.copy()

    df = df.merge(macro_df, on="Date", how="left")
    df[["repo_rate", "cpi_inflation", "inr_usd"]] = df[
        ["repo_rate", "cpi_inflation", "inr_usd"]
    ].ffill()

    df = df.merge(nifty_df, on="Date", how="left")
    df["nifty_change"] = df["nifty_change"].ffill().fillna(0)

    c = df["Close"]

    df["log_return"]    = np.log(c / c.shift(1))
    df["hl_spread"]     = (df["High"] - df["Low"]) / c
    df["oc_spread"]     = (c - df["Open"]) / df["Open"]
    df["volume_change"] = df["Volume"].pct_change().clip(-5, 5)

    df["sma_20"] = (c / c.rolling(20).mean()) - 1
    df["sma_50"] = (c / c.rolling(50).mean()) - 1
    df["ema_20"] = (c / c.ewm(span=20, adjust=False).mean()) - 1

    delta        = c.diff()
    gain         = delta.clip(lower=0).rolling(14).mean()
    loss         = (-delta.clip(upper=0)).rolling(14).mean()
    rs           = gain / (loss + 1e-9)
    df["rsi_14"] = (100 - (100 / (1 + rs)) - 50) / 50

    ema12       = c.ewm(span=12, adjust=False).mean()
    ema26       = c.ewm(span=26, adjust=False).mean()
    df["macd"]  = (ema12 - ema26) / (c + 1e-9)

    log_ret              = df["log_return"]
    df["rolling_std_20"] = log_ret.rolling(20).std()
    df["rolling_std_60"] = log_ret.rolling(60).std()

    high, low, prev_c = df["High"], df["Low"], c.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_c).abs(),
        (low - prev_c).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"]   = tr.rolling(14).mean() / (c + 1e-9)
    df["bb_width"] = (2 * c.rolling(20).std()) / (c.rolling(20).mean() + 1e-9)

    roll_max       = c.rolling(60).max()
    df["drawdown"] = (c - roll_max) / (roll_max + 1e-9)

    mean_ret        = log_ret.rolling(20).mean()
    std_ret         = log_ret.rolling(20).std()
    df["sharpe_20"] = ((mean_ret / (std_ret + 1e-9)) * np.sqrt(252)).clip(-5, 5)
    df["vol_ratio"] = (log_ret.rolling(5).std() / (
        df["rolling_std_20"] + 1e-9)
    ).clip(0, 5)

    df["inr_usd"]    = df["inr_usd"].pct_change().fillna(0).clip(-0.1, 0.1)
    df["close_norm"] = c / (c.rolling(252).mean() + 1e-9)

    FEATURE_COLS = [
        "log_return", "hl_spread", "oc_spread", "volume_change", "nifty_change",
        "sma_20", "sma_50", "ema_20", "rsi_14", "macd",
        "rolling_std_20", "rolling_std_60", "atr_14", "bb_width",
        "drawdown", "sharpe_20", "vol_ratio",
        "repo_rate", "cpi_inflation", "inr_usd",
        "close_norm",
    ]

    result = df[["Date"] + FEATURE_COLS].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna().reset_index(drop=True)
    return result, FEATURE_COLS
