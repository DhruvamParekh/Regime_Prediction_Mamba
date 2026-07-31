"""
labels/pseudo.py
=================
Generates the "pseudo" regime labels the detection model is trained on.

These labels are built from a FUTURE return window (looking `fwd_return_days`
days ahead of each date, then smoothing and thresholding), which is exactly
why this whole project is a DETECTION exercise and not a genuine forecasting
system — the labels themselves already use information from the future.

Logic is unchanged from the original notebook.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import PSEUDO_PATH, REGIME_INV_MAP


def generate_pseudo_labels(stock_df, pseudo_cfg, regime_map):
    """
    Build pseudo regime labels for one stock using a forward-looking,
    smoothed return window, then enforce a minimum regime duration.
    """
    df = stock_df[["Date", "Close"]].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    fwd   = pseudo_cfg["fwd_return_days"]
    sm    = pseudo_cfg["smooth_window"]
    bull  = pseudo_cfg["bull_threshold"]
    bear  = pseudo_cfg["bear_threshold"]
    mindr = pseudo_cfg["min_regime_duration"]
    mask  = pseudo_cfg["mask_last_days"]

    df["fwd_return"] = df["Close"].shift(-fwd) / df["Close"] - 1
    df["fwd_smooth"] = df["fwd_return"].rolling(sm, center=False).median()

    def label_row(r):
        if pd.isna(r):
            return np.nan
        if r > bull:
            return regime_map["Bullish"]
        if r < bear:
            return regime_map["Bearish"]
        return regime_map["Neutral"]

    df["pseudo_label"] = df["fwd_smooth"].apply(label_row)
    df.loc[df.index[-mask:], "pseudo_label"] = np.nan

    labels = df["pseudo_label"].values.copy()
    i = 0
    while i < len(labels):
        if np.isnan(labels[i]):
            i += 1
            continue
        current = labels[i]
        j = i + 1
        while j < len(labels) and not np.isnan(labels[j]) and labels[j] == current:
            j += 1
        duration = j - i
        if duration < mindr:
            prev = labels[i - 1] if i > 0 and not np.isnan(labels[i - 1]) else regime_map["Neutral"]
            labels[i:j] = prev
        i = j

    df["pseudo_label"] = labels
    result = df[["Date", "pseudo_label"]].dropna().copy()
    result["pseudo_label"] = result["pseudo_label"].astype(int)
    return result


def load_or_generate_pseudo(stock_name, stock_df, pseudo_cfg, regime_map):
    """Load cached pseudo labels for a stock if they exist, else generate + cache them."""
    save_path = Path(PSEUDO_PATH) / f"{stock_name}_pseudo.csv"

    if save_path.exists():
        df = pd.read_csv(save_path, parse_dates=["Date"])
        print(f"  {stock_name}: loaded existing pseudo labels ({len(df)} rows)")
        return df

    df = generate_pseudo_labels(stock_df, pseudo_cfg, regime_map)
    df.to_csv(save_path, index=False)
    counts = {REGIME_INV_MAP[k]: v for k, v in df["pseudo_label"].value_counts().to_dict().items()}
    print(f"  {stock_name}: generated pseudo labels ({len(df)} rows) | {counts}")
    return df
