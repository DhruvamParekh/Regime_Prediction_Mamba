"""
labels/pseudo.py
================
"Pseudo" labels — clean ground-truth regime labels built by looking
FORWARD N days from each date and classifying the forward return as
Bullish / Neutral / Bearish. These can only be computed for dates
where future price data exists, so the last `fwd_return_days` rows
of any stock are naturally left unlabelled.

Used two ways in this pipeline:
  1. As the legacy/reference label source (LABEL_SOURCE == 'pseudo').
  2. As the high-quality "confirmed" signal for the older half of the
     causal-confirmed label strategy (see labels/causal.py), where
     it's generated with CONFIRMATION_PSEUDO_CFG (30-day forward
     window) instead of the 60-day PSEUDO_CFG.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import PSEUDO_PATH, REGIME_MAP, REGIME_INV_MAP


def _enforce_min_duration(labels_arr, min_days):
    """
    Post-processing: merge any regime run shorter than `min_days`
    into whatever regime preceded it. Prevents label noise from
    creating unrealistically short regime segments.
    """
    labels = labels_arr.copy().astype(float)
    i = 0
    while i < len(labels):
        if np.isnan(labels[i]):
            i += 1
            continue
        current = labels[i]
        j = i + 1
        while j < len(labels) and not np.isnan(labels[j]) \
              and labels[j] == current:
            j += 1
        if j - i < min_days:
            prev = labels[i-1] if i > 0 \
                   and not np.isnan(labels[i-1]) \
                   else REGIME_MAP["Neutral"]
            labels[i:j] = prev
        i = j
    return labels


def load_or_generate_pseudo(stock_name, stock_df, cfg):
    """
    Load cached pseudo labels for `stock_name` if they exist on disk,
    otherwise generate them from forward returns and cache to CSV.

    Label logic per date T:
        forward_return = (Close[T+fwd] - Close[T]) / Close[T]
        Bullish  if forward_return >  cfg['bull_threshold']
        Bearish  if forward_return <  cfg['bear_threshold']
        Neutral  otherwise
    followed by a rolling-mode smoothing pass and a minimum-duration
    enforcement pass. The trailing `fwd_return_days` rows are dropped
    since no future data exists to label them.
    """
    save_path = Path(PSEUDO_PATH) / f"{stock_name}_pseudo_{cfg['fwd_return_days']}d.csv"

    if save_path.exists():
        df = pd.read_csv(save_path, parse_dates=["Date"])
        print(f"  {stock_name}: loaded pseudo-{cfg['fwd_return_days']}d labels ({len(df)} rows)")
        return df

    df  = stock_df[["Date", "Close"]].copy().sort_values("Date").reset_index(drop=True)
    n   = len(df)
    c   = df["Close"].values
    fwd = cfg['fwd_return_days']

    # ── Forward return labels — only where future data exists ────────
    raw = np.full(n, np.nan)
    for i in range(n - fwd):          # last `fwd` rows naturally stay NaN
        ret = (c[i + fwd] - c[i]) / c[i]
        if   ret >  cfg['bull_threshold']: raw[i] = REGIME_MAP['Bullish']
        elif ret <  cfg['bear_threshold']: raw[i] = REGIME_MAP['Bearish']
        else:                              raw[i] = REGIME_MAP['Neutral']

    # ── Smooth only over valid (non-NaN) range ───────────────────────
    sw     = cfg['smooth_window']
    smooth = np.full(n, np.nan)
    valid_end = n - fwd                # ✅ hard stop — no smoothing into future-data zone
    for i in range(sw - 1, valid_end):
        window = raw[i - sw + 1 : i + 1]
        valid  = window[~np.isnan(window)].astype(int)
        if len(valid) == 0:
            continue
        counts    = np.bincount(valid, minlength=3)
        smooth[i] = counts.argmax()

    # ── Min duration enforced only within valid range ────────────────
    smooth[:valid_end] = _enforce_min_duration(smooth[:valid_end], cfg['min_regime_duration'])
    # tail (last `fwd` rows) stays NaN — no masking needed, no future data used

    df["pseudo_label"] = smooth
    df = df[["Date", "pseudo_label"]].dropna().copy()   # NaN tail dropped naturally
    df["pseudo_label"] = df["pseudo_label"].astype(int)
    df.to_csv(save_path, index=False)

    dist = {REGIME_INV_MAP[k]: v for k, v in df["pseudo_label"].value_counts().to_dict().items()}
    print(f"  {stock_name}: generated pseudo-{fwd}d labels ({len(df)} rows) | {dist}")
    return df
