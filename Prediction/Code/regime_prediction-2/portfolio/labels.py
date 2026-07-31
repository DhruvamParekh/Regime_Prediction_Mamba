"""
portfolio/labels.py
====================
Hybrid, zero-future-data label helpers used to score how accurate
each stock's regime predictions have been, as of any given review
date.

At each review, a prediction date T gets scored against one of two
labels depending on how recent it is relative to the review date:

  Older predictions (T <= review_date - FWD_DAYS):
    fwd-N label — the N-day forward return from T has already
    resolved by review time, so this is a normal forward-return
    label with zero lookahead risk.

  Recent predictions (T > review_date - FWD_DAYS):
    causal-lagged label — the N-day forward return hasn't resolved
    yet, so instead this falls back to a fully causal 9-signal score
    computed only from prices between T-(lag+lookback) and T-lag.

Together they replace a naive "true_label" that would otherwise
require already knowing N days of future price — making portfolio
scoring fully honest and tradeable at the point of decision.
"""

import numpy as np
import pandas as pd


def compute_fwd30_label(price_series, date, fwd_days,
                         bull_thr=0.03, bear_thr=-0.03):
    """
    Label for prediction date T using price change from T to T+fwd_days.
    Only meaningful for dates where T+fwd_days is already in the past
    relative to the review date (callers enforce this via a
    review-time cutoff — see simulation.py). Returns 'Bullish',
    'Neutral', 'Bearish', or NaN if the price series doesn't cover
    the window.
    """
    try:
        p0_idx = price_series.index.searchsorted(date)
        p1_idx = price_series.index.searchsorted(date + pd.Timedelta(days=fwd_days))
        if p0_idx >= len(price_series) or p1_idx >= len(price_series):
            return np.nan
        p0  = price_series.iloc[p0_idx]
        p1  = price_series.iloc[p1_idx]
        ret = (p1 - p0) / (p0 + 1e-9)
        if   ret >  bull_thr: return "Bullish"
        elif ret <  bear_thr: return "Bearish"
        else:                 return "Neutral"
    except Exception:
        return np.nan


def compute_causal_lagged_label(price_series, date, lag, lookback=50,
                                 bull_score=3, bear_score=-3):
    """
    Label for prediction date T using prices from T-(lag+lookback) to
    T-lag only — fully causal, zero future data. Uses the same
    9-signal scoring system as labels/causal.py's causal_confirmed
    strategy in the main regime-prediction pipeline (SMA50/200,
    momentum, RSI, 52-window proximity, volatility ratio).
    Returns 'Bullish', 'Neutral', 'Bearish', or NaN if not enough
    history exists before T-lag.
    """
    end_idx   = price_series.index.searchsorted(date) - lag
    start_idx = end_idx - lookback
    if end_idx <= 0 or start_idx < 0:
        return np.nan
    window = price_series.iloc[start_idx:end_idx]
    if len(window) < 30:
        return np.nan

    c     = window.values
    close = pd.Series(c)

    sma50   = close.rolling(50).mean().iloc[-1]
    sma200  = close.rolling(min(200, len(close))).mean().iloc[-1]
    ret_60  = (c[-1] - c[max(-61, -len(c))]) / (c[max(-61, -len(c))] + 1e-9)
    ret_20  = (c[-1] - c[max(-21, -len(c))]) / (c[max(-21, -len(c))] + 1e-9)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss  = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    hi252   = close.rolling(min(252, len(close))).max().iloc[-1]
    lo252   = close.rolling(min(252, len(close))).min().iloc[-1]
    hi_prox = c[-1] / (hi252 + 1e-9)
    lo_prox = c[-1] / (lo252 + 1e-9)

    vol5      = close.pct_change().rolling(5).std().iloc[-1]
    vol20     = close.pct_change().rolling(20).std().iloc[-1]
    vol_ratio = vol5 / (vol20 + 1e-9)

    s = 0
    s += 1 if c[-1] > sma50   else -1
    s += 1 if c[-1] > sma200  else -1
    s += 1 if sma50 > sma200  else -1
    s += 1 if ret_20  > 0     else -1
    s += 1 if ret_60  >  0.05 else (-1 if ret_60  < -0.05 else 0)
    s += 1 if rsi     >  55   else (-1 if rsi      <  45   else 0)
    s += 1 if hi_prox > 0.92  else 0
    s -= 1 if lo_prox < 1.10  else 0
    s += 1 if vol_ratio < 1.5 else 0

    if   s >= bull_score: return "Bullish"
    elif s <= bear_score: return "Bearish"
    else:                 return "Neutral"
