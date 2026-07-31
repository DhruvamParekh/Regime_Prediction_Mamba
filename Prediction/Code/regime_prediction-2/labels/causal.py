"""
labels/causal.py
================
"Causal-confirmed" labels — the label source actually used for
training (LABEL_SOURCE == 'causal_confirmed'). Uses ZERO future data,
so it is fully tradeable/reproducible at inference time.

At each date T the label is a majority vote between:
  Recent window [T-30 : T-1]  — a live 8-signal causal score built
                                 only from backward-looking price
                                 momentum, drawdown, and volume
                                 signals (see compute_causal_score_series).
  Older window  [T-60 : T-31] — high-quality pseudo-30 labels
                                 (settled, confirmed signal), weighted
                                 once; the recent window is weighted
                                 2x in the vote.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import CAUSAL_CFG, CONFIRMATION_PSEUDO_CFG, PSEUDO_PATH, REGIME_MAP, REGIME_INV_MAP
from labels.pseudo import _enforce_min_duration, load_or_generate_pseudo


def compute_causal_score_series(stock_df, cfg):
    """
    Computes an 8-signal trend score for every date, using only
    backward-looking price signals (moving averages, momentum, RSI,
    52-week proximity, volatility ratio). Returns a DataFrame with
    Date, score, score_smooth, and the resulting causal_label.
    """
    df = stock_df[["Date", "Close"]].copy() \
                 .sort_values("Date").reset_index(drop=True)
    c  = df["Close"]

    lookback = cfg["causal_lookback"]

    df["sma50"]     = c.rolling(50).mean()
    df["sma200"]    = c.rolling(200).mean()
    df["ret_20"]    = c.pct_change(20)
    df["ret_60"]    = c.pct_change(lookback)

    delta           = c.diff()
    gain            = delta.clip(lower=0).rolling(14).mean()
    loss            = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]       = 100 - (100 / (1 + gain / (loss + 1e-9)))

    df["hi252"]     = c.rolling(252).max()
    df["lo252"]     = c.rolling(252).min()
    df["hi_prox"]   = c / df["hi252"]
    df["lo_prox"]   = c / df["lo252"]

    ret1            = c.pct_change()
    df["vol5"]      = ret1.rolling(5).std()
    df["vol20"]     = ret1.rolling(20).std()
    df["vol_ratio"] = df["vol5"] / (df["vol20"] + 1e-9)

    bull_s = cfg["bull_score"]
    bear_s = cfg["bear_score"]

    def score_row(row):
        if pd.isna(row["sma200"]) or pd.isna(row["rsi"]) \
           or pd.isna(row["ret_60"]):
            return np.nan
        s = 0
        s += 1 if row["Close"]    > row["sma50"]   else -1
        s += 1 if row["Close"]    > row["sma200"]  else -1
        s += 1 if row["sma50"]    > row["sma200"]  else -1
        s += 1 if row["ret_20"]   > 0              else -1
        if   row["ret_60"]   >  0.05: s += 1
        elif row["ret_60"]   < -0.05: s -= 1
        if   row["rsi"]      >  55:   s += 1
        elif row["rsi"]      <  45:   s -= 1
        if   row["hi_prox"]  > 0.92:  s += 1
        elif row["lo_prox"]  < 1.10:  s -= 1
        if   row["vol_ratio"]< 1.5:   s += 1
        return s

    df["score"]        = df.apply(score_row, axis=1)
    df["score_smooth"] = df["score"].rolling(5, center=False).median()

    def to_label(s):
        if pd.isna(s):       return np.nan
        if s >= bull_s:      return REGIME_MAP["Bullish"]
        if s <= bear_s:      return REGIME_MAP["Bearish"]
        return               REGIME_MAP["Neutral"]

    df["causal_label"] = df["score_smooth"].apply(to_label)
    return df[["Date", "score", "score_smooth", "causal_label"]].copy()


def generate_causal_confirmed_labels(stock_df, cfg, pseudo_older_df=None):
    """
    Generates hybrid causal-confirmed labels. Zero future data at inference.

    At each date T:
      Older window [T-60 : T-31] → pseudo-30 labels (high quality, confirmed)
      Recent window [T-30 : T-1] → causal labels (live, no future data)
      Majority vote → final confirmed label at T

    pseudo_older_df: DataFrame with columns [Date, pseudo_label] from
                     CONFIRMATION_PSEUDO_CFG. If None, falls back to the
                     settled confirmed[] array (old behaviour).
    """
    score_df = compute_causal_score_series(stock_df, cfg)
    score_df = score_df.reset_index(drop=True)
    n        = len(score_df)

    confirm_recent = cfg["confirm_recent"]   # 30
    confirm_older  = cfg["confirm_older"]    # 30
    confirm_total  = confirm_recent + confirm_older  # 60
    min_days       = cfg["min_regime_days"]

    CONTINUE_THRESH = 0.55
    UPGRADE_THRESH  = 0.65

    # Build a date-indexed pseudo label lookup for the older window
    if pseudo_older_df is not None:
        pseudo_map = pseudo_older_df.set_index("Date")["pseudo_label"].to_dict()
    else:
        pseudo_map = {}

    confirmed = np.full(n, np.nan)

    for i in range(n):
        causal = score_df["causal_label"].iloc[i]
        if pd.isna(causal):
            continue

        # ── Older half: T-60 to T-31 ─────────────────────────────
        # Use pseudo-30 labels (high quality, confirmed signal)
        older_start = max(0, i - confirm_total)
        older_end   = max(0, i - confirm_recent)

        if older_end > older_start:
            if pseudo_map:
                # Look up pseudo labels by date for the older window
                older_dates = score_df["Date"].iloc[older_start:older_end].values
                older_labels = np.array([
                    pseudo_map.get(d, np.nan) for d in older_dates
                ], dtype=float)
            else:
                # Fallback: use already-settled confirmed[] (old behaviour)
                older_conf   = confirmed[older_start : older_end].copy()
                older_causal = score_df["causal_label"].iloc[
                    older_start : older_end].values
                older_labels = np.where(
                    np.isnan(older_conf), older_causal, older_conf
                )
        else:
            older_labels = np.array([])

        # ── Recent half: T-30 to T-1 ─────────────────────────────
        # Use causal labels — no future data
        recent_start = max(0, i - confirm_recent)
        recent_end   = i

        if recent_end > recent_start:
            recent_labels = score_df["causal_label"].iloc[
                recent_start : recent_end].values
        else:
            recent_labels = np.array([])

        # ── Combine — recent weighted 2x ─────────────────────────
        win = np.concatenate([
            older_labels,
            recent_labels,
            recent_labels
        ])
        win = win[~np.isnan(win)]

        if len(win) == 0:
            confirmed[i] = causal
            continue

        total     = len(win)
        bull_frac = (win == REGIME_MAP["Bullish"]).sum() / total
        bear_frac = (win == REGIME_MAP["Bearish"]).sum() / total

        c = int(causal)

        if c == REGIME_MAP["Bullish"]:
            if   bear_frac >= CONTINUE_THRESH:
                confirmed[i] = REGIME_MAP["Neutral"]
            else:
                confirmed[i] = REGIME_MAP["Bullish"]
        elif c == REGIME_MAP["Bearish"]:
            if   bull_frac >= CONTINUE_THRESH:
                confirmed[i] = REGIME_MAP["Neutral"]
            else:
                confirmed[i] = REGIME_MAP["Bearish"]
        else:
            if   bull_frac >= UPGRADE_THRESH:
                confirmed[i] = REGIME_MAP["Bullish"]
            elif bear_frac >= UPGRADE_THRESH:
                confirmed[i] = REGIME_MAP["Bearish"]
            else:
                confirmed[i] = REGIME_MAP["Neutral"]

    confirmed = _enforce_min_duration(confirmed, min_days)

    result = score_df[["Date", "causal_label"]].copy()
    result["confirmed_label"] = confirmed
    result = result.dropna(subset=["confirmed_label"]).copy()
    result["confirmed_label"] = result["confirmed_label"].astype(int)
    result["causal_label"]    = result["causal_label"].fillna(
        REGIME_MAP["Neutral"]).astype(int)

    out = result[["Date", "confirmed_label", "causal_label"]].copy()
    out = out.rename(columns={"confirmed_label": "regime_label"})
    return out


def load_or_generate_causal_confirmed(stock_name, stock_df, cfg):
    """
    Load cached causal-confirmed labels if present, otherwise
    generate them (which itself generates/loads the pseudo-30 labels
    needed for the older window) and cache to CSV.
    """
    save_path = Path(PSEUDO_PATH) / f"{stock_name}_causal_confirmed.csv"

    if save_path.exists():
        df = pd.read_csv(save_path, parse_dates=["Date"])
        print(f"  {stock_name}: loaded causal_confirmed labels ({len(df)} rows)")
        return df

    # Generate pseudo-30 labels for the older window (T-60 to T-31)
    pseudo_older_df = load_or_generate_pseudo(
        stock_name, stock_df, CONFIRMATION_PSEUDO_CFG
    )

    df = generate_causal_confirmed_labels(stock_df, cfg, pseudo_older_df)

    df.to_csv(save_path, index=False)

    dist = {REGIME_INV_MAP[k]: v
            for k, v in df["regime_label"].value_counts().to_dict().items()}
    print(f"  {stock_name}: generated causal_confirmed labels ({len(df)} rows) | {dist}")
    return df
