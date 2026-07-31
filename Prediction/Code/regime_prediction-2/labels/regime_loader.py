"""
labels/regime_loader.py
========================
Dispatches to the correct label-generation strategy based on
LABEL_SOURCE, and builds the forward-looking "prediction label"
(majority regime over the next `forward_window` days) that the
model is actually trained to predict.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config import CAUSAL_CFG, DETECTED_PATH, MODEL_CFG, PSEUDO_CFG, REGIME_MAP
from labels.causal import load_or_generate_causal_confirmed
from labels.pseudo import load_or_generate_pseudo


def load_regime_labels(stock_name, stock_df, label_source):
    """
    Returns a DataFrame with columns [Date, regime_label] for the
    requested label_source: 'causal_confirmed' (default/production),
    'detected' (externally-supplied label file), or 'pseudo' (legacy).
    """

    # ── causal_confirmed — new default, check FIRST ───────────────
    if label_source == "causal_confirmed":
        cc_df = load_or_generate_causal_confirmed(
            stock_name, stock_df, CAUSAL_CFG
        )
        return cc_df[["Date", "regime_label"]]

    # ── detected ──────────────────────────────────────────────────
    if label_source == "detected":
        det_path = Path(DETECTED_PATH) / f"{stock_name}_regime_predictions.csv"
        if det_path.exists():
            df = pd.read_csv(det_path, index_col=0)
            df.index.name = "Date"
            df = df.reset_index()
            df.columns = [c.strip() for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"])
            regime_col = [c for c in df.columns
                          if c.lower() != "date"][0]
            df = df.rename(columns={regime_col: "regime_label"})
            if df["regime_label"].dtype == object:
                df["regime_label"] = df["regime_label"].map(REGIME_MAP)
            df["regime_label"] = df["regime_label"].fillna(1).astype(int)
            return df[["Date", "regime_label"]] \
                .sort_values("Date").reset_index(drop=True)
        raise FileNotFoundError(
            f"  {stock_name}: detected label file not found at "
            f"{det_path}. Add the file or remove stock from STOCKS_TO_TRAIN."
        )

    # ── pseudo (legacy only — keep for reference) ─────────────────
    if label_source == "pseudo":
        pseudo_df = load_or_generate_pseudo(
            stock_name, stock_df, PSEUDO_CFG, REGIME_MAP
        )
        return pseudo_df.rename(columns={"pseudo_label": "regime_label"})

    # ── fallback ──────────────────────────────────────────────────
    raise ValueError(
        f"Unknown label_source: '{label_source}'. "
        f"Use 'causal_confirmed', 'detected', or 'pseudo'."
    )


def build_prediction_labels(regime_df):
    """
    Builds the label the model is trained to predict: the majority
    regime over the next `forward_window` days from each date.
    Rows near the end of the series (where fewer than forward_window
    future rows exist) are dropped if no valid majority can be found.
    """
    fwd        = MODEL_CFG["forward_window"]
    df         = regime_df.sort_values("Date") \
                          .reset_index(drop=True).copy()
    labels_arr = df["regime_label"].values
    n          = len(df)
    pred_labels = np.full(n, np.nan)

    for i in range(n - fwd):
        window = labels_arr[i+1 : i+1+fwd]
        valid  = window[~np.isnan(window.astype(float))]
        if len(valid) == 0:
            continue
        counts = Counter(valid.astype(int))
        pred_labels[i] = max(counts, key=counts.get)

    df["pred_label"] = pred_labels
    df = df.dropna(subset=["pred_label"]).copy()
    df["pred_label"] = df["pred_label"].astype(int)
    return df[["Date", "pred_label"]]
