"""
labels/targets.py
==================
Loads the regime label series that feeds the model as an INPUT signal
('pseudo' or 'detected' regime per day), and builds the actual training
TARGET: for each date, the majority regime over the next `forward_window`
days.

Logic is unchanged from the original notebook.
"""

from collections import Counter

import numpy as np
import pandas as pd

from pathlib import Path

from config import DETECTED_PATH, MODEL_CFG, PSEUDO_CFG, REGIME_MAP
from labels.pseudo import load_or_generate_pseudo


def load_regime_labels(stock_name, stock_df, label_source, pseudo_cfg=None):
    """
    Returns df with columns: Date, regime_label
    Source = 'pseudo'   → loads from pseudo_regimes/
    Source = 'detected' → loads from detected_regimes/
    Falls back to pseudo if detected file not found.
    """
    pseudo_cfg = pseudo_cfg if pseudo_cfg is not None else PSEUDO_CFG

    if label_source == "detected":
        det_path = Path(DETECTED_PATH) / f"{stock_name}_detected.csv"
        if not det_path.exists():
            print(f"  {stock_name}: detected file not found — falling back to pseudo.")
        else:
            df = pd.read_csv(det_path, parse_dates=["Date"])
            df.columns = [c.strip() for c in df.columns]
            regime_col = [c for c in df.columns if c.lower() != "date"][0]
            df = df.rename(columns={regime_col: "regime_label"})
            if df["regime_label"].dtype == object:
                df["regime_label"] = df["regime_label"].map(REGIME_MAP)
            df["regime_label"] = df["regime_label"].fillna(1).astype(int)
            return df[["Date", "regime_label"]].sort_values("Date").reset_index(drop=True)

    # Pseudo path
    pseudo_df = load_or_generate_pseudo(
        stock_name, stock_df, pseudo_cfg, REGIME_MAP
    )
    return pseudo_df.rename(columns={"pseudo_label": "regime_label"})


def build_prediction_labels(regime_df):
    """
    For each date t, prediction target = majority regime
    over next 7 trading days [t+1, t+7].
    This is what the model is trained to predict.
    """
    fwd        = MODEL_CFG["forward_window"]
    df         = regime_df.sort_values("Date").reset_index(drop=True).copy()
    labels_arr = df["regime_label"].values
    n          = len(df)
    pred_labels = np.full(n, np.nan)

    for i in range(n - fwd):
        window = labels_arr[i + 1: i + 1 + fwd]
        valid  = window[~np.isnan(window.astype(float))]
        if len(valid) == 0:
            continue
        counts = Counter(valid.astype(int))
        pred_labels[i] = max(counts, key=counts.get)

    df["pred_label"] = pred_labels
    df = df.dropna(subset=["pred_label"]).copy()
    df["pred_label"] = df["pred_label"].astype(int)
    return df[["Date", "pred_label"]]
