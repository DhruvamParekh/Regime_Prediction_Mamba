"""
inference.py
============
Runs a trained checkpoint over data to produce predictions:
  - get_full_predictions: sliding-window predictions over a stock's
    entire assembled history (train+val+test), used for charts,
    analysis, and backtesting.
  - predict_specific_week: a single forward-looking prediction for
    one target date, using only history up to that date (no future
    data), for "what regime does the model predict next week" style
    queries.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import CHECKPOINT_PATH, LABEL_SOURCE, MODEL_CFG, REGIME_INV_MAP, RUN_ID, DEVICE
from model import RegimePredictionMamba


@torch.no_grad()
def get_full_predictions(stock_name, merged_df, feat_cols):
    """
    Loads the best checkpoint + scaler for `stock_name` and runs
    sliding-window inference over the entire assembled DataFrame,
    stepping by `prediction_freq` days (matching how val/test
    datasets are windowed).
    """
    ckpt_path   = Path(CHECKPOINT_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}_best.pt"
    scaler_path = Path(CHECKPOINT_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}_scaler.pkl"

    ckpt      = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    saved_cfg = ckpt.get("model_cfg", MODEL_CFG)
    model     = RegimePredictionMamba(cfg=saved_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    feat_arr    = scaler.transform(merged_df[feat_cols].values).astype(np.float32)
    regime_arr  = merged_df["detected_regime"].values.astype(np.int64)
    true_labels = merged_df["pred_label"].values.astype(np.int64)
    dates       = merged_df["Date"].values

    lookback = saved_cfg["lookback_window"]
    step     = saved_cfg["prediction_freq"]
    indices  = list(range(lookback, len(merged_df), step))

    all_dates, all_preds, all_probs, all_true = [], [], [], []

    for end in indices:
        start  = end - lookback
        feat   = torch.tensor(
            feat_arr[start:end], dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)
        regime = torch.tensor(
            regime_arr[start:end], dtype=torch.long
        ).unsqueeze(0).to(DEVICE)

        logits = model(feat, regime)
        probs  = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred   = int(probs.argmax())

        all_dates.append(dates[end - 1])
        all_preds.append(pred)
        all_probs.append(probs)
        all_true.append(true_labels[end - 1])


    return pd.DataFrame({
        "Date"      : pd.to_datetime(all_dates),
        "true_idx"  : all_true,
        "pred_idx"  : all_preds,
        "true_label": [REGIME_INV_MAP[l] for l in all_true],
        "pred_label": [REGIME_INV_MAP[p] for p in all_preds],
        "p_bearish" : [p[0] for p in all_probs],
        "p_neutral" : [p[1] for p in all_probs],
        "p_bullish" : [p[2] for p in all_probs],
        "confidence": [float(p.max()) for p in all_probs],
    })


@torch.no_grad()
def predict_specific_week(stock_name, target_date_str, assemble_stock_data_fn):
    """
    Predicts regime for the week starting at target_date.
    Uses 90 days of history leading up to that date.
    No future data used — only history up to target date.

    assemble_stock_data_fn: the assemble_stock_data() function from
    main.py, injected to avoid a circular import (assembling data
    depends on config/data/labels, not on inference).
    """
    target_date = pd.Timestamp(target_date_str)
    lookback    = MODEL_CFG["lookback_window"]

    # ── Load checkpoint & scaler ──────────────────────────────────
    ckpt_path   = Path(CHECKPOINT_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}_best.pt"
    scaler_path = Path(CHECKPOINT_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}_scaler.pkl"

    if not ckpt_path.exists():
        print(f"  {stock_name}: no checkpoint found.")
        return None

    ckpt       = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    saved_cfg  = ckpt.get("model_cfg", MODEL_CFG)
    model      = RegimePredictionMamba(cfg=saved_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # ── Assemble full data ────────────────────────────────────────
    merged_df, feat_cols = assemble_stock_data_fn(stock_name)

    # ── Get 90 days leading up to target date ─────────────────────
    history = merged_df[merged_df["Date"] <= target_date].tail(lookback)

    if len(history) < lookback:
        print(f"  {stock_name}: not enough history before {target_date_str} "
              f"(found {len(history)}, need {lookback})")
        return None

    actual_start = history["Date"].iloc[0].date()
    actual_end   = history["Date"].iloc[-1].date()

    # ── Prepare tensors ───────────────────────────────────────────
    feat_arr    = scaler.transform(
        history[feat_cols].values
    ).astype(np.float32)
    regime_arr  = history["detected_regime"].values.astype(np.int64)

    feat_tensor   = torch.tensor(
        feat_arr, dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)
    regime_tensor = torch.tensor(
        regime_arr, dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    # ── Inference ─────────────────────────────────────────────────
    logits = model(feat_tensor, regime_tensor)
    probs  = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred   = int(probs.argmax())

    current_regime   = REGIME_INV_MAP[int(history["detected_regime"].iloc[-1])]
    predicted_regime = REGIME_INV_MAP[pred]
    transition       = current_regime != predicted_regime

    return {
        "stock"           : stock_name,
        "history_from"    : str(actual_start),
        "history_to"      : str(actual_end),
        "predicting_for"  : target_date_str,
        "current_regime"  : current_regime,
        "predicted_regime": predicted_regime,
        "transition"      : transition,
        "p_bearish"       : round(float(probs[0]), 4),
        "p_neutral"       : round(float(probs[1]), 4),
        "p_bullish"       : round(float(probs[2]), 4),
        "confidence"      : round(float(probs.max()), 4),
    }
