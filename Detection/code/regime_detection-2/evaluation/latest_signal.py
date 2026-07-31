"""
evaluation/latest_signal.py
=============================
Runs inference for a single target date using only the `lookback_window`
days of history up to (and including) that date — i.e. no future data is
used at inference time, even though the model was trained on pseudo labels
that were themselves derived from future returns (see config.py's module
docstring for that distinction). This produces the "latest detected
regime" chart.

Logic is unchanged from the original notebook. Structural (not logical)
change: functions take `device`, `all_stock_files`, `macro_df`, `nifty_df`,
`label_source` as explicit arguments instead of reading module-level
globals.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import CHECKPOINT_PATH, LABEL_SOURCE, MODEL_CFG, REGIME_INV_MAP, RUN_ID
from data.loaders import load_stock_csv
from dataset import assemble_stock_data
from model import RegimePredictionMamba


@torch.no_grad()
def predict_specific_week(stock_name, target_date_str, all_stock_files, macro_df, nifty_df, device):
    """
    Predicts regime for the week starting at target_date.
    Uses `lookback_window` days of history leading up to that date.
    No future data used — only history up to target date.
    """
    target_date = pd.Timestamp(target_date_str)
    lookback    = MODEL_CFG["lookback_window"]

    # ── Load checkpoint & scaler ──────────────────────────────────
    ckpt_path   = Path(CHECKPOINT_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}_best.pt"
    scaler_path = Path(CHECKPOINT_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}_scaler.pkl"

    if not ckpt_path.exists():
        print(f"  {stock_name}: no checkpoint found.")
        return None

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg  = ckpt.get("model_cfg", MODEL_CFG)
    model      = RegimePredictionMamba(cfg=saved_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # ── Assemble full data ────────────────────────────────────────
    merged_df, feat_cols = assemble_stock_data(
        stock_name, all_stock_files, macro_df, nifty_df, LABEL_SOURCE
    )

    # ── Get lookback_window days leading up to target date ─────────
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
    ).unsqueeze(0).to(device)
    regime_tensor = torch.tensor(
        regime_arr, dtype=torch.long
    ).unsqueeze(0).to(device)

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


def plot_prediction_context(stock_name, result, save_dir, all_stock_files):
    """Shows the lookback-window price history leading up to the latest detected regime call."""
    COLORS = {
        "Bullish": "#2ecc71",
        "Neutral": "#f39c12",
        "Bearish": "#e74c3c"
    }

    stock_df         = load_stock_csv(all_stock_files[stock_name])
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    history_start = pd.Timestamp(result["history_from"])
    history_end   = pd.Timestamp(result["history_to"])

    window = stock_df[
        (stock_df["Date"] >= history_start) &
        (stock_df["Date"] <= history_end)
    ].copy()

    pred_color = COLORS[result["predicted_regime"]]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(window["Date"], window["Close"],
            color="black", linewidth=1.5, zorder=3)
    ax.fill_between(window["Date"], window["Close"].min() * 0.98,
                    window["Close"],
                    color=pred_color, alpha=0.08, zorder=1)

    # Detected regime banner at top
    ax.axhspan(
        window["Close"].max() * 1.01,
        window["Close"].max() * 1.03,
        color=pred_color, alpha=0.6
    )

    transition_text = (
        f"⚠️  TRANSITION: {result['current_regime']} → {result['predicted_regime']}"
        if result["transition"]
        else f"✓  CONTINUATION: {result['predicted_regime']}"
    )

    ax.set_title(
        f"{stock_name} — Regime Detection for week of {result['predicting_for']}\n"
        f"{transition_text}  |  Confidence: {result['confidence']}",
        fontsize=11, fontweight="bold"
    )

    # Probability annotations
    textstr = (
        f"P(Bullish) = {result['p_bullish']:.3f}\n"
        f"P(Neutral) = {result['p_neutral']:.3f}\n"
        f"P(Bearish) = {result['p_bearish']:.3f}"
    )
    ax.text(
        0.02, 0.97, textstr,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.set_xlabel(f"Date ({MODEL_CFG['lookback_window']}-day lookback window)")
    ax.set_ylabel("Price (INR)")
    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter("%b %Y")
    )
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        Path(save_dir) / f"{stock_name}_nov2025_prediction.png",
        dpi=120, bbox_inches="tight"
    )
    plt.show()
    plt.close()
