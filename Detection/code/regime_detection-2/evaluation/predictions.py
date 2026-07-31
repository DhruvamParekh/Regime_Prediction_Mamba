"""
evaluation/predictions.py
==========================
Reloads a stock's best checkpoint and re-runs it over its ENTIRE history
(not just the test period) to build a full prediction timeline, then plots
it as a 4-panel chart (price + regime dots, predicted regime timeline,
stacked probability area, confidence).

Logic is unchanged from the original notebook. Structural (not logical)
change: functions now take `device`, `all_stock_files`, `val_acc`/`test_acc`
as explicit arguments instead of reading module-level globals — the numbers
computed are identical either way.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import CHECKPOINT_PATH, LABEL_SOURCE, MODEL_CFG, REGIME_INV_MAP, RESULTS_PATH, RUN_ID, SPLIT_DATES
from data.loaders import load_stock_csv
from model import RegimePredictionMamba


@torch.no_grad()
def get_full_predictions(stock_name, merged_df, feat_cols, device):
    """Reload the best checkpoint for a stock and predict over its full history."""
    ckpt_path   = Path(CHECKPOINT_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}_best.pt"
    scaler_path = Path(CHECKPOINT_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}_scaler.pkl"

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("model_cfg", MODEL_CFG)
    model     = RegimePredictionMamba(cfg=saved_cfg).to(device)
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
        ).unsqueeze(0).to(device)
        regime = torch.tensor(
            regime_arr[start:end], dtype=torch.long
        ).unsqueeze(0).to(device)

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


def plot_full_regime_chart(stock_name, full_pred_df, merged_df, all_stock_files, val_acc, test_acc):
    """4-panel chart: price+regime dots, predicted regime timeline, probability area, confidence."""
    result_dir = Path(RESULTS_PATH) / \
        f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}"
    result_dir.mkdir(parents=True, exist_ok=True)

    COLORS = {
        "Bullish": "#2ecc71",
        "Neutral": "#aaaaaa",
        "Bearish": "#e74c3c"
    }

    stock_df = load_stock_csv(all_stock_files[stock_name])
    stock_df = stock_df[
        stock_df["Date"] >= full_pred_df["Date"].min()
    ].copy()

    val_start  = pd.Timestamp(SPLIT_DATES["train_end"])
    test_start = pd.Timestamp(SPLIT_DATES["test_start"])

    fig, axes = plt.subplots(
        4, 1, figsize=(18, 14),
        gridspec_kw={"height_ratios": [3, 1, 1.5, 1]}
    )
    fig.suptitle(
        f"Regime Detection — {stock_name}  "
        f"(Val Acc: {val_acc*100:.1f}%  |  Test Acc: {test_acc*100:.1f}%)",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: Price + coloured dots ───────────────────────────
    ax = axes[0]
    ax.plot(stock_df["Date"], stock_df["Close"],
            color="black", linewidth=0.8, alpha=0.5, zorder=2)

    for regime, color in COLORS.items():
        mask     = full_pred_df["true_label"] == regime
        dates_r  = full_pred_df.loc[mask, "Date"]
        prices_r = []
        for d in dates_r:
            row = stock_df[stock_df["Date"] == d]
            prices_r.append(
                row["Close"].values[0] if len(row) > 0 else np.nan
            )
        ax.scatter(dates_r, prices_r, color=color, s=18, zorder=4,
                   label=f"{regime} ({mask.sum()})")

    ax.axvline(val_start,  color="blue",   linestyle="--",
               linewidth=1, alpha=0.7, label="Val start")
    ax.axvline(test_start, color="purple", linestyle="--",
               linewidth=1, alpha=0.7, label="Test start")
    ax.set_ylabel("Price (INR)")
    ax.set_title("Detected Regime on Price")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.2)

    # ── Panel 2: Predicted regime timeline ───────────────────────
    ax = axes[1]
    regime_y = {"Neutral": 2, "Bullish": 1, "Bearish": 0}
    for _, row in full_pred_df.iterrows():
        ax.scatter(row["Date"], regime_y[row["pred_label"]],
                   color=COLORS[row["pred_label"]], s=25,
                   marker="s", zorder=3)
    ax.axvline(val_start,  color="blue",   linestyle="--",
               linewidth=1, alpha=0.7)
    ax.axvline(test_start, color="purple", linestyle="--",
               linewidth=1, alpha=0.7)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Bear", "Bull", "Neut"], fontsize=8)
    ax.set_title("Detected Regime Timeline")
    ax.grid(alpha=0.2)

    # ── Panel 3: Stacked probability area ────────────────────────
    ax    = axes[2]
    dates = full_pred_df["Date"]
    ax.fill_between(dates, 0,
                    full_pred_df["p_bearish"],
                    color="#e74c3c", alpha=0.6, label="Bearish")
    ax.fill_between(dates,
                    full_pred_df["p_bearish"],
                    full_pred_df["p_bearish"] + full_pred_df["p_bullish"],
                    color="#2ecc71", alpha=0.6, label="Bullish")
    ax.fill_between(dates,
                    full_pred_df["p_bearish"] + full_pred_df["p_bullish"],
                    1.0, color="#aaaaaa", alpha=0.4, label="Neutral")
    ax.axvline(val_start,  color="blue",   linestyle="--",
               linewidth=1, alpha=0.7)
    ax.axvline(test_start, color="purple", linestyle="--",
               linewidth=1, alpha=0.7)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Regime Probabilities")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.2)

    # ── Panel 4: Confidence ───────────────────────────────────────
    ax = axes[3]
    ax.fill_between(dates, 0, full_pred_df["confidence"],
                    color="#9b59b6", alpha=0.5)
    ax.plot(dates, full_pred_df["confidence"],
            color="#9b59b6", linewidth=0.8)
    ax.axhline(0.33, color="red",   linestyle="--",
               linewidth=1, label="Random (0.33)")
    ax.axhline(0.60, color="green", linestyle="--",
               linewidth=1, label="Threshold (0.6)")
    ax.axvline(val_start,  color="blue",   linestyle="--",
               linewidth=1, alpha=0.7)
    ax.axvline(test_start, color="purple", linestyle="--",
               linewidth=1, alpha=0.7)
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    ax.set_title("Prediction Confidence")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)

    for ax in axes:
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%b %Y")
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.set_xlim(
            full_pred_df["Date"].min(),
            full_pred_df["Date"].max()
        )

    plt.tight_layout()
    plt.savefig(
        result_dir / f"{stock_name}_full_regime_chart.png",
        dpi=130, bbox_inches="tight"
    )
    plt.show()
    plt.close()
