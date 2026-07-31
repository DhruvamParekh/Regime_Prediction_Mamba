"""
visualise.py
============
All matplotlib chart functions used across the pipeline: the full
regime chart (price + predicted regime + probabilities + confidence),
the period-accuracy / duration / return analysis panel, confusion
matrices, the specific-week prediction-context chart, and the
cross-stock summary chart.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import LABEL_SOURCE, RESULTS_PATH, RUN_ID, SPLIT_DATES
from data.loader import load_stock_csv


def plot_full_regime_chart(stock_name, full_pred_df, merged_df, all_stock_files,
                            best_val_acc, test_acc):
    """
    Four-panel chart: (1) price with coloured true-regime dots,
    (2) predicted-regime timeline, (3) stacked regime-probability
    area, (4) prediction confidence over time.
    """
    # ── result_dir defined once here — used throughout ────────────
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
        f"Regime Prediction — {stock_name}  "
        f"(Val Acc: {best_val_acc*100:.1f}%  |  Test Acc: {test_acc*100:.1f}%)",
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
    ax.set_title("Predicted Regime on Price")
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
    ax.set_title("Predicted Regime Timeline")
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


def plot_analysis(stock_name, period_acc_df, quality_dict,
                  period_ret_df, regime_ret_df, save_dir):
    """
    Four-panel analysis chart: accuracy by period & regime class,
    true vs predicted regime duration, model vs buy&hold return per
    period, and cumulative return by predicted regime (test period).
    """

    COLORS = {
        "Bullish": "#2ecc71",
        "Neutral": "#f39c12",
        "Bearish": "#e74c3c"
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"{stock_name} — Performance Analysis",
        fontsize=13, fontweight="bold"
    )

    # ── Plot 1: Accuracy by period ────────────────────────────────
    ax    = axes[0, 0]
    x     = np.arange(len(period_acc_df))
    width = 0.18

    ax.bar(x - width*1.5, period_acc_df["overall_acc"],
           width, label="Overall", color="#3498db")
    for i, regime in enumerate(["Bearish", "Neutral", "Bullish"]):
        col = f"{regime}_acc"
        if col in period_acc_df.columns:
            ax.bar(x + (i - 0.5) * width,
                   period_acc_df[col].fillna(0),
                   width, label=regime,
                   color=COLORS[regime], alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(period_acc_df["period"])
    ax.axhline(33.3, color="red", linestyle="--",
               linewidth=1, alpha=0.5)
    ax.set_ylabel("Accuracy %")
    ax.set_title("Accuracy by Period & Regime Class")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 2: Regime duration ───────────────────────────────────
    ax     = axes[0, 1]
    dur_df = quality_dict["duration_df"]
    x      = np.arange(len(dur_df))
    ax.bar(x - 0.2, dur_df["true_avg_days"],
           0.4, label="True", color="#3498db", alpha=0.8)
    ax.bar(x + 0.2, dur_df["pred_avg_days"],
           0.4, label="Predicted", color="#e67e22", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dur_df["regime"])
    ax.set_ylabel("Average Duration (days)")
    ax.set_title("True vs Predicted Regime Duration")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.98, 0.98,
        f"Transition detection : {quality_dict['transition_rate']}%\n"
        f"({quality_dict['detected']} / {quality_dict['total_transitions']})\n"
        f"Window : ±7 days",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    # ── Plot 3: Model vs BnH return per period ────────────────────
    ax    = axes[1, 0]
    x     = np.arange(len(period_ret_df))
    width = 0.3
    bars1 = ax.bar(x - width/2,
                   period_ret_df["model_return_pct"],
                   width, label="Model (Long/Short)",
                   color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width/2,
                   period_ret_df["bnh_return_pct"],
                   width, label="Buy & Hold",
                   color="#e74c3c", alpha=0.85)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + 0.5 if h >= 0 else h - 2,
                f"{h:.1f}%", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + 0.5 if h >= 0 else h - 2,
                f"{h:.1f}%", ha="center", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(period_ret_df["period"])
    ax.set_ylabel("Total Return %")
    ax.set_title("Model (L/S) vs Buy & Hold Return per Period")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Annotate alpha
    for i, row in period_ret_df.iterrows():
        color = "#2ecc71" if row["alpha_pct"] >= 0 else "#e74c3c"
        ax.text(i, min(row["model_return_pct"],
                       row["bnh_return_pct"]) - 4,
                f"α={row['alpha_pct']:+.1f}%",
                ha="center", fontsize=8, color=color, fontweight="bold")

    # ── Plot 4: Cumulative return by predicted regime (Test) ──────
    ax       = axes[1, 1]
    test_reg = regime_ret_df[regime_ret_df["period"] == "Test"]
    if len(test_reg) > 0:
        bar_colors = [COLORS.get(r, "grey") for r in test_reg["pred_regime"]]
        bars = ax.bar(test_reg["pred_regime"],
                      test_reg["cum_return"],
                      color=bar_colors, alpha=0.85, edgecolor="none")
        for bar, val in zip(bars, test_reg["cum_return"]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2,
                    f"{val:.1f}%", ha="center", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative Return %")
    ax.set_title("Actual Returns During Predicted Regimes (Test)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        Path(save_dir) / f"{stock_name}_analysis.png",
        dpi=120, bbox_inches="tight"
    )
    plt.show()
    plt.close()


def plot_confusion_matrices(stock_name, full_pred_df, split_dates, save_dir):
    """Plots Train/Val/Test confusion matrices (counts + row-normalised %)."""
    from sklearn.metrics import confusion_matrix

    train_end  = pd.Timestamp(split_dates["train_end"])
    val_end    = pd.Timestamp(split_dates["val_end"])
    test_start = pd.Timestamp(split_dates["test_start"])

    periods = {
        "Train": full_pred_df[full_pred_df["Date"] <= train_end],
        "Val"  : full_pred_df[(full_pred_df["Date"] > train_end) &
                              (full_pred_df["Date"] <= val_end)],
        "Test" : full_pred_df[full_pred_df["Date"] >= test_start],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{stock_name} — Confusion Matrices",
                 fontsize=12, fontweight="bold")

    for ax, (period_name, pdf) in zip(axes, periods.items()):
        if len(pdf) < 2:
            ax.set_title(f"{period_name} (no data)")
            continue
        cm     = confusion_matrix(
            pdf["true_idx"], pdf["pred_idx"], labels=[0, 1, 2]
        )
        cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100
        im     = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label="%")
        classes = ["Bearish", "Neutral", "Bullish"]
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(classes, rotation=30, fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{period_name} (n={len(pdf)})")
        for i in range(3):
            for j in range(3):
                ax.text(j, i,
                        f"{cm[i,j]}\n({cm_pct[i,j]:.0f}%)",
                        ha="center", va="center",
                        color="white" if cm_pct[i,j] > 50 else "black",
                        fontsize=8)

    plt.tight_layout()
    plt.savefig(
        Path(save_dir) / f"{stock_name}_confusion_matrices.png",
        dpi=120, bbox_inches="tight"
    )
    plt.show()
    plt.close()


def plot_prediction_context(stock_name, result, save_dir, all_stock_files):
    """
    Shows 90-day price history leading up to prediction
    with predicted regime displayed clearly.
    """
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

    # Predicted regime banner at top
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
        f"{stock_name} — Prediction for week of {result['predicting_for']}\n"
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

    ax.set_xlabel("Date (90-day lookback window)")
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


def plot_cross_stock_summary(summary_df):
    """
    Three-panel cross-stock summary: val vs test accuracy, model vs
    buy&hold total return, and alpha vs buy&hold (green=positive,
    red=negative), one bar per stock.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Cross-Stock Summary",
        fontsize=13, fontweight="bold"
    )

    stocks = summary_df["stock"].tolist()
    x      = np.arange(len(stocks))

    # ── Plot 1: Val vs Test accuracy ──────────────────────────────
    ax = axes[0]
    ax.bar(x - 0.2, summary_df["val_acc"],
           0.4, label="Val Acc", color="#3498db", alpha=0.8)
    ax.bar(x + 0.2, summary_df["test_acc"].fillna(0),
           0.4, label="Test Acc", color="#e67e22", alpha=0.8)
    ax.axhline(33.3, color="red", linestyle="--",
               linewidth=1, alpha=0.5, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(stocks, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy %")
    ax.set_title("Val vs Test Accuracy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 2: Model vs BnH total return ─────────────────────────
    ax = axes[1]
    ax.bar(x - 0.2, summary_df["model_ret_pct"].fillna(0),
           0.4, label="Model", color="#2ecc71", alpha=0.8)
    ax.bar(x + 0.2, summary_df["bnh_ret_pct"].fillna(0),
           0.4, label="Buy & Hold", color="#e74c3c", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stocks, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total Return %")
    ax.set_title("Model vs Buy & Hold (Test Period)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 3: Alpha vs BnH ──────────────────────────────────────
    ax     = axes[2]
    alphas = summary_df["alpha_vs_bnh"].fillna(0)
    colors = ["#2ecc71" if a >= 0 else "#e74c3c" for a in alphas]
    ax.bar(x, alphas, color=colors, alpha=0.85, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8)
    for i, (val, stock) in enumerate(zip(alphas, stocks)):
        ax.text(i, val + 0.3 if val >= 0 else val - 1.5,
                f"{val:+.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(stocks, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Alpha %")
    ax.set_title("Alpha vs Buy & Hold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        Path(RESULTS_PATH) / f"cross_stock_summary_{LABEL_SOURCE.upper()}_Run{RUN_ID}.png",
        dpi=120, bbox_inches="tight"
    )
    plt.show()
    plt.close()
