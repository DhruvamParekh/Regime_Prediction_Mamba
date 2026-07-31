"""
evaluation/analysis.py
========================
Post-training analysis: accuracy by period/class, regime "quality" (how
long true vs. detected regimes actually last, and how quickly transitions
are picked up), returns if you'd traded on the detected regime, and the
matching plots (2x2 analysis panel, confusion matrices).

Logic is unchanged from the original notebook.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import REGIME_INV_MAP


def analyze_period_accuracy(full_pred_df, split_dates):
    """Accuracy overall and per-class, split by Train/Val/Test."""
    train_end  = pd.Timestamp(split_dates["train_end"])
    val_end    = pd.Timestamp(split_dates["val_end"])
    test_start = pd.Timestamp(split_dates["test_start"])

    periods = {
        "Train": full_pred_df[full_pred_df["Date"] <= train_end],
        "Val"  : full_pred_df[(full_pred_df["Date"] > train_end) &
                              (full_pred_df["Date"] <= val_end)],
        "Test" : full_pred_df[full_pred_df["Date"] >= test_start],
    }

    rows = []
    for period_name, pdf in periods.items():
        if len(pdf) == 0:
            continue
        overall_acc = (pdf["true_idx"] == pdf["pred_idx"]).mean()
        row = {
            "period"     : period_name,
            "n_samples"  : len(pdf),
            "overall_acc": round(overall_acc * 100, 2),
        }
        for cls in range(3):
            name  = REGIME_INV_MAP[cls]
            mask  = pdf["true_idx"] == cls
            if mask.sum() > 0:
                acc = (pdf.loc[mask, "pred_idx"] == cls).mean()
                row[f"{name}_acc"] = round(acc * 100, 2)
                row[f"{name}_n"]   = int(mask.sum())
            else:
                row[f"{name}_acc"] = None
                row[f"{name}_n"]   = 0
        rows.append(row)

    return pd.DataFrame(rows)


def analyze_regime_quality(full_pred_df, stock_df):
    """
    Compares true vs. detected regime "runs" (consecutive stretches of the
    same regime): average duration per class, and how often a true regime
    transition is picked up by the model within a ±7 day window.
    """
    def get_regime_runs(labels, dates):
        runs = []
        if len(labels) == 0:
            return runs
        current = labels[0]
        start   = dates[0]
        for i in range(1, len(labels)):
            if labels[i] != current:
                runs.append({
                    "regime"  : REGIME_INV_MAP[int(current)],
                    "start"   : pd.Timestamp(start),
                    "end"     : pd.Timestamp(dates[i-1]),
                    "duration": int((dates[i-1] - start) / np.timedelta64(1, "D"))
                })
                current = labels[i]
                start   = dates[i]
        runs.append({
            "regime"  : REGIME_INV_MAP[int(current)],
            "start"   : pd.Timestamp(start),
            "end"     : pd.Timestamp(dates[-1]),
            "duration": int((dates[-1] - start) / np.timedelta64(1, "D"))
        })
        return runs

    true_runs = get_regime_runs(
        full_pred_df["true_idx"].values,
        full_pred_df["Date"].values
    )
    pred_runs = get_regime_runs(
        full_pred_df["pred_idx"].values,
        full_pred_df["Date"].values
    )

    true_runs_df = pd.DataFrame(true_runs)
    pred_runs_df = pd.DataFrame(pred_runs)

    # Average duration per regime
    duration_stats = []
    for regime in ["Bullish", "Neutral", "Bearish"]:
        t_mask = true_runs_df["regime"] == regime
        p_mask = pred_runs_df["regime"] == regime
        duration_stats.append({
            "regime"        : regime,
            "true_avg_days" : round(true_runs_df.loc[t_mask, "duration"].mean(), 1) if t_mask.sum() > 0 else 0,
            "pred_avg_days" : round(pred_runs_df.loc[p_mask, "duration"].mean(), 1) if p_mask.sum() > 0 else 0,
            "true_count"    : int(t_mask.sum()),
            "pred_count"    : int(p_mask.sum()),
        })

    # Transition detection — ±7 days window
    # A transition is detected if model changes regime
    # within 7 days before or after the true transition
    transitions_detected = 0
    total_transitions    = len(true_runs) - 1

    for i in range(total_transitions):
        true_date   = true_runs[i]["end"]
        window_start = true_date - pd.Timedelta(days=7)
        window_end   = true_date + pd.Timedelta(days=7)

        pred_in_window = full_pred_df[
            (full_pred_df["Date"] >= window_start) &
            (full_pred_df["Date"] <= window_end)
        ]
        # Check if predicted regime changes within window
        if len(pred_in_window) > 1:
            if pred_in_window["pred_idx"].nunique() > 1:
                transitions_detected += 1

    transition_rate = (transitions_detected / total_transitions * 100
                       if total_transitions > 0 else 0)

    return {
        "duration_df"       : pd.DataFrame(duration_stats),
        "true_runs_df"      : true_runs_df,
        "pred_runs_df"      : pred_runs_df,
        "transition_rate"   : round(transition_rate, 2),
        "total_transitions" : total_transitions,
        "detected"          : transitions_detected,
    }


def analyze_returns_by_regime(full_pred_df, stock_df, split_dates):
    """
    For each period computes:
      - Total return while model is IN MARKET (predicted Bullish)
      - Total return of Buy and Hold in that period
      - Average daily return per predicted regime
    """
    stock_df         = stock_df.copy()
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df         = stock_df.sort_values("Date").reset_index(drop=True)
    stock_df["daily_return"] = stock_df["Close"].pct_change()

    # Forward fill predictions to every trading day
    merged = stock_df.merge(
        full_pred_df[["Date", "pred_label", "true_label",
                      "pred_idx", "true_idx"]],
        on="Date", how="left"
    )
    merged["pred_label"] = merged["pred_label"].ffill()
    merged["true_label"] = merged["true_label"].ffill()
    merged["pred_idx"]   = merged["pred_idx"].ffill()
    merged               = merged.dropna(
        subset=["pred_label", "daily_return"]
    ).reset_index(drop=True)

    train_end  = pd.Timestamp(split_dates["train_end"])
    val_end    = pd.Timestamp(split_dates["val_end"])
    test_start = pd.Timestamp(split_dates["test_start"])

    periods = {
        "Train": merged[merged["Date"] <= train_end],
        "Val"  : merged[(merged["Date"] > train_end) &
                        (merged["Date"] <= val_end)],
        "Test" : merged[merged["Date"] >= test_start],
    }

    period_rows  = []
    regime_rows  = []

    for period_name, pdf in periods.items():
        if len(pdf) == 0:
            continue

        # ── Model strategy: only in market when Bullish ───────────
        in_market     = pdf["pred_label"] == "Bullish"
        model_returns = pdf.loc[in_market, "daily_return"]
        model_total   = round(((1 + model_returns).prod() - 1) * 100, 3)

        # ── Buy and hold total return ─────────────────────────────
        bnh_total = round(((1 + pdf["daily_return"]).prod() - 1) * 100, 3)

        # ── Days in market ────────────────────────────────────────
        days_in    = int(in_market.sum())
        days_total = len(pdf)

        period_rows.append({
            "period"          : period_name,
            "model_return_pct": model_total,
            "bnh_return_pct"  : bnh_total,
            "alpha_pct"       : round(model_total - bnh_total, 3),
            "days_in_market"  : days_in,
            "days_total"      : days_total,
            "pct_in_market"   : round(days_in / days_total * 100, 1),
        })

        # ── Per regime returns ────────────────────────────────────
        for regime in ["Bullish", "Neutral", "Bearish"]:
            mask = pdf["pred_label"] == regime
            if mask.sum() == 0:
                continue
            rets = pdf.loc[mask, "daily_return"].dropna()
            regime_rows.append({
                "period"      : period_name,
                "pred_regime" : regime,
                "n_days"      : len(rets),
                "avg_daily"   : round(rets.mean() * 100, 4),
                "cum_return"  : round(((1 + rets).prod() - 1) * 100, 3),
                "win_rate"    : round((rets > 0).mean() * 100, 2),
            })

    return pd.DataFrame(period_rows), pd.DataFrame(regime_rows)


def plot_analysis(stock_name, period_acc_df, quality_dict,
                  period_ret_df, regime_ret_df, save_dir):
    """2x2 panel: accuracy by period/class, regime duration, model vs BnH return, regime returns."""

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
           0.4, label="Detected", color="#e67e22", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dur_df["regime"])
    ax.set_ylabel("Average Duration (days)")
    ax.set_title("True vs Detected Regime Duration")
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
                   width, label="Model (Long only)",
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
    ax.set_title("Model vs Buy & Hold Return per Period")
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
    ax.set_title("Actual Returns During Detected Regimes (Test)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        Path(save_dir) / f"{stock_name}_analysis.png",
        dpi=120, bbox_inches="tight"
    )
    plt.show()
    plt.close()


def plot_confusion_matrices(stock_name, full_pred_df, split_dates, save_dir):
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
