"""
evaluate.py
===========
Post-prediction analysis functions: per-period (Train/Val/Test)
accuracy breakdowns, regime-duration and transition-detection
quality metrics, and return-by-regime analysis comparing the
model's implied strategy against buy-and-hold.
"""

import numpy as np
import pandas as pd

from config import REGIME_INV_MAP


def analyze_period_accuracy(full_pred_df, split_dates):
    """
    Splits full_pred_df into Train/Val/Test by date and computes
    overall accuracy plus per-class (Bearish/Neutral/Bullish)
    accuracy and sample count within each period.
    """
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
    Computes true-vs-predicted regime run statistics: average
    duration per regime, and a transition-detection rate — the
    fraction of true regime transitions the model catches within
    a ±7-day window.
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

        # ── Model strategy: Long on Bullish, Short on Bearish ─────
        pos_map       = {"Bullish": 1, "Bearish": -1, "Neutral": 0}
        model_pos_ser = pdf["pred_label"].map(pos_map).shift(1).fillna(0)
        model_ret_ser = model_pos_ser * pdf["daily_return"]
        model_total   = round(((1 + model_ret_ser).prod() - 1) * 100, 3)

        # ── Buy and hold total return ─────────────────────────────
        bnh_total = round(((1 + pdf["daily_return"]).prod() - 1) * 100, 3)

        # ── Days in market (long or short) ────────────────────────
        days_in    = int((model_pos_ser != 0).sum())
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
