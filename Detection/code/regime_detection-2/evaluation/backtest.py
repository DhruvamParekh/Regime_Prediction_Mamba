"""
evaluation/backtest.py
========================
Simple long/cash backtest over the test period only: Model (long when
detected regime is Bullish), Perfect Foresight (long when the true label
was Bullish — the theoretical ceiling), and Buy & Hold.

Logic is unchanged from the original notebook. Structural (not logical)
change: `run_backtest` takes `all_stock_files` as an explicit argument
instead of reading a module-level global.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import LABEL_SOURCE, SPLIT_DATES
from data.loaders import load_stock_csv


def run_backtest(stock_name, full_pred_df, all_stock_files):
    """
    Long/Cash backtest on test period only.

    Strategy A — Model   : Long when predicted Bullish, cash otherwise
    Strategy B — Pseudo  : Long when true label Bullish, cash otherwise
    Strategy C — Buy&Hold: Always in market

    Only changes position on regime change — no daily churning.
    No transaction costs. No shorting.
    """
    stock_df         = load_stock_csv(all_stock_files[stock_name])
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df         = stock_df.sort_values("Date").reset_index(drop=True)
    stock_df["daily_return"] = stock_df["Close"].pct_change()

    # ── Filter to test period ─────────────────────────────────────
    test_start = pd.Timestamp(SPLIT_DATES["test_start"])
    pred_test  = full_pred_df[
        full_pred_df["Date"] >= test_start
    ].copy().reset_index(drop=True)

    price_test = stock_df[
        stock_df["Date"] >= test_start
    ].copy().reset_index(drop=True)

    if len(pred_test) < 2:
        print(f"  {stock_name}: not enough test data.")
        return None

    # ── Forward fill predictions to every trading day ─────────────
    merged = price_test.merge(
        pred_test[["Date", "pred_label", "true_label",
                   "pred_idx", "true_idx"]],
        on="Date", how="left"
    )
    merged["pred_label"] = merged["pred_label"].ffill()
    merged["true_label"] = merged["true_label"].ffill()
    merged["pred_idx"]   = merged["pred_idx"].ffill()
    merged["true_idx"]   = merged["true_idx"].ffill()
    merged = merged.dropna(
        subset=["pred_label", "daily_return"]
    ).reset_index(drop=True)

    # ── Strategy A: Model ─────────────────────────────────────────
    model_pos  = []
    a_prev_pos = 0
    a_prev     = None
    for _, row in merged.iterrows():
        if row["pred_label"] != a_prev:
            a_prev_pos = 1 if row["pred_label"] == "Bullish" else 0
            a_prev     = row["pred_label"]
        model_pos.append(a_prev_pos)

    merged["model_pos"] = model_pos
    merged["model_ret"] = merged["model_pos"].shift(1).fillna(0) * merged["daily_return"]

    # ── Strategy B: Perfect label foresight ──────────────────────
    b_pos      = []
    b_prev_pos = 0
    b_prev     = None
    for _, row in merged.iterrows():
        if row["true_label"] != b_prev:
            b_prev_pos = 1 if row["true_label"] == "Bullish" else 0
            b_prev     = row["true_label"]
        b_pos.append(b_prev_pos)

    merged["pseudo_pos"] = b_pos
    merged["pseudo_ret"] = merged["pseudo_pos"].shift(1).fillna(0) * merged["daily_return"]

    strategy_b_label = f"Perfect {LABEL_SOURCE.capitalize()} Foresight"

    # ── Strategy C: Buy and Hold ──────────────────────────────────
    merged["bnh_ret"] = merged["daily_return"]

    # ── Cumulative returns ────────────────────────────────────────
    merged["model_cum"]  = (1 + merged["model_ret"]).cumprod()
    merged["pseudo_cum"] = (1 + merged["pseudo_ret"]).cumprod()
    merged["bnh_cum"]    = (1 + merged["bnh_ret"]).cumprod()

    # ── Metrics ───────────────────────────────────────────────────
    def metrics(cum_ret, daily_ret, label):
        total_ret    = cum_ret.iloc[-1] - 1
        n_days       = len(daily_ret)
        annual_ret   = (1 + total_ret) ** (252 / n_days) - 1
        vol          = daily_ret.std() * np.sqrt(252)
        sharpe       = annual_ret / (vol + 1e-9)
        roll_max     = cum_ret.cummax()
        max_dd       = ((cum_ret - roll_max) / roll_max).min()
        in_mkt       = daily_ret[daily_ret != 0]
        win_rate     = (in_mkt > 0).mean() if len(in_mkt) > 0 else 0
        return {
            "strategy"      : label,
            "total_ret_pct" : round(total_ret * 100, 2),
            "annual_ret_pct": round(annual_ret * 100, 2),
            "volatility_pct": round(vol * 100, 2),
            "sharpe"        : round(sharpe, 3),
            "max_drawdown"  : round(max_dd * 100, 2),
            "win_rate_pct"  : round(win_rate * 100, 2),
            "days_in_market": int((daily_ret != 0).sum()),
        }

    metrics_df = pd.DataFrame([
        metrics(merged["model_cum"],  merged["model_ret"],  "Model"),
        metrics(merged["pseudo_cum"], merged["pseudo_ret"], strategy_b_label),
        metrics(merged["bnh_cum"],    merged["bnh_ret"],    "Buy and Hold"),
    ])

    return {
        "stock"  : stock_name,
        "merged" : merged,
        "metrics": metrics_df,
    }


def plot_backtest(result, save_dir):
    """Cumulative return curves for all three strategies, plus a position (long/cash) strip."""
    stock_name = result["stock"]
    merged     = result["merged"]
    metrics_df = result["metrics"]

    def get_ret(label):
        return metrics_df.loc[
            metrics_df["strategy"] == label, "total_ret_pct"
        ].values[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(
        f"{stock_name} — Backtest (Test Period: "
        f"{SPLIT_DATES['test_start']} onwards)",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: Cumulative returns ───────────────────────────────
    ax1.plot(merged["Date"], merged["model_cum"],
             color="#3498db", linewidth=2,
             label=f"Model ({get_ret('Model'):+.1f}%)")
    ax1.plot(merged["Date"], merged["pseudo_cum"],
         color="#2ecc71", linewidth=2, linestyle="--",
         label=f"Perfect {LABEL_SOURCE.capitalize()} Foresight ({get_ret(f'Perfect {LABEL_SOURCE.capitalize()} Foresight'):+.1f}%)")
    ax1.plot(merged["Date"], merged["bnh_cum"],
             color="#e74c3c", linewidth=2, linestyle=":",
             label=f"Buy and Hold ({get_ret('Buy and Hold'):+.1f}%)")
    ax1.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)

    # Shade in-market periods
    in_market = merged["model_pos"] > 0
    for i in range(len(merged) - 1):
        if in_market.iloc[i]:
            ax1.axvspan(
                merged["Date"].iloc[i],
                merged["Date"].iloc[i+1],
                color="#3498db", alpha=0.04
            )

    ax1.set_ylabel("Cumulative Return (1 = start)")
    ax1.set_title("Cumulative Returns — Model vs Pseudo Label vs Buy & Hold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Model position ───────────────────────────────────
    ax2.fill_between(
        merged["Date"], merged["model_pos"],
        step="post", color="#3498db", alpha=0.5
    )
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Cash", "Long"])
    ax2.set_ylabel("Position")
    ax2.set_title("Model Position")
    ax2.grid(alpha=0.3)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%b %Y")
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(
        Path(save_dir) / f"{stock_name}_backtest.png",
        dpi=120, bbox_inches="tight"
    )
    plt.show()
    plt.close()

    # ── Metrics table ─────────────────────────────────────────────
    print(f"\n  {stock_name} — Backtest Metrics (Test Period):")
    print(metrics_df.to_string(index=False))
