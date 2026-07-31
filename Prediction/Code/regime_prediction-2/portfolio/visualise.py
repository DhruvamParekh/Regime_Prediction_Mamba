"""
portfolio/visualise.py
=========================
All chart functions for the rolling portfolio: equity curves +
portfolio size, stock selection frequency, a presence heatmap,
streak-length bars, and per-stock return contribution.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "EqWt_LongOnly"  : "#3498db",
    "EqWt_LongShort" : "#2980b9",
    "AccWt_LongOnly" : "#2ecc71",
    "AccWt_LongShort": "#27ae60",
    "BnH_EqWt"       : "#e74c3c",
}
STYLES = {
    "EqWt_LongOnly"  : "-",
    "EqWt_LongShort" : "--",
    "AccWt_LongOnly" : "-",
    "AccWt_LongShort": "--",
    "BnH_EqWt"       : ":",
}


def plot_equity_and_size(equity, review_log, all_trading_days, top_k,
                          run_id, lookback_days, review_days, fwd_days, causal_lag,
                          results_path):
    """Equity curves for all 5 strategies (top) + active-stock count over time (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"Rolling Portfolio Simulation — Run {run_id}\n"
        f"Lookback={lookback_days}d | Review={review_days}d | Top-K={top_k} "
        f"| Hybrid causal labels (fwd-{fwd_days}d + causal-lag-{causal_lag}d)",
        fontsize=12, fontweight="bold"
    )

    ax1 = axes[0]
    for k, ser in equity.items():
        total_ret = (ser.iloc[-1] - 1) * 100
        ax1.plot(ser.index, ser.values,
                 color=COLORS[k], linestyle=STYLES[k], linewidth=1.8,
                 label=f"{k} ({total_ret:+.1f}%)")

    ax1.axhline(1.0, color="black", linewidth=0.6, alpha=0.4)
    ax1.set_ylabel("Cumulative Return (1 = start)")
    ax1.set_title("Equity Curves")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    active_n = pd.Series(index=all_trading_days, dtype=float)
    current_count = 0
    for i, today in enumerate(all_trading_days):
        for entry in review_log:
            if entry["date"] == today:
                current_count = len(entry["stocks"])
        active_n.iloc[i] = current_count

    ax2 = axes[1]
    ax2.fill_between(all_trading_days, active_n.values,
                     step="post", color="#3498db", alpha=0.4)
    ax2.set_ylabel("Active stocks")
    ax2.set_ylim(0, top_k + 2)
    ax2.set_title("Portfolio Size")
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(Path(results_path) / f"rolling_portfolio_Run{run_id}.png",
                dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_selection_frequency(freq_df, n_reviews, run_id, results_path):
    """Bar chart of how often each stock was selected, colour-tiered by rank."""
    fig, ax = plt.subplots(figsize=(18, 6))
    top_n = min(25, len(freq_df))
    df_plot = freq_df.head(top_n)
    colors_bar = ["#1F3864" if r <= 5 else "#2E75B6" if r <= 10 else "#9DC3E6"
                  for r in df_plot["rank"]]
    bars = ax.bar(df_plot["stock"], df_plot["selections"], color=colors_bar, alpha=0.87)
    for bar, val in zip(bars, df_plot["selections"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(int(val)), ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(n_reviews * 0.5, color="red", linestyle="--", alpha=0.5,
               label=f"50% of reviews ({n_reviews//2})")
    ax.axhline(n_reviews * 0.25, color="orange", linestyle=":", alpha=0.5,
               label=f"25% of reviews ({n_reviews//4})")
    ax.set_xlabel("Stock")
    ax.set_ylabel("Times Selected")
    ax.set_title(f"Stock Selection Frequency — Run {run_id} ({n_reviews} reviews)\n"
                 f"Dark = top 5 most selected, Medium = top 10, Light = rest")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(Path(results_path) / f"selection_frequency_Run{run_id}.png",
                dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_presence_heatmap(freq_df, daily_active, all_trading_days, run_id, results_path):
    """Heatmap of which of the top-20 most-selected stocks were active on each trading day."""
    top_stocks_heatmap = freq_df.head(20)["stock"].tolist()
    n_td = len(all_trading_days)
    heatmap = np.zeros((len(top_stocks_heatmap), n_td))

    for j, (today, active_set) in enumerate(daily_active):
        for i_s, s in enumerate(top_stocks_heatmap):
            if s in active_set:
                heatmap[i_s, j] = 1.0

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.imshow(heatmap, aspect="auto", cmap="Blues", vmin=0, vmax=1,
              extent=[0, n_td, len(top_stocks_heatmap)-0.5, -0.5])
    ax.set_yticks(range(len(top_stocks_heatmap)))
    ax.set_yticklabels(top_stocks_heatmap, fontsize=9)
    ax.set_xlabel("Trading Day Index")
    ax.set_title(f"Portfolio Presence Heatmap — Run {run_id}\n"
                 f"Blue = stock active in portfolio on that day (top 20 by selection freq)")
    tick_positions = np.linspace(0, n_td-1, 10, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(all_trading_days[p].date()) for p in tick_positions],
                       rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(results_path) / f"portfolio_heatmap_Run{run_id}.png",
                dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_streak_analysis(streak_df, run_id, results_path):
    """Two side-by-side bar charts: longest single streak and average streak length (top 15 stocks)."""
    streak_plot = streak_df.head(15)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Streak Analysis — Run {run_id}", fontsize=12, fontweight="bold")

    ax1.barh(streak_plot["stock"][::-1], streak_plot["max_streak_wks"][::-1],
             color="#1F3864", alpha=0.85)
    ax1.set_xlabel("Max Streak (weeks)")
    ax1.set_title("Longest Single Streak (weeks in portfolio)")
    ax1.grid(axis="x", alpha=0.25)
    for i, (val, s) in enumerate(zip(streak_plot["max_streak_wks"][::-1],
                                      streak_plot["stock"][::-1])):
        ax1.text(val + 0.2, i, f"{val}wk", va="center", fontsize=9)

    ax2.barh(streak_plot["stock"][::-1], streak_plot["avg_streak"][::-1],
             color="#2E75B6", alpha=0.85)
    ax2.set_xlabel("Avg Streak Length (reviews)")
    ax2.set_title("Average Streak Length (review periods)")
    ax2.grid(axis="x", alpha=0.25)
    for i, (val, s) in enumerate(zip(streak_plot["avg_streak"][::-1],
                                      streak_plot["stock"][::-1])):
        ax2.text(val + 0.05, i, f"{val:.1f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(results_path) / f"streak_analysis_Run{run_id}.png",
                dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_stock_contribution(contrib_df, run_id, results_path):
    """Horizontal bar chart of each stock's cumulative contribution to the EqWt_LongOnly return."""
    contrib_plot = contrib_df.sort_values("total_contrib_pct", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(12, 7))
    bar_colors = ["#1E7B34" if v >= 0 else "#C00000" for v in contrib_plot["total_contrib_pct"]]
    ax.barh(contrib_plot["stock"], contrib_plot["total_contrib_pct"],
            color=bar_colors, alpha=0.87)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cumulative Return Contribution (%)")
    ax.set_title(f"Per-Stock Return Contribution — EqWt Long-Only — Run {run_id}\n"
                 f"Sum of daily weighted position returns across all active days")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(Path(results_path) / f"stock_contribution_Run{run_id}.png",
                dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()
