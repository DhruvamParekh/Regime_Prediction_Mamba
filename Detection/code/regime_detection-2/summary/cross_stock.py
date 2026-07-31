"""
summary/cross_stock.py
========================
Pulls together per-stock accuracy, backtest, and feature-importance results
into one cross-stock summary table + three comparison charts, and prints a
readable report.

Logic is unchanged from the original notebook. Structural (not logical)
change: functions take `all_results`, `all_backtest`, `all_importance` as
explicit arguments instead of reading module-level globals.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import LABEL_SOURCE, RESULTS_PATH, RUN_ID


def build_cross_stock_summary(all_results, all_backtest, all_importance):
    rows = []
    for stock_name, res in all_results.items():

        bt     = all_backtest.get(stock_name)
        imp_df = all_importance.get(stock_name)

        # ── Accuracy from saved CSV ───────────────────────────────
        acc_path = Path(RESULTS_PATH) / \
            f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}" / \
            "period_accuracy.csv"
        acc_df   = pd.read_csv(acc_path) if acc_path.exists() else None
        test_acc = None
        if acc_df is not None:
            test_row = acc_df[acc_df["period"] == "Test"]
            if len(test_row) > 0:
                test_acc = test_row["overall_acc"].values[0]

        # ── Backtest metrics ──────────────────────────────────────
        model_ret  = None
        pseudo_ret = None
        bnh_ret    = None
        sharpe     = None
        max_dd     = None

        if bt is not None:
            m            = bt["metrics"]
            strategy_b   = f"Perfect {LABEL_SOURCE.capitalize()} Foresight"

            model_rows   = m.loc[m["strategy"] == "Model", "total_ret_pct"]
            pseudo_rows  = m.loc[m["strategy"] == strategy_b, "total_ret_pct"]
            bnh_rows     = m.loc[m["strategy"] == "Buy and Hold", "total_ret_pct"]
            sharpe_rows  = m.loc[m["strategy"] == "Model", "sharpe"]
            maxdd_rows   = m.loc[m["strategy"] == "Model", "max_drawdown"]

            model_ret    = model_rows.values[0]  if len(model_rows)  > 0 else None
            pseudo_ret   = pseudo_rows.values[0] if len(pseudo_rows) > 0 else None
            bnh_ret      = bnh_rows.values[0]    if len(bnh_rows)    > 0 else None
            sharpe       = sharpe_rows.values[0] if len(sharpe_rows) > 0 else None
            max_dd       = maxdd_rows.values[0]  if len(maxdd_rows)  > 0 else None

        # ── Top feature ───────────────────────────────────────────
        top_feature = None
        if imp_df is not None:
            top_feature = imp_df.iloc[0]["feature"]

        # ── Latest signal (the "nov2025"-style prediction) ────────
        nov_path = Path(RESULTS_PATH) / \
            f"nov2025_predictions_{LABEL_SOURCE.upper()}_Run{RUN_ID}.csv"
        nov_pred = None
        if nov_path.exists():
            nov_df  = pd.read_csv(nov_path)
            nov_row = nov_df[nov_df["stock"] == stock_name]
            if len(nov_row) > 0:
                nov_pred = nov_row["predicted_regime"].values[0]

        rows.append({
            "stock"         : stock_name,
            "val_acc"       : round(res["best_val_acc"] * 100, 2),
            "test_acc"      : test_acc,
            "model_ret_pct" : model_ret,
            "pseudo_ret_pct": pseudo_ret,
            "bnh_ret_pct"   : bnh_ret,
            "alpha_vs_bnh"  : round(model_ret - bnh_ret, 2)
                              if model_ret is not None and bnh_ret is not None
                              else None,
            "sharpe"        : sharpe,
            "max_drawdown"  : max_dd,
            "top_feature"   : top_feature,
            "nov2025_pred"  : nov_pred,
            "label_source"  : LABEL_SOURCE.upper(),
        })

    summary_df = pd.DataFrame(rows).sort_values(
        "test_acc", ascending=False
    ).reset_index(drop=True)

    return summary_df


def plot_cross_stock_summary(summary_df):
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


def print_cross_stock_summary(summary_df, all_importance):
    print("\n" + "="*90)
    print("                    CROSS-STOCK SUMMARY")
    print(f"                    Label Source : {LABEL_SOURCE.upper()}")
    print("="*90)
    print(summary_df[[
        "stock", "val_acc", "test_acc",
        "model_ret_pct", "bnh_ret_pct", "alpha_vs_bnh",
        "sharpe", "max_drawdown", "nov2025_pred"
    ]].to_string(index=False))
    print("="*90)

    # Best and worst
    best  = summary_df.iloc[0]
    worst = summary_df.iloc[-1]
    print(f"\n  Best  stock : {best['stock']:15s} "
          f"test_acc={best['test_acc']}%  "
          f"alpha={best['alpha_vs_bnh']:+.1f}%")
    print(f"  Worst stock : {worst['stock']:15s} "
          f"test_acc={worst['test_acc']}%  "
          f"alpha={worst['alpha_vs_bnh']:+.1f}%")

    # Feature importance across stocks
    print(f"\n  Cross-stock feature importance:")
    all_imp = pd.concat(
        [df.assign(stock=s) for s, df in all_importance.items()],
        ignore_index=True
    )
    agg_imp = all_imp.groupby("feature")["importance_pct"].mean(
    ).sort_values(ascending=False).reset_index()
    agg_imp["rank"] = agg_imp.index + 1

    print(f"\n  Top 5 features across all stocks:")
    print(agg_imp.head(5).to_string(index=False))
    print(f"\n  Bottom 5 features (removal candidates):")
    print(agg_imp.tail(5).to_string(index=False))

    # Save
    summary_df.to_csv(
        Path(RESULTS_PATH) / f"cross_stock_summary_{LABEL_SOURCE.upper()}_Run{RUN_ID}.csv",
        index=False
    )
    agg_imp.to_csv(
        Path(RESULTS_PATH) / f"cross_stock_feature_importance_{LABEL_SOURCE.upper()}_Run{RUN_ID}.csv",
        index=False
    )
    print(f"\n  Saved to results/cross_stock_summary.csv")
    print(f"  Saved to results/cross_stock_feature_importance.csv")
