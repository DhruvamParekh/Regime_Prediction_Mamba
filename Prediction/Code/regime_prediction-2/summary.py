"""
summary.py
==========
Consolidates every stock's accuracy, backtest, and feature-importance
results (plus the specific-week prediction, if available) into one
cross-stock summary table, and prints/saves it alongside a
cross-stock feature-importance ranking.
"""

from pathlib import Path

import pandas as pd

from config import LABEL_SOURCE, RESULTS_PATH, RUN_ID


def build_cross_stock_summary(all_results, all_backtest, all_importance):
    """
    Builds one row per stock combining: validation/test accuracy
    (from the saved period_accuracy.csv), backtest total return /
    Sharpe / max drawdown (from ALL_BACKTEST), the single most
    important feature (from ALL_IMPORTANCE), and the November-2025
    specific-week prediction if that CSV was saved.
    """
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

        # ── Nov 2025 prediction ───────────────────────────────────
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


def print_cross_stock_summary(summary_df, all_importance):
    """Prints the summary table plus best/worst stocks and cross-stock feature importance, then saves both to CSV."""
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
