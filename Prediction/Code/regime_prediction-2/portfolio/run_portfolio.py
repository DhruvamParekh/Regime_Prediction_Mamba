"""
portfolio/run_portfolio.py
============================
ENTRY POINT for the rolling portfolio. Reads each stock's saved
`full_predictions.csv` from the regime-prediction pipeline and
nothing else from that side of the repo — it never touches model
weights, training code, or label-generation code directly.

Steps:
  1. Load predictions + raw prices for every stock in
     config.STOCKS_TO_TRAIN, attach hybrid fwd30/causal-lagged labels.
  2. Run the full daily rolling-rebalance simulation.
  3. Print/save metrics, quarterly breakdown, rotation log, and
     portfolio-composition analysis (selection frequency, streaks,
     turnover, per-stock contribution).
  4. Plot everything.
  5. Run a single ad-hoc "as of today" review for whatever date is
     configured, so you can ask "what would the portfolio look like
     if I rebalanced right now?" without re-running the whole
     simulation or writing a one-off script per date.

Usage:
    python portfolio/run_portfolio.py
"""

import sys
from pathlib import Path

# Allow running this file directly (`python portfolio/run_portfolio.py`)
# as well as as a module (`python -m portfolio.run_portfolio`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import LABEL_SOURCE, RAW_DATA_PATH, RESULTS_PATH, RUN_ID, STOCKS_TO_TRAIN
from portfolio.config import (
    CAUSAL_LAG, FWD_DAYS, LOOKBACK_DAYS, REVIEW_DAYS, SIM_END, SIM_START, TOP_K,
)
from portfolio.loader import load_stock_predictions_and_prices
from portfolio.simulation import run_simulation, run_single_review
from portfolio.metrics import build_metrics_table, print_quarterly_breakdown
from portfolio.composition import (
    build_selection_frequency, build_stock_contribution, build_streak_analysis,
    build_turnover_analysis, print_rolling_accuracy_table, print_rotation_log,
)
from portfolio.visualise import (
    plot_equity_and_size, plot_presence_heatmap, plot_selection_frequency,
    plot_stock_contribution, plot_streak_analysis,
)

# Date to run the ad-hoc "review as of today" check for. Change this
# to any date you want a live recommendation for — it does not need
# to be inside [SIM_START, SIM_END].
AD_HOC_REVIEW_DATE = SIM_END


def main():
    # ── Step 1: load predictions + prices ───────────────────────────
    stock_preds, stock_prices, universe = load_stock_predictions_and_prices(
        STOCKS_TO_TRAIN, RESULTS_PATH, LABEL_SOURCE, RUN_ID, RAW_DATA_PATH
    )

    if len(universe) < TOP_K:
        print(f"  Universe ({len(universe)} stocks) is smaller than TOP_K "
              f"({TOP_K}) — cannot run simulation. Make sure main.py has "
              f"been run and full_predictions.csv exists for enough stocks.")
        return

    # ── Step 2: run the full simulation ──────────────────────────────
    equity, review_log, daily_active = run_simulation(
        stock_preds, stock_prices, universe,
        SIM_START, SIM_END, REVIEW_DAYS, LOOKBACK_DAYS, FWD_DAYS, TOP_K,
    )
    all_trading_days = pd.bdate_range(SIM_START, SIM_END)

    # ── Step 3a: metrics + quarterly breakdown ───────────────────────
    metrics_df = build_metrics_table(equity)
    print("\n" + "=" * 65)
    print("  STRATEGY COMPARISON  (hybrid causal labels — no future data)")
    print(f"  Run {RUN_ID} | Lookback={LOOKBACK_DAYS}d | Review={REVIEW_DAYS}d | Top-K={TOP_K}")
    print(f"  Label: fwd-{FWD_DAYS}d for older half / causal-lag-{CAUSAL_LAG}d for recent half")
    print("=" * 65)
    print(metrics_df.to_string())

    print_quarterly_breakdown(equity, SIM_START, SIM_END)

    # ── Step 3b: rotation log + composition analysis ─────────────────
    print_rotation_log(review_log)

    freq_df, selection_counts = build_selection_frequency(review_log, universe)
    streak_df = build_streak_analysis(review_log, universe, REVIEW_DAYS)
    print_rolling_accuracy_table(review_log, freq_df, top_n=10)
    build_turnover_analysis(review_log, TOP_K)
    contrib_df = build_stock_contribution(
        daily_active, stock_prices, stock_preds, selection_counts
    )

    # ── Step 4: plots ─────────────────────────────────────────────────
    plot_equity_and_size(
        equity, review_log, all_trading_days, TOP_K,
        RUN_ID, LOOKBACK_DAYS, REVIEW_DAYS, FWD_DAYS, CAUSAL_LAG, RESULTS_PATH
    )
    plot_selection_frequency(freq_df, len(review_log), RUN_ID, RESULTS_PATH)
    plot_presence_heatmap(freq_df, daily_active, all_trading_days, RUN_ID, RESULTS_PATH)
    plot_streak_analysis(streak_df, RUN_ID, RESULTS_PATH)
    plot_stock_contribution(contrib_df, RUN_ID, RESULTS_PATH)

    # ── Step 5: save results ──────────────────────────────────────────
    out_dir = Path(RESULTS_PATH)

    equity_df = pd.DataFrame(equity)
    equity_df.index.name = "Date"
    equity_df.to_csv(out_dir / f"rolling_equity_Run{RUN_ID}.csv")
    metrics_df.to_csv(out_dir / f"rolling_metrics_Run{RUN_ID}.csv")

    log_rows = []
    for entry in review_log:
        for s in entry["stocks"]:
            log_rows.append({
                "date"      : entry["date"].date(),
                "stock"     : s,
                "acc_score" : entry["acc_scores"][s],
                "conf_score": entry["conf_scores"].get(s, None),
                "acc_weight": entry["acc_wts"][s],
            })
    pd.DataFrame(log_rows).to_csv(
        out_dir / f"rolling_rotation_log_Run{RUN_ID}.csv", index=False
    )
    freq_df.to_csv(out_dir / f"selection_frequency_Run{RUN_ID}.csv", index=False)
    streak_df.to_csv(out_dir / f"streak_analysis_Run{RUN_ID}.csv", index=False)
    contrib_df.to_csv(out_dir / f"stock_contribution_Run{RUN_ID}.csv", index=False)

    print(f"\nSaved to {RESULTS_PATH}/")
    print(f"  rolling_equity_Run{RUN_ID}.csv")
    print(f"  rolling_metrics_Run{RUN_ID}.csv")
    print(f"  rolling_rotation_log_Run{RUN_ID}.csv")
    print(f"  selection_frequency_Run{RUN_ID}.csv")
    print(f"  streak_analysis_Run{RUN_ID}.csv")
    print(f"  stock_contribution_Run{RUN_ID}.csv")
    print(f"  rolling_portfolio_Run{RUN_ID}.png")
    print(f"  selection_frequency_Run{RUN_ID}.png")
    print(f"  portfolio_heatmap_Run{RUN_ID}.png")
    print(f"  streak_analysis_Run{RUN_ID}.png")
    print(f"  stock_contribution_Run{RUN_ID}.png")

    # ── Step 6: ad-hoc "review as of today" check ─────────────────────
    # Generalises the original notebook's repeated one-off "portfolio
    # review as of DATE" cells into a single reusable call — change
    # AD_HOC_REVIEW_DATE above to check any date.
    print("\n" + "=" * 65)
    print(f"  AD-HOC PORTFOLIO REVIEW AS OF {pd.Timestamp(AD_HOC_REVIEW_DATE).date()}")
    print("=" * 65)
    result = run_single_review(
        stock_preds, stock_prices, universe, AD_HOC_REVIEW_DATE,
        LOOKBACK_DAYS, FWD_DAYS, TOP_K, hold_days=REVIEW_DAYS,
    )
    if result is not None:
        print(f"  Hold until : {result['hold_until'].date()}")
        print(f"\n  {'Stock':<14} {'Signal':>9} {'Score':>8} {'Conf':>7} "
              f"{'EqWt':>7} {'AccWt':>7}")
        print(f"  {'-'*56}")
        for s in result["top_names"]:
            sig = result["last_signal"].get(s)
            pred_label = sig["pred_label"] if sig else "Unknown"
            conf       = sig["confidence"] if sig else 0.5
            print(
                f"  {s:<14}"
                f" {pred_label:>9}"
                f" {result['scores'][s]:>7.1%}"
                f" {conf:>7.2f}"
                f" {result['eq_weights'][s]:>6.1%}"
                f" {result['acc_weights'][s]:>6.1%}"
            )

        review_rows = [
            {
                "stock"      : s,
                "signal"     : (result["last_signal"].get(s) or {}).get("pred_label", "Unknown"),
                "confidence" : (result["last_signal"].get(s) or {}).get("confidence", 0.5),
                "score"      : result["scores"][s],
                "eq_weight"  : result["eq_weights"][s],
                "acc_weight" : result["acc_weights"][s],
                "review_date": str(result["review_date"].date()),
                "hold_until" : str(result["hold_until"].date()),
            }
            for s in result["top_names"]
        ]
        review_df = pd.DataFrame(review_rows)
        review_save_path = out_dir / (
            f"ad_hoc_review_{pd.Timestamp(AD_HOC_REVIEW_DATE).date()}_Run{RUN_ID}.csv"
        )
        review_df.to_csv(review_save_path, index=False)
        print(f"\n  Saved: {review_save_path.name}")


if __name__ == "__main__":
    main()
