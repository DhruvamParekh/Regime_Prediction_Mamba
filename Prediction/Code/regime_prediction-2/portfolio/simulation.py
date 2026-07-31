"""
portfolio/simulation.py
=========================
The daily rolling-rebalance simulation: every REVIEW_DAYS, scores
all stocks in the universe on rolling prediction accuracy (using the
hybrid fwd30/causal-lagged label — see labels.py) over the last
LOOKBACK_DAYS, picks the top TOP_K, and tracks four strategies plus
an equal-weight buy-and-hold benchmark day by day:

  EqWt_LongOnly   — equal weight,   long only  (0 or +1 per stock)
  EqWt_LongShort  — equal weight,   long/short (-1, 0, or +1)
  AccWt_LongOnly  — accuracy weighted (softmax), long only
  AccWt_LongShort — accuracy weighted (softmax), long/short
  BnH_EqWt        — equal-weight buy & hold across the full universe
"""

import numpy as np
import pandas as pd
from scipy.special import softmax


def run_simulation(stock_preds, stock_prices, universe,
                    sim_start, sim_end, review_days, lookback_days,
                    fwd_days, top_k):
    """
    Runs the full daily simulation from sim_start to sim_end.

    Returns (equity, review_log, daily_active):
      equity       — dict of strategy name -> pd.Series (indexed by
                     trading day, starting at 1.0)
      review_log   — list of dicts, one per rebalance, with the
                     selected stocks, their scores/weights, and the
                     full score ranking for that review
      daily_active — list of (date, set_of_active_stocks) for every
                     trading day, used by composition analysis/plots
    """
    all_trading_days = pd.bdate_range(sim_start, sim_end)

    curves = {
        "EqWt_LongOnly"  : np.ones(len(all_trading_days)),
        "EqWt_LongShort" : np.ones(len(all_trading_days)),
        "AccWt_LongOnly" : np.ones(len(all_trading_days)),
        "AccWt_LongShort": np.ones(len(all_trading_days)),
        "BnH_EqWt"       : np.ones(len(all_trading_days)),
    }

    current_portfolio = []
    prev_review       = all_trading_days[0]
    review_log        = []

    # Track which stocks are active on each day (for heatmap)
    daily_active = []   # list of (date, set_of_active_stocks)

    for i, today in enumerate(all_trading_days):
        if i == 0:
            daily_active.append((today, set()))
            continue

        # ── Rebalance if due ─────────────────────────────────────────
        if (today - prev_review).days >= review_days or i == 1:
            prev_review       = today
            lookback_start    = today - pd.Timedelta(days=lookback_days)
            cutoff_at_review  = today - pd.Timedelta(days=fwd_days)  # ← KEY FIX

            scores      = {}
            conf_scores = {}
            for stock_name in universe:
                pred_df = stock_preds[stock_name]
                window  = pred_df[
                    (pred_df.index >= lookback_start) &
                    (pred_df.index <  today)
                ]
                if len(window) < 2:
                    continue

                # Pick label based on review-relative cutoff — no future data
                def correct_at_review(row):
                    lbl = row["fwd30_label"] if row.name <= cutoff_at_review \
                          else row["causal_label"]
                    return float(row["pred_label"] == lbl) if pd.notna(lbl) else np.nan

                valid_correct = window.apply(correct_at_review, axis=1).dropna()
                if len(valid_correct) < 2:
                    continue
                acc  = valid_correct.mean()
                conf = window["confidence"].mean() if "confidence" in window.columns else 0.5
                scores[stock_name]      = acc
                conf_scores[stock_name] = conf

            if len(scores) >= top_k:
                ranked    = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_names = [s for s, _ in ranked[:top_k]]
                top_acc   = np.array([scores[s] for s in top_names])

                eq_weights  = np.full(top_k, 1.0 / top_k)
                acc_weights = softmax(top_acc * 10)

                current_portfolio = list(zip(top_names, eq_weights, acc_weights))

                review_log.append({
                    "date"       : today,
                    "stocks"     : top_names,
                    "acc_scores" : {s: round(scores[s], 3) for s in top_names},
                    "conf_scores": {s: round(conf_scores[s], 3) for s in top_names},
                    "acc_wts"    : {s: round(w, 3) for s, w in zip(top_names, acc_weights)},
                    "all_scores" : {s: round(scores[s], 3) for s in scores},  # full ranking
                })

        active_stocks = set(s for s, _, _ in current_portfolio)
        daily_active.append((today, active_stocks))

        # ── Daily portfolio return ────────────────────────────────────
        eq_lo = eq_ls = acc_lo = acc_ls = 0.0

        for stock_name, eq_w, acc_w in current_portfolio:
            price_df = stock_prices[stock_name]
            if today not in price_df.index:
                continue
            day_ret = price_df.loc[today, "daily_return"]
            if pd.isna(day_ret):
                continue

            past = stock_preds[stock_name]
            past = past[past.index <= today]
            if len(past) == 0:
                continue
            pred = past.iloc[-1]["pred_label"]

            lo_pos = 1 if pred == "Bullish" else 0
            ls_pos = (1 if pred == "Bullish" else
                     -1 if pred == "Bearish" else 0)

            eq_lo  += eq_w  * lo_pos * day_ret
            eq_ls  += eq_w  * ls_pos * day_ret
            acc_lo += acc_w * lo_pos * day_ret
            acc_ls += acc_w * ls_pos * day_ret

        bnh_rets = []
        for s in universe:
            price_df = stock_prices[s]
            if today in price_df.index:
                r = price_df.loc[today, "daily_return"]
                if not pd.isna(r):
                    bnh_rets.append(r)
        bnh_ret = float(np.mean(bnh_rets)) if bnh_rets else 0.0

        curves["EqWt_LongOnly"][i]   = curves["EqWt_LongOnly"][i-1]   * (1 + eq_lo)
        curves["EqWt_LongShort"][i]  = curves["EqWt_LongShort"][i-1]  * (1 + eq_ls)
        curves["AccWt_LongOnly"][i]  = curves["AccWt_LongOnly"][i-1]  * (1 + acc_lo)
        curves["AccWt_LongShort"][i] = curves["AccWt_LongShort"][i-1] * (1 + acc_ls)
        curves["BnH_EqWt"][i]        = curves["BnH_EqWt"][i-1]        * (1 + bnh_ret)

    equity = {k: pd.Series(v, index=all_trading_days) for k, v in curves.items()}

    return equity, review_log, daily_active


def run_single_review(stock_preds, stock_prices, universe, review_date,
                       lookback_days, fwd_days, top_k, hold_days=21):
    """
    Scores every stock in the universe as of a single `review_date`
    (using the same hybrid-label / accuracy-scoring logic as
    run_simulation's rebalance step) and returns the ranked selection
    for the next `hold_days` — i.e. "what would today's portfolio
    review recommend right now?".

    This generalises the one-off "portfolio review as of DATE" cells
    from the original notebook into a single reusable function that
    can be called for ANY date, instead of copy-pasting one script
    per historical review date.

    Returns a dict with the top_k selection, per-stock scores, and
    weights (or None if fewer than top_k stocks have enough history).
    """
    review_date    = pd.Timestamp(review_date)
    lookback_start = review_date - pd.Timedelta(days=lookback_days)
    cutoff_at_review = review_date - pd.Timedelta(days=fwd_days)

    scores      = {}
    conf_scores = {}
    last_signal = {}

    for stock_name in universe:
        pred_df = stock_preds[stock_name]
        window  = pred_df[
            (pred_df.index >= lookback_start) &
            (pred_df.index <  review_date)
        ]
        if len(window) < 2:
            continue

        def correct_at_review(row):
            lbl = row["fwd30_label"] if row.name <= cutoff_at_review \
                  else row["causal_label"]
            return float(row["pred_label"] == lbl) if pd.notna(lbl) else np.nan

        valid_correct = window.apply(correct_at_review, axis=1).dropna()
        if len(valid_correct) < 2:
            continue

        scores[stock_name]      = valid_correct.mean()
        conf_scores[stock_name] = window["confidence"].mean() \
            if "confidence" in window.columns else 0.5

        past = pred_df[pred_df.index <= review_date]
        if len(past) > 0:
            last_signal[stock_name] = {
                "pred_label": past.iloc[-1]["pred_label"],
                "confidence": past.iloc[-1].get("confidence", 0.5),
                "last_date" : past.index[-1],
            }

    if len(scores) < top_k:
        print(f"  Not enough scored stocks ({len(scores)}) for top_k={top_k} "
              f"as of {review_date.date()}.")
        return None

    ranked      = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_names   = [s for s, _ in ranked[:top_k]]
    top_acc     = np.array([scores[s] for s in top_names])
    eq_weights  = np.full(top_k, 1.0 / top_k)
    acc_weights = softmax(top_acc * 10)

    return {
        "review_date"  : review_date,
        "hold_until"   : review_date + pd.Timedelta(days=hold_days),
        "top_names"    : top_names,
        "scores"       : {s: scores[s] for s in top_names},
        "conf_scores"  : {s: conf_scores[s] for s in top_names},
        "eq_weights"   : dict(zip(top_names, eq_weights)),
        "acc_weights"  : dict(zip(top_names, acc_weights)),
        "last_signal"  : {s: last_signal.get(s) for s in top_names},
        "next_tier"    : ranked[top_k:top_k + 4],
        "all_scores"   : scores,
    }
