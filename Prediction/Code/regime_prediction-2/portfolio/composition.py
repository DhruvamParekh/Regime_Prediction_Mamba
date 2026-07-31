"""
portfolio/composition.py
===========================
Analysis of how the portfolio behaved over the simulation:
  - the raw rotation log (what was picked at each review, and why)
  - selection frequency per stock
  - streak analysis (longest/average consecutive reviews held)
  - rolling accuracy at each review, for the most-selected stocks
  - rotation turnover between consecutive reviews
  - per-stock contribution to the EqWt_LongOnly strategy's return
"""

from collections import defaultdict

import numpy as np
import pandas as pd


def print_rotation_log(review_log):
    """Prints what was selected at every review, with accuracy/confidence/weight."""
    print("\n" + "=" * 65)
    print("  PORTFOLIO ROTATION LOG")
    print("=" * 65)
    for entry in review_log:
        print(f"\n  {entry['date'].date()}")
        print(f"  {'Stock':<14} {'Acc%':>8} {'Conf%':>8} {'AccWt':>8}")
        print(f"  {'-'*42}")
        for s in entry['stocks']:
            print(
                f"  {s:<14}"
                f" {entry['acc_scores'][s]:>7.1%}"
                f" {entry['conf_scores'].get(s, 0):>7.1%}"
                f" {entry['acc_wts'][s]:>8.3f}"
            )


def build_selection_frequency(review_log, universe):
    """How often each stock was selected across all reviews, ranked descending."""
    n_reviews = len(review_log)
    selection_counts = defaultdict(int)
    for entry in review_log:
        for s in entry["stocks"]:
            selection_counts[s] += 1

    freq_df = pd.DataFrame([
        {"stock": s, "selections": c, "pct_reviews": round(c / n_reviews * 100, 1)}
        for s, c in selection_counts.items()
    ]).sort_values("selections", ascending=False).reset_index(drop=True)
    freq_df["rank"] = freq_df.index + 1

    print("\n" + "=" * 65)
    print(f"  STOCK SELECTION FREQUENCY  ({n_reviews} total reviews)")
    print("=" * 65)
    print(f"  {'Rank':<6} {'Stock':<16} {'Times Selected':>16} {'% of Reviews':>14}")
    print(f"  {'-'*55}")
    for _, row in freq_df.iterrows():
        bar = "█" * int(row["selections"])
        print(f"  {int(row['rank']):<6} {row['stock']:<16} {int(row['selections']):>16} {row['pct_reviews']:>13.1f}%  {bar}")

    never_selected = [s for s in universe if s not in selection_counts]
    print(f"\n  Stocks never selected ({len(never_selected)}): {', '.join(sorted(never_selected))}")

    return freq_df, selection_counts


def build_streak_analysis(review_log, universe, review_days):
    """Consecutive-review streaks each stock stayed in the portfolio (max/avg length)."""
    streak_data = defaultdict(list)
    in_streak   = defaultdict(lambda: {"start": None, "length": 0})

    for i, entry in enumerate(review_log):
        active = set(entry["stocks"])
        for s in universe:
            if s in active:
                if in_streak[s]["start"] is None:
                    in_streak[s]["start"]  = entry["date"]
                    in_streak[s]["length"] = 1
                else:
                    in_streak[s]["length"] += 1
            else:
                if in_streak[s]["start"] is not None:
                    streak_data[s].append({
                        "start" : in_streak[s]["start"],
                        "length": in_streak[s]["length"],
                    })
                    in_streak[s] = {"start": None, "length": 0}

    # Close any open streaks at end
    for s in universe:
        if in_streak[s]["start"] is not None:
            streak_data[s].append({
                "start" : in_streak[s]["start"],
                "length": in_streak[s]["length"],
            })

    streak_summary = []
    for s in universe:
        streaks = streak_data.get(s, [])
        if not streaks:
            continue
        lengths = [x["length"] for x in streaks]
        streak_summary.append({
            "stock"         : s,
            "n_streaks"     : len(streaks),
            "max_streak"    : max(lengths),
            "avg_streak"    : round(np.mean(lengths), 1),
            "total_reviews" : sum(lengths),
            "max_streak_wks": max(lengths) * (review_days // 7),
        })

    streak_df = pd.DataFrame(streak_summary).sort_values(
        "max_streak", ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("  STREAK ANALYSIS  (consecutive review periods in portfolio)")
    print("=" * 65)
    print(f"  {'Stock':<14} {'N Streaks':>10} {'Max Streak':>12} {'Max(wks)':>10} {'Avg Streak':>12} {'Total Reviews':>15}")
    print(f"  {'-'*77}")
    for _, row in streak_df.iterrows():
        print(
            f"  {row['stock']:<14}"
            f" {int(row['n_streaks']):>10}"
            f" {int(row['max_streak']):>12}"
            f" {int(row['max_streak_wks']):>10}"
            f" {row['avg_streak']:>12.1f}"
            f" {int(row['total_reviews']):>15}"
        )

    return streak_df


def print_rolling_accuracy_table(review_log, freq_df, top_n=10):
    """Prints a Date x Stock accuracy table for the top_n most-selected stocks, ★ = was selected."""
    print("\n" + "=" * 65)
    print(f"  ACCURACY AT EACH REVIEW — TOP {top_n} STOCKS BY SELECTION FREQ")
    print("=" * 65)

    top_stocks = freq_df.head(top_n)["stock"].tolist()

    print(f"  {'Date':<12}", end="")
    for s in top_stocks:
        print(f" {s[:10]:>11}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in top_stocks:
        print(f" {'-'*11}", end="")
    print()

    for entry in review_log:
        print(f"  {str(entry['date'].date()):<12}", end="")
        for s in top_stocks:
            acc = entry["all_scores"].get(s, None)
            selected = s in entry["stocks"]
            if acc is None:
                print(f" {'—':>11}", end="")
            else:
                marker = "★" if selected else " "
                print(f" {acc:>9.0%}{marker}", end="")
        print()

    print("\n  ★ = stock was selected into portfolio that review")


def build_turnover_analysis(review_log, top_k):
    """Entered/exited/kept stocks between every pair of consecutive reviews, plus summary stats."""
    print("\n" + "=" * 65)
    print("  ROTATION TURNOVER ANALYSIS")
    print("=" * 65)

    turnovers = []
    for i in range(1, len(review_log)):
        prev_set = set(review_log[i-1]["stocks"])
        curr_set = set(review_log[i]["stocks"])
        entered  = curr_set - prev_set
        exited   = prev_set - curr_set
        kept     = curr_set & prev_set
        turnovers.append({
            "date"    : review_log[i]["date"].date(),
            "entered" : sorted(entered),
            "exited"  : sorted(exited),
            "kept"    : sorted(kept),
            "n_change": len(entered),
        })

    n_reviews   = len(review_log)
    avg_changes = np.mean([t["n_change"] for t in turnovers]) if turnovers else 0.0
    zero_change = sum(1 for t in turnovers if t["n_change"] == 0)
    full_change = sum(1 for t in turnovers if t["n_change"] == top_k)

    print(f"  Total reviews       : {n_reviews}")
    print(f"  Avg stocks rotated  : {avg_changes:.1f} / {top_k} per review")
    print(f"  No-change reviews   : {zero_change}")
    print(f"  Full-churn reviews  : {full_change}")
    print()
    print(f"  {'Date':<12} {'Entered':<35} {'Exited':<35} {'Kept'}")
    print(f"  {'-'*105}")
    for t in turnovers:
        entered_str = ", ".join(t["entered"]) if t["entered"] else "—"
        exited_str  = ", ".join(t["exited"])  if t["exited"]  else "—"
        kept_str    = ", ".join(t["kept"])
        print(f"  {str(t['date']):<12} {entered_str:<35} {exited_str:<35} {kept_str}")

    return turnovers


def build_stock_contribution(daily_active, stock_prices, stock_preds, selection_counts):
    """Per-stock cumulative contribution to the EqWt_LongOnly strategy's return."""
    print("\n" + "=" * 65)
    print("  PER-STOCK RETURN CONTRIBUTION  (EqWt_LongOnly)")
    print("=" * 65)

    stock_contrib = defaultdict(float)
    stock_days_in = defaultdict(int)

    for i, (today, active_set) in enumerate(daily_active):
        if i == 0 or not active_set:
            continue
        eq_w = 1.0 / len(active_set)
        for stock_name in active_set:
            price_df = stock_prices.get(stock_name)
            if price_df is None or today not in price_df.index:
                continue
            day_ret = price_df.loc[today, "daily_return"]
            if pd.isna(day_ret):
                continue
            past = stock_preds[stock_name]
            past = past[past.index <= today]
            if len(past) == 0:
                continue
            pred   = past.iloc[-1]["pred_label"]
            lo_pos = 1 if pred == "Bullish" else 0
            contrib = eq_w * lo_pos * day_ret
            stock_contrib[stock_name] += contrib
            stock_days_in[stock_name] += lo_pos

    contrib_df = pd.DataFrame([
        {
            "stock"      : s,
            "total_contrib_pct": round(stock_contrib[s] * 100, 2),
            "days_long"  : stock_days_in[s],
            "selections" : selection_counts.get(s, 0),
        }
        for s in selection_counts
    ]).sort_values("total_contrib_pct", ascending=False).reset_index(drop=True)

    print(f"  {'Stock':<14} {'Contribution%':>15} {'Days Long':>12} {'Selections':>12}")
    print(f"  {'-'*57}")
    for _, row in contrib_df.iterrows():
        bar_len = int(abs(row["total_contrib_pct"]) * 3)
        bar = ("+" if row["total_contrib_pct"] >= 0 else "-") * min(bar_len, 30)
        print(
            f"  {row['stock']:<14}"
            f" {row['total_contrib_pct']:>14.2f}%"
            f" {int(row['days_long']):>12}"
            f" {int(row['selections']):>12}  {bar}"
        )

    return contrib_df
