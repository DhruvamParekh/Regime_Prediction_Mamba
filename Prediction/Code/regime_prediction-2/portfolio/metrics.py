"""
portfolio/metrics.py
=====================
Standard performance metrics (total/annualised return, Sharpe, max
drawdown) plus a quarter-by-quarter return breakdown for each
strategy's equity curve.
"""

import pandas as pd


def calc_metrics(ser, label):
    """Total/annualised return, Sharpe (rf=0), and max drawdown for one equity curve."""
    total  = ser.iloc[-1] - 1
    n      = len(ser)
    dr     = ser.pct_change().dropna()
    ann    = (1 + total) ** (252 / n) - 1
    vol    = dr.std() * (252 ** 0.5)
    sharpe = ann / (vol + 1e-9)
    mdd    = ((ser - ser.cummax()) / ser.cummax()).min()
    return {
        "Strategy"   : label,
        "Total Ret %" : round(total * 100, 2),
        "Ann Ret %"   : round(ann   * 100, 2),
        "Sharpe"      : round(sharpe, 3),
        "Max DD %"    : round(mdd   * 100, 2),
    }


def build_metrics_table(equity):
    """Builds the Strategy-indexed metrics DataFrame for every equity curve in `equity`."""
    metrics_df = pd.DataFrame([calc_metrics(equity[k], k) for k in equity])
    return metrics_df.set_index("Strategy")


def print_quarterly_breakdown(equity, sim_start, sim_end):
    """Prints a quarter-by-quarter % return table, one row per quarter, one column per strategy."""
    print("\n" + "=" * 65)
    print("  QUARTERLY BREAKDOWN")
    print("=" * 65)
    strats = list(equity.keys())
    header = f"  {'Quarter':<10}" + "".join(f"  {s[:14]:>14}" for s in strats)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for q in pd.period_range(sim_start, sim_end, freq="Q"):
        qs  = q.start_time
        qe  = q.end_time
        row = f"  {str(q):<10}"
        for k in strats:
            sl = equity[k][(equity[k].index >= qs) & (equity[k].index <= qe)]
            if len(sl) < 2:
                row += f"  {'N/A':>14}"
            else:
                qr = (sl.iloc[-1] / sl.iloc[0] - 1) * 100
                row += f"  {qr:>13.1f}%"
        print(row)
