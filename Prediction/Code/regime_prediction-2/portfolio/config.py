"""
portfolio/config.py
====================
Portfolio-specific constants, layered on top of the top-level
config.py (which still owns BASE, RAW_DATA_PATH, RESULTS_PATH,
RUN_ID, LABEL_SOURCE, STOCKS_TO_TRAIN — the portfolio layer reads
the *same* run's full_predictions.csv files, it does not duplicate
or override them).
"""

from config import PORTFOLIO_CFG

# ── Re-exported for convenience so other portfolio/ modules can do
#    `from portfolio.config import SIM_START, TOP_K, ...` directly ──
SIM_START     = PORTFOLIO_CFG['sim_start']
SIM_END       = PORTFOLIO_CFG['sim_end']
REVIEW_DAYS   = PORTFOLIO_CFG['review_days']
LOOKBACK_DAYS = PORTFOLIO_CFG['lookback_days']
TOP_K         = PORTFOLIO_CFG['top_k']
FWD_DAYS      = PORTFOLIO_CFG['fwd_days']
CAUSAL_LAG    = PORTFOLIO_CFG['causal_lag']

EXCLUDE_FILES = {"macro_features.csv", "nifty50_index.csv"}
