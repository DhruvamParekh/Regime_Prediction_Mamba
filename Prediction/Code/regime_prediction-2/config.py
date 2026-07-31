"""
config.py
=========
Single source of truth for every path, hyperparameter, and constant
used across the regime-prediction pipeline.

Nothing else in this codebase hardcodes a path or a hyperparameter —
every other file imports what it needs from here. To change which
stocks are trained, which run ID is used, or any model/training
setting, edit this file only.
"""

import torch

# ── Toggles ───────────────────────────────────────────────────────
# List of NSE stock tickers (must match raw CSV filenames under
# RAW_DATA_PATH) that the pipeline will train/evaluate on.
STOCKS_TO_TRAIN = [
    'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK', 'SBIN',
    'INDUSINDBK', 'BAJFINANCE', 'BAJAJFINSV', 'SHRIRAMFIN',
    'TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM',
    'MARUTI', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT',
    'RELIANCE', 'ONGC', 'BPCL', 'NTPC', 'POWERGRID', 'COALINDIA', 'BEL',
    'TATASTEEL', 'JSWSTEEL', 'HINDALCO',
    'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'TATACONSUM',
    'ASIANPAINT', 'TITAN', 'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB',
    'LT', 'GRASIM', 'ULTRACEMCO', 'APOLLOHOSP', 'ADANIPORTS',
]

# Which label source is used to train the model.
#   'causal_confirmed' — the production label source (zero future data
#                         at inference; recommended)
#   'pseudo'           — legacy forward-looking labels, kept only for
#                         reference / comparison
LABEL_SOURCE = 'causal_confirmed'

# Run identifier — used to namespace checkpoints and results so
# multiple experiments don't overwrite each other.
RUN_ID = 21

# ── Paths ─────────────────────────────────────────────────────────
# BASE assumes Google Drive is mounted at /content/drive (Colab).
# Change this single line to relocate the entire project.
BASE            = '/content/drive/MyDrive/Regime_Prediction_Final'
RAW_DATA_PATH   = f'{BASE}/data/raw'
PROCESSED_PATH  = f'{BASE}/data/processed'
PSEUDO_PATH     = f'{BASE}/pseudo_regimes'
DETECTED_PATH   = f'{BASE}/detected_regimes'
RESULTS_PATH    = f'{BASE}/results/Run{RUN_ID}'
CHECKPOINT_PATH = f'{BASE}/checkpoints'

# ── File names ────────────────────────────────────────────────────
MACRO_FILE = 'macro_features.csv'
NIFTY_FILE = 'NIFTY50_INDEX.csv'

# ── Causal label config ──────────────────────────────────────────
# Governs the causal 8-signal score used for the recent (T-30 to T-1)
# window of the causal-confirmed label strategy. Uses zero future
# data — fully tradeable at inference time.
CAUSAL_CFG = {
    'causal_lookback'   : 60,
    'confirm_recent'    : 30,
    'confirm_older'     : 30,
    'bull_score'        : 3,    # updated from 4
    'bear_score'        : -3,   # updated from -4
    'min_regime_days'   : 10,
}

# Pseudo-30 label config — used as the "confirmed" ground truth for
# the older T-60 to T-31 window inside the causal-confirmed strategy.
CONFIRMATION_PSEUDO_CFG = {
    'fwd_return_days'    : 30,
    'bull_threshold'     : 0.05,
    'bear_threshold'     : -0.03,
    'smooth_window'      : 5,
    'min_regime_duration': 15,
    'mask_last_days'     : 0,
}

# Pseudo label config (legacy / reference label source).
PSEUDO_CFG = {
    'fwd_return_days'    : 60,
    'smooth_window'      : 15,
    'bull_threshold'     : 0.05,
    'bear_threshold'     : -0.05,
    'min_regime_duration': 15,
    'mask_last_days'     : 0,
}

# ── Train / Val / Test split dates ────────────────────────────────
SPLIT_DATES = {
    'train_end'  : '2022-06-30',
    'val_end'    : '2023-12-31',
    'test_start' : '2024-01-01',
    'test_end'   : '2026-07-24',
}

# ── Model config ──────────────────────────────────────────────────
# Architecture hyperparameters for RegimePredictionMamba.
MODEL_CFG = {
    'lookback_window' : 60,
    'forward_window'  : 7,
    'prediction_freq' : 7,
    'n_raw_features'  : 13,    # was 21 — 5 new features added
    'n_classes'       : 3,
    'd_model'         : 32,    # was 32
    'd_state'         : 8,     # was 8
    'd_conv'          : 4,
    'expand'          : 4,
    'n_mamba_layers'  : 3,     # was 2
    'regime_embed_dim': 8,     # was 8
    'dropout'         : 0.35,  # was 0.40 — reduced since we need capacity
}

# ── Training config ───────────────────────────────────────────────
TRAIN_CFG = {
    'batch_size'          : 32,
    'lr'                  : 2e-4,
    'weight_decay'        : 1e-2,
    'n_epochs'            : 150,
    'early_stop_patience' : 70,
    'label_smoothing'     : 0.05,
    'warmup_epochs'       : 5,
}

# ── Regime encoding ───────────────────────────────────────────────
REGIME_MAP     = {'Bearish': 0, 'Neutral': 1, 'Bullish': 2}
REGIME_INV_MAP = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}

# ── Reproducibility / device ──────────────────────────────────────
SEED   = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Target date used by the "specific week" prediction step ──────
TARGET_DATE = "2025-11-01"

# ── Portfolio config (Phase 3 — rolling portfolio) ─────────────────
# Not used by anything in Phase 2 (regime prediction) — read by
# portfolio/config.py, which re-exports it alongside portfolio-only
# derived paths (e.g. the meta-model checkpoint path).
PORTFOLIO_CFG = {
    'sim_start'     : '2023-01-01',  # simulation start date
    'sim_end'       : '2025-03-31',  # simulation end date
    'review_days'   : 28,            # rebalance every N days
    'lookback_days' : 56,            # scoring window (accuracy lookback)
    'top_k'         : 5,             # how many stocks to hold at once

    # Hybrid label parameters (zero future data at review time)
    'fwd_days'      : 30,   # forward window for the older-half label
    'causal_lag'    : 10,   # days ignored at the recent end for the causal label
}
