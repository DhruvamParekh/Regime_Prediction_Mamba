"""
config.py
=========
All constants, toggles, and hyperparameters for the regime DETECTION
pipeline live here, in one place, instead of being scattered across the
code.

IMPORTANT — this is a DETECTION model, not a forecasting model:
The model is trained to predict the majority regime over the next
`forward_window` (7) trading days, and it is trained on PSEUDO LABELS that
are themselves built using *future* price data (a forward return window,
smoothed and thresholded — see labels/pseudo.py). Because the labels the
model is trained on are derived from the future, this whole pipeline is a
"detection" exercise (verifying that the Mamba architecture can pick up a
regime signal at all) rather than a genuine walk-forward prediction system.
Keep that framing in mind when interpreting any results.

Only this file should need editing to point the pipeline at your own data
folder, change which stocks are trained, or tweak hyperparameters.
"""

import os
from pathlib import Path

# ── Base data directory ────────────────────────────────────────────
# Original Colab code hardcoded this to a Google Drive path
# ('/content/drive/MyDrive/Regime_Prediction2'). Locally, this defaults to
# a `project_data/` folder next to this file, but can be overridden with
# the REGIME_DETECTION_BASE environment variable, e.g.:
#   export REGIME_DETECTION_BASE=/path/to/your/data
BASE = os.environ.get(
    "REGIME_DETECTION_BASE",
    str(Path(__file__).resolve().parent / "project_data"),
)

RAW_DATA_PATH   = f"{BASE}/data/raw"
PROCESSED_PATH  = f"{BASE}/data/processed"
PSEUDO_PATH     = f"{BASE}/pseudo_regimes"
DETECTED_PATH   = f"{BASE}/detected_regimes"
RESULTS_PATH    = f"{BASE}/results"
CHECKPOINT_PATH = f"{BASE}/checkpoints"

# Folders that must exist before the pipeline runs. The original notebook
# only created PROCESSED_PATH/RESULTS_PATH/CHECKPOINT_PATH and relied on
# pseudo_regimes/ already existing on Drive from an earlier run — that
# doesn't hold on a fresh clone, so PSEUDO_PATH is created here too.
# RAW_DATA_PATH is NOT created — you're expected to populate it yourself.
FOLDERS_TO_CREATE = [
    PROCESSED_PATH,
    RESULTS_PATH,
    CHECKPOINT_PATH,
    PSEUDO_PATH,
]

# ── Raw file names ──────────────────────────────────────────────────
MACRO_FILE = "macro_features.csv"
NIFTY_FILE = "NIFTY50_INDEX.csv"

# ── Toggles ─────────────────────────────────────────────────────────
STOCKS_TO_TRAIN = [
    "ICICIBANK", "HDFCBANK", "AXISBANK", "BAJFINANCE", "BAJAJFINSV",
    "RELIANCE", "TCS", "HCLTECH", "WIPRO", "TATASTEEL",
    "MARUTI", "LT", "NESTLEIND", "HINDUNILVR", "SUNPHARMA",
    "DRREDDY", "CIPLA", "EICHERMOT", "GRASIM",
]

LABEL_SOURCE = "pseudo"   # 'pseudo' or 'detected'
RUN_ID = 5

# Date used for the "latest signal" chart (originally called the
# "November 2025 prediction" cell in the notebook).
TARGET_DATE = "2025-11-01"

# ── Pseudo label settings ──────────────────────────────────────────
PSEUDO_CFG = {
    "fwd_return_days"    : 60,
    "smooth_window"      : 15,
    "bull_threshold"     : 0.05,
    "bear_threshold"     : -0.05,
    "min_regime_duration": 15,
    "mask_last_days"     : 75,
}

# ── Train / Val / Test split dates ─────────────────────────────────
SPLIT_DATES = {
    "train_end"  : "2022-12-31",
    "val_end"    : "2023-12-31",
    "test_start" : "2024-01-01",
}

# ── Model config ────────────────────────────────────────────────────
MODEL_CFG = {
    "lookback_window" : 60,
    "forward_window"  : 7,
    "prediction_freq" : 7,
    "n_raw_features"  : 21,
    "n_classes"       : 3,
    "d_model"         : 32,
    "d_state"         : 8,
    "d_conv"          : 4,
    "expand"          : 2,
    "n_mamba_layers"  : 2,
    "regime_embed_dim": 8,
    "dropout"         : 0.4,
}

# ── Training config ─────────────────────────────────────────────────
TRAIN_CFG = {
    "batch_size"          : 16,
    "lr"                  : 3e-4,
    "weight_decay"        : 5e-3,
    "n_epochs"            : 100,
    "early_stop_patience" : 60,
    "label_smoothing"     : 0.2,
}

# ── Regime encoding ─────────────────────────────────────────────────
REGIME_MAP     = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
REGIME_INV_MAP = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

SEED = 42
