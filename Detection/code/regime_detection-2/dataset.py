"""
dataset.py
==========
Assembles per-stock model-ready data (features + input regime + prediction
target), splits it into train/val/test by date, and wraps it in a PyTorch
Dataset that produces sliding-window sequences.

Logic is unchanged from the original notebook.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config import MODEL_CFG, REGIME_INV_MAP, SPLIT_DATES
from data.features import compute_features
from data.loaders import load_stock_csv
from labels.targets import build_prediction_labels, load_regime_labels


def assemble_stock_data(stock_name, all_stock_files, macro_df, nifty_df, label_source):
    """
    Loads and merges everything for one stock:
      - 21 engineered features
      - detected/pseudo regime as input signal
      - 7-day forward prediction labels as targets
    """
    stock_df = load_stock_csv(all_stock_files[stock_name])
    feat_df, feat_cols = compute_features(stock_df, macro_df, nifty_df)

    regime_df     = load_regime_labels(stock_name, stock_df, label_source)
    pred_label_df = build_prediction_labels(regime_df)

    # Features + prediction targets
    merged = feat_df.merge(pred_label_df, on="Date", how="inner")

    # Input regime sequence
    merged = merged.merge(
        regime_df.rename(columns={"regime_label": "detected_regime"}),
        on="Date", how="left"
    )
    merged["detected_regime"] = merged["detected_regime"].ffill().fillna(1).astype(int)
    merged = merged.dropna(subset=feat_cols).reset_index(drop=True)

    dist = {REGIME_INV_MAP[k]: v
            for k, v in merged["pred_label"].value_counts().to_dict().items()}
    print(f"  {stock_name}: {len(merged)} rows | {dist}")

    return merged, feat_cols


def make_splits(df):
    """Split a merged stock dataframe into train/val/test by date, per config.SPLIT_DATES."""
    train_end  = pd.Timestamp(SPLIT_DATES["train_end"])
    val_end    = pd.Timestamp(SPLIT_DATES["val_end"])
    test_start = pd.Timestamp(SPLIT_DATES["test_start"])

    train_df = df[df["Date"] <= train_end].copy()
    val_df   = df[(df["Date"] > train_end) & (df["Date"] <= val_end)].copy()
    test_df  = df[df["Date"] >= test_start].copy()

    print(f"    Train : {len(train_df)} rows | "
          f"{train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"    Val   : {len(val_df)} rows | "
          f"{val_df['Date'].min().date()} to {val_df['Date'].max().date()}")
    print(f"    Test  : {len(test_df)} rows | "
          f"{test_df['Date'].min().date()} to {test_df['Date'].max().date()}")

    return train_df, val_df, test_df


class RegimePredictionDataset(Dataset):
    """
    Produces sliding-window sequences of shape (lookback_window, n_features)
    plus the corresponding input-regime sequence and the target label at the
    end of each window.
    """

    def __init__(self, df, feature_cols, scaler=None,
                 fit_scaler=False, is_train=False):
        self.df       = df.reset_index(drop=True)
        self.lookback = MODEL_CFG["lookback_window"]
        self.step     = 3 if is_train else MODEL_CFG["prediction_freq"]

        self.scaler = scaler if scaler is not None else StandardScaler()
        feat_arr    = df[feature_cols].values
        if fit_scaler:
            self.scaler.fit(feat_arr)
        self.features_scaled = self.scaler.transform(feat_arr).astype(np.float32)

        self.regime_seq = df["detected_regime"].values.astype(np.int64)
        self.labels     = df["pred_label"].values.astype(np.int64)
        self.dates      = df["Date"].values

        n            = len(df)
        self.indices = list(range(self.lookback, n, self.step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end   = self.indices[idx]
        start = end - self.lookback
        feat   = torch.tensor(self.features_scaled[start:end], dtype=torch.float32)
        regime = torch.tensor(self.regime_seq[start:end],      dtype=torch.long)
        label  = torch.tensor(self.labels[end - 1],             dtype=torch.long)
        return feat, regime, label
