"""
dataset.py
==========
PyTorch Dataset that slices each stock's feature/regime/label series
into fixed-length lookback windows, plus a post-processing helper
used at inference time to smooth over unrealistically short
predicted regimes.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config import MODEL_CFG


def enforce_min_regime_duration(predictions, min_duration=21):
    """
    Post-processing: merge regime runs shorter than min_duration
    into their neighbours. Prevents the model's oscillation problem
    where predicted durations are 3-5x shorter than actual.
    Applied only at inference time, not during training.
    """
    if len(predictions) == 0:
        return predictions
    result = predictions.copy()
    i = 0
    while i < len(result):
        current = result[i]
        j = i + 1
        while j < len(result) and result[j] == current:
            j += 1
        run_length = j - i
        if run_length < min_duration and i > 0:
            # Replace short run with previous regime
            result[i:j] = result[i - 1]
        i = j
    return result


class RegimePredictionDataset(Dataset):
    """
    Slices a stock's assembled DataFrame (features + detected_regime
    + pred_label, indexed by Date) into overlapping lookback windows
    of length MODEL_CFG['lookback_window'].

    Training windows (is_train=True) step by 2 days and apply light
    augmentation (Gaussian noise + random feature masking) to reduce
    overfitting. Val/test windows step by prediction_freq days
    (non-overlapping, matching how the model is actually queried at
    inference) and use no augmentation.
    """

    def __init__(self, df, feature_cols, scaler=None,
                 fit_scaler=False, is_train=False):
        self.df       = df.reset_index(drop=True)
        self.lookback = MODEL_CFG["lookback_window"]
        self.step     = 2 if is_train else MODEL_CFG["prediction_freq"]
        self.is_train = is_train
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
        end    = self.indices[idx]
        start  = end - self.lookback
        feat   = torch.tensor(self.features_scaled[start:end], dtype=torch.float32)
        regime = torch.tensor(self.regime_seq[start:end],      dtype=torch.long)
        label  = torch.tensor(self.labels[end - 1],            dtype=torch.long)
        if self.is_train:
            # Gaussian noise
            feat = feat + 0.01 * torch.randn_like(feat)
            # Feature masking — randomly zero out ~10% of features
            # forces model not to over-rely on any single indicator
            feat_mask = (torch.rand(feat.shape[-1]) > 0.15).float()
            feat = feat * feat_mask.unsqueeze(0)
        return feat, regime, label
