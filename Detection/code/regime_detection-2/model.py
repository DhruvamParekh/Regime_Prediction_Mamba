"""
model.py
========
The Mamba-based regime classifier. A stack of Mamba state-space blocks
processes the feature sequence (with the input regime embedded and added
in), and a small MLP head classifies the final timestep into
Bearish / Neutral / Bullish.

Logic is unchanged from the original notebook.

NOTE: `mamba-ssm` requires an NVIDIA GPU with a working CUDA toolkit to
install and run (it compiles custom CUDA kernels). This will not run on
CPU-only machines. See README.md for details.
"""

import torch.nn as nn
from mamba_ssm import Mamba

from config import MODEL_CFG


class MambaBlock(nn.Module):
    """Pre-norm residual wrapper around a single Mamba layer."""

    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.mamba   = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class RegimePredictionMamba(nn.Module):
    """Feature projection + regime embedding -> stacked Mamba blocks -> classifier head."""

    def __init__(self, cfg=None):
        super().__init__()
        cfg            = cfg if cfg is not None else MODEL_CFG
        d_model        = cfg["d_model"]
        d_state        = cfg["d_state"]
        d_conv         = cfg["d_conv"]
        expand         = cfg["expand"]
        n_layers       = cfg["n_mamba_layers"]
        dropout        = cfg["dropout"]
        n_features     = cfg["n_raw_features"]
        n_classes      = cfg["n_classes"]
        regime_emb_dim = cfg["regime_embed_dim"]

        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        self.regime_embed = nn.Embedding(n_classes, regime_emb_dim)
        self.regime_proj  = nn.Linear(regime_emb_dim, d_model)

        self.input_dropout = nn.Dropout(dropout)

        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, features, regime_seq):
        x = self.feature_proj(features)
        r = self.regime_embed(regime_seq)
        r = self.regime_proj(r)
        x = self.input_dropout(x + r)
        for block in self.mamba_blocks:
            x = block(x)
        x = self.final_norm(x[:, -1, :])
        return self.classifier(x)
