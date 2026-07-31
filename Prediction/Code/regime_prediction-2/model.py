"""
model.py
========
The RegimePredictionMamba architecture: a stack of Mamba SSM blocks
with per-layer regime-embedding conditioning, followed by a
classification head that combines the last-timestep representation
with a mean-pooled representation over the lookback window.

⚠️  Requires an NVIDIA GPU with CUDA 12.6 — the `mamba_ssm` package
    used here (Mamba) only runs on GPU. See setup_env.sh.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from config import MODEL_CFG


class MambaBlock(nn.Module):
    """
    One residual Mamba layer with LayerNorm, dropout, and an
    additional sigmoid gate conditioned on the mean regime embedding
    for the current window (per-layer regime conditioning).
    """

    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.mamba   = Mamba(
            d_model = d_model,
            d_state = d_state,
            d_conv  = d_conv,
            expand  = expand,
        )
        self.dropout    = nn.Dropout(dropout)
        # Projects regime context into d_model for per-layer conditioning
        self.regime_gate = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, regime_context=None):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        out = x + residual
        if regime_context is not None:
            # regime_context: (batch, d_model) — broadcast across time
            gate = torch.sigmoid(self.regime_gate(regime_context))
            out  = out * gate.unsqueeze(1)
        return out


class RegimePredictionMamba(nn.Module):
    """
    Full model: raw-feature projection + regime-embedding injection,
    a stack of MambaBlocks, then a classifier over
    [last-timestep || mean-pooled] representations.
    """

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

        self.final_norm = nn.LayerNorm(d_model * 2)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),   # 2*d_model input from last+pooled
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
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

        # Regime context for per-layer conditioning:
        # use the mean regime embedding across the lookback window
        regime_context = r.mean(dim=1)   # (batch, d_model)

        for block in self.mamba_blocks:
            x = block(x, regime_context=regime_context)

        # Last timestep + mean pool — both carry information
        last   = x[:, -1, :]         # (batch, d_model)
        pooled = x.mean(dim=1)        # (batch, d_model)
        x      = self.final_norm(torch.cat([last, pooled], dim=-1))  # (batch, 2*d_model)
        return self.classifier(x)
