"""
train.py
========
Training utilities for a single stock's RegimePredictionMamba model:
class-weight computation (with a Bearish boost), one training epoch,
evaluation, gradient-based feature importance, and the full
`train_model` driver (checkpointing, early stopping, LR schedule).
"""

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from config import CHECKPOINT_PATH, DEVICE, LABEL_SOURCE, MODEL_CFG, REGIME_INV_MAP, RUN_ID, TRAIN_CFG


def compute_class_weights(dataset):
    """
    Inverse-frequency class weights for cross-entropy, with an extra
    1.5x boost on the Bearish class (historically the hardest/rarest
    to predict correctly), renormalised to sum to n_classes.
    """
    all_labels = [dataset[i][2].item() for i in range(len(dataset))]
    counts     = Counter(all_labels)
    total      = sum(counts.values())
    weights    = [total / (MODEL_CFG["n_classes"] * counts.get(c, 1))
                  for c in range(MODEL_CFG["n_classes"])]

    # ── Boost Bearish (class 0) by 1.5x extra ─────────────────────
    BEAR_BOOST = 1.5
    weights[0] *= BEAR_BOOST

    # Renormalise so weights sum to n_classes
    w_sum = sum(weights)
    weights = [w * MODEL_CFG["n_classes"] / w_sum for w in weights]

    print(f"  Class weights (boosted): "
          f"{ {REGIME_INV_MAP[i]: round(w,3) for i,w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float32)


def compute_feature_importance(model, loader, feature_cols, n_batches=20):
    """
    Gradient-magnitude feature importance: averages |d(loss)/d(feature)|
    over up to `n_batches` batches, then ranks and normalises to
    percentages for cross-stock comparability.
    """
    model.eval()
    importance = np.zeros(len(feature_cols))
    total      = 0
    for i, (feat, regime, label) in enumerate(loader):
        if i >= n_batches:
            break
        feat   = feat.to(DEVICE).requires_grad_(True)
        regime = regime.to(DEVICE)
        label  = label.to(DEVICE)
        logits = model(feat, regime)
        loss   = F.cross_entropy(logits, label)
        loss.backward()
        importance += feat.grad.abs().mean(dim=(0,1)).cpu().numpy()
        total      += 1
    importance /= total
    imp_df = pd.DataFrame({
        "feature"       : feature_cols,
        "importance"    : importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df["rank"]           = imp_df.index + 1
    imp_df["importance_pct"] = (
        imp_df["importance"] / imp_df["importance"].sum() * 100
    ).round(2)
    return imp_df


def train_one_epoch(model, loader, optimizer, criterion):
    """Runs one full pass over the training loader."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for feat, regime, label in loader:
        feat, regime, label = feat.to(DEVICE), regime.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        logits = model(feat, regime)
        loss   = criterion(logits, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(label)
        correct    += (logits.argmax(1) == label).sum().item()
        total      += len(label)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluates the model on a loader with no gradient tracking."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels      = [], []
    for feat, regime, label in loader:
        feat, regime, label = feat.to(DEVICE), regime.to(DEVICE), label.to(DEVICE)
        logits = model(feat, regime)
        loss   = criterion(logits, label)
        total_loss += loss.item() * len(label)
        preds       = logits.argmax(1)
        correct    += (preds == label).sum().item()
        total      += len(label)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def train_model(model, train_ds, val_ds, stock_name, feature_cols):
    """
    Full training loop for one stock: class-weighted cross-entropy
    with label smoothing, AdamW, linear-warmup + cosine-annealing LR,
    early stopping on validation accuracy, checkpointing the best
    model + scaler, and a final feature-importance report.
    """
    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_CFG["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAIN_CFG["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True
    )

    class_weights = compute_class_weights(train_ds).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight          = class_weights,
        label_smoothing = TRAIN_CFG["label_smoothing"]
    )
    optimizer = AdamW(
        model.parameters(),
        lr           = TRAIN_CFG["lr"],
        weight_decay = TRAIN_CFG["weight_decay"]
    )
    warmup_epochs = TRAIN_CFG.get('warmup_epochs', 5)
    warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=warmup_epochs)
    cosine    = CosineAnnealingLR(optimizer,
                                   T_max=TRAIN_CFG["n_epochs"] - warmup_epochs)
    scheduler = SequentialLR(optimizer,
                              schedulers=[warmup, cosine],
                              milestones=[warmup_epochs])

    best_val_acc   = 0
    best_val_loss  = float('inf')   # ← add this line
    patience_count = 0
    history        = {
        "train_loss": [], "val_loss": [],
        "train_acc" : [], "val_acc" : []
    }

    ckpt_path   = Path(CHECKPOINT_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}_best.pt"
    scaler_path = Path(CHECKPOINT_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}_scaler.pkl"

    print(f"  Training {stock_name} | "
          f"train={len(train_ds)} val={len(val_ds)} samples")

    for epoch in range(1, TRAIN_CFG["n_epochs"] + 1):
        tr_loss, tr_acc        = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _  = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_loss  = vl_loss
            best_val_acc   = vl_acc   # ← add this line
            patience_count = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_acc"     : vl_acc,
                "label_source": LABEL_SOURCE,
                "model_cfg"   : MODEL_CFG,
            }, ckpt_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(train_ds.scaler, f)
        else:
            patience_count += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
                  f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.3f} | "
                  f"Best: {best_val_acc:.3f}")

        if patience_count >= TRAIN_CFG["early_stop_patience"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Feature importance ────────────────────────────────────────
    print(f"\n  Computing feature importance...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    imp_df = compute_feature_importance(model, train_loader, feature_cols)
    print(f"\n  Top 5 features:")
    print(imp_df[["rank","feature","importance_pct"]].head(5).to_string(index=False))
    print(f"\n  Bottom 5 features:")
    print(imp_df[["rank","feature","importance_pct"]].tail(5).to_string(index=False))
    print(f"\n  Best val acc: {best_val_acc:.4f}")

    return history, str(ckpt_path), imp_df
