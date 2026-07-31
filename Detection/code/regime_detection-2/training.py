"""
training.py
===========
Class-weight computation, one training epoch, evaluation, gradient-based
feature importance, and the full per-stock training loop (with checkpointing
and early stopping).

Logic is unchanged from the original notebook, with ONE structural (not
logical) change: `compute_feature_importance` and `train_model` now take
`feat_cols` as an explicit argument instead of silently reading a
module-level global `FEATURE_COLS` variable (as the original notebook did).
The feature list is identical for every stock either way, so this does not
change any computed numbers — it just avoids fragile cross-file global
state, which doesn't translate well outside a single notebook.
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import CHECKPOINT_PATH, LABEL_SOURCE, MODEL_CFG, REGIME_INV_MAP, RUN_ID, TRAIN_CFG


def compute_class_weights(dataset):
    """Inverse-frequency class weights, used to counter regime class imbalance."""
    all_labels = [dataset[i][2].item() for i in range(len(dataset))]
    counts     = Counter(all_labels)
    total      = sum(counts.values())
    weights    = [total / (MODEL_CFG["n_classes"] * counts.get(c, 1))
                  for c in range(MODEL_CFG["n_classes"])]
    print(f"  Class weights : "
          f"{ {REGIME_INV_MAP[i]: round(w, 3) for i, w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float32)


def compute_feature_importance(model, loader, feat_cols, device, n_batches=20):
    """Gradient-based feature importance: average |d(loss)/d(feature)| over a few batches."""
    model.eval()
    importance = np.zeros(len(feat_cols))
    total      = 0
    for i, (feat, regime, label) in enumerate(loader):
        if i >= n_batches:
            break
        feat   = feat.to(device).requires_grad_(True)
        regime = regime.to(device)
        label  = label.to(device)
        logits = model(feat, regime)
        loss   = F.cross_entropy(logits, label)
        loss.backward()
        importance += feat.grad.abs().mean(dim=(0, 1)).cpu().numpy()
        total      += 1
    importance /= total
    imp_df = pd.DataFrame({
        "feature"   : feat_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df["rank"]           = imp_df.index + 1
    imp_df["importance_pct"] = (
        imp_df["importance"] / imp_df["importance"].sum() * 100
    ).round(2)
    return imp_df


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for feat, regime, label in loader:
        feat, regime, label = feat.to(device), regime.to(device), label.to(device)
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels      = [], []
    for feat, regime, label in loader:
        feat, regime, label = feat.to(device), regime.to(device), label.to(device)
        logits = model(feat, regime)
        loss   = criterion(logits, label)
        total_loss += loss.item() * len(label)
        preds       = logits.argmax(1)
        correct    += (preds == label).sum().item()
        total      += len(label)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def train_model(model, train_ds, val_ds, stock_name, feat_cols, device):
    """Full training loop for one stock: trains, checkpoints on best val acc,
    early-stops, then computes feature importance from the best checkpoint."""
    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_CFG["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAIN_CFG["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True
    )

    class_weights = compute_class_weights(train_ds).to(device)
    criterion     = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=TRAIN_CFG["label_smoothing"]
    )
    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CFG["lr"],
        weight_decay=TRAIN_CFG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CFG["n_epochs"])

    best_val_acc   = 0
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
        tr_loss, tr_acc        = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _  = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc   = vl_acc
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    imp_df = compute_feature_importance(model, train_loader, feat_cols, device)
    print(f"\n  Top 5 features:")
    print(imp_df[["rank", "feature", "importance_pct"]].head(5).to_string(index=False))
    print(f"\n  Bottom 5 features:")
    print(imp_df[["rank", "feature", "importance_pct"]].tail(5).to_string(index=False))
    print(f"\n  Best val acc: {best_val_acc:.4f}")

    return history, str(ckpt_path), imp_df
