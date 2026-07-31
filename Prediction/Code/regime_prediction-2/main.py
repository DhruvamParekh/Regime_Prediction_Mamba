"""
main.py
=======
END-TO-END RUNNER for the regime-prediction pipeline.

Imports every module below and runs the full sequence: mount
storage → load shared data → for each stock in
config.STOCKS_TO_TRAIN, assemble features + labels, train, evaluate,
run the full prediction history, chart it, analyze it, backtest it,
and predict the specific target week → finally build the
cross-stock summary.

This file contains NO modelling logic of its own — it only calls
functions defined in config.py, data/, labels/, dataset.py, model.py,
train.py, inference.py, evaluate.py, backtest.py, visualise.py, and
summary.py, in the same order as the original 18-cell notebook.

⚠️  GPU REQUIRED — this pipeline depends on the `mamba_ssm` package
    (see model.py), which only runs on an NVIDIA GPU with CUDA 12.6.
    Run `setup_env.sh` once before this script (see README.md).

Usage:
    python main.py
"""

import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

import config
from config import (
    CHECKPOINT_PATH, DEVICE, LABEL_SOURCE, MACRO_FILE, MODEL_CFG, NIFTY_FILE,
    PROCESSED_PATH, RAW_DATA_PATH, REGIME_INV_MAP, RESULTS_PATH, RUN_ID,
    SEED, SPLIT_DATES, STOCKS_TO_TRAIN, TARGET_DATE, TRAIN_CFG,
)

from data.loader import load_stock_csv, load_macro, load_nifty
from data.features import compute_features

from labels.pseudo import load_or_generate_pseudo
from labels.causal import load_or_generate_causal_confirmed
from labels.regime_loader import load_regime_labels, build_prediction_labels

from dataset import RegimePredictionDataset
from model import RegimePredictionMamba
from train import compute_class_weights, evaluate as evaluate_model, train_model

from inference import get_full_predictions, predict_specific_week
from evaluate import analyze_period_accuracy, analyze_regime_quality, analyze_returns_by_regime
from backtest import run_backtest, plot_backtest
from visualise import (
    plot_full_regime_chart, plot_analysis, plot_confusion_matrices,
    plot_prediction_context, plot_cross_stock_summary,
)
from summary import build_cross_stock_summary, print_cross_stock_summary

import torch.nn as nn

warnings.filterwarnings('ignore')


# ── Reproducibility & device ───────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)


def check_gpu():
    """Warns loudly (but does not hard-exit) if no GPU/mamba_ssm is available."""
    print(f"Device : {DEVICE}")
    if DEVICE.type != "cuda":
        print("=" * 70)
        print("⚠️  WARNING: No GPU detected. Mamba SSM REQUIRES an NVIDIA GPU")
        print("   with CUDA 12.6. Run setup_env.sh on a GPU runtime first.")
        print("=" * 70)
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        print("=" * 70)
        print("⚠️  WARNING: mamba_ssm is not importable. Run setup_env.sh")
        print("   (GPU + CUDA 12.6 required) before running this script.")
        print("=" * 70)


def mount_and_prepare_folders():
    """
    Mounts Google Drive (Colab) if not already mounted, and creates
    the processed/results/checkpoint output folders.
    """
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive')
        else:
            print("Drive already mounted.")
    except ImportError:
        print("Not running on Colab — skipping drive mount "
              "(ensure BASE in config.py points to a valid local path).")

    for folder in [PROCESSED_PATH, RESULTS_PATH, CHECKPOINT_PATH]:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {folder}")


def discover_stock_files():
    """
    Scans RAW_DATA_PATH for per-stock CSVs (excluding the macro/NIFTY
    files) and validates that every stock in STOCKS_TO_TRAIN has a
    matching file.
    """
    exclude_files = {MACRO_FILE.lower(), NIFTY_FILE.lower()}
    all_files     = sorted(Path(RAW_DATA_PATH).glob('*.csv'))
    all_stock_files = {
        f.stem: f for f in all_files
        if f.name.lower() not in exclude_files
    }

    macro_path = Path(RAW_DATA_PATH) / MACRO_FILE
    nifty_path = Path(RAW_DATA_PATH) / NIFTY_FILE

    print(f"\nTotal stocks found : {len(all_stock_files)}")
    print(f"Stocks to train    : {STOCKS_TO_TRAIN}")
    print(f"Macro exists       : {macro_path.exists()}")
    print(f"NIFTY exists       : {nifty_path.exists()}")

    for s in STOCKS_TO_TRAIN:
        if s not in all_stock_files:
            print(f"  WARNING: {s} not found in raw folder")
        else:
            print(f"  ✓ {s}")

    return all_stock_files, macro_path, nifty_path


def assemble_stock_data(stock_name, all_stock_files, macro_df, nifty_df):
    """
    Loads a stock's price data, computes features, generates/loads
    both pseudo-60 (for the forward prediction label) and the
    configured LABEL_SOURCE (for the model's input regime signal),
    then merges everything into one DataFrame ready for splitting.
    """
    stock_df  = load_stock_csv(all_stock_files[stock_name])
    feat_df, feat_cols = compute_features(stock_df, macro_df, nifty_df)

    pseudo60_df     = load_or_generate_pseudo(stock_name, stock_df, config.PSEUDO_CFG)
    pseudo60_regime = pseudo60_df.rename(columns={"pseudo_label": "regime_label"})
    pred_label_df   = build_prediction_labels(pseudo60_regime)

    if LABEL_SOURCE == "causal_confirmed":
        cc_df = load_or_generate_causal_confirmed(stock_name, stock_df, config.CAUSAL_CFG)
        input_signal = cc_df[["Date", "regime_label"]].rename(
            columns={"regime_label": "detected_regime"})
    elif LABEL_SOURCE == "pseudo":
        input_signal = pseudo60_regime[["Date", "regime_label"]].rename(
            columns={"regime_label": "detected_regime"})
    elif LABEL_SOURCE == "detected":
        detected_regime = load_regime_labels(stock_name, stock_df, "detected")
        input_signal = detected_regime.rename(columns={"regime_label": "detected_regime"})
    else:
        raise ValueError(f"Unhandled LABEL_SOURCE: {LABEL_SOURCE}")

    # ── LEFT join so tail rows beyond pseudo-60 range are kept ───────
    merged = feat_df.merge(pred_label_df, on="Date", how="left")
    merged = merged.merge(input_signal,   on="Date", how="left")

    merged["detected_regime"] = merged["detected_regime"].ffill().fillna(1).astype(int)

    # ── Fill pred_label tail with last known label (ffill) ───────────
    # Rows beyond pseudo-60 range have NaN pred_label.
    # These are TEST-only rows — model predicts on them but they have
    # no ground-truth forward label. Fill with last known for dataset
    # construction; true_label in output will show this honestly.
    merged["pred_label"] = merged["pred_label"].ffill()

    merged = merged.dropna(subset=feat_cols).reset_index(drop=True)
    merged["pred_label"] = merged["pred_label"].astype(int)

    dist = {REGIME_INV_MAP[k]: v
            for k, v in merged["pred_label"].value_counts().to_dict().items()}
    print(f"  {stock_name}: {len(merged)} rows | {dist}")

    return merged, feat_cols


def make_splits(df):
    """Splits an assembled stock DataFrame into Train/Val/Test by SPLIT_DATES."""
    train_end  = pd.Timestamp(SPLIT_DATES["train_end"])
    val_end    = pd.Timestamp(SPLIT_DATES["val_end"])
    test_start = pd.Timestamp(SPLIT_DATES["test_start"])

    train_df = df[df["Date"] <= train_end].copy()
    val_df   = df[(df["Date"] > train_end) & (df["Date"] <= val_end)].copy()
    test_df  = df[df["Date"] >= test_start].copy()

    test_end = pd.Timestamp(SPLIT_DATES["test_end"]) if SPLIT_DATES.get("test_end") else None
    if test_end is not None:
        test_df = test_df[test_df["Date"] <= test_end].copy()

    print(f"    Train : {len(train_df)} rows | "
          f"{train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"    Val   : {len(val_df)} rows | "
          f"{val_df['Date'].min().date()} to {val_df['Date'].max().date()}")
    print(f"    Test  : {len(test_df)} rows | "
          f"{test_df['Date'].min().date()} to {test_df['Date'].max().date()}")

    return train_df, val_df, test_df


def main():
    check_gpu()

    # ── Step 1: mount storage & create output folders ─────────────
    mount_and_prepare_folders()

    # ── Step 2: discover stock files & load shared data ────────────
    all_stock_files, macro_path, nifty_path = discover_stock_files()
    macro_df = load_macro(macro_path)
    nifty_df = load_nifty(nifty_path)

    print("Macro:")
    print(f"  Shape : {macro_df.shape}")
    print(f"  Dates : {macro_df['Date'].min().date()} to {macro_df['Date'].max().date()}")
    print("\nNIFTY:")
    print(f"  Shape : {nifty_df.shape}")
    print(f"  Dates : {nifty_df['Date'].min().date()} to {nifty_df['Date'].max().date()}")

    # ── Step 3: train + evaluate + save predictions per stock ──────
    ALL_RESULTS    = {}
    ALL_IMPORTANCE = {}

    for stock_name in STOCKS_TO_TRAIN:
        print(f"\n{'='*60}")
        print(f"  {stock_name} | Label Source: {LABEL_SOURCE.upper()}")
        print(f"{'='*60}")

        try:
            # ── Assemble ─────────────────────────────────────────
            merged_df, feat_cols = assemble_stock_data(
                stock_name, all_stock_files, macro_df, nifty_df
            )

            if len(merged_df) < MODEL_CFG["lookback_window"] * 3:
                print(f"  SKIP: not enough data ({len(merged_df)} rows)")
                continue

            # ── Split ────────────────────────────────────────────
            train_df, val_df, test_df = make_splits(merged_df)

            if len(train_df) < MODEL_CFG["lookback_window"] * 2:
                print(f"  SKIP: not enough train data ({len(train_df)} rows)")
                continue

            # ── Datasets ─────────────────────────────────────────
            train_ds = RegimePredictionDataset(
                train_df, feat_cols, fit_scaler=True, is_train=True
            )
            val_ds = RegimePredictionDataset(
                val_df, feat_cols, scaler=train_ds.scaler, is_train=False
            )
            test_ds = RegimePredictionDataset(
                test_df, feat_cols, scaler=train_ds.scaler, is_train=False
            )

            if len(train_ds) < 10 or len(val_ds) < 2:
                print(f"  SKIP: too few samples "
                      f"(train={len(train_ds)}, val={len(val_ds)})")
                continue

            # ── Train ────────────────────────────────────────────
            model = RegimePredictionMamba().to(DEVICE)
            history, ckpt_path, imp_df = train_model(
                model, train_ds, val_ds, stock_name, feat_cols
            )

            # ── Test evaluation ──────────────────────────────────
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt["model_state"])

            test_loader   = DataLoader(
                test_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=False
            )
            class_weights = compute_class_weights(test_ds).to(DEVICE)
            criterion     = nn.CrossEntropyLoss(weight=class_weights)

            _, test_acc, test_preds, test_labels = evaluate_model(
                model, test_loader, criterion
            )

            print(f"\n  Test Accuracy : {test_acc:.4f}")
            print(classification_report(
                test_labels, test_preds,
                labels=[0, 1, 2],
                target_names=["Bearish", "Neutral", "Bullish"],
                zero_division=0
            ))

            # ── Save predictions ─────────────────────────────────
            result_dir = Path(RESULTS_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}"
            result_dir.mkdir(parents=True, exist_ok=True)

            test_dates = test_df["Date"].values[
                MODEL_CFG["lookback_window"]::MODEL_CFG["prediction_freq"]
            ]
            n_preds = min(len(test_dates), len(test_preds))

            pred_df = pd.DataFrame({
                "Date"      : test_dates[:n_preds],
                "true_label": [REGIME_INV_MAP[l] for l in test_labels[:n_preds]],
                "pred_label": [REGIME_INV_MAP[p] for p in test_preds[:n_preds]],
                "true_idx"  : test_labels[:n_preds],
                "pred_idx"  : test_preds[:n_preds],
            })
            pred_df.to_csv(result_dir / "predictions.csv", index=False)

            pd.DataFrame(history).to_csv(
                result_dir / "training_history.csv", index=False
            )
            imp_df.to_csv(
                result_dir / "feature_importance.csv", index=False
            )

            # ── Store ────────────────────────────────────────────
            ALL_RESULTS[stock_name] = {
                "test_acc"    : test_acc,
                "best_val_acc": ckpt["val_acc"],
                "test_preds"  : test_preds,
                "test_labels" : test_labels,
                "history"     : history,
                "pred_df"     : pred_df,
                "merged_df"   : merged_df,
                "feat_cols"   : feat_cols,
                "ckpt_path"   : ckpt_path,
                "scaler_path" : str(ckpt_path).replace("_best.pt", "_scaler.pkl"),
            }
            ALL_IMPORTANCE[stock_name] = imp_df

        except Exception as e:
            print(f"  FAILED : {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Done: {len(ALL_RESULTS)}/{len(STOCKS_TO_TRAIN)} stocks trained")
    print(f"{'='*60}")

    # ── Step 4: full prediction history + regime chart per stock ───
    for stock_name, res in ALL_RESULTS.items():
        print(f"Generating full chart: {stock_name}")
        try:
            full_pred_df = get_full_predictions(
                stock_name, res["merged_df"], res["feat_cols"]
            )
            ALL_RESULTS[stock_name]["full_pred_df"] = full_pred_df
            plot_full_regime_chart(
                stock_name, full_pred_df, res["merged_df"], all_stock_files,
                res["best_val_acc"], res["test_acc"]
            )
        except Exception as e:
            print(f"  FAILED {stock_name}: {e}")
            traceback.print_exc()
    print("\nAll charts saved.")

    # ── Step 5: period accuracy, regime quality, returns analysis ───
    for stock_name, res in ALL_RESULTS.items():
        print(f"\n{'='*55}")
        print(f"  Analyzing : {stock_name}")
        print(f"{'='*55}")

        result_dir = Path(RESULTS_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}"
        result_dir.mkdir(parents=True, exist_ok=True)

        full_pred_df = get_full_predictions(
            stock_name, res["merged_df"], res["feat_cols"]
        )
        ALL_RESULTS[stock_name]["full_pred_df"] = full_pred_df

        stock_df = load_stock_csv(all_stock_files[stock_name])

        # ── Charts ────────────────────────────────────────────────
        plot_full_regime_chart(
            stock_name, full_pred_df, res["merged_df"], all_stock_files,
            res["best_val_acc"], res["test_acc"]
        )

        # ── Accuracy ──────────────────────────────────────────────
        period_acc_df = analyze_period_accuracy(full_pred_df, SPLIT_DATES)
        print(f"\n  Period Accuracy:")
        print(period_acc_df.to_string(index=False))

        # ── Regime quality ────────────────────────────────────────
        quality_dict = analyze_regime_quality(full_pred_df, stock_df)
        print(f"\n  Regime Duration (avg days):")
        print(quality_dict["duration_df"].to_string(index=False))
        print(f"\n  Transition Detection : {quality_dict['transition_rate']}% "
              f"({quality_dict['detected']}/{quality_dict['total_transitions']}) "
              f"within ±7 days")

        # ── Returns ───────────────────────────────────────────────
        period_ret_df, regime_ret_df = analyze_returns_by_regime(
            full_pred_df, stock_df, SPLIT_DATES
        )
        print(f"\n  Returns by Period:")
        print(period_ret_df.to_string(index=False))
        print(f"\n  Returns by Predicted Regime (Test):")
        print(regime_ret_df[regime_ret_df["period"] == "Test"].to_string(index=False))

        # ── Analysis plots ────────────────────────────────────────
        plot_analysis(
            stock_name, period_acc_df, quality_dict,
            period_ret_df, regime_ret_df, result_dir
        )
        plot_confusion_matrices(stock_name, full_pred_df, SPLIT_DATES, result_dir)

        # ── Save CSVs ─────────────────────────────────────────────
        full_pred_df.to_csv(result_dir / "full_predictions.csv", index=False)
        period_acc_df.to_csv(result_dir / "period_accuracy.csv", index=False)
        quality_dict["duration_df"].to_csv(result_dir / "regime_duration.csv", index=False)
        quality_dict["true_runs_df"].to_csv(result_dir / "true_regime_runs.csv", index=False)
        quality_dict["pred_runs_df"].to_csv(result_dir / "pred_regime_runs.csv", index=False)
        period_ret_df.to_csv(result_dir / "period_returns.csv", index=False)
        regime_ret_df.to_csv(result_dir / "regime_returns.csv", index=False)

        print(f"\n  Saved to results/{stock_name}/")

    # ── Step 6: backtesting ─────────────────────────────────────────
    print("=" * 60)
    print("  BACKTESTING — TEST PERIOD")
    print("=" * 60)

    ALL_BACKTEST = {}

    for stock_name, res in ALL_RESULTS.items():
        print(f"\nRunning backtest : {stock_name}")
        result_dir = Path(RESULTS_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}"

        full_pred_df = res["full_pred_df"]

        # ── Compute Bear precision (important: prediction-based) ──
        bear_mask = full_pred_df["pred_label"] == "Bearish"

        if bear_mask.sum() > 0:
            bear_accuracy = (
                (full_pred_df.loc[bear_mask, "true_label"] == "Bearish").mean()
            )
        else:
            bear_accuracy = 0.0

        allow_short = bear_accuracy > 0.45
        bt = run_backtest(stock_name, res["full_pred_df"], allow_short, all_stock_files)
        if bt is None:
            continue

        plot_backtest(bt, result_dir)
        bt["metrics"].to_csv(result_dir / "backtest_metrics.csv", index=False)
        ALL_BACKTEST[stock_name] = bt

    # ── Cross stock backtest summary ─────────────────────────────────
    print(f"\n{'='*65}")
    print("  CROSS-STOCK BACKTEST SUMMARY")
    print(f"{'='*65}")

    summary_rows = []
    for stock_name, bt in ALL_BACKTEST.items():
        for _, row in bt["metrics"].iterrows():
            summary_rows.append({
                "stock"         : stock_name,
                "strategy"      : row["strategy"],
                "total_ret_pct" : row["total_ret_pct"],
                "annual_ret_pct": row["annual_ret_pct"],
                "sharpe"        : row["sharpe"],
                "max_drawdown"  : row["max_drawdown"],
                "days_in_market": row["days_in_market"],
            })

    bt_summary = pd.DataFrame(summary_rows)

    pivot = bt_summary.pivot_table(
        index="stock", columns="strategy", values="total_ret_pct"
    ).round(2)

    print("\nTotal Return % by Strategy (Test Period):")
    print(pivot.to_string())

    bt_summary.to_csv(Path(RESULTS_PATH) / "backtest_summary.csv", index=False)
    print(f"\nSaved to results/backtest_summary.csv")

    # ── Step 7: specific-week prediction ─────────────────────────────
    print("=" * 60)
    print(f"  REGIME PREDICTION — Week of {TARGET_DATE}")
    print(f"  Label Source : {LABEL_SOURCE.upper()}")
    print("=" * 60)

    def _assemble_for_inference(stock_name):
        return assemble_stock_data(stock_name, all_stock_files, macro_df, nifty_df)

    nov_predictions = []

    for stock_name in ALL_RESULTS.keys():
        result = predict_specific_week(stock_name, TARGET_DATE, _assemble_for_inference)
        if result is None:
            continue

        nov_predictions.append(result)
        result_dir = Path(RESULTS_PATH) / f"{stock_name}_{LABEL_SOURCE.upper()}_Run{RUN_ID}"

        print(f"\n  {stock_name}")
        print(f"    History     : {result['history_from']} to {result['history_to']}")
        print(f"    Current     : {result['current_regime']}")
        print(f"    Predicted   : {result['predicted_regime']}")
        if result["transition"]:
            print(f"    ⚠️  TRANSITION DETECTED")
        else:
            print(f"    ✓  Continuation")
        print(f"    P(Bull/Neut/Bear) : "
              f"{result['p_bullish']} / "
              f"{result['p_neutral']} / "
              f"{result['p_bearish']}")
        print(f"    Confidence  : {result['confidence']}")

        plot_prediction_context(stock_name, result, result_dir, all_stock_files)

    if nov_predictions:
        nov_df = pd.DataFrame(nov_predictions)

        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(nov_df[[
            "stock", "current_regime", "predicted_regime",
            "transition", "confidence"
        ]].to_string(index=False))

        transitions = nov_df[nov_df["transition"] == True]
        print(f"\n  Transition alerts : {len(transitions)}/{len(nov_df)} stocks")
        for _, row in transitions.iterrows():
            print(f"    ⚠️  {row['stock']:15s} "
                  f"{row['current_regime']} → {row['predicted_regime']} "
                  f"(conf={row['confidence']})")

        # NOTE: saving nov_df to CSV was commented out (disabled) in the
        # original notebook — preserved as disabled here for zero logic
        # changes. Uncomment to enable:
        # nov_df.to_csv(
        #     Path(RESULTS_PATH) / f"nov2025_predictions_{LABEL_SOURCE.upper()}_Run{RUN_ID}.csv",
        #     index=False
        # )
        # print(f"\n  Saved to results/nov2025_predictions.csv")

    # ── Step 8: cross-stock summary ──────────────────────────────────
    summary_df = build_cross_stock_summary(ALL_RESULTS, ALL_BACKTEST, ALL_IMPORTANCE)
    print_cross_stock_summary(summary_df, ALL_IMPORTANCE)
    plot_cross_stock_summary(summary_df)


if __name__ == "__main__":
    main()
