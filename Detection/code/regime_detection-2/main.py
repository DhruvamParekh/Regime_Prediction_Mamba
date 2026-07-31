"""
main.py
=======
Orchestrates the full regime DETECTION pipeline, end to end, in the same
order the original notebook ran its cells:

  1. Discover stock files, load shared macro/NIFTY data
  2. For each stock: assemble data, split train/val/test, train the Mamba
     model, save predictions/history/feature-importance
  3. For each trained stock: run full-history predictions + regime chart
  4. For each trained stock: accuracy/quality/returns analysis + confusion
     matrices
  5. For each trained stock: backtest over the test period, then a
     cross-stock backtest summary
  6. For each trained stock: latest detected-regime signal for TARGET_DATE
  7. Cross-stock summary table + charts

Run with:
    python main.py

All results are written under config.RESULTS_PATH, in the same per-stock
folder structure as the original notebook
(`{STOCK}_{LABEL_SOURCE}_Run{RUN_ID}/...`).
"""

import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

import config
from data.loaders import discover_stock_files, load_macro, load_nifty, load_stock_csv
from dataset import RegimePredictionDataset, assemble_stock_data, make_splits
from model import RegimePredictionMamba
from training import compute_class_weights, evaluate, train_model
from evaluation.predictions import get_full_predictions, plot_full_regime_chart
from evaluation.analysis import (
    analyze_period_accuracy,
    analyze_regime_quality,
    analyze_returns_by_regime,
    plot_analysis,
    plot_confusion_matrices,
)
from evaluation.backtest import plot_backtest, run_backtest
from evaluation.latest_signal import plot_prediction_context, predict_specific_week
from summary.cross_stock import (
    build_cross_stock_summary,
    plot_cross_stock_summary,
    print_cross_stock_summary,
)

warnings.filterwarnings("ignore")


def setup():
    """Seed, device, folder creation, and shared data loading."""
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: running on CPU. mamba-ssm is built around CUDA "
              "kernels — training will likely be extremely slow or may "
              "fail outright without a GPU. See README.md.")

    for folder in config.FOLDERS_TO_CREATE:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {folder}")

    all_stock_files, macro_path, nifty_path = discover_stock_files()

    print(f"\nTotal stocks found : {len(all_stock_files)}")
    print(f"Stocks to train    : {config.STOCKS_TO_TRAIN}")
    print(f"Macro exists       : {macro_path.exists()}")
    print(f"NIFTY exists       : {nifty_path.exists()}")

    for s in config.STOCKS_TO_TRAIN:
        if s not in all_stock_files:
            print(f"  WARNING: {s} not found in raw folder")

    macro_df = load_macro(macro_path)
    nifty_df = load_nifty(nifty_path)

    return device, all_stock_files, macro_df, nifty_df


def run_training(all_stock_files, macro_df, nifty_df, device):
    """Assemble data, split, train, and save predictions/history/importance for every stock."""
    all_results    = {}
    all_importance = {}

    for stock_name in config.STOCKS_TO_TRAIN:
        print(f"\n{'='*60}")
        print(f"  {stock_name} | Label Source: {config.LABEL_SOURCE.upper()}")
        print(f"{'='*60}")

        try:
            # ── Assemble ─────────────────────────────────────────────
            merged_df, feat_cols = assemble_stock_data(
                stock_name, all_stock_files, macro_df, nifty_df, config.LABEL_SOURCE
            )

            if len(merged_df) < config.MODEL_CFG["lookback_window"] * 3:
                print(f"  SKIP: not enough data ({len(merged_df)} rows)")
                continue

            # ── Split ────────────────────────────────────────────────
            train_df, val_df, test_df = make_splits(merged_df)

            if len(train_df) < config.MODEL_CFG["lookback_window"] * 2:
                print(f"  SKIP: not enough train data ({len(train_df)} rows)")
                continue

            # ── Datasets ─────────────────────────────────────────────
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

            # ── Train ────────────────────────────────────────────────
            model = RegimePredictionMamba().to(device)
            history, ckpt_path, imp_df = train_model(
                model, train_ds, val_ds, stock_name, feat_cols, device
            )

            # ── Test evaluation ──────────────────────────────────────
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state"])

            test_loader   = DataLoader(
                test_ds, batch_size=config.TRAIN_CFG["batch_size"], shuffle=False
            )
            class_weights = compute_class_weights(test_ds).to(device)
            criterion     = nn.CrossEntropyLoss(weight=class_weights)

            _, test_acc, test_preds, test_labels = evaluate(
                model, test_loader, criterion, device
            )

            print(f"\n  Test Accuracy : {test_acc:.4f}")
            print(classification_report(
                test_labels, test_preds,
                labels=[0, 1, 2],
                target_names=["Bearish", "Neutral", "Bullish"],
                zero_division=0
            ))

            # ── Save predictions ─────────────────────────────────────
            result_dir = Path(config.RESULTS_PATH) / \
                f"{stock_name}_{config.LABEL_SOURCE.upper()}_Run{config.RUN_ID}"
            result_dir.mkdir(parents=True, exist_ok=True)

            test_dates = test_df["Date"].values[
                config.MODEL_CFG["lookback_window"]::config.MODEL_CFG["prediction_freq"]
            ]
            n_preds = min(len(test_dates), len(test_preds))

            pred_df = pd.DataFrame({
                "Date"      : test_dates[:n_preds],
                "true_label": [config.REGIME_INV_MAP[l] for l in test_labels[:n_preds]],
                "pred_label": [config.REGIME_INV_MAP[p] for p in test_preds[:n_preds]],
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

            # ── Store ────────────────────────────────────────────────
            all_results[stock_name] = {
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
            all_importance[stock_name] = imp_df

        except Exception as e:
            print(f"  FAILED : {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Done: {len(all_results)}/{len(config.STOCKS_TO_TRAIN)} stocks trained")
    print(f"{'='*60}")

    return all_results, all_importance


def run_full_history_charts(all_results, all_stock_files, device):
    """Full-history predictions + regime chart for every trained stock."""
    for stock_name, res in all_results.items():
        print(f"Generating full chart: {stock_name}")
        try:
            full_pred_df = get_full_predictions(
                stock_name, res["merged_df"], res["feat_cols"], device
            )
            all_results[stock_name]["full_pred_df"] = full_pred_df
            plot_full_regime_chart(
                stock_name, full_pred_df, res["merged_df"], all_stock_files,
                res["best_val_acc"], res["test_acc"],
            )
        except Exception as e:
            print(f"  FAILED {stock_name}: {e}")
            traceback.print_exc()

    print("\nAll charts saved.")


def run_analysis(all_results, all_stock_files, device):
    """Accuracy / quality / returns analysis + confusion matrices for every trained stock."""
    for stock_name, res in all_results.items():
        print(f"\n{'='*55}")
        print(f"  Analyzing : {stock_name}")
        print(f"{'='*55}")

        result_dir = Path(config.RESULTS_PATH) / \
            f"{stock_name}_{config.LABEL_SOURCE.upper()}_Run{config.RUN_ID}"
        result_dir.mkdir(parents=True, exist_ok=True)

        full_pred_df = get_full_predictions(
            stock_name, res["merged_df"], res["feat_cols"], device
        )
        all_results[stock_name]["full_pred_df"] = full_pred_df

        stock_df = load_stock_csv(all_stock_files[stock_name])

        # ── Charts ────────────────────────────────────────────────────
        plot_full_regime_chart(
            stock_name, full_pred_df, res["merged_df"], all_stock_files,
            res["best_val_acc"], res["test_acc"],
        )

        # ── Accuracy ──────────────────────────────────────────────────
        period_acc_df = analyze_period_accuracy(full_pred_df, config.SPLIT_DATES)
        print(f"\n  Period Accuracy:")
        print(period_acc_df.to_string(index=False))

        # ── Regime quality ────────────────────────────────────────────
        quality_dict = analyze_regime_quality(full_pred_df, stock_df)
        print(f"\n  Regime Duration (avg days):")
        print(quality_dict["duration_df"].to_string(index=False))
        print(f"\n  Transition Detection : {quality_dict['transition_rate']}% "
              f"({quality_dict['detected']}/{quality_dict['total_transitions']}) "
              f"within ±7 days")

        # ── Returns ───────────────────────────────────────────────────
        period_ret_df, regime_ret_df = analyze_returns_by_regime(
            full_pred_df, stock_df, config.SPLIT_DATES
        )
        print(f"\n  Returns by Period:")
        print(period_ret_df.to_string(index=False))
        print(f"\n  Returns by Detected Regime (Test):")
        print(regime_ret_df[regime_ret_df["period"] == "Test"].to_string(index=False))

        # ── Analysis plots ────────────────────────────────────────────
        plot_analysis(
            stock_name, period_acc_df, quality_dict,
            period_ret_df, regime_ret_df, result_dir
        )
        plot_confusion_matrices(stock_name, full_pred_df, config.SPLIT_DATES, result_dir)

        # ── Save CSVs ─────────────────────────────────────────────────
        full_pred_df.to_csv(result_dir / "full_predictions.csv", index=False)
        period_acc_df.to_csv(result_dir / "period_accuracy.csv", index=False)
        quality_dict["duration_df"].to_csv(result_dir / "regime_duration.csv", index=False)
        quality_dict["true_runs_df"].to_csv(result_dir / "true_regime_runs.csv", index=False)
        quality_dict["pred_runs_df"].to_csv(result_dir / "pred_regime_runs.csv", index=False)
        period_ret_df.to_csv(result_dir / "period_returns.csv", index=False)
        regime_ret_df.to_csv(result_dir / "regime_returns.csv", index=False)

        print(f"\n  Saved to results/{stock_name}/")


def run_backtests(all_results, all_stock_files):
    """Backtest every trained stock over the test period, then a cross-stock summary."""
    print("=" * 60)
    print("  BACKTESTING — TEST PERIOD")
    print("=" * 60)

    all_backtest = {}

    for stock_name, res in all_results.items():
        print(f"\nRunning backtest : {stock_name}")
        result_dir = Path(config.RESULTS_PATH) / \
            f"{stock_name}_{config.LABEL_SOURCE.upper()}_Run{config.RUN_ID}"

        bt = run_backtest(stock_name, res["full_pred_df"], all_stock_files)
        if bt is None:
            continue

        plot_backtest(bt, result_dir)
        bt["metrics"].to_csv(result_dir / "backtest_metrics.csv", index=False)
        all_backtest[stock_name] = bt

    # ── Cross stock summary ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  CROSS-STOCK BACKTEST SUMMARY")
    print(f"{'='*65}")

    summary_rows = []
    for stock_name, bt in all_backtest.items():
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

    bt_summary.to_csv(Path(config.RESULTS_PATH) / "backtest_summary.csv", index=False)
    print(f"\nSaved to results/backtest_summary.csv")

    return all_backtest


def run_latest_signal(all_results, all_stock_files, macro_df, nifty_df, device):
    """Latest detected-regime signal (config.TARGET_DATE) for every trained stock."""
    print("=" * 60)
    print(f"  LATEST REGIME SIGNAL — Week of {config.TARGET_DATE}")
    print(f"  Label Source : {config.LABEL_SOURCE.upper()}")
    print("=" * 60)

    latest_predictions = []

    for stock_name in all_results.keys():
        result = predict_specific_week(
            stock_name, config.TARGET_DATE, all_stock_files, macro_df, nifty_df, device
        )
        if result is None:
            continue

        latest_predictions.append(result)
        result_dir = Path(config.RESULTS_PATH) / \
            f"{stock_name}_{config.LABEL_SOURCE.upper()}_Run{config.RUN_ID}"

        print(f"\n  {stock_name}")
        print(f"    History     : {result['history_from']} to {result['history_to']}")
        print(f"    Current     : {result['current_regime']}")
        print(f"    Detected    : {result['predicted_regime']}")
        if result["transition"]:
            print(f"    ⚠️  TRANSITION DETECTED")
        else:
            print(f"    ✓  Continuation")
        print(f"    P(Bull/Neut/Bear) : "
              f"{result['p_bullish']} / {result['p_neutral']} / {result['p_bearish']}")
        print(f"    Confidence  : {result['confidence']}")

        plot_prediction_context(stock_name, result, result_dir, all_stock_files)

    # ── Summary table ─────────────────────────────────────────────────
    if latest_predictions:
        latest_df = pd.DataFrame(latest_predictions)

        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(latest_df[[
            "stock", "current_regime", "predicted_regime",
            "transition", "confidence"
        ]].to_string(index=False))

        transitions = latest_df[latest_df["transition"] == True]
        print(f"\n  Transition alerts : {len(transitions)}/{len(latest_df)} stocks")
        for _, row in transitions.iterrows():
            print(f"    ⚠️  {row['stock']:15s} "
                  f"{row['current_regime']} → {row['predicted_regime']} "
                  f"(conf={row['confidence']})")

        # NOTE: the original notebook has this save commented out — kept
        # that way here too, since the summary CSV built at the end
        # (build_cross_stock_summary) is what's actually used downstream.
        # latest_df.to_csv(
        #     Path(config.RESULTS_PATH) /
        #     f"nov2025_predictions_{config.LABEL_SOURCE.upper()}_Run{config.RUN_ID}.csv",
        #     index=False,
        # )


def main():
    device, all_stock_files, macro_df, nifty_df = setup()

    all_results, all_importance = run_training(all_stock_files, macro_df, nifty_df, device)
    if not all_results:
        print("No stocks were trained successfully — stopping.")
        return

    run_full_history_charts(all_results, all_stock_files, device)
    run_analysis(all_results, all_stock_files, device)
    all_backtest = run_backtests(all_results, all_stock_files)
    run_latest_signal(all_results, all_stock_files, macro_df, nifty_df, device)

    summary_df = build_cross_stock_summary(all_results, all_backtest, all_importance)
    print_cross_stock_summary(summary_df, all_importance)
    plot_cross_stock_summary(summary_df)


if __name__ == "__main__":
    main()
