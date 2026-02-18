#!/usr/bin/env python3
"""
Hyperparameter sweep with 58 expanded features.

Tests CatBoost and LightGBM on the expanded feature set:
- 23 original features
- 35 new features (microstructure, regime, divergences, volume-price)

Total: 58 features on 4,795 daily samples from 115K hourly candles
"""

import sys

sys.path.insert(0, "src")

# Force unbuffered output
import os

os.environ["PYTHONUNBUFFERED"] = "1"

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from sparky.backtest.costs import TransactionCostModel


def load_data():
    """Load 58-feature hourly dataset."""
    print("Loading 58-feature dataset...", flush=True)

    features = pd.read_parquet("data/processed/feature_matrix_btc_hourly.parquet")
    target = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")

    # Target is a DataFrame with 'target' column, extract as Series
    if isinstance(target, pd.DataFrame):
        target = target["target"]

    print(f"Features: {features.shape[0]} samples, {features.shape[1]} features", flush=True)
    print(f"Targets: {target.shape[0]} samples", flush=True)
    print(f"Date range: {features.index.min()} to {features.index.max()}", flush=True)
    print(f"Target balance: {target.value_counts(normalize=True).to_dict()}", flush=True)

    return features, target


def validate_config(features, target, model_name, params):
    """Yearly walk-forward validation (2020-2023).

    Note: Data goes back to 2012, so 2020 has 8 years of training data.
    """
    # Filter to in-sample period only (up to 2024-06-01 embargo boundary)
    features_in = features.loc[:"2024-05-31"]
    target_in = target.loc[:"2024-05-31"]

    years = [2020, 2021, 2022, 2023]
    results = []

    for year in years:
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        X_train = features_in.loc[:train_end]
        y_train = target_in.loc[:train_end]
        X_test = features_in.loc[test_start:test_end]
        y_test = target_in.loc[test_start:test_end]

        print(f"    Year {year}: Train={len(X_train)}, Test={len(X_test)}")

        if len(X_test) == 0 or len(X_train) == 0:
            print(f"    Year {year}: Skipping (no data)")
            continue

        # Train
        if model_name == "CatBoost":
            model = CatBoostClassifier(**params, verbose=0, random_state=42)
        else:  # LightGBM
            model = LGBMClassifier(**params, verbose=-1, random_state=42)

        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5

        # Trading simulation
        signals = (y_proba > 0.52).astype(int)  # 52% threshold (slight edge needed)
        signals = pd.Series(signals, index=y_test.index)

        # Load prices for this year
        prices_hourly = pd.read_parquet("data/raw/btc/ohlcv_hourly_max_coverage.parquet")
        prices_daily = prices_hourly.resample("D").last()
        prices_test = prices_daily.loc[test_start:test_end]

        if len(prices_test) == 0:
            print(f"    Year {year}: No price data, skipping trading sim")
            continue

        # Align signals with prices
        common_idx = signals.index.intersection(prices_test.index)
        signals = signals.loc[common_idx]
        prices_test = prices_test.loc[common_idx]

        returns = prices_test["close"].pct_change()
        strategy_returns = signals.shift(1) * returns  # Signal at T trades at T+1

        # Apply transaction costs
        position_changes = signals.diff().abs()
        cost_model = TransactionCostModel.for_btc()
        costs = position_changes * cost_model.total_cost_pct
        net_returns = strategy_returns - costs

        # Compute Sharpe
        if len(net_returns.dropna()) > 0 and net_returns.std() > 0:
            sharpe = net_returns.mean() / net_returns.std() * np.sqrt(365)
        else:
            sharpe = 0.0

        results.append(
            {
                "year": year,
                "accuracy": acc,
                "auc": auc,
                "sharpe": sharpe,
                "test_samples": len(X_test),
            }
        )

        print(f"    Year {year}: Acc={acc:.3f}, AUC={auc:.3f}, Sharpe={sharpe:.3f}")

    # Aggregate
    if len(results) == 0:
        return {
            "mean_sharpe": 0.0,
            "mean_accuracy": 0.5,
            "mean_auc": 0.5,
            "yearly_results": [],
        }

    mean_sharpe = np.mean([r["sharpe"] for r in results])
    mean_acc = np.mean([r["accuracy"] for r in results])
    mean_auc = np.mean([r["auc"] for r in results])

    return {
        "mean_sharpe": mean_sharpe,
        "mean_accuracy": mean_acc,
        "mean_auc": mean_auc,
        "yearly_results": results,
    }


def run_sweep():
    """Run focused hyperparameter sweep."""
    features, target = load_data()

    # Test configs (27 CatBoost + 27 LightGBM = 54 total)
    configs = []

    # CatBoost configs
    for depth in [3, 4, 5]:
        for lr in [0.01, 0.03, 0.05]:
            for l2 in [1.0, 3.0, 5.0]:
                configs.append(
                    {
                        "model": "CatBoost",
                        "params": {
                            "iterations": 200,
                            "depth": depth,
                            "learning_rate": lr,
                            "l2_leaf_reg": l2,
                            "task_type": "GPU",
                            "devices": "0",
                        },
                    }
                )

    # LightGBM configs
    for depth in [3, 4, 5]:
        for lr in [0.01, 0.03, 0.05]:
            for l1 in [0.0, 0.5, 1.0]:
                configs.append(
                    {
                        "model": "LightGBM",
                        "params": {
                            "n_estimators": 200,
                            "max_depth": depth,
                            "learning_rate": lr,
                            "reg_lambda": 1.0,
                            "reg_alpha": l1,
                            "device": "gpu",
                            "gpu_platform_id": 0,
                            "gpu_device_id": 0,
                        },
                    }
                )

    print(f"\nRunning sweep: {len(configs)} configurations")
    print("=" * 80)

    all_results = []

    for i, config in enumerate(configs, 1):
        print(f"\nConfig {i}/{len(configs)}: {config['model']} - {config['params']}", flush=True)

        try:
            result = validate_config(features, target, config["model"], config["params"])
            result["config"] = config
            all_results.append(result)

            print(f"  RESULT: Sharpe={result['mean_sharpe']:.3f}, Acc={result['mean_accuracy']:.3f}", flush=True)

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append(
                {
                    "config": config,
                    "error": str(e),
                    "mean_sharpe": 0.0,
                }
            )

    # Save results
    output_path = Path("results/validation/sweep_58_features.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"SWEEP COMPLETE - {len(all_results)} configs tested")
    print(f"Results saved to {output_path}")

    # Report top 10
    valid_results = [r for r in all_results if "error" not in r]
    valid_results.sort(key=lambda x: x["mean_sharpe"], reverse=True)

    print("\nTOP 10 CONFIGS BY SHARPE:")
    for i, r in enumerate(valid_results[:10], 1):
        model = r["config"]["model"]
        params = r["config"]["params"]
        print(f"{i}. {model} - Sharpe={r['mean_sharpe']:.3f}, Acc={r['mean_accuracy']:.3f}")
        print(f"   Params: {params}")

    # Compare to baseline
    baseline_sharpe = 1.062  # Multi-TF Donchian (corrected)
    best_sharpe = valid_results[0]["mean_sharpe"] if valid_results else 0.0

    print("\nBASELINE COMPARISON:")
    print(f"  Baseline (Donchian): {baseline_sharpe:.3f}")
    print(f"  Best ML: {best_sharpe:.3f}")
    print(f"  Improvement: {best_sharpe - baseline_sharpe:+.3f} ({100 * (best_sharpe / baseline_sharpe - 1):+.1f}%)")

    if best_sharpe > baseline_sharpe:
        print("\n✅ ML BEATS BASELINE - Continue with feature ablation + ensemble")
    elif best_sharpe > 0.7:
        print("\n⚠️  ML shows promise but below baseline - Try different models or ensembles")
    else:
        print("\n❌ ML does not beat baseline - Consider alternative approaches")


if __name__ == "__main__":
    run_sweep()
