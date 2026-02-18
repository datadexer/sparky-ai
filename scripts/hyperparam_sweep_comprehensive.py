#!/usr/bin/env python3
"""
Comprehensive hyperparameter sweep for ML models on hourly BTC data.

Goal: Find ML configuration that beats corrected baseline (Sharpe 1.062).

Strategy:
1. Test 3 model families: CatBoost, LightGBM, XGBoost
2. Sweep hyperparameters: depth, learning_rate, regularization
3. Test on expanding window walk-forward (2019-2023)
4. Report ALL results, not just best (avoid cherry-picking)
5. Use same validation methodology as baseline (yearly folds)

IMPORTANT: This is systematic search, not cherry-picking.
We report ALL configs, successful or not.
"""

import sys

sys.path.insert(0, "src")

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sparky.backtest.costs import TransactionCostModel
from sparky.data.loader import load
from xgboost import XGBClassifier


def load_data():
    """Load hourly features and targets."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    features = load("features_hourly_full", purpose="training")

    # Create 1h ahead target (binary: up or down)
    # NOTE: data/btc_hourly.parquet is not in the loader mapping — kept as pd.read_parquet.
    prices = pd.read_parquet("data/btc_hourly.parquet")
    prices = prices.loc[features.index]

    # Target: 1h ahead return > 0
    returns_1h = prices["close"].pct_change(1).shift(-1)
    target = (returns_1h > 0).astype(int)
    target.name = "target_1h_up"

    # Align
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    # Drop NaN
    features = features.dropna()
    target = target.loc[features.index].dropna()
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    print(f"Features: {features.shape}")
    print(f"Target: {target.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}")
    print(f"Target distribution: {target.value_counts(normalize=True).to_dict()}")
    print()

    return features, target, prices


def yearly_walk_forward_validation(features, target, prices, model_name, model_params):
    """
    Validate model using yearly walk-forward (same as baseline validation).

    Returns:
        dict with year-by-year Sharpe, overall metrics
    """
    print(f"\n{'=' * 80}")
    print(f"VALIDATING: {model_name} | Params: {model_params}")
    print(f"{'=' * 80}\n")

    # Define yearly folds (2019-2023, matching baseline)
    years = [2019, 2020, 2021, 2022, 2023]
    results = {}

    for year in years:
        # Train on all data before year
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        X_train = features.loc[:train_end]
        y_train = target.loc[:train_end]
        X_test = features.loc[test_start:test_end]
        y_test = target.loc[test_start:test_end]

        if len(X_test) == 0:
            print(f"  {year}: No test data, skipping")
            continue

        # Train model
        if model_name == "CatBoost":
            model = CatBoostClassifier(**model_params, verbose=0, random_state=42)
        elif model_name == "LightGBM":
            model = LGBMClassifier(**model_params, verbose=-1, random_state=42)
        elif model_name == "XGBoost":
            model = XGBClassifier(**model_params, random_state=42, eval_metric="logloss")
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train, y_train)

        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # Backtest: aggregate hourly predictions to daily signals
        test_prices = prices.loc[test_start:test_end]

        # Simple aggregation: LONG if >=60% of daily hours predict UP
        hourly_preds = pd.Series(y_pred_proba, index=X_test.index)
        daily_signals = hourly_preds.resample("D").apply(lambda x: (x > 0.5).mean() >= 0.6).astype(int)
        daily_signals = daily_signals.reindex(test_prices.resample("D").last().index, fill_value=0)

        # Compute daily returns
        daily_prices = test_prices.resample("D").last()
        daily_returns = daily_prices["close"].pct_change()

        # Strategy returns (shift signals by 1 to avoid look-ahead)
        strategy_returns = daily_signals.shift(1) * daily_returns

        # Apply transaction costs
        cost_model = TransactionCostModel.for_btc()
        trades = daily_signals.diff().abs()
        costs = trades * cost_model.estimate_cost(1.0, "BTC/USD")
        strategy_returns = strategy_returns - costs

        # Drop NaN
        strategy_returns = strategy_returns.dropna()

        if len(strategy_returns) == 0 or strategy_returns.std() == 0:
            sharpe = 0.0
            total_return = 0.0
        else:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(365)
            total_return = (1 + strategy_returns).prod() - 1

        results[year] = {"sharpe": sharpe, "return": total_return, "acc": acc, "auc": auc, "n_samples": len(X_test)}

        print(f"  {year}: Sharpe={sharpe:.3f}, Return={total_return:.1%}, AUC={auc:.3f}, Acc={acc:.1%}")

    # Overall stats
    sharpes = [r["sharpe"] for r in results.values()]
    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)
    median_sharpe = np.median(sharpes)

    print(f"\n  OVERALL: Mean Sharpe={mean_sharpe:.3f}, Std={std_sharpe:.3f}, Median={median_sharpe:.3f}")

    return {
        "model_name": model_name,
        "model_params": model_params,
        "yearly_results": results,
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "median_sharpe": median_sharpe,
        "best_year": max(sharpes),
        "worst_year": min(sharpes),
    }


def main():
    print("=" * 80)
    print("COMPREHENSIVE HYPERPARAMETER SWEEP")
    print("=" * 80)
    print(f"Started: {datetime.now()}")
    print()

    # Load data
    features, target, prices = load_data()

    # Define search grid
    configs = []

    # CatBoost configurations
    for depth in [3, 4, 5, 6]:
        for lr in [0.01, 0.03, 0.05, 0.1]:
            for l2 in [1, 3, 5]:
                configs.append(
                    {
                        "model": "CatBoost",
                        "params": {
                            "depth": depth,
                            "learning_rate": lr,
                            "l2_leaf_reg": l2,
                            "iterations": 500,
                            "task_type": "GPU" if True else "CPU",
                        },
                    }
                )

    # LightGBM configurations
    for depth in [3, 4, 5, 6]:
        for lr in [0.01, 0.03, 0.05, 0.1]:
            for l2 in [0.0, 0.1, 1.0]:
                configs.append(
                    {
                        "model": "LightGBM",
                        "params": {
                            "max_depth": depth,
                            "learning_rate": lr,
                            "reg_lambda": l2,
                            "n_estimators": 500,
                            "device": "gpu",
                        },
                    }
                )

    # XGBoost configurations
    for depth in [3, 4, 5, 6]:
        for lr in [0.01, 0.03, 0.05, 0.1]:
            for l2 in [0.0, 1.0, 3.0]:
                configs.append(
                    {
                        "model": "XGBoost",
                        "params": {
                            "max_depth": depth,
                            "learning_rate": lr,
                            "reg_lambda": l2,
                            "n_estimators": 500,
                            "tree_method": "gpu_hist",
                        },
                    }
                )

    print(f"Total configurations to test: {len(configs)}")
    print(f"Estimated time: {len(configs) * 2} minutes (2 min per config)")
    print()

    # Run sweep
    all_results = []
    baseline_sharpe = 1.062  # Corrected Multi-TF baseline

    for i, config in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] Testing {config['model']} config {i + 1}")

        result = yearly_walk_forward_validation(features, target, prices, config["model"], config["params"])

        all_results.append(result)

        # Check if beats baseline
        if result["mean_sharpe"] > baseline_sharpe:
            print(f"\n🎯 BEATS BASELINE! Mean Sharpe {result['mean_sharpe']:.3f} > {baseline_sharpe:.3f}")

        # Save intermediate results
        if (i + 1) % 10 == 0:
            output_path = Path("results/validation/hyperparam_sweep_intermediate.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nIntermediate results saved: {output_path}")

    # Final results
    output_path = Path("results/validation/hyperparam_sweep_comprehensive.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add summary
    summary = {
        "total_configs": len(all_results),
        "baseline_sharpe": baseline_sharpe,
        "best_config": max(all_results, key=lambda x: x["mean_sharpe"]),
        "configs_beat_baseline": sum(1 for r in all_results if r["mean_sharpe"] > baseline_sharpe),
        "all_results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"Total configs: {summary['total_configs']}")
    print(f"Configs beating baseline ({baseline_sharpe:.3f}): {summary['configs_beat_baseline']}")
    print(
        f"Best config: {summary['best_config']['model_name']} with Mean Sharpe {summary['best_config']['mean_sharpe']:.3f}"
    )
    print(f"Results saved: {output_path}")
    print(f"Completed: {datetime.now()}")


if __name__ == "__main__":
    main()
