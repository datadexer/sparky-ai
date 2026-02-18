#!/usr/bin/env python3
"""
Ensemble strategy combining top-performing configs from sweep.

Approach:
1. Train top-N configs (by Sharpe) on full training data
2. Combine predictions via:
   - Weighted average (by validation Sharpe)
   - Voting (majority vote)
   - Stacking (meta-learner on predictions)
3. Validate ensemble on yearly walk-forward

Expected: Ensemble reduces overfitting, improves generalization
"""

import sys

sys.path.insert(0, "src")

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from sparky.backtest.costs import TransactionCostModel

# Top 5 configs from sweep (update after sweep completes)
TOP_CONFIGS = [
    ("CatBoost", {"iterations": 200, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 1.0, "task_type": "GPU"}),
    ("CatBoost", {"iterations": 200, "depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 3.0, "task_type": "GPU"}),
    ("CatBoost", {"iterations": 200, "depth": 4, "learning_rate": 0.01, "l2_leaf_reg": 1.0, "task_type": "GPU"}),
    ("LightGBM", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03, "reg_lambda": 1.0, "device": "gpu"}),
    ("LightGBM", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03, "reg_lambda": 3.0, "device": "gpu"}),
]


def load_data():
    """Load 58-feature dataset."""
    features = pd.read_parquet("data/processed/feature_matrix_btc_hourly.parquet")
    target = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")

    if isinstance(target, pd.DataFrame):
        target = target["target"]

    return features, target


def train_ensemble(X_train, y_train):
    """Train all models in ensemble."""
    models = []

    for i, (name, params) in enumerate(TOP_CONFIGS):
        print(f"Training model {i + 1}/{len(TOP_CONFIGS)}: {name}")

        if name == "CatBoost":
            model = CatBoostClassifier(**params, verbose=0, random_state=42)
        else:
            model = LGBMClassifier(**params, verbose=-1, random_state=42)

        model.fit(X_train, y_train)
        models.append(model)

    return models


def ensemble_predict(models, X_test, method="weighted"):
    """Combine predictions from multiple models."""
    all_proba = []

    for model in models:
        proba = model.predict_proba(X_test)[:, 1]
        all_proba.append(proba)

    all_proba = np.array(all_proba)

    if method == "weighted":
        # Simple average (can weight by validation Sharpe later)
        ensemble_proba = all_proba.mean(axis=0)
    elif method == "voting":
        # Majority vote
        ensemble_proba = (all_proba > 0.5).sum(axis=0) / len(models)
    else:
        raise ValueError(f"Unknown method: {method}")

    return ensemble_proba


def calculate_sharpe(signals, prices, cost_model):
    """Calculate Sharpe ratio with transaction costs."""
    # Ensure alignment
    common_idx = signals.index.intersection(prices.index)
    signals = signals.loc[common_idx]
    prices = prices.loc[common_idx]

    # Positions: shift signals to avoid look-ahead
    positions = signals.shift(1).fillna(0)

    # Returns
    price_returns = prices.pct_change().fillna(0)

    # Apply costs on position changes
    position_changes = positions.diff().abs()
    costs = position_changes * cost_model.round_trip_cost

    # Strategy returns
    strategy_returns = (positions * price_returns) - costs

    # Sharpe
    if strategy_returns.std() == 0:
        return 0.0

    sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(365)
    return sharpe


def main():
    print("=== Ensemble Strategy Validation ===\n")

    features, target = load_data()
    print(f"Features: {features.shape}, Target: {target.shape}\n")

    # In-sample only
    features_in = features.loc[:"2024-05-31"]
    target_in = target.loc[:"2024-05-31"]

    # Load prices
    prices_hourly = pd.read_parquet("data/raw/btc/ohlcv_hourly_max_coverage.parquet")
    prices_daily = prices_hourly.resample("D").last()["close"]

    cost_model = TransactionCostModel.for_btc()

    # Yearly walk-forward
    years = [2020, 2021, 2022, 2023]
    results = []

    for year in years:
        print(f"\n--- Year {year} ---")

        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        X_train = features_in.loc[:train_end]
        y_train = target_in.loc[:train_end]
        X_test = features_in.loc[test_start:test_end]
        y_test = target_in.loc[test_start:test_end]

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        # Train ensemble
        models = train_ensemble(X_train, y_train)

        # Predict
        ensemble_proba = ensemble_predict(models, X_test, method="weighted")

        # Metrics
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        acc = accuracy_score(y_test, ensemble_pred)
        auc = roc_auc_score(y_test, ensemble_proba) if len(np.unique(y_test)) > 1 else 0.5

        # Trading simulation
        signals = (ensemble_proba > 0.52).astype(int)
        signals = pd.Series(signals, index=y_test.index)

        prices_year = prices_daily.loc[test_start:test_end]
        sharpe = calculate_sharpe(signals, prices_year, cost_model)

        print(f"Acc={acc:.3f}, AUC={auc:.3f}, Sharpe={sharpe:.3f}")

        results.append({"year": year, "accuracy": acc, "auc": auc, "sharpe": sharpe})

    # Overall metrics
    print("\n=== Overall Results ===")
    sharpe_mean = np.mean([r["sharpe"] for r in results])
    acc_mean = np.mean([r["accuracy"] for r in results])
    print(f"Mean Sharpe: {sharpe_mean:.3f}")
    print(f"Mean Accuracy: {acc_mean:.3f}")
    print("Baseline to beat: 1.062 (Multi-TF Donchian)")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_models": len(TOP_CONFIGS),
        "method": "weighted_average",
        "yearly_results": results,
        "mean_sharpe": sharpe_mean,
        "mean_accuracy": acc_mean,
        "baseline_sharpe": 1.062,
    }

    outpath = Path("results/validation/ensemble_weighted_58features.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
