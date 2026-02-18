#!/usr/bin/env python3
"""
Stacking ensemble combining top-performing configs from sweep.

Approach:
1. Train base models (Level 0) on training data
2. Generate out-of-fold predictions for meta-learner training
3. Train meta-learner (Level 1) on base model predictions
4. Final predictions: meta-learner combines base model outputs

Expected: Meta-learner learns optimal combination weights
"""

import sys

sys.path.insert(0, "src")

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

from sparky.backtest.costs import TransactionCostModel

# Top 5 base models from sweep
BASE_CONFIGS = [
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


def train_base_models(X_train, y_train):
    """Train all base models."""
    models = []

    for i, (name, params) in enumerate(BASE_CONFIGS):
        print(f"  Training base model {i + 1}/{len(BASE_CONFIGS)}: {name}")

        if name == "CatBoost":
            model = CatBoostClassifier(**params, verbose=0, random_state=42)
        else:
            model = LGBMClassifier(**params, verbose=-1, random_state=42)

        model.fit(X_train, y_train)
        models.append(model)

    return models


def generate_oof_predictions(X_train, y_train, n_folds=5):
    """Generate out-of-fold predictions for meta-learner training."""
    kf = KFold(n_splits=n_folds, shuffle=False)  # Time series, no shuffle

    oof_preds = np.zeros((len(X_train), len(BASE_CONFIGS)))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold_idx + 1}/{n_folds}")

        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        for model_idx, (name, params) in enumerate(BASE_CONFIGS):
            if name == "CatBoost":
                model = CatBoostClassifier(**params, verbose=0, random_state=42)
            else:
                model = LGBMClassifier(**params, verbose=-1, random_state=42)

            model.fit(X_fold_train, y_fold_train)
            oof_preds[val_idx, model_idx] = model.predict_proba(X_fold_val)[:, 1]

    return oof_preds


def calculate_sharpe(signals, prices, cost_model):
    """Calculate Sharpe ratio with transaction costs."""
    common_idx = signals.index.intersection(prices.index)
    signals = signals.loc[common_idx]
    prices = prices.loc[common_idx]

    positions = signals.shift(1).fillna(0)
    price_returns = prices.pct_change().fillna(0)

    position_changes = positions.diff().abs()
    costs = position_changes * cost_model.round_trip_cost

    strategy_returns = (positions * price_returns) - costs

    if strategy_returns.std() == 0:
        return 0.0

    sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    return sharpe


def main():
    print("=== Stacking Ensemble Validation ===\n")

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

        # Step 1: Generate OOF predictions for meta-learner
        print("Generating OOF predictions for meta-learner...")
        oof_preds = generate_oof_predictions(X_train, y_train, n_folds=5)

        # Step 2: Train meta-learner on OOF predictions
        print("Training meta-learner...")
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        meta_learner.fit(oof_preds, y_train)

        print(f"Meta-learner weights: {meta_learner.coef_[0]}")

        # Step 3: Train base models on full training data
        print("Training base models on full training data...")
        base_models = train_base_models(X_train, y_train)

        # Step 4: Generate test predictions from base models
        test_preds = np.zeros((len(X_test), len(BASE_CONFIGS)))
        for i, model in enumerate(base_models):
            test_preds[:, i] = model.predict_proba(X_test)[:, 1]

        # Step 5: Meta-learner final predictions
        ensemble_proba = meta_learner.predict_proba(test_preds)[:, 1]

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
        "n_base_models": len(BASE_CONFIGS),
        "meta_learner": "LogisticRegression",
        "n_folds": 5,
        "yearly_results": results,
        "mean_sharpe": sharpe_mean,
        "mean_accuracy": acc_mean,
        "baseline_sharpe": 1.062,
    }

    outpath = Path("results/validation/ensemble_stacking_58features.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
