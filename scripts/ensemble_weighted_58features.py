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
from sparky.data.loader import load
from sparky.tracking.guardrails import has_blocking_failure, log_results, run_post_checks, run_pre_checks
from sparky.tracking.metrics import compute_all_metrics

# Top 5 configs from sweep (selected in Stage 1 screening, evaluated here on
# walk-forward 2020-2023 for secondary validation only — not primary results).
# This is a legacy exploration script; sweep_two_stage.py is the canonical tool.
TOP_CONFIGS = [
    ("CatBoost", {"iterations": 200, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 1.0, "task_type": "GPU"}),
    ("CatBoost", {"iterations": 200, "depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 3.0, "task_type": "GPU"}),
    ("CatBoost", {"iterations": 200, "depth": 4, "learning_rate": 0.01, "l2_leaf_reg": 1.0, "task_type": "GPU"}),
    ("LightGBM", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03, "reg_lambda": 1.0, "device": "gpu"}),
    ("LightGBM", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03, "reg_lambda": 3.0, "device": "gpu"}),
]


def load_data():
    """Load 58-feature dataset via holdout-enforced loader."""
    features = load("feature_matrix_btc_hourly", purpose="training")
    target_df = load("targets_btc_hourly_1d", purpose="training")

    if isinstance(target_df, pd.DataFrame):
        target = target_df["target"]
    else:
        target = target_df

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
    prices_hourly = load("ohlcv_hourly_max_coverage", purpose="training")
    prices_daily = prices_hourly.resample("D").last()["close"]

    cost_model = TransactionCostModel.for_btc()

    # Pre-experiment guardrail checks
    config = {"model": "weighted_ensemble", "cost_bps": 10}
    pre_results = run_pre_checks(features_in, config)
    if has_blocking_failure(pre_results):
        print("PRE-CHECK BLOCKED — aborting.")
        return

    # Yearly walk-forward
    years = [2020, 2021, 2022, 2023]
    results = []
    all_net_returns = []

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

        # Collect net returns for DSR computation
        common_idx = signals.index.intersection(prices_year.index)
        pos = signals.loc[common_idx].shift(1).fillna(0)
        rets = prices_year.loc[common_idx].pct_change().fillna(0)
        pos_chg = pos.diff().abs()
        year_net_returns = (pos * rets) - pos_chg * cost_model.round_trip_cost
        all_net_returns.append(year_net_returns)

        print(f"Acc={acc:.3f}, AUC={auc:.3f}, Sharpe={sharpe:.3f}")

        results.append({"year": year, "accuracy": acc, "auc": auc, "sharpe": sharpe})

    # Overall metrics
    print("\n=== Overall Results ===")
    sharpe_mean = np.mean([r["sharpe"] for r in results])
    acc_mean = np.mean([r["accuracy"] for r in results])
    print(f"Mean Sharpe: {sharpe_mean:.3f}")
    print(f"Mean Accuracy: {acc_mean:.3f}")
    print("Baseline to beat: 1.062 (Multi-TF Donchian)")

    # DSR computation with cumulative n_trials (250 prior + this ensemble = 251 configs)
    net_returns_all = pd.concat(all_net_returns) if all_net_returns else pd.Series(dtype=float)
    n_trials_cumulative = 251
    dsr_metrics = (
        compute_all_metrics(net_returns_all.values, n_trials=n_trials_cumulative)
        if len(net_returns_all) > 30
        else {"dsr": 0.0, "psr": 0.0}
    )
    print(f"DSR (n_trials={n_trials_cumulative}): {dsr_metrics['dsr']:.3f}")

    # Post-experiment guardrail checks
    equity_curve = (1 + net_returns_all).cumprod() if len(net_returns_all) > 0 else pd.Series([1.0])
    running_max = equity_curve.cummax()
    actual_dd = float(((equity_curve - running_max) / running_max).min()) if len(equity_curve) > 0 else -1.0
    post_results = run_post_checks(
        net_returns_all.values,
        {"sharpe": sharpe_mean, "max_drawdown": actual_dd},
        config,
        n_trials=n_trials_cumulative,
    )
    log_results(pre_results + post_results, run_id="ensemble_weighted")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_models": len(TOP_CONFIGS),
        "method": "weighted_average",
        "yearly_results": results,
        "mean_sharpe": sharpe_mean,
        "mean_accuracy": acc_mean,
        "baseline_sharpe": 1.062,
        "dsr": dsr_metrics["dsr"],
        "psr": dsr_metrics["psr"],
        "n_trials": n_trials_cumulative,
    }

    outpath = Path("results/validation/ensemble_weighted_58features.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
