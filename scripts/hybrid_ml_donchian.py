#!/usr/bin/env python3
"""
Hybrid strategy: ML + Donchian baseline.

Approach:
1. Donchian(40/20) provides base signal (proven Sharpe 1.243)
2. ML model acts as filter (only take trades when ML agrees)
3. ML model adjusts position sizing (high confidence = full size, low = reduced)

Expected: ML improves Donchian by filtering false breakouts
"""

import sys

sys.path.insert(0, "src")

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from sparky.backtest.costs import TransactionCostModel
from sparky.data.loader import load


def load_data():
    """Load features and target via holdout-enforced loader."""
    features = load("feature_matrix_btc_hourly", purpose="training")
    target_df = load("targets_btc_hourly_1d", purpose="training")

    if isinstance(target_df, pd.DataFrame):
        target = target_df["target"]
    else:
        target = target_df

    return features, target


def compute_donchian_signals(prices, window_high=40, window_low=20):
    """Compute Donchian channel breakout signals.

    Long: price breaks above 40-day high
    Short: price breaks below 20-day low
    """
    signals = pd.Series(0, index=prices.index)

    rolling_high = prices.rolling(window_high, min_periods=1).max()
    rolling_low = prices.rolling(window_low, min_periods=1).min()

    # Long breakout
    signals[prices >= rolling_high] = 1

    # No short positions (crypto long-only)

    return signals


def calculate_sharpe(signals, prices, cost_model):
    """Calculate Sharpe with transaction costs."""
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

    sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(365)
    return sharpe


def main():
    print("=== Hybrid ML + Donchian Strategy ===\n")

    features, target = load_data()
    print(f"Features: {features.shape}, Target: {target.shape}\n")

    # In-sample only
    features_in = features.loc[:"2024-05-31"]
    target_in = target.loc[:"2024-05-31"]

    # Load prices
    prices_hourly = load("ohlcv_hourly_max_coverage", purpose="training")
    prices_daily = prices_hourly.resample("D").last()["close"]

    cost_model = TransactionCostModel.for_btc()

    # Best CatBoost config selected in Stage 1 screening from the prior sweep.
    # Validated here on walk-forward 2020-2023 as secondary exploration only.
    # This is a legacy exploration script; sweep_two_stage.py is the canonical tool.
    ml_params = {
        "iterations": 200,
        "depth": 4,
        "learning_rate": 0.03,
        "l2_leaf_reg": 1.0,
        "task_type": "GPU",
        "verbose": 0,
        "random_state": 42,
    }

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

        # Train ML model
        print("Training ML model...")
        ml_model = CatBoostClassifier(**ml_params)
        ml_model.fit(X_train, y_train)

        # Get ML predictions
        ml_proba = ml_model.predict_proba(X_test)[:, 1]
        ml_proba = pd.Series(ml_proba, index=y_test.index)

        # Get Donchian signals
        prices_year = prices_daily.loc[:test_end]  # Need history for rolling
        donchian_signals = compute_donchian_signals(prices_year, window_high=40, window_low=20)
        donchian_signals = donchian_signals.loc[test_start:test_end]

        # Ensure alignment
        common_idx = ml_proba.index.intersection(donchian_signals.index)
        ml_proba = ml_proba.loc[common_idx]
        donchian_signals = donchian_signals.loc[common_idx]

        # --- Strategy 1: ML Filter (only trade when ML > 0.6) ---
        ml_filter_signals = donchian_signals.copy()
        ml_filter_signals[ml_proba < 0.6] = 0

        prices_test = prices_daily.loc[test_start:test_end]
        sharpe_filter = calculate_sharpe(ml_filter_signals, prices_test, cost_model)

        # --- Strategy 2: ML Position Sizing ---
        # Scale position by ML confidence: proba=0.5 → 0%, proba=1.0 → 100%
        ml_position_size = ((ml_proba - 0.5) / 0.5).clip(0, 1)
        ml_sized_signals = donchian_signals * ml_position_size

        sharpe_sized = calculate_sharpe(ml_sized_signals, prices_test, cost_model)

        # --- Baseline: Pure Donchian ---
        sharpe_donchian = calculate_sharpe(donchian_signals, prices_test, cost_model)

        print(f"Donchian only: Sharpe={sharpe_donchian:.3f}")
        print(f"ML Filter (>0.6): Sharpe={sharpe_filter:.3f}")
        print(f"ML Position Sizing: Sharpe={sharpe_sized:.3f}")

        results.append(
            {
                "year": year,
                "sharpe_donchian": sharpe_donchian,
                "sharpe_ml_filter": sharpe_filter,
                "sharpe_ml_sized": sharpe_sized,
            }
        )

    # Overall metrics
    print("\n=== Overall Results ===")
    sharpe_donchian_mean = np.mean([r["sharpe_donchian"] for r in results])
    sharpe_filter_mean = np.mean([r["sharpe_ml_filter"] for r in results])
    sharpe_sized_mean = np.mean([r["sharpe_ml_sized"] for r in results])

    print(f"Donchian baseline: {sharpe_donchian_mean:.3f}")
    print(f"ML Filter: {sharpe_filter_mean:.3f}")
    print(f"ML Position Sizing: {sharpe_sized_mean:.3f}")
    print("\nTarget: 1.062 (Multi-TF Donchian)")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "donchian_params": {"window_high": 40, "window_low": 20},
        "ml_model": "CatBoost",
        "ml_params": ml_params,
        "yearly_results": results,
        "mean_sharpe_donchian": sharpe_donchian_mean,
        "mean_sharpe_ml_filter": sharpe_filter_mean,
        "mean_sharpe_ml_sized": sharpe_sized_mean,
        "baseline_sharpe": 1.062,
    }

    outpath = Path("results/validation/hybrid_ml_donchian_58features.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
