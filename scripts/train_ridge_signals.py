#!/usr/bin/env python3
"""
Ridge regression for continuous signal strength (not binary classification).
Hypothesis: Classification threshold (0.5) loses information. Use regression → threshold at 0.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sparky.oversight.time_tracker import TaskTimer

SELECTED_FEATURES = [
    "volume_surge_4h",
    "tick_direction_ratio_24h",
    "rsi_divergence_14h_168h",
    "momentum_24h",
    "volatility_regime",
    "breakout_proximity_upper",
    "intraday_range",
    "vwap_deviation_24h",
    "macd_histogram",
    "intraday_momentum_reversal",
    "higher_highs_lower_lows_5h",
    "high_low_ratio_20h",
    "distance_from_sma_200h",
    "obv",
    "price_acceleration_10h",
    "momentum_168h",
    "mfi_14h",
    "momentum_divergence_24h_168h",
    "rsi_6h",
    "volume_momentum_30h",
]


def walk_forward_split(index, train_min=1095, test_len=365, step=365):
    folds = []
    for i in range(0, len(index), step):
        test_start = train_min + i
        test_end = test_start + test_len
        if test_end > len(index):
            break
        train_idx = index[:test_start]
        test_idx = index[test_start:test_end]
        if len(train_idx) >= train_min and len(test_idx) == test_len:
            folds.append((train_idx, test_idx))
    return folds


def main():
    timer = TaskTimer(agent_id="research")
    timer.start("train_ridge_signals")

    print("=" * 80)
    print("RIDGE REGRESSION: Continuous signal strength")
    print(f"Started: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 80)

    features = pd.read_parquet("data/processed/feature_matrix_btc_hourly_clean.parquet")
    targets = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")

    features_daily = features.resample("D").last().dropna()
    targets_daily = targets.resample("D").last().dropna()

    common = features_daily.index.intersection(targets_daily.index)
    X = features_daily.loc[common][SELECTED_FEATURES]
    y_binary = targets_daily.loc[common]["target"]

    # Convert target to continuous (-1 to +1)
    y_continuous = 2 * y_binary - 1

    print(f"\nSamples: {len(X)}, Features: {X.shape[1]}")
    print(f"Target range: [{y_continuous.min():.1f}, {y_continuous.max():.1f}]")

    print("\nWalk-forward validation...")
    accs = []
    mses = []

    for train_dates, test_dates in walk_forward_split(X.index):
        model = Ridge(alpha=1.0)
        model.fit(X.loc[train_dates], y_continuous.loc[train_dates])

        y_pred_continuous = model.predict(X.loc[test_dates])
        y_pred_binary = (y_pred_continuous > 0).astype(int)

        acc = (y_pred_binary == y_binary.loc[test_dates]).mean()
        mse = mean_squared_error(y_continuous.loc[test_dates], y_pred_continuous)

        print(f"  {test_dates[0].year}: Acc={acc:.3f} MSE={mse:.3f}")
        accs.append(acc)
        mses.append(mse)

    mean_acc = np.mean(accs)

    print(f"\n{'=' * 80}")
    print(f"Ridge: Mean Acc={mean_acc:.3f}, MSE={np.mean(mses):.3f}")
    print("CatBoost: Acc=0.530")
    print("Donchian: Sharpe=1.062")

    result = {"mean_acc": mean_acc, "mean_mse": np.mean(mses), "yearly_accs": accs}

    with open("results/validation/ridge_signals.json", "w") as f:
        json.dump(result, f, indent=2)

    if mean_acc > 0.530:
        print(f"\n✓ Ridge improves over CatBoost by {((mean_acc / 0.530 - 1) * 100):.1f}%")
    else:
        print(f"\n✗ Ridge underperforms CatBoost by {((1 - mean_acc / 0.530) * 100):.1f}%")

    timer.end(claimed_duration_minutes=5)


if __name__ == "__main__":
    main()
