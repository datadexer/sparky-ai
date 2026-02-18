#!/usr/bin/env python3
"""
Try 40 features (double from 20) via two-stage selection.
Hypothesis: 20 features might be too few — try 40.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sparky.data.loader import load
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sparky.oversight.time_tracker import TaskTimer


def walk_forward_split(index, train_min=1095, test_len=365, step=365):
    """Generate WF splits."""
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
    timer.start("train_boosted_features")

    print("=" * 80)
    print("BOOSTED FEATURES: 40 features (up from 20)")
    print(f"Started: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 80)

    # Load
    features = load("feature_matrix_btc_hourly", purpose="training")
    targets = load("targets_btc_hourly_1d", purpose="training")

    # Daily resample
    features_daily = features.resample("D").last().dropna()
    targets_daily = targets.resample("D").last().dropna()

    common = features_daily.index.intersection(targets_daily.index)
    X_full = features_daily.loc[common]
    y = targets_daily.loc[common]["target"]

    print(f"\nFull dataset: {len(X_full)} samples, {X_full.shape[1]} features")
    print(f"Date range: {X_full.index[0]} to {X_full.index[-1]}")

    # Feature selection with default XGBoost
    print("\nFeature selection (keeping top 40)...")
    from xgboost import XGBClassifier

    selector = XGBClassifier(n_estimators=100, max_depth=3, tree_method="hist", device="cuda")

    # Use first 80% for selection
    split_idx = int(len(X_full) * 0.8)
    selector.fit(X_full.iloc[:split_idx], y.iloc[:split_idx])

    importances = pd.Series(selector.feature_importances_, index=X_full.columns).sort_values(ascending=False)
    top_40 = importances.head(40).index.tolist()

    print(f"Selected: {top_40[:5]}... (40 total)")

    X = X_full[top_40]

    # Walk-forward validation
    config = {
        "iterations": 200,
        "depth": 4,
        "learning_rate": 0.01,
        "l2_leaf_reg": 1.0,
        "task_type": "GPU",
        "devices": "0",
        "verbose": 0,
    }

    print("\nWalk-forward validation...")
    sharpes, accs = [], []

    # Load prices for backtest
    prices = load("ohlcv_hourly_max_coverage", purpose="analysis")
    prices_daily = prices.resample("D").last().dropna()
    prices_common = prices_daily[prices_daily.index.isin(X.index)].loc[X.index]

    for train_dates, test_dates in walk_forward_split(X.index):
        model = CatBoostClassifier(**config)
        model.fit(X.loc[train_dates], y.loc[train_dates])

        y_pred_proba = model.predict_proba(X.loc[test_dates])[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        acc = accuracy_score(y.loc[test_dates], y_pred)
        auc = roc_auc_score(y.loc[test_dates], y_pred_proba)

        # Backtest
        signals = pd.Series(2 * y_pred - 1, index=test_dates)
        returns = prices_common.loc[test_dates, "close"].pct_change()

        strat = signals.shift(1).fillna(0) * returns
        strat = strat.replace([np.inf, -np.inf], 0).fillna(0)

        sharpe = strat.mean() / strat.std() * np.sqrt(365) if strat.std() > 0 else 0.0

        print(f"  {test_dates[0].year}: Sharpe={sharpe:.3f} Acc={acc:.3f} AUC={auc:.3f}")
        sharpes.append(sharpe)
        accs.append(acc)

    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)

    print(f"\n{'=' * 80}")
    print(f"40-feature: Sharpe={mean_sharpe:.3f} ± {std_sharpe:.3f}, Acc={np.mean(accs):.3f}")
    print("20-feature: Sharpe=0.982")
    print("Donchian: Sharpe=1.062")

    result = {
        "n_features": 40,
        "selected_features": top_40,
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "mean_acc": np.mean(accs),
        "yearly_sharpes": sharpes,
    }

    with open("results/validation/boosted_features_40.json", "w") as f:
        json.dump(result, f, indent=2)

    if mean_sharpe > 1.062:
        print(f"✓ BEATS baseline by {((mean_sharpe / 1.062 - 1) * 100):.1f}%")
    else:
        print(f"✗ Below baseline by {((1 - mean_sharpe / 1.062) * 100):.1f}%")

    timer.end(claimed_duration_minutes=20)


if __name__ == "__main__":
    main()
