#!/usr/bin/env python3
"""Backtest Aggregated Hourly Signals on 2024-2025 Holdout.

Load trained CatBoost 1h-ahead model, generate hourly predictions,
aggregate to daily trading signals, and evaluate performance.

Model: CatBoost with 23 base features (AUC 0.5599 on 2023 test set)
Aggregation: HourlyToDailyAggregator with mean method
Holdout: 2024-01-01 to 2025-12-31 (never seen during training)

Performance Metrics:
- Sharpe ratio (daily returns, annualized)
- Total return
- Max drawdown
- Win rate
- Comparison to Buy & Hold baseline
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from sparky.models.signal_aggregator import HourlyToDailyAggregator
from sparky.features.returns import simple_returns, annualized_sharpe, max_drawdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data():
    """Load hourly features and targets."""
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)

    features_path = Path("data/processed/features_hourly_full.parquet")
    targets_path = Path("data/processed/targets_hourly_1h.parquet")

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Targets not found: {targets_path}")

    logger.info(f"Loading features: {features_path}")
    X = pd.read_parquet(features_path)

    logger.info(f"Loading targets: {targets_path}")
    y = pd.read_parquet(targets_path)

    # Clean data
    logger.info(f"Raw data: {X.shape} features, {y.shape} targets")

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop NaN rows
    nan_mask = X.isna().any(axis=1)
    logger.info(f"Dropping {nan_mask.sum()} rows with NaN values")
    X = X[~nan_mask]

    # Align targets
    y = y.loc[X.index]

    logger.info(f"Clean data: {X.shape} features, {y.shape} targets")
    logger.info(f"Date range: {X.index.min()} to {X.index.max()}")
    logger.info(f"Features: {list(X.columns)}")
    logger.info("")

    return X, y


def create_splits(X, y):
    """Split into train/val (2017-2023) and holdout (2024-2025).

    Train/Val periods (same as original training):
    - Train: 2017-2020
    - Val: 2021-2022
    - Test: 2023

    Holdout period (never seen):
    - Holdout: 2024-2025
    """
    logger.info("=" * 80)
    logger.info("CREATING SPLITS")
    logger.info("=" * 80)

    # Training splits
    train_mask = (X.index >= "2017-01-01") & (X.index < "2021-01-01")
    val_mask = (X.index >= "2021-01-01") & (X.index < "2023-01-01")
    test_mask = (X.index >= "2023-01-01") & (X.index < "2024-01-01")

    # Holdout split (2024-2025)
    holdout_mask = (X.index >= "2024-01-01") & (X.index < "2026-01-01")

    X_train = X[train_mask]
    y_train = y[train_mask]["target"]

    X_val = X[val_mask]
    y_val = y[val_mask]["target"]

    X_test = X[test_mask]
    y_test = y[test_mask]["target"]

    X_holdout = X[holdout_mask]
    y_holdout = y[holdout_mask]["target"]

    logger.info(f"Train: {X_train.shape} ({X_train.index.min()} to {X_train.index.max()})")
    logger.info(f"Val: {X_val.shape} ({X_val.index.min()} to {X_val.index.max()})")
    logger.info(f"Test: {X_test.shape} ({X_test.index.min()} to {X_test.index.max()})")
    logger.info(f"Holdout: {X_holdout.shape} ({X_holdout.index.min()} to {X_holdout.index.max()})")
    logger.info("")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_holdout": X_holdout,
        "y_holdout": y_holdout,
    }


def train_model(X_train, y_train):
    """Train CatBoost with best configuration (base features, AUC 0.5599).

    Hyperparameters from train_expanded_features_1h.py:
    - depth=5
    - learning_rate=0.05
    - iterations=200
    - l2_leaf_reg=3.0
    - subsample=0.8
    - rsm=0.8
    """
    logger.info("=" * 80)
    logger.info("TRAINING MODEL")
    logger.info("=" * 80)

    model = CatBoostClassifier(
        depth=5,
        learning_rate=0.05,
        iterations=200,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=50,
        subsample=0.8,
        rsm=0.8,
    )

    logger.info(f"Training on {len(X_train):,} samples with {X_train.shape[1]} features")
    model.fit(X_train, y_train)

    logger.info("Training complete")
    logger.info("")

    return model


def generate_predictions(model, X, y, split_name):
    """Generate hourly predictions and compute AUC."""
    logger.info(f"Generating predictions on {split_name} ({len(X):,} samples)")

    # Get P(up) predictions
    probas = model.predict_proba(X)[:, 1]
    probas_series = pd.Series(probas, index=X.index, name="proba")

    # Compute AUC
    auc = roc_auc_score(y, probas)
    logger.info(f"{split_name} AUC: {auc:.4f}")

    return probas_series, auc


def aggregate_to_daily(hourly_probas, method="mean", threshold=0.5):
    """Aggregate hourly predictions to daily signals.

    Args:
        hourly_probas: Series of hourly P(up) predictions
        method: Aggregation method (mean, weighted, regime)
        threshold: Probability threshold for LONG signal

    Returns:
        DataFrame with daily_proba, signal, n_hours, std
    """
    logger.info("=" * 80)
    logger.info("AGGREGATING TO DAILY SIGNALS")
    logger.info("=" * 80)

    aggregator = HourlyToDailyAggregator(method=method, threshold=threshold)
    daily_signals = aggregator.aggregate(hourly_probas)

    logger.info(f"Aggregated {len(hourly_probas):,} hourly predictions to {len(daily_signals)} daily signals")
    logger.info(f"Method: {method}, Threshold: {threshold}")
    logger.info(f"LONG signals: {daily_signals['signal'].sum()} / {len(daily_signals)} ({daily_signals['signal'].sum() / len(daily_signals) * 100:.1f}%)")
    logger.info("")

    return daily_signals


def load_price_data(start_date, end_date):
    """Load BTC price data for performance calculation.

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        DataFrame with daily close prices
    """
    price_path = Path("data/raw/btc/ohlcv_hourly.parquet")

    if not price_path.exists():
        raise FileNotFoundError(f"Price data not found: {price_path}")

    logger.info(f"Loading price data: {price_path}")
    prices = pd.read_parquet(price_path)

    # Resample to daily (take last close of each day)
    prices_daily = prices['close'].resample('D').last()

    # Remove timezone for consistency
    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)

    # Filter date range
    prices_daily = prices_daily.loc[start_date:end_date]

    logger.info(f"Price data: {len(prices_daily)} days ({prices_daily.index.min()} to {prices_daily.index.max()})")

    return prices_daily


def compute_strategy_returns(daily_signals, prices):
    """Compute strategy returns based on daily signals.

    Strategy:
    - signal=1 (LONG): Hold BTC for the day, earn daily return
    - signal=0 (FLAT): Hold cash, earn 0% return

    Execution:
    - Signal generated at day T close using data up to T
    - Position taken at T+1 open (approximately T close)
    - Return earned from T+1 open to T+1 close

    Args:
        daily_signals: DataFrame with 'signal' column (0 or 1)
        prices: Series of daily close prices

    Returns:
        Series of daily strategy returns
    """
    logger.info("=" * 80)
    logger.info("COMPUTING STRATEGY RETURNS")
    logger.info("=" * 80)

    # Align signals and prices
    common_dates = daily_signals.index.intersection(prices.index)
    daily_signals = daily_signals.loc[common_dates]
    prices = prices.loc[common_dates]

    logger.info(f"Computing returns on {len(common_dates)} days")

    # Compute daily returns
    daily_returns = prices.pct_change()

    # Strategy returns: position * market return
    # Lag signal by 1 day (today's signal determines tomorrow's position)
    positions = daily_signals['signal'].shift(1).fillna(0)
    strategy_returns = positions * daily_returns

    # Remove NaN (first day has no prior signal)
    strategy_returns = strategy_returns.dropna()

    logger.info(f"Strategy returns computed: {len(strategy_returns)} days")
    logger.info("")

    return strategy_returns


def compute_performance_metrics(returns, name="Strategy"):
    """Compute Sharpe, total return, max drawdown, win rate.

    Args:
        returns: Series of daily returns
        name: Strategy name for logging

    Returns:
        Dictionary of performance metrics
    """
    logger.info(f"Computing metrics for: {name}")

    # Total return
    cumulative_return = (1 + returns).prod() - 1
    total_return_pct = cumulative_return * 100

    # Sharpe ratio (annualized)
    sharpe = annualized_sharpe(returns, periods_per_year=365)

    # Max drawdown
    cumulative_wealth = (1 + returns).cumprod()
    max_dd = max_drawdown(cumulative_wealth)

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    # Number of trades
    # Count number of position changes (excluding first day)
    positions = (returns != 0).astype(int)
    n_trades = (positions.diff().abs()).sum()

    # Average daily return
    avg_daily_return = returns.mean()

    # Volatility (annualized)
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(365)

    metrics = {
        "total_return_pct": float(total_return_pct),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd * 100),
        "win_rate": float(win_rate),
        "n_trades": int(n_trades),
        "avg_daily_return_pct": float(avg_daily_return * 100),
        "annual_volatility_pct": float(annual_vol * 100),
        "n_days": len(returns),
    }

    logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    logger.info(f"  Win Rate: {metrics['win_rate']:.2f}")
    logger.info(f"  Trades: {metrics['n_trades']}")
    logger.info(f"  Avg Daily Return: {metrics['avg_daily_return_pct']:.4f}%")
    logger.info(f"  Annual Volatility: {metrics['annual_volatility_pct']:.2f}%")
    logger.info("")

    return metrics


def compute_baseline_performance(prices):
    """Compute Buy & Hold baseline performance.

    Args:
        prices: Series of daily close prices

    Returns:
        Dictionary of baseline metrics
    """
    logger.info("=" * 80)
    logger.info("BASELINE: BUY & HOLD")
    logger.info("=" * 80)

    # Daily returns (always invested)
    daily_returns = prices.pct_change().dropna()

    metrics = compute_performance_metrics(daily_returns, name="Buy & Hold")

    return metrics


def save_results(results, output_path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def print_comparison_table(strategy_metrics, baseline_metrics):
    """Print comparison table between strategy and baseline."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: Aggregated Signals vs Buy & Hold")
    print("=" * 80)
    print()
    print(f"{'Metric':<30s} {'Strategy':>15s} {'Buy & Hold':>15s} {'Delta':>15s}")
    print("-" * 80)

    metrics_to_compare = [
        ("Total Return (%)", "total_return_pct"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Max Drawdown (%)", "max_drawdown_pct"),
        ("Win Rate", "win_rate"),
        ("Trades", "n_trades"),
        ("Avg Daily Return (%)", "avg_daily_return_pct"),
        ("Annual Volatility (%)", "annual_volatility_pct"),
    ]

    for label, key in metrics_to_compare:
        strat_val = strategy_metrics[key]
        base_val = baseline_metrics[key]

        if key == "n_trades":
            delta_str = f"{int(strat_val - base_val):+d}"
            print(f"{label:<30s} {int(strat_val):>15d} {int(base_val):>15d} {delta_str:>15s}")
        elif key in ["total_return_pct", "max_drawdown_pct", "avg_daily_return_pct", "annual_volatility_pct"]:
            delta = strat_val - base_val
            delta_str = f"{delta:+.2f}"
            print(f"{label:<30s} {strat_val:>15.2f} {base_val:>15.2f} {delta_str:>15s}")
        else:
            delta = strat_val - base_val
            delta_str = f"{delta:+.3f}"
            print(f"{label:<30s} {strat_val:>15.3f} {base_val:>15.3f} {delta_str:>15s}")

    print("=" * 80)
    print()


def main():
    """Main backtest pipeline."""
    logger.info("=" * 80)
    logger.info("BACKTEST: HOURLY → DAILY AGGREGATED SIGNALS")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.utcnow().isoformat()}")
    logger.info("")

    # Load data
    X, y = load_data()

    # Create splits
    splits = create_splits(X, y)

    # Train model
    model = train_model(splits["X_train"], splits["y_train"])

    # Validate on val/test to confirm model quality
    _, val_auc = generate_predictions(model, splits["X_val"], splits["y_val"], "Validation")
    _, test_auc = generate_predictions(model, splits["X_test"], splits["y_test"], "Test")
    logger.info("")

    # Generate predictions on HOLDOUT (2024-2025)
    logger.info("=" * 80)
    logger.info("HOLDOUT EVALUATION (2024-2025)")
    logger.info("=" * 80)

    holdout_probas, holdout_auc = generate_predictions(
        model, splits["X_holdout"], splits["y_holdout"], "Holdout"
    )
    logger.info("")

    # Aggregate to daily signals
    daily_signals = aggregate_to_daily(holdout_probas, method="mean", threshold=0.5)

    # Load price data for holdout period
    holdout_start = splits["X_holdout"].index.min().date()
    holdout_end = splits["X_holdout"].index.max().date()
    prices = load_price_data(holdout_start, holdout_end)

    # Compute strategy returns
    strategy_returns = compute_strategy_returns(daily_signals, prices)

    # Compute strategy metrics
    logger.info("=" * 80)
    logger.info("STRATEGY PERFORMANCE")
    logger.info("=" * 80)
    strategy_metrics = compute_performance_metrics(strategy_returns, name="Aggregated Signals")

    # Compute baseline metrics
    baseline_metrics = compute_baseline_performance(prices)

    # Print comparison
    print_comparison_table(strategy_metrics, baseline_metrics)

    # Save results
    results = {
        "model": "CatBoost 1h-ahead (base features)",
        "aggregation_method": "mean",
        "aggregation_threshold": 0.5,
        "holdout_period": {
            "start": str(holdout_start),
            "end": str(holdout_end),
            "n_days": len(daily_signals),
        },
        "model_performance": {
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            "holdout_auc": float(holdout_auc),
        },
        "signal_stats": {
            "total_signals": len(daily_signals),
            "long_signals": int(daily_signals['signal'].sum()),
            "long_pct": float(daily_signals['signal'].sum() / len(daily_signals) * 100),
            "avg_daily_proba": float(daily_signals['daily_proba'].mean()),
            "std_daily_proba": float(daily_signals['daily_proba'].std()),
        },
        "strategy_metrics": strategy_metrics,
        "baseline_metrics": baseline_metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }

    output_path = Path("results/signal_aggregation/backtest_2024_2025_holdout.json")
    save_results(results, output_path)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nModel: CatBoost 1h-ahead (23 base features)")
    print(f"Aggregation: Mean of 24 hourly predictions, threshold=0.5")
    print(f"Holdout Period: {holdout_start} to {holdout_end} ({len(daily_signals)} days)")
    print(f"\nModel Quality:")
    print(f"  Val AUC (2021-2022): {val_auc:.4f}")
    print(f"  Test AUC (2023): {test_auc:.4f}")
    print(f"  Holdout AUC (2024-2025): {holdout_auc:.4f}")
    print(f"\nStrategy Performance:")
    print(f"  Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.3f}")
    print(f"  Total Return: {strategy_metrics['total_return_pct']:.2f}%")
    print(f"  Max Drawdown: {strategy_metrics['max_drawdown_pct']:.2f}%")
    print(f"\nBaseline (Buy & Hold):")
    print(f"  Sharpe Ratio: {baseline_metrics['sharpe_ratio']:.3f}")
    print(f"  Total Return: {baseline_metrics['total_return_pct']:.2f}%")
    print(f"  Max Drawdown: {baseline_metrics['max_drawdown_pct']:.2f}%")
    print(f"\nDelta:")
    print(f"  Sharpe: {strategy_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']:+.3f}")
    print(f"  Return: {strategy_metrics['total_return_pct'] - baseline_metrics['total_return_pct']:+.2f}%")
    print(f"\nResults saved to: {output_path}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
