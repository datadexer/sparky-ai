#!/usr/bin/env python3
"""Backtest Regime-Aware Trading Strategy on 2024-2025 Holdout.

PHASE 2A: REGIME-AWARE TRADING
===============================

Load trained CatBoost 1h-ahead model (Phase 1 cross-asset pooled),
apply regime-aware position sizing and dynamic thresholds,
and evaluate performance vs static baseline.

Regime Rules:
- HIGH (>60% vol): 50% position, threshold 0.55
- MEDIUM (30-60%): 75% position, threshold 0.52
- LOW (<30%): 100% position, threshold 0.50

Target: Sharpe ≥ 0.95 (vs Buy & Hold 0.950, Static Strategy 0.646)

Performance Metrics:
- Sharpe ratio by regime
- Total return
- Max drawdown by regime
- Win rate by regime
- Comparison to Buy & Hold and Static baseline
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.signal_aggregator import HourlyToDailyAggregator, RegimeAwareAggregator

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
    logger.info("")

    return X, y


def create_splits(X, y):
    """Split into train/val (2017-2023) and holdout (2024-2025)."""
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
    """Train CatBoost with Phase 1 configuration (cross-asset pooled).

    Hyperparameters:
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


def load_price_data(start_date, end_date):
    """Load BTC price data for performance calculation."""
    price_path = Path("data/raw/btc/ohlcv_hourly.parquet")

    if not price_path.exists():
        raise FileNotFoundError(f"Price data not found: {price_path}")

    logger.info(f"Loading price data: {price_path}")
    prices = pd.read_parquet(price_path)

    # Hourly close prices
    prices_hourly = prices["close"]

    # Remove timezone for consistency (do this first)
    if prices_hourly.index.tz is not None:
        prices_hourly.index = prices_hourly.index.tz_localize(None)

    # Convert start/end dates to tz-naive if needed
    if hasattr(start_date, "tz") and start_date.tz is not None:
        start_date = start_date.tz_localize(None)
    if hasattr(end_date, "tz") and end_date.tz is not None:
        end_date = end_date.tz_localize(None)

    # Filter date range
    prices_hourly = prices_hourly.loc[start_date:end_date]

    logger.info(f"Price data: {len(prices_hourly)} hours ({prices_hourly.index.min()} to {prices_hourly.index.max()})")

    return prices_hourly


def aggregate_regime_aware(hourly_probas, prices):
    """Aggregate hourly predictions with regime awareness.

    Args:
        hourly_probas: Series of hourly P(up) predictions
        prices: Hourly close prices for regime computation

    Returns:
        DataFrame with regime-aware daily signals
    """
    logger.info("=" * 80)
    logger.info("REGIME-AWARE AGGREGATION")
    logger.info("=" * 80)

    aggregator = RegimeAwareAggregator(regime_window=30 * 24, frequency="1h")
    daily_signals = aggregator.aggregate_to_daily(hourly_probas, prices)

    logger.info(f"Regime-aware signals generated: {len(daily_signals)} days")
    logger.info("")

    return daily_signals


def aggregate_static(hourly_probas):
    """Aggregate hourly predictions with static threshold (baseline).

    Args:
        hourly_probas: Series of hourly P(up) predictions

    Returns:
        DataFrame with static daily signals
    """
    logger.info("=" * 80)
    logger.info("STATIC AGGREGATION (BASELINE)")
    logger.info("=" * 80)

    aggregator = HourlyToDailyAggregator(method="mean", threshold=0.5)
    daily_signals = aggregator.aggregate(hourly_probas)

    logger.info(f"Static signals generated: {len(daily_signals)} days")
    logger.info("")

    return daily_signals


def compute_strategy_returns(daily_signals, prices, use_position_sizing=False):
    """Compute strategy returns based on daily signals.

    Args:
        daily_signals: DataFrame with 'signal' and optionally 'position_size' columns
        prices: Series of hourly close prices
        use_position_sizing: If True, use 'position_size' column for fractional positions

    Returns:
        Series of daily strategy returns
    """
    logger.info("=" * 80)
    logger.info("COMPUTING STRATEGY RETURNS")
    logger.info("=" * 80)

    # Resample prices to daily
    prices_daily = prices.resample("D").last()

    # Remove timezone for consistency
    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)

    # Align signals and prices
    common_dates = daily_signals.index.intersection(prices_daily.index)
    daily_signals = daily_signals.loc[common_dates]
    prices_daily = prices_daily.loc[common_dates]

    logger.info(f"Computing returns on {len(common_dates)} days")

    # Compute daily returns
    daily_returns = prices_daily.pct_change()

    # Strategy returns
    if use_position_sizing and "position_size" in daily_signals.columns:
        # Regime-aware: use position_size directly (already accounts for signal)
        positions = daily_signals["position_size"].shift(1).fillna(0)
        logger.info("Using regime-aware position sizing")
    else:
        # Static: signal is 0 or 1, position is 100% when signal=1
        positions = daily_signals["signal"].shift(1).fillna(0)
        logger.info("Using static position sizing")

    strategy_returns = positions * daily_returns

    # Remove NaN (first day has no prior signal)
    strategy_returns = strategy_returns.dropna()

    logger.info(f"Strategy returns computed: {len(strategy_returns)} days")
    logger.info("")

    return strategy_returns


def compute_performance_metrics(returns, name="Strategy"):
    """Compute Sharpe, total return, max drawdown, win rate."""
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

    # Number of trades (position changes)
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
    logger.info("")

    return metrics


def compute_baseline_performance(prices):
    """Compute Buy & Hold baseline performance."""
    logger.info("=" * 80)
    logger.info("BASELINE: BUY & HOLD")
    logger.info("=" * 80)

    # Resample to daily
    prices_daily = prices.resample("D").last()

    # Remove timezone
    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)

    # Daily returns (always invested)
    daily_returns = prices_daily.pct_change().dropna()

    metrics = compute_performance_metrics(daily_returns, name="Buy & Hold")

    return metrics


def analyze_regime_performance(daily_signals, strategy_returns):
    """Analyze performance by volatility regime.

    Args:
        daily_signals: DataFrame with 'regime' column
        strategy_returns: Series of daily strategy returns

    Returns:
        Dictionary with regime-specific metrics
    """
    logger.info("=" * 80)
    logger.info("REGIME-SPECIFIC PERFORMANCE")
    logger.info("=" * 80)

    # Align returns with signals
    common_dates = daily_signals.index.intersection(strategy_returns.index)
    daily_signals = daily_signals.loc[common_dates]
    strategy_returns = strategy_returns.loc[common_dates]

    regime_analysis = {}

    for regime in ["low", "medium", "high"]:
        regime_mask = daily_signals["regime"] == regime
        regime_returns = strategy_returns[regime_mask]

        if len(regime_returns) == 0:
            logger.info(f"\nRegime: {regime.upper()} (NO DATA)")
            regime_analysis[regime] = None
            continue

        logger.info(f"\nRegime: {regime.upper()} ({len(regime_returns)} days)")

        # Compute metrics for this regime
        cumulative_return = (1 + regime_returns).prod() - 1
        sharpe = annualized_sharpe(regime_returns, periods_per_year=365)
        win_rate = (regime_returns > 0).sum() / len(regime_returns)

        cumulative_wealth = (1 + regime_returns).cumprod()
        max_dd = max_drawdown(cumulative_wealth)

        logger.info(f"  Return: {cumulative_return * 100:.2f}%")
        logger.info(f"  Sharpe: {sharpe:.3f}")
        logger.info(f"  Max DD: {max_dd * 100:.2f}%")
        logger.info(f"  Win Rate: {win_rate:.2f}")

        regime_analysis[regime] = {
            "n_days": len(regime_returns),
            "total_return_pct": float(cumulative_return * 100),
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": float(max_dd * 100),
            "win_rate": float(win_rate),
        }

    logger.info("")

    return regime_analysis


def print_comparison_table(regime_metrics, static_metrics, baseline_metrics):
    """Print comparison table between regime-aware, static, and baseline."""
    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON: Regime-Aware vs Static vs Buy & Hold")
    print("=" * 100)
    print()
    print(f"{'Metric':<25s} {'Regime-Aware':>18s} {'Static':>18s} {'Buy & Hold':>18s} {'Delta (R-S)':>15s}")
    print("-" * 100)

    metrics_to_compare = [
        ("Total Return (%)", "total_return_pct"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Max Drawdown (%)", "max_drawdown_pct"),
        ("Win Rate", "win_rate"),
        ("Trades", "n_trades"),
    ]

    for label, key in metrics_to_compare:
        regime_val = regime_metrics[key]
        static_val = static_metrics[key]
        base_val = baseline_metrics[key]

        if key == "n_trades":
            delta_str = f"{int(regime_val - static_val):+d}"
            print(f"{label:<25s} {int(regime_val):>18d} {int(static_val):>18d} {int(base_val):>18d} {delta_str:>15s}")
        elif key in ["total_return_pct", "max_drawdown_pct"]:
            delta = regime_val - static_val
            delta_str = f"{delta:+.2f}"
            print(f"{label:<25s} {regime_val:>18.2f} {static_val:>18.2f} {base_val:>18.2f} {delta_str:>15s}")
        else:
            delta = regime_val - static_val
            delta_str = f"{delta:+.3f}"
            print(f"{label:<25s} {regime_val:>18.3f} {static_val:>18.3f} {base_val:>18.3f} {delta_str:>15s}")

    print("=" * 100)
    print()


def save_results(results, output_path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def main():
    """Main backtest pipeline."""
    logger.info("=" * 80)
    logger.info("PHASE 2A: REGIME-AWARE TRADING BACKTEST")
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

    holdout_probas, holdout_auc = generate_predictions(model, splits["X_holdout"], splits["y_holdout"], "Holdout")
    logger.info("")

    # Load price data for holdout period
    holdout_start = splits["X_holdout"].index.min()
    holdout_end = splits["X_holdout"].index.max()
    prices = load_price_data(holdout_start, holdout_end)

    # Aggregate with regime awareness
    regime_signals = aggregate_regime_aware(holdout_probas, prices)

    # Aggregate with static threshold (baseline)
    static_signals = aggregate_static(holdout_probas)

    # Compute regime-aware strategy returns
    logger.info("REGIME-AWARE STRATEGY")
    regime_returns = compute_strategy_returns(regime_signals, prices, use_position_sizing=True)

    # Compute static strategy returns
    logger.info("STATIC STRATEGY (BASELINE)")
    static_returns = compute_strategy_returns(static_signals, prices, use_position_sizing=False)

    # Compute regime-aware metrics
    logger.info("=" * 80)
    logger.info("REGIME-AWARE PERFORMANCE")
    logger.info("=" * 80)
    regime_metrics = compute_performance_metrics(regime_returns, name="Regime-Aware")

    # Compute static metrics
    logger.info("=" * 80)
    logger.info("STATIC PERFORMANCE")
    logger.info("=" * 80)
    static_metrics = compute_performance_metrics(static_returns, name="Static")

    # Compute baseline metrics
    baseline_metrics = compute_baseline_performance(prices)

    # Analyze regime-specific performance
    regime_analysis = analyze_regime_performance(regime_signals, regime_returns)

    # Print comparison
    print_comparison_table(regime_metrics, static_metrics, baseline_metrics)

    # Save results
    results = {
        "model": "CatBoost 1h-ahead (Phase 1 cross-asset pooled)",
        "regime_aware": {
            "regime_window": 30 * 24,
            "thresholds": {"high": 0.55, "medium": 0.52, "low": 0.50},
            "position_sizes": {"high": 0.50, "medium": 0.75, "low": 1.00},
        },
        "holdout_period": {
            "start": str(holdout_start.date()),
            "end": str(holdout_end.date()),
            "n_days": len(regime_signals),
        },
        "model_performance": {
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            "holdout_auc": float(holdout_auc),
        },
        "regime_aware_metrics": regime_metrics,
        "static_metrics": static_metrics,
        "baseline_metrics": baseline_metrics,
        "regime_analysis": regime_analysis,
        "signal_stats": {
            "regime_aware_long_pct": float(regime_signals["signal"].sum() / len(regime_signals) * 100),
            "static_long_pct": float(static_signals["signal"].sum() / len(static_signals) * 100),
        },
        "verdict": "SUCCESS"
        if regime_metrics["sharpe_ratio"] >= 0.95
        else "PARTIAL"
        if regime_metrics["sharpe_ratio"] >= 0.85
        else "FAILED",
        "timestamp": datetime.utcnow().isoformat(),
    }

    output_path = Path("results/regime_aware/backtest_2024_2025_holdout.json")
    save_results(results, output_path)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nPhase 2A: Regime-Aware Trading")
    print(f"Holdout Period: {holdout_start.date()} to {holdout_end.date()} ({len(regime_signals)} days)")
    print("\nModel Quality:")
    print(f"  Holdout AUC (2024-2025): {holdout_auc:.4f}")
    print("\nRegime-Aware Strategy:")
    print(f"  Sharpe Ratio: {regime_metrics['sharpe_ratio']:.3f}")
    print(f"  Total Return: {regime_metrics['total_return_pct']:.2f}%")
    print(f"  Max Drawdown: {regime_metrics['max_drawdown_pct']:.2f}%")
    print("\nStatic Strategy:")
    print(f"  Sharpe Ratio: {static_metrics['sharpe_ratio']:.3f}")
    print(f"  Total Return: {static_metrics['total_return_pct']:.2f}%")
    print("\nBaseline (Buy & Hold):")
    print(f"  Sharpe Ratio: {baseline_metrics['sharpe_ratio']:.3f}")
    print(f"  Total Return: {baseline_metrics['total_return_pct']:.2f}%")
    print("\nImprovement:")
    print(f"  Sharpe Delta (Regime - Static): {regime_metrics['sharpe_ratio'] - static_metrics['sharpe_ratio']:+.3f}")
    print(
        f"  Sharpe Delta (Regime - Buy&Hold): {regime_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']:+.3f}"
    )
    print(f"\nVerdict: {results['verdict']}")
    if results["verdict"] == "SUCCESS":
        print("✅ Sharpe ≥ 0.95 → Proceed to paper trading (SKIP Phase 2B)")
    elif results["verdict"] == "PARTIAL":
        print("⚠️ Sharpe 0.85-0.94 → Execute Phase 2B to push over 1.0")
    else:
        print("❌ Sharpe < 0.85 → Reassess approach")
    print(f"\nResults saved to: {output_path}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
