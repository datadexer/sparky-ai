#!/usr/bin/env python3
"""Evaluate ML model Sharpe ratio with proper walk-forward methodology.

Computes Sharpe from XGBoost predictions using:
- Walk-forward expanding window (train on past, predict next year)
- signal.shift(1) to avoid look-ahead bias
- Transaction costs (26 bps per trade)
- Comparison against Multi-TF Donchian baseline (Sharpe 1.062)

This is the TRUE test of whether ML adds value over simple baselines.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data():
    """Load hourly feature matrix and targets."""
    features_path = Path("data/processed/feature_matrix_btc_hourly.parquet")
    targets_path = Path("data/processed/targets_btc_hourly_1d.parquet")

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(targets_path)["target"]

    logger.info(f"Loaded {len(X):,} samples, {X.shape[1]} features")
    return X, y


def load_daily_prices():
    """Load BTC daily prices for return computation."""
    prices = pd.read_parquet("data/btc_daily.parquet")["close"]
    prices.index = pd.to_datetime(prices.index)
    # Ensure timezone-aware (UTC)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")
    return prices


def walk_forward_ml_backtest(X, y, prices, test_years, min_train_years=2):
    """Walk-forward backtest with expanding training window.

    For each test year:
    1. Train on all data before test year (expanding window)
    2. Predict on test year
    3. Convert predictions to positions (1=long, 0=flat)
    4. Compute returns with signal.shift(1) and transaction costs

    Args:
        X: Feature matrix (daily)
        y: Targets (daily)
        prices: Daily close prices
        test_years: List of years to test on
        min_train_years: Minimum training years before first test

    Returns:
        Dict with yearly Sharpe ratios and overall metrics
    """
    from sparky.models.xgboost_model import XGBoostModel
    from sparky.backtest.costs import TransactionCostModel
    from sparky.features.returns import annualized_sharpe

    cost_model = TransactionCostModel.for_btc()
    results = []

    for year in test_years:
        # Training data: everything before this year
        train_mask = X.index < f"{year}-01-01"
        test_mask = (X.index >= f"{year}-01-01") & (X.index < f"{year + 1}-01-01")

        X_train = X[train_mask].replace([np.inf, -np.inf], np.nan).dropna()
        y_train = y[train_mask].loc[X_train.index]
        X_test = X[test_mask].replace([np.inf, -np.inf], np.nan).dropna()
        y_test = y[test_mask].loc[X_test.index]

        if len(X_train) < 365 * min_train_years:
            logger.warning(f"Skipping {year}: insufficient training data ({len(X_train)} samples)")
            continue

        if len(X_test) == 0:
            logger.warning(f"Skipping {year}: no test data")
            continue

        # Train model
        model = XGBoostModel(
            max_depth=3,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test).mean()

        # Convert to trading signals (1=long, 0=flat)
        signals = pd.Series(predictions, index=X_test.index)

        # Get price returns for test period
        # Align indices: feature dates may not exactly match price dates
        price_returns = prices.pct_change()
        common_idx = signals.index.intersection(price_returns.index)
        signals = signals.loc[common_idx]
        test_returns = price_returns.loc[common_idx]

        # Apply signal.shift(1) — NO look-ahead bias
        # Signal at T uses features from T, position taken at T+1
        actual_positions = signals.shift(1).fillna(0)
        strategy_returns = actual_positions * test_returns

        # Transaction costs
        position_changes = actual_positions.diff().abs()
        transaction_costs = position_changes * cost_model.total_cost_pct
        strategy_returns_after_costs = strategy_returns - transaction_costs

        # Sharpe ratio
        sharpe = annualized_sharpe(strategy_returns_after_costs)
        n_trades = int(position_changes.sum())
        total_return = ((1 + strategy_returns_after_costs).cumprod().iloc[-1] - 1) * 100

        logger.info(
            f"{year}: Sharpe={sharpe:.3f}, Acc={accuracy:.3f}, "
            f"Return={total_return:.1f}%, Trades={n_trades}, "
            f"Train={len(X_train)}, Test={len(X_test)}"
        )

        results.append({
            "year": year,
            "sharpe": sharpe,
            "accuracy": accuracy,
            "total_return": total_return,
            "n_trades": n_trades,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })

    return results


def run_baseline_comparison(prices, test_years):
    """Run Multi-TF Donchian baseline for comparison (with corrected methodology)."""
    from sparky.models.simple_baselines import donchian_channel_strategy
    from sparky.backtest.costs import TransactionCostModel
    from sparky.features.returns import annualized_sharpe

    cost_model = TransactionCostModel.for_btc()

    # Generate Donchian signals
    s20 = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
    s40 = donchian_channel_strategy(prices, entry_period=40, exit_period=20)
    s60 = donchian_channel_strategy(prices, entry_period=60, exit_period=30)
    signals = ((s20 + s40 + s60) >= 2).astype(int)

    price_returns = prices.pct_change()
    results = []

    for year in test_years:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        period_signals = signals.loc[start:end]
        period_returns = price_returns.loc[start:end]

        # signal.shift(1) — corrected methodology
        actual_positions = period_signals.shift(1).fillna(0)
        strategy_returns = actual_positions * period_returns

        position_changes = actual_positions.diff().abs()
        transaction_costs = position_changes * cost_model.total_cost_pct
        strategy_returns_after_costs = strategy_returns - transaction_costs

        sharpe = annualized_sharpe(strategy_returns_after_costs)
        total_return = ((1 + strategy_returns_after_costs).cumprod().iloc[-1] - 1) * 100

        results.append({
            "year": year,
            "sharpe": sharpe,
            "total_return": total_return,
        })

    return results


def main():
    logger.info("=" * 80)
    logger.info("ML vs BASELINE SHARPE COMPARISON (Walk-Forward)")
    logger.info("=" * 80)

    X, y = load_data()
    prices = load_daily_prices()

    # Test on 2021-2023 (in-sample validation years)
    # Do NOT touch 2024+ (holdout)
    test_years = [2021, 2022, 2023]

    logger.info("\n--- XGBoost ML Model ---")
    ml_results = walk_forward_ml_backtest(X, y, prices, test_years)

    logger.info("\n--- Multi-TF Donchian Baseline ---")
    baseline_results = run_baseline_comparison(prices, test_years)

    # Summary comparison
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 80)
    logger.info(f"{'Year':<8} {'ML Sharpe':<12} {'Baseline Sharpe':<16} {'ML Wins?':<10}")
    logger.info("-" * 50)

    ml_sharpes = []
    baseline_sharpes = []

    for ml, bl in zip(ml_results, baseline_results):
        wins = "YES" if ml["sharpe"] > bl["sharpe"] else "NO"
        logger.info(f"{ml['year']:<8} {ml['sharpe']:<12.3f} {bl['sharpe']:<16.3f} {wins}")
        ml_sharpes.append(ml["sharpe"])
        baseline_sharpes.append(bl["sharpe"])

    ml_mean = np.mean(ml_sharpes) if ml_sharpes else 0
    bl_mean = np.mean(baseline_sharpes) if baseline_sharpes else 0

    logger.info("-" * 50)
    logger.info(f"{'Mean':<8} {ml_mean:<12.3f} {bl_mean:<16.3f} {'YES' if ml_mean > bl_mean else 'NO'}")

    logger.info("\n" + "=" * 80)
    if ml_mean > bl_mean:
        logger.info(f"RESULT: ML BEATS BASELINE ({ml_mean:.3f} vs {bl_mean:.3f})")
        logger.info("Next: Consider holdout validation (Human Gate required)")
    else:
        logger.info(f"RESULT: BASELINE WINS ({bl_mean:.3f} vs {ml_mean:.3f})")
        logger.info("ML with hourly features does NOT beat Multi-TF Donchian.")
        logger.info("Consider: cross-asset training, different model architecture, or accept baseline.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
