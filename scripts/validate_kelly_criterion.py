#!/usr/bin/env python3
"""
Kelly Criterion Position Sizing Validation

Apply Kelly Criterion to Multi-Timeframe Donchian Ensemble (20/40/60).
Compare fixed 100% sizing vs Kelly 0.25x vs Kelly 0.5x.

Target: Sharpe ≥0.85 (vs current 0.772 with fixed sizing).

Validation:
- 6 yearly walk-forward folds (2018-2023)
- Transaction costs: 0.26% round-trip
- Block bootstrap Monte Carlo (≥75% win rate)
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sparky.backtest.costs import TransactionCostModel
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.portfolio.kelly_criterion import (
    apply_fixed_sizing,
    apply_kelly_sizing,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_prices():
    """Load BTC daily prices (2017-2023)."""
    price_path = Path("data/raw/btc/ohlcv_hourly.parquet")
    prices = pd.read_parquet(price_path)
    prices_daily = prices["close"].resample("D").last()
    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)
    return prices_daily.loc["2017-01-01":"2023-12-31"]


def strategy_multi_timeframe_ensemble(prices):
    """Multi-Timeframe Donchian Ensemble (20/40/60) - baseline Sharpe 0.772"""
    signals_20 = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
    signals_40 = donchian_channel_strategy(prices, entry_period=40, exit_period=20)
    signals_60 = donchian_channel_strategy(prices, entry_period=60, exit_period=30)

    # Ensemble: LONG if ANY timeframe signals LONG
    ensemble = ((signals_20 + signals_40 + signals_60) > 0).astype(int)
    return ensemble


def backtest_with_sizing(
    prices: pd.Series,
    signals: pd.Series,
    position_sizes: pd.Series,
    cost_model: TransactionCostModel,
) -> dict:
    """Backtest strategy with variable position sizing.

    Args:
        prices: Daily close prices.
        signals: Binary signals (1 = LONG, 0 = FLAT).
        position_sizes: Position sizes (0.0 to 2.0).
        cost_model: Transaction cost model.

    Returns:
        Dictionary with performance metrics.
    """
    # Daily returns
    returns = prices.pct_change()

    # Strategy returns: position_size[t-1] * returns[t]
    # (Use yesterday's position for today's return)
    strategy_returns = position_sizes.shift(1) * returns

    # Transaction costs: charge on position changes
    position_changes = position_sizes.diff().abs()
    cost_returns = -position_changes * cost_model.total_cost_pct

    # Net returns after costs
    net_returns = strategy_returns + cost_returns

    # Performance metrics
    total_return = (1 + net_returns).prod() - 1
    sharpe = annualized_sharpe(net_returns)
    max_dd = max_drawdown(net_returns)

    # Count trades (position changes)
    n_trades = (position_changes > 0.01).sum()

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "mean_position_size": position_sizes[position_sizes > 0].mean() if (position_sizes > 0).any() else 0.0,
        "max_position_size": position_sizes.max(),
    }


def yearly_walk_forward_validation(prices: pd.Series) -> dict:
    """Yearly walk-forward validation comparing sizing strategies.

    Tests:
    - Fixed 100% sizing (baseline)
    - Kelly 0.25x (conservative)
    - Kelly 0.5x (moderate)

    Returns:
        Dictionary with results for each strategy across 6 years.
    """
    results = {
        "Fixed 100%": {"folds": []},
        "Kelly 0.25x": {"folds": []},
        "Kelly 0.5x": {"folds": []},
    }

    cost_model = TransactionCostModel.for_btc()
    years = [2018, 2019, 2020, 2021, 2022, 2023]

    for year in years:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"YEAR {year}")
        logger.info(f"{'=' * 60}")

        # Train period: all data before this year
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        train_prices = prices.loc[:train_end]
        test_prices = prices.loc[test_start:test_end]

        if len(test_prices) < 30:
            logger.warning(f"Skipping {year}: insufficient test data ({len(test_prices)} days)")
            continue

        # Generate signals on test period (no look-ahead)
        # Use full history up to each point for signal generation
        full_prices_to_test_end = prices.loc[:test_end]
        signals = strategy_multi_timeframe_ensemble(full_prices_to_test_end)
        signals_test = signals.loc[test_start:test_end]

        # Calculate returns
        returns = prices.pct_change()
        returns_test = returns.loc[test_start:test_end]

        # === FIXED 100% SIZING ===
        logger.info("\n--- Fixed 100% Sizing ---")
        position_sizes_fixed = apply_fixed_sizing(signals_test, fixed_size=1.0)
        metrics_fixed = backtest_with_sizing(test_prices, signals_test, position_sizes_fixed, cost_model)

        results["Fixed 100%"]["folds"].append(
            {
                "year": str(year),
                "sharpe": float(metrics_fixed["sharpe"]),
                "return_pct": float(metrics_fixed["total_return"] * 100),
                "max_drawdown": float(metrics_fixed["max_drawdown"]),
                "n_trades": int(metrics_fixed["n_trades"]),
            }
        )

        logger.info(
            f"Fixed 100%: Sharpe={metrics_fixed['sharpe']:.3f}, "
            f"Return={metrics_fixed['total_return'] * 100:.2f}%, "
            f"Trades={metrics_fixed['n_trades']}"
        )

        # === KELLY 0.25x SIZING ===
        logger.info("\n--- Kelly 0.25x Sizing ---")

        # Need historical data to calculate Kelly parameters
        # Use expanding window: train on all data before test period
        if len(train_prices) < 252:
            logger.warning(f"Insufficient training data for Kelly ({len(train_prices)} days), using fixed sizing")
            position_sizes_kelly_025 = apply_fixed_sizing(signals_test, fixed_size=1.0)
        else:
            # Calculate Kelly on full dataset up to each point
            position_sizes_kelly_025 = apply_kelly_sizing(
                signals=signals.loc[:test_end],
                returns=returns.loc[:test_end],
                fraction=0.25,
                max_leverage=2.0,
                lookback=252,
            ).loc[test_start:test_end]

        metrics_kelly_025 = backtest_with_sizing(test_prices, signals_test, position_sizes_kelly_025, cost_model)

        results["Kelly 0.25x"]["folds"].append(
            {
                "year": str(year),
                "sharpe": float(metrics_kelly_025["sharpe"]),
                "return_pct": float(metrics_kelly_025["total_return"] * 100),
                "max_drawdown": float(metrics_kelly_025["max_drawdown"]),
                "n_trades": int(metrics_kelly_025["n_trades"]),
                "mean_position_size": float(metrics_kelly_025["mean_position_size"]),
                "max_position_size": float(metrics_kelly_025["max_position_size"]),
            }
        )

        logger.info(
            f"Kelly 0.25x: Sharpe={metrics_kelly_025['sharpe']:.3f}, "
            f"Return={metrics_kelly_025['total_return'] * 100:.2f}%, "
            f"Mean Size={metrics_kelly_025['mean_position_size']:.3f}, "
            f"Trades={metrics_kelly_025['n_trades']}"
        )

        # === KELLY 0.5x SIZING ===
        logger.info("\n--- Kelly 0.5x Sizing ---")

        if len(train_prices) < 252:
            position_sizes_kelly_050 = apply_fixed_sizing(signals_test, fixed_size=1.0)
        else:
            position_sizes_kelly_050 = apply_kelly_sizing(
                signals=signals.loc[:test_end],
                returns=returns.loc[:test_end],
                fraction=0.5,
                max_leverage=2.0,
                lookback=252,
            ).loc[test_start:test_end]

        metrics_kelly_050 = backtest_with_sizing(test_prices, signals_test, position_sizes_kelly_050, cost_model)

        results["Kelly 0.5x"]["folds"].append(
            {
                "year": str(year),
                "sharpe": float(metrics_kelly_050["sharpe"]),
                "return_pct": float(metrics_kelly_050["total_return"] * 100),
                "max_drawdown": float(metrics_kelly_050["max_drawdown"]),
                "n_trades": int(metrics_kelly_050["n_trades"]),
                "mean_position_size": float(metrics_kelly_050["mean_position_size"]),
                "max_position_size": float(metrics_kelly_050["max_position_size"]),
            }
        )

        logger.info(
            f"Kelly 0.5x: Sharpe={metrics_kelly_050['sharpe']:.3f}, "
            f"Return={metrics_kelly_050['total_return'] * 100:.2f}%, "
            f"Mean Size={metrics_kelly_050['mean_position_size']:.3f}, "
            f"Trades={metrics_kelly_050['n_trades']}"
        )

    # Calculate summary statistics
    for strategy_name in results.keys():
        folds = results[strategy_name]["folds"]
        if len(folds) == 0:
            continue

        sharpes = [f["sharpe"] for f in folds]
        returns = [f["return_pct"] for f in folds]

        results[strategy_name]["mean_sharpe"] = np.mean(sharpes)
        results[strategy_name]["std_sharpe"] = np.std(sharpes)
        results[strategy_name]["min_sharpe"] = np.min(sharpes)
        results[strategy_name]["max_sharpe"] = np.max(sharpes)
        results[strategy_name]["median_sharpe"] = np.median(sharpes)
        results[strategy_name]["positive_count"] = sum(1 for s in sharpes if s > 0)
        results[strategy_name]["mean_return"] = np.mean(returns)

    return results


def block_bootstrap_monte_carlo(
    prices: pd.Series,
    signals: pd.Series,
    position_sizes: pd.Series,
    cost_model: TransactionCostModel,
    n_simulations: int = 1000,
    block_size: int = 21,
) -> dict:
    """Block bootstrap Monte Carlo simulation.

    Preserves autocorrelation in returns by resampling blocks of consecutive days.

    Args:
        prices: Daily prices.
        signals: Binary signals.
        position_sizes: Position sizes.
        cost_model: Transaction cost model.
        n_simulations: Number of bootstrap samples.
        block_size: Block size for bootstrap (default 21 = ~1 month).

    Returns:
        Dictionary with bootstrap statistics.
    """
    returns = prices.pct_change()
    strategy_returns = position_sizes.shift(1) * returns
    position_changes = position_sizes.diff().abs()
    cost_returns = -position_changes * cost_model.total_cost_pct
    net_returns = strategy_returns + cost_returns

    # Remove NaN
    net_returns = net_returns.dropna()

    sharpe_samples = []
    for _ in range(n_simulations):
        # Block bootstrap
        n_blocks = len(net_returns) // block_size
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)

        bootstrap_returns = []
        for block_idx in block_indices:
            start = block_idx * block_size
            end = start + block_size
            bootstrap_returns.extend(net_returns.iloc[start:end].values)

        bootstrap_returns = pd.Series(bootstrap_returns[: len(net_returns)])
        sharpe_samples.append(annualized_sharpe(bootstrap_returns))

    win_rate = sum(1 for s in sharpe_samples if s > 0) / len(sharpe_samples)

    return {
        "n_simulations": n_simulations,
        "mean_sharpe": np.mean(sharpe_samples),
        "std_sharpe": np.std(sharpe_samples),
        "ci_lower": np.percentile(sharpe_samples, 2.5),
        "ci_upper": np.percentile(sharpe_samples, 97.5),
        "win_rate": win_rate,
    }


def main():
    logger.info("=" * 70)
    logger.info("KELLY CRITERION POSITION SIZING VALIDATION")
    logger.info("=" * 70)

    # Load data
    logger.info("\n1. Loading BTC daily prices (2017-2023)...")
    prices = load_prices()
    logger.info(f"Loaded {len(prices)} days of data")

    # Yearly walk-forward validation
    logger.info("\n2. Running yearly walk-forward validation (6 years)...")
    results = yearly_walk_forward_validation(prices)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY - Yearly Walk-Forward Validation")
    logger.info("=" * 70)

    for strategy_name in ["Fixed 100%", "Kelly 0.25x", "Kelly 0.5x"]:
        logger.info(f"\n{strategy_name}:")
        logger.info(f"  Mean Sharpe:   {results[strategy_name]['mean_sharpe']:.3f}")
        logger.info(f"  Median Sharpe: {results[strategy_name]['median_sharpe']:.3f}")
        logger.info(f"  Std Sharpe:    {results[strategy_name]['std_sharpe']:.3f}")
        logger.info(f"  Min Sharpe:    {results[strategy_name]['min_sharpe']:.3f}")
        logger.info(f"  Max Sharpe:    {results[strategy_name]['max_sharpe']:.3f}")
        logger.info(f"  Positive:      {results[strategy_name]['positive_count']}/6 years")
        logger.info(f"  Mean Return:   {results[strategy_name]['mean_return']:.2f}%")

    # Determine best strategy
    best_strategy = max(["Fixed 100%", "Kelly 0.25x", "Kelly 0.5x"], key=lambda s: results[s]["mean_sharpe"])

    logger.info(f"\n{'=' * 70}")
    logger.info(f"BEST STRATEGY: {best_strategy} (Sharpe {results[best_strategy]['mean_sharpe']:.3f})")
    logger.info(f"{'=' * 70}")

    # Monte Carlo on best strategy
    logger.info(f"\n3. Running block bootstrap Monte Carlo on {best_strategy}...")

    # Generate signals and position sizes for full period (2018-2023)
    test_prices = prices.loc["2018-01-01":"2023-12-31"]
    signals = strategy_multi_timeframe_ensemble(prices)
    signals_test = signals.loc["2018-01-01":"2023-12-31"]

    if best_strategy == "Fixed 100%":
        position_sizes = apply_fixed_sizing(signals_test, fixed_size=1.0)
    elif best_strategy == "Kelly 0.25x":
        returns = prices.pct_change()
        position_sizes = apply_kelly_sizing(
            signals=signals, returns=returns, fraction=0.25, max_leverage=2.0, lookback=252
        ).loc["2018-01-01":"2023-12-31"]
    else:  # Kelly 0.5x
        returns = prices.pct_change()
        position_sizes = apply_kelly_sizing(
            signals=signals, returns=returns, fraction=0.5, max_leverage=2.0, lookback=252
        ).loc["2018-01-01":"2023-12-31"]

    cost_model = TransactionCostModel.for_btc()
    mc_results = block_bootstrap_monte_carlo(test_prices, signals_test, position_sizes, cost_model, n_simulations=1000)

    logger.info(f"\nMonte Carlo Results ({best_strategy}):")
    logger.info(f"  Mean Sharpe:   {mc_results['mean_sharpe']:.3f}")
    logger.info(f"  95% CI:        [{mc_results['ci_lower']:.3f}, {mc_results['ci_upper']:.3f}]")
    logger.info(f"  Win Rate:      {mc_results['win_rate'] * 100:.1f}%")

    # Validation criteria check
    logger.info(f"\n{'=' * 70}")
    logger.info("VALIDATION CRITERIA CHECK")
    logger.info(f"{'=' * 70}")

    target_sharpe = 0.85
    target_mc_win_rate = 0.75
    target_positive_years = 5

    sharpe_pass = results[best_strategy]["mean_sharpe"] >= target_sharpe
    mc_pass = mc_results["win_rate"] >= target_mc_win_rate
    years_pass = results[best_strategy]["positive_count"] >= target_positive_years

    logger.info(
        f"\n✓ Sharpe ≥{target_sharpe}:     {results[best_strategy]['mean_sharpe']:.3f} {'✓ PASS' if sharpe_pass else '✗ FAIL'}"
    )
    logger.info(
        f"✓ Monte Carlo ≥{target_mc_win_rate * 100:.0f}%: {mc_results['win_rate'] * 100:.1f}% {'✓ PASS' if mc_pass else '✗ FAIL'}"
    )
    logger.info(
        f"✓ Positive ≥{target_positive_years}/6:      {results[best_strategy]['positive_count']}/6 {'✓ PASS' if years_pass else '✗ FAIL'}"
    )

    all_pass = sharpe_pass and mc_pass and years_pass

    if all_pass:
        logger.info(f"\n{'=' * 70}")
        logger.info("✓ SUCCESS: All validation criteria passed!")
        logger.info(f"{'=' * 70}")
    else:
        logger.info(f"\n{'=' * 70}")
        logger.info("✗ FAIL: Some validation criteria not met")
        logger.info(f"{'=' * 70}")

    # Save results
    output_path = Path("results/validation/kelly_criterion_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "yearly_walkforward": results,
        "best_strategy": best_strategy,
        "monte_carlo": mc_results,
        "validation_criteria": {
            "target_sharpe": float(target_sharpe),
            "target_mc_win_rate": float(target_mc_win_rate),
            "target_positive_years": int(target_positive_years),
            "sharpe_pass": bool(sharpe_pass),
            "mc_pass": bool(mc_pass),
            "years_pass": bool(years_pass),
            "all_pass": bool(all_pass),
        },
        "improvement_vs_baseline": {
            "baseline_sharpe": 0.772,  # Multi-TF fixed 100% from previous validation
            "kelly_sharpe": results[best_strategy]["mean_sharpe"],
            "delta_sharpe": results[best_strategy]["mean_sharpe"] - 0.772,
            "pct_improvement": (results[best_strategy]["mean_sharpe"] / 0.772 - 1) * 100,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n✓ Results saved to: {output_path}")

    # Final verdict
    baseline_sharpe = 0.772
    improvement_pct = (results[best_strategy]["mean_sharpe"] / baseline_sharpe - 1) * 100

    logger.info(f"\n{'=' * 70}")
    logger.info("FINAL VERDICT")
    logger.info(f"{'=' * 70}")
    logger.info(f"Baseline (Fixed 100%):     Sharpe {baseline_sharpe:.3f}")
    logger.info(f"Best Kelly Strategy:       Sharpe {results[best_strategy]['mean_sharpe']:.3f} ({best_strategy})")
    logger.info(f"Improvement:               {improvement_pct:+.1f}%")

    if results[best_strategy]["mean_sharpe"] >= target_sharpe:
        logger.info(f"\n✓ Target achieved: Sharpe {results[best_strategy]['mean_sharpe']:.3f} ≥ {target_sharpe}")
    else:
        logger.info(f"\n✗ Target missed: Sharpe {results[best_strategy]['mean_sharpe']:.3f} < {target_sharpe}")


if __name__ == "__main__":
    main()
