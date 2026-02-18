#!/usr/bin/env python3
"""
Recompute Multi-Timeframe Ensemble metrics with block bootstrap Monte Carlo.

CRITICAL FIX:
- OLD: Simple resampling Monte Carlo (destroys autocorrelation) → inflated 83% win rate
- NEW: Block bootstrap Monte Carlo (preserves momentum structure) → realistic win rate

Expected impact: Monte Carlo win rate drops from 83% to 70-75%
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sparky.backtest.statistics import BacktestStatistics
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.simple_baselines import donchian_channel_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_prices():
    """Load BTC daily prices."""
    price_path = Path("data/raw/btc/ohlcv_hourly.parquet")
    prices = pd.read_parquet(price_path)
    prices_daily = prices["close"].resample("D").last()
    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)
    return prices_daily.loc["2017-01-01":"2023-12-31"]


def compute_ensemble_signals(prices):
    """Multi-timeframe ensemble: LONG if 2+ of 3 agree."""
    signals_20 = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
    signals_40 = donchian_channel_strategy(prices, entry_period=40, exit_period=20)
    signals_60 = donchian_channel_strategy(prices, entry_period=60, exit_period=30)
    ensemble = (signals_20 + signals_40 + signals_60) >= 2
    return ensemble.astype(int)


def compute_returns(signals, prices, tc=0.0026):
    """Compute strategy returns with transaction costs."""
    daily_returns = prices.pct_change()
    positions = signals.shift(1).fillna(0)
    strategy_returns = positions * daily_returns
    position_changes = positions.diff().abs()
    transaction_costs = position_changes * tc
    return (strategy_returns - transaction_costs).dropna()


def metrics(returns):
    """Compute strategy metrics."""
    if len(returns) == 0:
        return {}
    rf_daily = 0.045 / 365
    sharpe_rf0 = annualized_sharpe(returns, risk_free_rate=0.0, periods_per_year=365)
    sharpe_rf = annualized_sharpe(returns, risk_free_rate=rf_daily, periods_per_year=365)
    cum_ret = (1 + returns).prod() - 1
    max_dd = max_drawdown((1 + returns).cumprod())
    win_rate = (returns > 0).sum() / len(returns)
    positions = (returns != 0).astype(int)
    n_trades = (positions.diff().abs()).sum()
    return {
        "sharpe": float(sharpe_rf0),
        "sharpe_rf45": float(sharpe_rf),
        "return_pct": float(cum_ret * 100),
        "max_dd_pct": float(max_dd * 100),
        "win_rate": float(win_rate),
        "n_trades": int(n_trades),
    }


def bootstrap_ci(returns, n_resamples=1000):
    """Simple bootstrap CI (for comparison)."""
    sharpe_samples = []
    returns_array = returns.values
    for i in range(n_resamples):
        sample = np.random.choice(returns_array, size=len(returns_array), replace=True)
        sharpe = annualized_sharpe(pd.Series(sample), risk_free_rate=0.0, periods_per_year=365)
        sharpe_samples.append(sharpe)
    sharpe_samples = np.array(sharpe_samples)
    return {
        "mean": float(np.mean(sharpe_samples)),
        "lower": float(np.percentile(sharpe_samples, 2.5)),
        "upper": float(np.percentile(sharpe_samples, 97.5)),
    }


def simple_monte_carlo(strategy_returns, market_returns, n_simulations=1000):
    """OLD: Simple resampling Monte Carlo (for comparison)."""
    strategy_array = strategy_returns.values
    market_array = market_returns.values

    min_len = min(len(strategy_array), len(market_array))
    strategy_array = strategy_array[:min_len]
    market_array = market_array[:min_len]

    wins = 0
    for i in range(n_simulations):
        strategy_sample = np.random.choice(strategy_array, size=len(strategy_array), replace=True)
        market_sample = np.random.choice(market_array, size=len(market_array), replace=True)

        strategy_sharpe = annualized_sharpe(pd.Series(strategy_sample), risk_free_rate=0.0, periods_per_year=365)
        market_sharpe = annualized_sharpe(pd.Series(market_sample), risk_free_rate=0.0, periods_per_year=365)

        if strategy_sharpe > market_sharpe:
            wins += 1

    return {
        "win_rate": wins / n_simulations,
        "wins": wins,
        "total": n_simulations,
    }


def main():
    logger.info("=" * 70)
    logger.info("REVALIDATION: Block Bootstrap Monte Carlo")
    logger.info("=" * 70)

    # Load data
    prices = load_prices()
    logger.info(f"Loaded {len(prices)} days of BTC prices (2017-2023)")

    # Compute signals and returns
    signals = compute_ensemble_signals(prices)
    strategy_returns = compute_returns(signals, prices, tc=0.0026)

    # Buy & Hold returns
    market_returns = prices.pct_change().dropna()

    # Align returns
    common_idx = strategy_returns.index.intersection(market_returns.index)
    strategy_returns = strategy_returns.loc[common_idx]
    market_returns = market_returns.loc[common_idx]

    logger.info(f"Strategy returns: {len(strategy_returns)} days")

    # Compute metrics
    strategy_metrics = metrics(strategy_returns)
    market_metrics = metrics(market_returns)

    logger.info("")
    logger.info("=" * 70)
    logger.info("PERFORMANCE METRICS")
    logger.info("=" * 70)
    logger.info(f"Strategy Sharpe (rf=0): {strategy_metrics['sharpe']:.3f}")
    logger.info(f"Market Sharpe (rf=0): {market_metrics['sharpe']:.3f}")
    logger.info(f"Strategy Return: {strategy_metrics['return_pct']:.1f}%")
    logger.info(f"Market Return: {market_metrics['return_pct']:.1f}%")

    # Bootstrap CI
    logger.info("")
    logger.info("=" * 70)
    logger.info("BOOTSTRAP CI (1000 resamples)")
    logger.info("=" * 70)
    boot_ci = bootstrap_ci(strategy_returns, n_resamples=1000)
    logger.info(f"Mean: {boot_ci['mean']:.3f}")
    logger.info(f"95% CI: [{boot_ci['lower']:.3f}, {boot_ci['upper']:.3f}]")

    # OLD: Simple Monte Carlo
    logger.info("")
    logger.info("=" * 70)
    logger.info("OLD: SIMPLE RESAMPLING MONTE CARLO (1000 simulations)")
    logger.info("=" * 70)
    simple_mc = simple_monte_carlo(strategy_returns, market_returns, n_simulations=1000)
    logger.info(f"Win rate: {simple_mc['win_rate'] * 100:.1f}% ({simple_mc['wins']}/{simple_mc['total']} trials)")

    # NEW: Block Bootstrap Monte Carlo
    logger.info("")
    logger.info("=" * 70)
    logger.info("NEW: BLOCK BOOTSTRAP MONTE CARLO (1000 simulations)")
    logger.info("=" * 70)

    block_mc = BacktestStatistics.block_bootstrap_monte_carlo(
        strategy_returns=strategy_returns,
        market_returns=market_returns,
        n_simulations=1000,
        block_size=None,  # Auto: sqrt(n) ≈ 50 for 2555 days
        risk_free_rate=0.0,
        periods_per_year=365,
    )

    logger.info(f"Block size: {block_mc['block_size']} days (auto-selected via sqrt(n) rule)")
    logger.info(f"Win rate: {block_mc['win_rate'] * 100:.1f}% ({block_mc['wins']}/{block_mc['n_simulations']} trials)")
    logger.info(f"Baseline strategy Sharpe: {block_mc['baseline_strategy_sharpe']:.3f}")
    logger.info(f"Baseline market Sharpe: {block_mc['baseline_market_sharpe']:.3f}")

    # Comparison
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON: Simple vs Block Bootstrap")
    logger.info("=" * 70)
    logger.info(f"Simple resampling win rate: {simple_mc['win_rate'] * 100:.1f}%")
    logger.info(f"Block bootstrap win rate: {block_mc['win_rate'] * 100:.1f}%")
    degradation = (simple_mc["win_rate"] - block_mc["win_rate"]) * 100
    logger.info(f"Degradation: {degradation:.1f} percentage points")

    if degradation > 0:
        logger.info("✅ Block bootstrap is more conservative (as expected)")
    else:
        logger.info("⚠️ Block bootstrap gave HIGHER win rate (unexpected)")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "period": "2017-2023",
        "strategy": "Multi-Timeframe Donchian Ensemble (20/40/60)",
        "metrics": {
            "strategy": strategy_metrics,
            "market": market_metrics,
        },
        "bootstrap_ci": boot_ci,
        "monte_carlo": {
            "simple_resampling": simple_mc,
            "block_bootstrap": block_mc,
            "degradation_percentage_points": float(degradation),
        },
        "conclusion": {
            "sharpe": strategy_metrics["sharpe"],
            "monte_carlo_win_rate": block_mc["win_rate"],
            "bootstrap_ci_lower": boot_ci["lower"],
            "honest_assessment": "Block bootstrap gives more realistic uncertainty estimates",
        },
    }

    output_path = Path("results/validation/block_bootstrap_revalidation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info(f"Results saved to: {output_path}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("REVALIDATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
