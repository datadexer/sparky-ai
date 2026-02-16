#!/usr/bin/env python3
"""Backtest Simple Baseline Strategies on 2024-2025 Holdout.

Test three simple trend-following strategies to establish a "complexity floor":
1. SMA(200) Crossover
2. Donchian Channel Breakout (20/10)
3. ATR-Filtered Momentum

Research shows simple Donchian + ATR can achieve Sharpe 1.5+ on Bitcoin.
If these simple strategies beat our ML models, we're overcomplicating.

Baseline comparison:
- Buy & Hold: Sharpe 0.950
- Static ML: Sharpe 0.646
- Regime-Aware ML: Sharpe 0.158 (failed)

Target: Find if simple strategies beat Sharpe 0.646
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sparky.models.simple_baselines import (
    sma_crossover_strategy,
    donchian_channel_strategy,
    atr_filtered_momentum_strategy,
)
from sparky.features.returns import annualized_sharpe, max_drawdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_price_data(start_date="2024-01-01", end_date="2025-12-31"):
    """Load BTC daily prices for holdout period."""
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)

    price_path = Path("data/raw/btc/ohlcv_hourly.parquet")

    if not price_path.exists():
        raise FileNotFoundError(f"Price data not found: {price_path}")

    logger.info(f"Loading price data: {price_path}")
    prices = pd.read_parquet(price_path)

    # Resample hourly to daily (close of day)
    prices_daily = prices['close'].resample('D').last()

    # Remove timezone
    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)

    # Filter to holdout period
    prices_daily = prices_daily.loc[start_date:end_date]

    logger.info(f"Daily prices: {len(prices_daily)} days ({prices_daily.index.min()} to {prices_daily.index.max()})")
    logger.info(f"Price range: ${prices_daily.min():.2f} - ${prices_daily.max():.2f}")
    logger.info("")

    return prices_daily


def compute_strategy_returns(signals, prices):
    """Compute strategy returns.

    Args:
        signals: Series of signals (0 or 1), indexed by date
        prices: Series of close prices, same index

    Returns:
        Series of daily strategy returns
    """
    # Align signals and prices
    common_dates = signals.index.intersection(prices.index)
    signals = signals.loc[common_dates]
    prices = prices.loc[common_dates]

    # Compute daily returns
    daily_returns = prices.pct_change()

    # Strategy returns: position * market return
    # Lag signal by 1 day (today's signal determines tomorrow's position)
    positions = signals.shift(1).fillna(0)
    strategy_returns = positions * daily_returns

    # Remove NaN (first day)
    strategy_returns = strategy_returns.dropna()

    return strategy_returns


def compute_performance_metrics(returns, name="Strategy"):
    """Compute Sharpe, total return, max drawdown, win rate."""
    if len(returns) == 0:
        logger.warning(f"{name}: No returns to compute metrics")
        return {}

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

    logger.info(f"{name}:")
    logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    logger.info(f"  Win Rate: {metrics['win_rate']:.2f}")
    logger.info(f"  Trades: {metrics['n_trades']}")
    logger.info("")

    return metrics


def compute_baseline_performance(prices):
    """Compute Buy & Hold baseline."""
    logger.info("=" * 80)
    logger.info("BASELINE: BUY & HOLD")
    logger.info("=" * 80)

    daily_returns = prices.pct_change().dropna()
    metrics = compute_performance_metrics(daily_returns, name="Buy & Hold")

    return metrics


def print_comparison_table(results):
    """Print comparison table for all strategies."""
    print("\n" + "=" * 120)
    print("PERFORMANCE COMPARISON: Simple Baselines vs ML vs Buy & Hold")
    print("=" * 120)
    print()
    print(f"{'Strategy':<35s} {'Sharpe':>12s} {'Return (%)':>12s} {'Max DD (%)':>12s} {'Win Rate':>12s} {'Trades':>10s}")
    print("-" * 120)

    # Sort by Sharpe ratio (best first)
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('sharpe_ratio', -999), reverse=True)

    for name, metrics in sorted_results:
        if not metrics:
            continue

        sharpe = metrics.get('sharpe_ratio', 0)
        ret = metrics.get('total_return_pct', 0)
        dd = metrics.get('max_drawdown_pct', 0)
        wr = metrics.get('win_rate', 0)
        trades = metrics.get('n_trades', 0)

        print(f"{name:<35s} {sharpe:>12.3f} {ret:>12.2f} {dd:>12.2f} {wr:>12.2f} {trades:>10d}")

    print("=" * 120)
    print()


def save_results(results, output_path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def main():
    """Main backtest pipeline."""
    logger.info("=" * 80)
    logger.info("BACKTEST: SIMPLE BASELINE STRATEGIES")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info("")

    # Load daily prices for holdout period
    prices = load_price_data("2024-01-01", "2025-12-31")

    # Compute Buy & Hold baseline
    baseline_metrics = compute_baseline_performance(prices)

    # Test Strategy 1: SMA(200) Crossover
    logger.info("=" * 80)
    logger.info("STRATEGY 1: SMA(200) CROSSOVER")
    logger.info("=" * 80)

    sma_signals = sma_crossover_strategy(prices, sma_period=200)
    sma_returns = compute_strategy_returns(sma_signals, prices)
    sma_metrics = compute_performance_metrics(sma_returns, name="SMA(200) Crossover")

    # Test Strategy 2: Donchian Channel Breakout
    logger.info("=" * 80)
    logger.info("STRATEGY 2: DONCHIAN CHANNEL BREAKOUT")
    logger.info("=" * 80)

    donchian_signals = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
    donchian_returns = compute_strategy_returns(donchian_signals, prices)
    donchian_metrics = compute_performance_metrics(donchian_returns, name="Donchian(20/10) Breakout")

    # Test Strategy 3: ATR-Filtered Momentum
    logger.info("=" * 80)
    logger.info("STRATEGY 3: ATR-FILTERED MOMENTUM")
    logger.info("=" * 80)

    atr_signals = atr_filtered_momentum_strategy(prices, momentum_period=30, atr_period=14)
    atr_returns = compute_strategy_returns(atr_signals, prices)
    atr_metrics = compute_performance_metrics(atr_returns, name="ATR-Filtered Momentum")

    # Collect all results
    all_results = {
        "Buy & Hold (Baseline)": baseline_metrics,
        "SMA(200) Crossover": sma_metrics,
        "Donchian(20/10) Breakout": donchian_metrics,
        "ATR-Filtered Momentum": atr_metrics,
        # Add ML baselines for comparison (from previous results)
        "Static ML (Phase 1)": {"sharpe_ratio": 0.646, "total_return_pct": 42.15, "max_drawdown_pct": 31.42, "win_rate": 0.322, "n_trades": 312, "n_days": 730},
        "Regime-Aware ML (Phase 2A)": {"sharpe_ratio": 0.158, "total_return_pct": 2.41, "max_drawdown_pct": 19.44, "win_rate": 0.148, "n_trades": 290, "n_days": 730},
    }

    # Print comparison
    print_comparison_table(all_results)

    # Save results
    results_dict = {
        "holdout_period": {
            "start": "2024-01-01",
            "end": "2025-12-31",
            "n_days": len(prices),
        },
        "strategies": {
            "buy_and_hold": baseline_metrics,
            "sma_crossover": sma_metrics,
            "donchian_breakout": donchian_metrics,
            "atr_momentum": atr_metrics,
        },
        "ml_comparison": {
            "static_ml": {"sharpe_ratio": 0.646, "total_return_pct": 42.15},
            "regime_aware_ml": {"sharpe_ratio": 0.158, "total_return_pct": 2.41},
        },
        "timestamp": datetime.now().isoformat(),
    }

    output_path = Path("results/simple_baselines/backtest_2024_2025_holdout.json")
    save_results(results_dict, output_path)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nHoldout Period: 2024-01-01 to 2025-12-31 ({len(prices)} days)")
    print(f"\nBest Strategy: ", end="")

    best_strategy = max(
        [("SMA(200)", sma_metrics), ("Donchian(20/10)", donchian_metrics), ("ATR-Momentum", atr_metrics)],
        key=lambda x: x[1].get('sharpe_ratio', -999)
    )
    print(f"{best_strategy[0]} (Sharpe {best_strategy[1]['sharpe_ratio']:.3f})")

    print(f"\nComparison to ML:")
    print(f"  Best Simple vs Static ML: {best_strategy[1]['sharpe_ratio']:.3f} vs 0.646")
    print(f"  Best Simple vs Regime ML: {best_strategy[1]['sharpe_ratio']:.3f} vs 0.158")

    if best_strategy[1]['sharpe_ratio'] > 0.646:
        print(f"\n✅ SIMPLE STRATEGIES WIN - We were overcomplicating!")
        print(f"   Best simple strategy beats ML by {best_strategy[1]['sharpe_ratio'] - 0.646:+.3f} Sharpe")
    else:
        print(f"\n⚠️ ML still leads, but simple strategies provide strong baseline")

    print(f"\nResults saved to: {output_path}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
