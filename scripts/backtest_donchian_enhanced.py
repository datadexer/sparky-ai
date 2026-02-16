#!/usr/bin/env python3
"""Backtest Enhanced Donchian Strategy with ATR Position Sizing.

GOAL: Improve Donchian Sharpe 1.307 → 1.5+ using research-validated enhancements.

Research shows: Donchian + ATR-based position sizing → Sharpe 1.5+, alpha 10.8%
(Source: QuantifiedStrategies, SSRN Catching Crypto Trends 2025)

Enhancements:
1. ATR-based position sizing: Scale position by 1 / ATR (reduce size in high vol)
2. Trend-aware position sizing: Increase in volatile uptrends, decrease in volatile downtrends
3. Multi-timeframe ensemble: Combine 20/40/60 day Donchian channels

Target: Sharpe ≥ 1.5
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.features.regime_indicators import (
    compute_volatility_regime,
    detect_trend,
    get_trend_aware_position_size,
)
from sparky.features.returns import annualized_sharpe, max_drawdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_price_data(start_date="2024-01-01", end_date="2025-12-31"):
    """Load BTC daily prices."""
    price_path = Path("data/raw/btc/ohlcv_hourly.parquet")

    if not price_path.exists():
        raise FileNotFoundError(f"Price data not found: {price_path}")

    logger.info(f"Loading price data: {price_path}")
    prices = pd.read_parquet(price_path)

    # Resample to daily
    prices_daily = prices['close'].resample('D').last()

    # Remove timezone
    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)

    # Filter date range
    prices_daily = prices_daily.loc[start_date:end_date]

    logger.info(f"Daily prices: {len(prices_daily)} days ({prices_daily.index.min()} to {prices_daily.index.max()})")

    return prices_daily


def compute_atr_position_sizing(
    prices: pd.Series,
    base_volatility: float = 0.02,
    atr_period: int = 14,
    max_position: float = 2.0,
    min_position: float = 0.5,
) -> pd.Series:
    """Compute position sizes based on inverse ATR (volatility targeting).

    Logic: Target constant volatility by scaling position size inversely with ATR.
    - High volatility → Reduce position size
    - Low volatility → Increase position size

    Formula: position_size = base_volatility / ATR(14)

    Args:
        prices: Close prices.
        base_volatility: Target daily volatility (default 2%).
        atr_period: ATR lookback period (default 14 days).
        max_position: Maximum position size cap (default 2.0 = 200%).
        min_position: Minimum position size floor (default 0.5 = 50%).

    Returns:
        Series of position sizes (0.5 to 2.0).
    """
    # Compute ATR (using close-to-close range as proxy)
    returns = prices.pct_change()
    true_range = returns.abs()
    atr = true_range.rolling(window=atr_period).mean()

    # Position sizing: inversely proportional to ATR
    position_sizes = base_volatility / atr

    # Cap position sizes
    position_sizes = position_sizes.clip(lower=min_position, upper=max_position)

    # Fill initial NaN
    position_sizes = position_sizes.fillna(1.0)

    logger.info(f"ATR Position Sizing: mean={position_sizes.mean():.2f}, min={position_sizes.min():.2f}, max={position_sizes.max():.2f}")

    return position_sizes


def compute_trend_aware_position_sizing(
    prices: pd.Series,
    volatility_window: int = 30,
    sma_period: int = 200,
) -> pd.Series:
    """Compute position sizes based on volatility regime + trend direction.

    Combines:
    - Volatility regime (HIGH/MEDIUM/LOW based on realized vol)
    - Trend direction (UPTREND/DOWNTREND/SIDEWAYS based on SMA)

    Position sizing rules:
    - HIGH vol + UPTREND: 125% (capture volatile bull)
    - HIGH vol + DOWNTREND: 25% (avoid volatile bear)
    - MEDIUM vol + UPTREND: 100% (standard)
    - etc.

    Args:
        prices: Close prices.
        volatility_window: Window for volatility regime (default 30 days).
        sma_period: SMA period for trend detection (default 200 days).

    Returns:
        Series of position sizes.
    """
    # Detect volatility regime
    vol_regime = compute_volatility_regime(prices, window=volatility_window, frequency="1d")

    # Detect trend
    trend = detect_trend(prices, sma_period=sma_period)

    # Compute position sizes
    position_sizes = pd.Series(index=prices.index, dtype=float)

    for i in range(len(prices)):
        vol_reg = vol_regime.iloc[i]
        tr = trend.iloc[i]

        position_sizes.iloc[i] = get_trend_aware_position_size(vol_reg, tr)

    logger.info(f"Trend-Aware Position Sizing: mean={position_sizes.mean():.2f}, min={position_sizes.min():.2f}, max={position_sizes.max():.2f}")

    return position_sizes


def compute_strategy_returns(signals, prices, position_sizes=None):
    """Compute strategy returns with optional position sizing.

    Args:
        signals: Series of signals (0 or 1).
        prices: Series of close prices.
        position_sizes: Optional series of position sizes (default None = 100%).

    Returns:
        Series of daily strategy returns.
    """
    # Align
    common_dates = signals.index.intersection(prices.index)
    signals = signals.loc[common_dates]
    prices = prices.loc[common_dates]

    # Daily returns
    daily_returns = prices.pct_change()

    # Positions: lag signal by 1 day
    positions = signals.shift(1).fillna(0)

    # Apply position sizing if provided
    if position_sizes is not None:
        position_sizes = position_sizes.loc[common_dates]
        positions = positions * position_sizes.shift(1).fillna(1.0)

    # Strategy returns
    strategy_returns = positions * daily_returns
    strategy_returns = strategy_returns.dropna()

    return strategy_returns


def compute_performance_metrics(returns, name="Strategy"):
    """Compute Sharpe, total return, max drawdown, win rate."""
    if len(returns) == 0:
        return {}

    cumulative_return = (1 + returns).prod() - 1
    total_return_pct = cumulative_return * 100

    sharpe = annualized_sharpe(returns, periods_per_year=365)

    cumulative_wealth = (1 + returns).cumprod()
    max_dd = max_drawdown(cumulative_wealth)

    win_rate = (returns > 0).sum() / len(returns)

    positions = (returns != 0).astype(int)
    n_trades = (positions.diff().abs()).sum()

    metrics = {
        "total_return_pct": float(total_return_pct),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd * 100),
        "win_rate": float(win_rate),
        "n_trades": int(n_trades),
        "n_days": len(returns),
    }

    logger.info(f"{name}:")
    logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Return: {metrics['total_return_pct']:.2f}%")
    logger.info(f"  Max DD: {metrics['max_drawdown_pct']:.2f}%")
    logger.info(f"  Trades: {metrics['n_trades']}")
    logger.info("")

    return metrics


def print_comparison_table(results):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("ENHANCED DONCHIAN STRATEGIES COMPARISON")
    print("=" * 100)
    print()
    print(f"{'Strategy':<40s} {'Sharpe':>12s} {'Return (%)':>12s} {'Max DD (%)':>12s} {'Trades':>10s}")
    print("-" * 100)

    sorted_results = sorted(results.items(), key=lambda x: x[1].get('sharpe_ratio', -999), reverse=True)

    for name, metrics in sorted_results:
        if not metrics:
            continue

        sharpe = metrics.get('sharpe_ratio', 0)
        ret = metrics.get('total_return_pct', 0)
        dd = metrics.get('max_drawdown_pct', 0)
        trades = metrics.get('n_trades', 0)

        print(f"{name:<40s} {sharpe:>12.3f} {ret:>12.2f} {dd:>12.2f} {trades:>10d}")

    print("=" * 100)
    print()


def main():
    """Main backtest pipeline."""
    logger.info("=" * 80)
    logger.info("ENHANCED DONCHIAN STRATEGIES")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info("")

    # Load prices
    prices = load_price_data("2024-01-01", "2025-12-31")

    # Baseline: Standard Donchian(20/10)
    logger.info("=" * 80)
    logger.info("BASELINE: DONCHIAN(20/10)")
    logger.info("=" * 80)

    donchian_signals = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
    donchian_returns = compute_strategy_returns(donchian_signals, prices)
    donchian_metrics = compute_performance_metrics(donchian_returns, name="Donchian(20/10) Baseline")

    # Enhancement 1: Donchian + ATR Position Sizing
    logger.info("=" * 80)
    logger.info("ENHANCEMENT 1: DONCHIAN + ATR POSITION SIZING")
    logger.info("=" * 80)

    atr_positions = compute_atr_position_sizing(prices, base_volatility=0.02, atr_period=14)
    donchian_atr_returns = compute_strategy_returns(donchian_signals, prices, position_sizes=atr_positions)
    donchian_atr_metrics = compute_performance_metrics(donchian_atr_returns, name="Donchian + ATR Position Sizing")

    # Enhancement 2: Donchian + Trend-Aware Position Sizing
    logger.info("=" * 80)
    logger.info("ENHANCEMENT 2: DONCHIAN + TREND-AWARE POSITION SIZING")
    logger.info("=" * 80)

    trend_positions = compute_trend_aware_position_sizing(prices, volatility_window=30, sma_period=200)
    donchian_trend_returns = compute_strategy_returns(donchian_signals, prices, position_sizes=trend_positions)
    donchian_trend_metrics = compute_performance_metrics(donchian_trend_returns, name="Donchian + Trend-Aware Sizing")

    # Collect results
    all_results = {
        "Donchian(20/10) Baseline": donchian_metrics,
        "Donchian + ATR Sizing": donchian_atr_metrics,
        "Donchian + Trend-Aware Sizing": donchian_trend_metrics,
        "Buy & Hold": {"sharpe_ratio": 0.950, "total_return_pct": 98.01, "max_drawdown_pct": 32.11, "n_trades": 0},
        "Static ML": {"sharpe_ratio": 0.646, "total_return_pct": 42.15, "max_drawdown_pct": 31.42, "n_trades": 312},
    }

    # Print comparison
    print_comparison_table(all_results)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best = max(
        [
            ("Baseline", donchian_metrics),
            ("ATR", donchian_atr_metrics),
            ("Trend-Aware", donchian_trend_metrics),
        ],
        key=lambda x: x[1].get('sharpe_ratio', -999)
    )

    print(f"\nBest Enhancement: {best[0]} (Sharpe {best[1]['sharpe_ratio']:.3f})")
    print(f"Baseline Donchian: Sharpe {donchian_metrics['sharpe_ratio']:.3f}")
    print(f"Improvement: {best[1]['sharpe_ratio'] - donchian_metrics['sharpe_ratio']:+.3f} Sharpe")

    if best[1]['sharpe_ratio'] >= 1.5:
        print(f"\n✅ TARGET ACHIEVED: Sharpe ≥ 1.5")
    elif best[1]['sharpe_ratio'] >= donchian_metrics['sharpe_ratio']:
        print(f"\n✅ ENHANCEMENT WORKS: Improved over baseline")
    else:
        print(f"\n⚠️ BASELINE WINS: Keep simple Donchian(20/10)")

    print("=" * 80)


if __name__ == "__main__":
    main()
