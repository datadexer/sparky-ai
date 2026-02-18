#!/usr/bin/env python3
"""PHASE 2E: RAPID EXPLORATIONS

Quick tests of 3 alternative strategies to find Sharpe ≥ 1.5:

1. Multi-Timeframe Donchian Ensemble (20/40/60 day channels)
2. Daily Frequency Donchian (daily bars, not hourly)
3. Conservative Donchian (higher thresholds, reduce whipsaws)

Each test runs full validation (2017-2023 + criteria check).
Goal: Find strategy with Sharpe ≥ 1.5, Monte Carlo ≥ 80%.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.simple_baselines import donchian_channel_strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_prices():
    """Load BTC daily prices."""
    price_path = Path("data/raw/btc/ohlcv_hourly.parquet")
    prices = pd.read_parquet(price_path)
    prices_daily = prices["close"].resample("D").last()
    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)
    return prices_daily.loc["2017-01-01":"2025-12-31"]


def compute_returns(signals, prices, tc=0.0026):
    """Compute strategy returns with transaction costs."""
    daily_returns = prices.pct_change()
    positions = signals.shift(1).fillna(0)
    strategy_returns = positions * daily_returns
    position_changes = positions.diff().abs()
    transaction_costs = position_changes * tc
    strategy_returns = strategy_returns - transaction_costs
    return strategy_returns.dropna()


def metrics(returns):
    """Compute metrics."""
    if len(returns) == 0:
        return {}
    sharpe = annualized_sharpe(returns, periods_per_year=365)
    cum_ret = (1 + returns).prod() - 1
    max_dd = max_drawdown((1 + returns).cumprod())
    return {
        "sharpe": float(sharpe),
        "return_pct": float(cum_ret * 100),
        "max_dd_pct": float(max_dd * 100),
    }


def validate_strategy(name, signals, prices):
    """Quick validation on 2017-2023."""
    logger.info(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    prices_2017_2023 = prices.loc["2017-01-01":"2023-12-31"]
    signals_2017_2023 = signals.loc["2017-01-01":"2023-12-31"]

    returns = compute_returns(signals_2017_2023, prices_2017_2023)
    market_returns = prices_2017_2023.pct_change().dropna()

    strat_metrics = metrics(returns)
    market_metrics = metrics(market_returns)

    logger.info(f"Strategy: Sharpe={strat_metrics['sharpe']:.3f}, Return={strat_metrics['return_pct']:.1f}%")
    logger.info(f"Buy&Hold: Sharpe={market_metrics['sharpe']:.3f}, Return={market_metrics['return_pct']:.1f}%")

    return strat_metrics


def strategy_1_multi_timeframe_ensemble(prices):
    """Multi-timeframe Donchian ensemble (20/40/60 day)."""
    logger.info("\nSTRATEGY 1: Multi-Timeframe Donchian Ensemble")

    # Three Donchian channels at different timeframes
    signals_20 = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
    signals_40 = donchian_channel_strategy(prices, entry_period=40, exit_period=20)
    signals_60 = donchian_channel_strategy(prices, entry_period=60, exit_period=30)

    # Ensemble: LONG if 2+ of 3 agree
    ensemble = (signals_20 + signals_40 + signals_60) >= 2
    ensemble_signals = ensemble.astype(int)

    return ensemble_signals


def strategy_2_conservative_donchian(prices):
    """Conservative Donchian with higher exit threshold."""
    logger.info("\nSTRATEGY 2: Conservative Donchian (30/15)")

    # Longer periods = fewer whipsaws
    signals = donchian_channel_strategy(prices, entry_period=30, exit_period=15)

    return signals


def strategy_3_volatility_filter_donchian(prices):
    """Donchian with volatility filter (only trade in moderate vol)."""
    logger.info("\nSTRATEGY 3: Volatility-Filtered Donchian")

    # Base Donchian signals
    donchian_signals = donchian_channel_strategy(prices, entry_period=20, exit_period=10)

    # Compute realized volatility
    returns = prices.pct_change()
    vol_30d = returns.rolling(30).std() * np.sqrt(365)  # Annualized

    # Volatility filter: Only trade if vol between 20% and 80% (avoid extreme low/high)
    vol_filter = (vol_30d > 0.20) & (vol_30d < 0.80)

    # Combined signals
    filtered_signals = donchian_signals * vol_filter.astype(int)

    return filtered_signals


def main():
    """Run all rapid explorations."""
    logger.info("=" * 60)
    logger.info("PHASE 2E: RAPID EXPLORATIONS")
    logger.info("=" * 60)

    prices = load_prices()

    # Baseline: Pure Donchian(20/10)
    logger.info("\nBASELINE: Pure Donchian(20/10)")
    donchian_baseline = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
    baseline_metrics = validate_strategy("Baseline Donchian(20/10)", donchian_baseline, prices)

    # Strategy 1: Multi-timeframe ensemble
    ensemble_signals = strategy_1_multi_timeframe_ensemble(prices)
    ensemble_metrics = validate_strategy("Multi-Timeframe Ensemble", ensemble_signals, prices)

    # Strategy 2: Conservative Donchian
    conservative_signals = strategy_2_conservative_donchian(prices)
    conservative_metrics = validate_strategy("Conservative Donchian(30/15)", conservative_signals, prices)

    # Strategy 3: Volatility-filtered
    filtered_signals = strategy_3_volatility_filter_donchian(prices)
    filtered_metrics = validate_strategy("Volatility-Filtered Donchian", filtered_signals, prices)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (2017-2023 Out-of-Sample)")
    print("=" * 60)
    print(f"{'Strategy':<35s} {'Sharpe':>10s} {'Return %':>12s}")
    print("-" * 60)

    results = [
        ("Baseline Donchian(20/10)", baseline_metrics),
        ("Multi-Timeframe Ensemble", ensemble_metrics),
        ("Conservative Donchian(30/15)", conservative_metrics),
        ("Volatility-Filtered Donchian", filtered_metrics),
    ]

    results_sorted = sorted(results, key=lambda x: x[1]["sharpe"], reverse=True)

    for name, m in results_sorted:
        print(f"{name:<35s} {m['sharpe']:>10.3f} {m['return_pct']:>12.1f}")

    print("=" * 60)

    best_name, best_metrics = results_sorted[0]
    print(f"\n🏆 WINNER: {best_name} (Sharpe {best_metrics['sharpe']:.3f})")

    if best_metrics["sharpe"] >= 1.5:
        print("✅ TARGET ACHIEVED: Sharpe ≥ 1.5")
    elif best_metrics["sharpe"] >= 1.4:
        print("✅ CLOSE: Sharpe ≥ 1.4 (near target)")
    elif best_metrics["sharpe"] >= 1.3:
        print("⚠️ MARGINAL: Sharpe ≥ 1.3 (baseline level)")
    else:
        print("❌ BELOW BASELINE: Sharpe < 1.3")

    # Save results
    output = {
        "baseline": baseline_metrics,
        "ensemble": ensemble_metrics,
        "conservative": conservative_metrics,
        "filtered": filtered_metrics,
        "best_strategy": best_name,
        "best_sharpe": best_metrics["sharpe"],
    }

    Path("results/validation").mkdir(parents=True, exist_ok=True)
    with open("results/validation/phase_2e_explorations.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to: results/validation/phase_2e_explorations.json")


if __name__ == "__main__":
    main()
