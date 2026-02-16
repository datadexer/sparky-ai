#!/usr/bin/env python3
"""
Test Regime-Aware Position Sizing for Multi-Timeframe Donchian

APPROACH: Apply DYNAMIC POSITION SIZING (50%-100%) instead of binary filtering.

Previous failure (STATE.yaml line 147):
- Regime-filtered Donchian went to FLAT during HIGH volatility
- Result: Sharpe -0.350 (79% worse than baseline)
- Problem: Missing too many profitable opportunities

New approach (STRATEGY_REPORT.md lines 117-128):
- Keep signals active in all regimes
- Adjust POSITION SIZE based on volatility:
  - HIGH regime (>60% vol): 50% position (reduce exposure)
  - MEDIUM regime (30-60% vol): 75% position (moderate caution)
  - LOW regime (<30% vol): 100% position (full exposure)

Expected impact: Sharpe 0.772 → 0.85-1.0 (better bear market protection)
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
from sparky.features.regime_indicators import compute_volatility_regime, get_regime_position_size
from sparky.features.returns import annualized_sharpe, max_drawdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_prices():
    """Load BTC daily prices."""
    price_path = Path("data/raw/btc/ohlcv_hourly.parquet")
    prices = pd.read_parquet(price_path)
    prices_daily = prices['close'].resample('D').last()
    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)
    return prices_daily.loc["2017-01-01":"2023-12-31"]


def strategy_multi_timeframe_baseline(prices):
    """Multi-Timeframe Donchian Ensemble (baseline, no regime adjustment)."""
    signals_20 = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
    signals_40 = donchian_channel_strategy(prices, entry_period=40, exit_period=20)
    signals_60 = donchian_channel_strategy(prices, entry_period=60, exit_period=30)

    # Ensemble: LONG if ANY timeframe signals LONG
    ensemble = ((signals_20 + signals_40 + signals_60) > 0).astype(int)
    return ensemble


def strategy_regime_aware_position_sizing(prices):
    """Multi-Timeframe Donchian with regime-aware position sizing.

    Returns:
        Series of FRACTIONAL positions (0.0, 0.5, 0.75, or 1.0)
    """
    # Get base signals
    base_signals = strategy_multi_timeframe_baseline(prices)

    # Compute volatility regime
    regime = compute_volatility_regime(prices, window=30, frequency="1d")

    # Apply position sizing
    position_sizes = pd.Series(0.0, index=prices.index)

    for idx in prices.index:
        if base_signals.loc[idx] == 1:
            regime_label = regime.loc[idx]
            position_sizes.loc[idx] = get_regime_position_size(regime_label)
        else:
            position_sizes.loc[idx] = 0.0

    return position_sizes


def backtest_fractional_positions(signals, prices, start, end, cost_pct=0.0026):
    """Backtest with fractional position sizes.

    Args:
        signals: Series of position sizes (0.0-1.0)
        prices: Price series
        start: Start date
        end: End date
        cost_pct: Round-trip transaction cost (0.26% = 0.0026)

    Returns:
        Dict with metrics
    """
    fold_signals = signals.loc[start:end]
    fold_prices = prices.loc[start:end]

    daily_returns = fold_prices.pct_change()

    # Positions lagged by 1 day (trade on next day's open)
    positions = fold_signals.shift(1).fillna(0)

    # Strategy returns = position_size × market_return
    strategy_returns = positions * daily_returns

    # Transaction costs = |position_change| × cost
    position_changes = positions.diff().abs()
    transaction_costs = position_changes * cost_pct

    # Net returns after costs
    net_returns = (strategy_returns - transaction_costs).dropna()

    if len(net_returns) == 0 or net_returns.std() == 0:
        return {
            "sharpe": 0.0,
            "return_pct": 0.0,
            "max_dd_pct": 0.0,
            "n_days": len(fold_signals),
            "n_trades": 0,
        }

    sharpe = annualized_sharpe(net_returns, risk_free_rate=0.0, periods_per_year=365)
    cum_ret = (1 + net_returns).prod() - 1
    max_dd = max_drawdown((1 + net_returns).cumprod())

    # Count trades (any position change > 0.1)
    n_trades = (position_changes > 0.1).sum()

    return {
        "sharpe": float(sharpe),
        "return_pct": float(cum_ret * 100),
        "max_dd_pct": float(max_dd * 100),
        "n_days": len(fold_signals),
        "n_trades": int(n_trades),
    }


def main():
    logger.info("=" * 80)
    logger.info("REGIME-AWARE POSITION SIZING TEST")
    logger.info("=" * 80)

    # Load data
    logger.info("Loading BTC daily prices...")
    prices = load_prices()
    logger.info(f"Loaded {len(prices)} days (2017-01-01 to 2023-12-31)")

    # Generate signals
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Generate signals")
    logger.info("=" * 80)

    logger.info("\n[1/2] Baseline Multi-Timeframe Donchian (fixed 100% position)...")
    baseline_signals = strategy_multi_timeframe_baseline(prices)
    logger.info(f"Baseline signals: {baseline_signals.sum()} LONG days ({baseline_signals.sum()/len(baseline_signals)*100:.1f}%)")

    logger.info("\n[2/2] Regime-Aware Position Sizing (50%-100% dynamic)...")
    regime_signals = strategy_regime_aware_position_sizing(prices)

    # Log regime distribution
    n_full = (regime_signals == 1.0).sum()
    n_75 = (regime_signals == 0.75).sum()
    n_50 = (regime_signals == 0.50).sum()
    n_flat = (regime_signals == 0.0).sum()
    logger.info(f"Position distribution:")
    logger.info(f"  100% (LOW vol):    {n_full} days ({n_full/len(regime_signals)*100:.1f}%)")
    logger.info(f"  75%  (MEDIUM vol): {n_75} days ({n_75/len(regime_signals)*100:.1f}%)")
    logger.info(f"  50%  (HIGH vol):   {n_50} days ({n_50/len(regime_signals)*100:.1f}%)")
    logger.info(f"  0%   (FLAT):       {n_flat} days ({n_flat/len(regime_signals)*100:.1f}%)")

    # IN-SAMPLE TEST: 2018-2020 (3 years)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: IN-SAMPLE TEST (2018-2020)")
    logger.info("=" * 80)
    logger.info("Objective: Verify regime sizing improves performance on train data")

    insample_results = {}

    for year in [2018, 2019, 2020]:
        start = f"{year}-01-01"
        end = f"{year}-12-31"

        logger.info(f"\n--- Year {year} ---")

        # Baseline
        baseline_metrics = backtest_fractional_positions(
            baseline_signals, prices, start, end
        )
        logger.info(f"Baseline:    Sharpe {baseline_metrics['sharpe']:.3f}, "
                   f"Return {baseline_metrics['return_pct']:+.1f}%, "
                   f"MaxDD {baseline_metrics['max_dd_pct']:.1f}%")

        # Regime-aware
        regime_metrics = backtest_fractional_positions(
            regime_signals, prices, start, end
        )
        logger.info(f"Regime-Aware: Sharpe {regime_metrics['sharpe']:.3f}, "
                   f"Return {regime_metrics['return_pct']:+.1f}%, "
                   f"MaxDD {regime_metrics['max_dd_pct']:.1f}%")

        # Delta
        delta_sharpe = regime_metrics['sharpe'] - baseline_metrics['sharpe']
        delta_return = regime_metrics['return_pct'] - baseline_metrics['return_pct']

        status = "✅ IMPROVED" if delta_sharpe > 0 else "❌ WORSE"
        logger.info(f"Delta:       Sharpe {delta_sharpe:+.3f}, "
                   f"Return {delta_return:+.1f}% {status}")

        insample_results[year] = {
            "baseline": baseline_metrics,
            "regime_aware": regime_metrics,
            "delta_sharpe": float(delta_sharpe),
            "delta_return": float(delta_return),
        }

    # Aggregate in-sample
    baseline_sharpes = [insample_results[y]["baseline"]["sharpe"] for y in [2018, 2019, 2020]]
    regime_sharpes = [insample_results[y]["regime_aware"]["sharpe"] for y in [2018, 2019, 2020]]

    logger.info("\n" + "=" * 80)
    logger.info("IN-SAMPLE SUMMARY (2018-2020)")
    logger.info("=" * 80)
    logger.info(f"Baseline Mean Sharpe:     {np.mean(baseline_sharpes):.3f}")
    logger.info(f"Regime-Aware Mean Sharpe: {np.mean(regime_sharpes):.3f}")
    logger.info(f"Delta:                    {np.mean(regime_sharpes) - np.mean(baseline_sharpes):+.3f}")

    if np.mean(regime_sharpes) <= np.mean(baseline_sharpes):
        logger.warning("\n❌ FAILED: Regime-aware does NOT improve in-sample!")
        logger.warning("This approach is NOT working. Debug before out-of-sample testing.")
        return
    else:
        logger.info("\n✅ SUCCESS: Regime-aware improves in-sample!")
        logger.info("Proceeding to out-of-sample validation...")

    # OUT-OF-SAMPLE TEST: 2021-2023 (yearly walk-forward)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: OUT-OF-SAMPLE YEARLY VALIDATION (2021-2023)")
    logger.info("=" * 80)

    oos_results = {}

    for year in [2021, 2022, 2023]:
        start = f"{year}-01-01"
        end = f"{year}-12-31"

        logger.info(f"\n--- Year {year} ---")

        # Baseline
        baseline_metrics = backtest_fractional_positions(
            baseline_signals, prices, start, end
        )
        logger.info(f"Baseline:     Sharpe {baseline_metrics['sharpe']:.3f}, "
                   f"Return {baseline_metrics['return_pct']:+.1f}%, "
                   f"MaxDD {baseline_metrics['max_dd_pct']:.1f}%")

        # Regime-aware
        regime_metrics = backtest_fractional_positions(
            regime_signals, prices, start, end
        )
        logger.info(f"Regime-Aware: Sharpe {regime_metrics['sharpe']:.3f}, "
                   f"Return {regime_metrics['return_pct']:+.1f}%, "
                   f"MaxDD {regime_metrics['max_dd_pct']:.1f}%")

        # Delta
        delta_sharpe = regime_metrics['sharpe'] - baseline_metrics['sharpe']
        delta_return = regime_metrics['return_pct'] - baseline_metrics['return_pct']

        status = "✅ IMPROVED" if delta_sharpe > 0 else "❌ WORSE"
        logger.info(f"Delta:        Sharpe {delta_sharpe:+.3f}, "
                   f"Return {delta_return:+.1f}% {status}")

        oos_results[year] = {
            "baseline": baseline_metrics,
            "regime_aware": regime_metrics,
            "delta_sharpe": float(delta_sharpe),
            "delta_return": float(delta_return),
        }

    # Aggregate out-of-sample
    oos_baseline_sharpes = [oos_results[y]["baseline"]["sharpe"] for y in [2021, 2022, 2023]]
    oos_regime_sharpes = [oos_results[y]["regime_aware"]["sharpe"] for y in [2021, 2022, 2023]]

    logger.info("\n" + "=" * 80)
    logger.info("OUT-OF-SAMPLE SUMMARY (2021-2023)")
    logger.info("=" * 80)
    logger.info(f"Baseline Mean Sharpe:     {np.mean(oos_baseline_sharpes):.3f}")
    logger.info(f"Regime-Aware Mean Sharpe: {np.mean(oos_regime_sharpes):.3f}")
    logger.info(f"Delta:                    {np.mean(oos_regime_sharpes) - np.mean(oos_baseline_sharpes):+.3f}")

    # Overall comparison (2018-2023)
    all_baseline_sharpes = baseline_sharpes + oos_baseline_sharpes
    all_regime_sharpes = regime_sharpes + oos_regime_sharpes

    logger.info("\n" + "=" * 80)
    logger.info("OVERALL SUMMARY (2018-2023, 6 years)")
    logger.info("=" * 80)
    logger.info(f"Baseline Mean Sharpe:     {np.mean(all_baseline_sharpes):.3f}")
    logger.info(f"Regime-Aware Mean Sharpe: {np.mean(all_regime_sharpes):.3f}")
    logger.info(f"Delta:                    {np.mean(all_regime_sharpes) - np.mean(all_baseline_sharpes):+.3f}")

    # Success criteria
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS CRITERIA CHECK")
    logger.info("=" * 80)

    target_sharpe = 0.85
    baseline_mean = np.mean(all_baseline_sharpes)
    regime_mean = np.mean(all_regime_sharpes)
    improvement_pct = (regime_mean / baseline_mean - 1) * 100 if baseline_mean != 0 else 0

    criteria = {
        "mean_sharpe_gte_0.85": regime_mean >= target_sharpe,
        "improvement_vs_baseline": regime_mean > baseline_mean,
        "bear_market_improvement": oos_results[2022]["delta_sharpe"] > 0,
    }

    logger.info(f"✓ Mean Sharpe ≥ 0.85:        {regime_mean:.3f} {'✅ PASS' if criteria['mean_sharpe_gte_0.85'] else '❌ FAIL'}")
    logger.info(f"✓ Improvement vs baseline:  {improvement_pct:+.1f}% {'✅ PASS' if criteria['improvement_vs_baseline'] else '❌ FAIL'}")
    logger.info(f"✓ Better 2022 bear market:  {oos_results[2022]['delta_sharpe']:+.3f} {'✅ PASS' if criteria['bear_market_improvement'] else '❌ FAIL'}")

    passed = sum(criteria.values())
    total = len(criteria)

    logger.info(f"\nPASSED: {passed}/{total} criteria")

    if passed == total:
        verdict = "✅ SUCCESS — Regime-aware position sizing WORKS!"
    elif passed >= 2:
        verdict = "⚠️ MARGINAL — Some improvement, but below target"
    else:
        verdict = "❌ FAILED — No meaningful improvement"

    logger.info(f"\nVERDICT: {verdict}")

    # Save results
    output_dir = Path("results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "regime_position_sizing_validation.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "Multi-Timeframe Donchian with Regime-Aware Position Sizing",
        "approach": "Dynamic position sizing (50%-100%) based on volatility regime",
        "insample": insample_results,
        "out_of_sample": oos_results,
        "summary": {
            "baseline_mean_sharpe": float(np.mean(all_baseline_sharpes)),
            "regime_aware_mean_sharpe": float(np.mean(all_regime_sharpes)),
            "delta_sharpe": float(np.mean(all_regime_sharpes) - np.mean(all_baseline_sharpes)),
            "improvement_pct": float(improvement_pct),
        },
        "criteria": {k: bool(v) for k, v in criteria.items()},  # Convert to bool for JSON
        "verdict": verdict,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
