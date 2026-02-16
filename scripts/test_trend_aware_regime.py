#!/usr/bin/env python3
"""
Test Trend-Aware Regime Position Sizing for Multi-Timeframe Donchian

HYPOTHESIS: The problem with simple volatility-based sizing is that high volatility
can be GOOD (volatile uptrend) or BAD (volatile downtrend). We need to consider BOTH.

NEW APPROACH (from regime_indicators.py get_trend_aware_position_size):
- HIGH vol + UPTREND: 125% position (capture volatile bull - max exposure)
- HIGH vol + DOWNTREND: 25% position (avoid volatile bear - min exposure)
- HIGH vol + SIDEWAYS: 50% position (avoid whipsaws)
- MEDIUM/LOW vol: Standard rules (50%-100%)

Expected: Better bear market protection without missing bull runs.
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
    ensemble = ((signals_20 + signals_40 + signals_60) > 0).astype(int)
    return ensemble


def strategy_trend_aware_position_sizing(prices):
    """Multi-Timeframe Donchian with trend-aware position sizing.

    Returns:
        Series of FRACTIONAL positions (0.0 to 1.25)
    """
    # Get base signals
    base_signals = strategy_multi_timeframe_baseline(prices)

    # Compute volatility regime and trend
    vol_regime = compute_volatility_regime(prices, window=30, frequency="1d")
    trend = detect_trend(prices, sma_period=200, trend_threshold=0.02)

    # Apply trend-aware position sizing
    position_sizes = pd.Series(0.0, index=prices.index)

    for idx in prices.index:
        if base_signals.loc[idx] == 1:
            vol = vol_regime.loc[idx]
            tr = trend.loc[idx]
            position_sizes.loc[idx] = get_trend_aware_position_size(vol, tr)
        else:
            position_sizes.loc[idx] = 0.0

    return position_sizes


def backtest_fractional_positions(signals, prices, start, end, cost_pct=0.0026):
    """Backtest with fractional position sizes (can exceed 1.0 for leverage)."""
    fold_signals = signals.loc[start:end]
    fold_prices = prices.loc[start:end]

    daily_returns = fold_prices.pct_change()
    positions = fold_signals.shift(1).fillna(0)
    strategy_returns = positions * daily_returns
    position_changes = positions.diff().abs()
    transaction_costs = position_changes * cost_pct
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
    logger.info("TREND-AWARE REGIME POSITION SIZING TEST")
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

    logger.info("\n[2/2] Trend-Aware Position Sizing (25%-125% dynamic)...")
    trend_signals = strategy_trend_aware_position_sizing(prices)

    # Log position distribution
    n_125 = (trend_signals == 1.25).sum()
    n_100 = (trend_signals == 1.00).sum()
    n_75 = (trend_signals == 0.75).sum()
    n_50 = (trend_signals == 0.50).sum()
    n_25 = (trend_signals == 0.25).sum()
    n_flat = (trend_signals == 0.0).sum()

    logger.info(f"Position distribution:")
    logger.info(f"  125% (HIGH vol + UPTREND):     {n_125} days ({n_125/len(trend_signals)*100:.1f}%)")
    logger.info(f"  100% (MEDIUM/LOW + UPTREND):   {n_100} days ({n_100/len(trend_signals)*100:.1f}%)")
    logger.info(f"  75%  (MEDIUM/LOW vol):         {n_75} days ({n_75/len(trend_signals)*100:.1f}%)")
    logger.info(f"  50%  (MEDIUM/HIGH + SIDEWAYS): {n_50} days ({n_50/len(trend_signals)*100:.1f}%)")
    logger.info(f"  25%  (HIGH vol + DOWNTREND):   {n_25} days ({n_25/len(trend_signals)*100:.1f}%)")
    logger.info(f"  0%   (FLAT):                   {n_flat} days ({n_flat/len(trend_signals)*100:.1f}%)")

    # IN-SAMPLE TEST: 2018-2020
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: IN-SAMPLE TEST (2018-2020)")
    logger.info("=" * 80)

    insample_results = {}

    for year in [2018, 2019, 2020]:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        logger.info(f"\n--- Year {year} ---")

        baseline_metrics = backtest_fractional_positions(baseline_signals, prices, start, end)
        logger.info(f"Baseline:     Sharpe {baseline_metrics['sharpe']:.3f}, "
                   f"Return {baseline_metrics['return_pct']:+.1f}%, "
                   f"MaxDD {baseline_metrics['max_dd_pct']:.1f}%")

        trend_metrics = backtest_fractional_positions(trend_signals, prices, start, end)
        logger.info(f"Trend-Aware:  Sharpe {trend_metrics['sharpe']:.3f}, "
                   f"Return {trend_metrics['return_pct']:+.1f}%, "
                   f"MaxDD {trend_metrics['max_dd_pct']:.1f}%")

        delta_sharpe = trend_metrics['sharpe'] - baseline_metrics['sharpe']
        delta_return = trend_metrics['return_pct'] - baseline_metrics['return_pct']
        status = "✅ IMPROVED" if delta_sharpe > 0 else "❌ WORSE"
        logger.info(f"Delta:        Sharpe {delta_sharpe:+.3f}, Return {delta_return:+.1f}% {status}")

        insample_results[year] = {
            "baseline": baseline_metrics,
            "trend_aware": trend_metrics,
            "delta_sharpe": float(delta_sharpe),
            "delta_return": float(delta_return),
        }

    # Aggregate in-sample
    baseline_sharpes = [insample_results[y]["baseline"]["sharpe"] for y in [2018, 2019, 2020]]
    trend_sharpes = [insample_results[y]["trend_aware"]["sharpe"] for y in [2018, 2019, 2020]]

    logger.info("\n" + "=" * 80)
    logger.info("IN-SAMPLE SUMMARY (2018-2020)")
    logger.info("=" * 80)
    logger.info(f"Baseline Mean Sharpe:     {np.mean(baseline_sharpes):.3f}")
    logger.info(f"Trend-Aware Mean Sharpe:  {np.mean(trend_sharpes):.3f}")
    logger.info(f"Delta:                    {np.mean(trend_sharpes) - np.mean(baseline_sharpes):+.3f}")

    if np.mean(trend_sharpes) <= np.mean(baseline_sharpes):
        logger.warning("\n❌ FAILED: Trend-aware does NOT improve in-sample!")
        logger.warning("This approach is NOT working. No point testing out-of-sample.")
        verdict = "❌ FAILED — No in-sample improvement"

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy": "Multi-Timeframe Donchian with Trend-Aware Position Sizing",
            "approach": "Dynamic position sizing (25%-125%) based on volatility regime AND trend",
            "insample": insample_results,
            "out_of_sample": {},
            "summary": {
                "baseline_mean_sharpe": float(np.mean(baseline_sharpes)),
                "trend_aware_mean_sharpe": float(np.mean(trend_sharpes)),
                "delta_sharpe": float(np.mean(trend_sharpes) - np.mean(baseline_sharpes)),
            },
            "verdict": verdict,
        }

        output_dir = Path("results/validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "trend_aware_regime_validation.json"

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"\nResults saved to: {output_file}")
        logger.info("\n" + "=" * 80)
        logger.info("TEST COMPLETE")
        logger.info("=" * 80)
        return

    logger.info("\n✅ SUCCESS: Trend-aware improves in-sample!")
    logger.info("Proceeding to out-of-sample validation...")

    # OUT-OF-SAMPLE TEST: 2021-2023
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: OUT-OF-SAMPLE YEARLY VALIDATION (2021-2023)")
    logger.info("=" * 80)

    oos_results = {}

    for year in [2021, 2022, 2023]:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        logger.info(f"\n--- Year {year} ---")

        baseline_metrics = backtest_fractional_positions(baseline_signals, prices, start, end)
        logger.info(f"Baseline:     Sharpe {baseline_metrics['sharpe']:.3f}, "
                   f"Return {baseline_metrics['return_pct']:+.1f}%, "
                   f"MaxDD {baseline_metrics['max_dd_pct']:.1f}%")

        trend_metrics = backtest_fractional_positions(trend_signals, prices, start, end)
        logger.info(f"Trend-Aware:  Sharpe {trend_metrics['sharpe']:.3f}, "
                   f"Return {trend_metrics['return_pct']:+.1f}%, "
                   f"MaxDD {trend_metrics['max_dd_pct']:.1f}%")

        delta_sharpe = trend_metrics['sharpe'] - baseline_metrics['sharpe']
        delta_return = trend_metrics['return_pct'] - baseline_metrics['return_pct']
        status = "✅ IMPROVED" if delta_sharpe > 0 else "❌ WORSE"
        logger.info(f"Delta:        Sharpe {delta_sharpe:+.3f}, Return {delta_return:+.1f}% {status}")

        oos_results[year] = {
            "baseline": baseline_metrics,
            "trend_aware": trend_metrics,
            "delta_sharpe": float(delta_sharpe),
            "delta_return": float(delta_return),
        }

    # Aggregate out-of-sample
    oos_baseline_sharpes = [oos_results[y]["baseline"]["sharpe"] for y in [2021, 2022, 2023]]
    oos_trend_sharpes = [oos_results[y]["trend_aware"]["sharpe"] for y in [2021, 2022, 2023]]

    logger.info("\n" + "=" * 80)
    logger.info("OUT-OF-SAMPLE SUMMARY (2021-2023)")
    logger.info("=" * 80)
    logger.info(f"Baseline Mean Sharpe:     {np.mean(oos_baseline_sharpes):.3f}")
    logger.info(f"Trend-Aware Mean Sharpe:  {np.mean(oos_trend_sharpes):.3f}")
    logger.info(f"Delta:                    {np.mean(oos_trend_sharpes) - np.mean(oos_baseline_sharpes):+.3f}")

    # Overall comparison
    all_baseline_sharpes = baseline_sharpes + oos_baseline_sharpes
    all_trend_sharpes = trend_sharpes + oos_trend_sharpes

    logger.info("\n" + "=" * 80)
    logger.info("OVERALL SUMMARY (2018-2023, 6 years)")
    logger.info("=" * 80)
    logger.info(f"Baseline Mean Sharpe:     {np.mean(all_baseline_sharpes):.3f}")
    logger.info(f"Trend-Aware Mean Sharpe:  {np.mean(all_trend_sharpes):.3f}")
    logger.info(f"Delta:                    {np.mean(all_trend_sharpes) - np.mean(all_baseline_sharpes):+.3f}")

    # Success criteria
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS CRITERIA CHECK")
    logger.info("=" * 80)

    target_sharpe = 0.85
    baseline_mean = np.mean(all_baseline_sharpes)
    trend_mean = np.mean(all_trend_sharpes)
    improvement_pct = (trend_mean / baseline_mean - 1) * 100 if baseline_mean != 0 else 0

    criteria = {
        "mean_sharpe_gte_0.85": trend_mean >= target_sharpe,
        "improvement_vs_baseline": trend_mean > baseline_mean,
        "bear_market_improvement": oos_results[2022]["delta_sharpe"] > 0,
    }

    logger.info(f"✓ Mean Sharpe ≥ 0.85:        {trend_mean:.3f} {'✅ PASS' if criteria['mean_sharpe_gte_0.85'] else '❌ FAIL'}")
    logger.info(f"✓ Improvement vs baseline:  {improvement_pct:+.1f}% {'✅ PASS' if criteria['improvement_vs_baseline'] else '❌ FAIL'}")
    logger.info(f"✓ Better 2022 bear market:  {oos_results[2022]['delta_sharpe']:+.3f} {'✅ PASS' if criteria['bear_market_improvement'] else '❌ FAIL'}")

    passed = sum(criteria.values())
    total = len(criteria)
    logger.info(f"\nPASSED: {passed}/{total} criteria")

    if passed == total:
        verdict = "✅ SUCCESS — Trend-aware position sizing WORKS!"
    elif passed >= 2:
        verdict = "⚠️ MARGINAL — Some improvement, but below target"
    else:
        verdict = "❌ FAILED — No meaningful improvement"

    logger.info(f"\nVERDICT: {verdict}")

    # Save results
    output_dir = Path("results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "trend_aware_regime_validation.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "Multi-Timeframe Donchian with Trend-Aware Position Sizing",
        "approach": "Dynamic position sizing (25%-125%) based on volatility regime AND trend",
        "insample": insample_results,
        "out_of_sample": oos_results,
        "summary": {
            "baseline_mean_sharpe": float(np.mean(all_baseline_sharpes)),
            "trend_aware_mean_sharpe": float(np.mean(all_trend_sharpes)),
            "delta_sharpe": float(np.mean(all_trend_sharpes) - np.mean(all_baseline_sharpes)),
            "improvement_pct": float(improvement_pct),
        },
        "criteria": {k: bool(v) for k, v in criteria.items()},
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
