#!/usr/bin/env python3
"""
Unified Strategy Validation Framework

Systematically tests multiple strategy classes through same rigorous walk-forward validation.

Strategies to test:
1. Pure Donchian(20/10) - simpler baseline
2. Conservative Donchian(30/15) - longer periods
3. RSI Mean Reversion - oversold/overbought
4. Bollinger Band Mean Reversion
5. SMA Crossover - momentum
6. Buy & Hold - benchmark

All tested with:
- 18-fold walk-forward validation (6 yearly + 12 quarterly)
- Transaction costs: 0.26%
- Comparison metrics: Sharpe, max DD, win rate
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sparky.models.simple_baselines import donchian_channel_strategy
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


def strategy_pure_donchian(prices: pd.Series) -> pd.Series:
    """Pure Donchian(20/10) - simplest baseline."""
    return donchian_channel_strategy(prices, entry_period=20, exit_period=10)


def strategy_conservative_donchian(prices: pd.Series) -> pd.Series:
    """Conservative Donchian(30/15) - longer periods for less whipsaw."""
    return donchian_channel_strategy(prices, entry_period=30, exit_period=15)


def strategy_rsi_mean_reversion(prices: pd.Series, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """RSI mean reversion: LONG when oversold, EXIT when overbought."""
    # Compute RSI
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Generate signals
    signals = pd.Series(0, index=prices.index, dtype=int)
    in_position = False

    for i in range(len(prices)):
        if pd.isna(rsi.iloc[i]):
            signals.iloc[i] = 0
            continue

        if not in_position:
            # Entry: RSI oversold
            if rsi.iloc[i] < oversold:
                in_position = True
                signals.iloc[i] = 1
            else:
                signals.iloc[i] = 0
        else:
            # Exit: RSI overbought
            if rsi.iloc[i] > overbought:
                in_position = False
                signals.iloc[i] = 0
            else:
                # Hold position
                signals.iloc[i] = 1

    logger.info(f"RSI({period}): {signals.sum()} LONG days ({signals.sum()/len(signals)*100:.1f}%)")
    return signals


def strategy_bollinger_mean_reversion(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger Band mean reversion: LONG at lower band, EXIT at upper band."""
    # Compute Bollinger Bands
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std

    # Generate signals
    signals = pd.Series(0, index=prices.index, dtype=int)
    in_position = False

    for i in range(len(prices)):
        if pd.isna(lower_band.iloc[i]) or pd.isna(upper_band.iloc[i]):
            signals.iloc[i] = 0
            continue

        if not in_position:
            # Entry: Price below lower band (oversold)
            if prices.iloc[i] < lower_band.iloc[i]:
                in_position = True
                signals.iloc[i] = 1
            else:
                signals.iloc[i] = 0
        else:
            # Exit: Price above upper band (overbought)
            if prices.iloc[i] > upper_band.iloc[i]:
                in_position = False
                signals.iloc[i] = 0
            else:
                # Hold position
                signals.iloc[i] = 1

    logger.info(f"Bollinger({period},{num_std}σ): {signals.sum()} LONG days ({signals.sum()/len(signals)*100:.1f}%)")
    return signals


def strategy_sma_crossover(prices: pd.Series, fast_period: int = 50, slow_period: int = 200) -> pd.Series:
    """SMA crossover: LONG when fast > slow, FLAT when fast < slow."""
    fast_sma = prices.rolling(window=fast_period).mean()
    slow_sma = prices.rolling(window=slow_period).mean()

    # LONG when fast SMA above slow SMA
    signals = (fast_sma > slow_sma).astype(int)
    signals = signals.fillna(0)

    logger.info(f"SMA({fast_period}/{slow_period}): {signals.sum()} LONG days ({signals.sum()/len(signals)*100:.1f}%)")
    return signals


def strategy_buy_and_hold(prices: pd.Series) -> pd.Series:
    """Buy & Hold benchmark - always LONG."""
    signals = pd.Series(1, index=prices.index, dtype=int)
    logger.info(f"Buy & Hold: {len(signals)} LONG days (100%)")
    return signals


def compute_fold_returns(signals, prices, start_date, end_date, tc=0.0026):
    """Compute strategy returns for a specific fold."""
    fold_signals = signals.loc[start_date:end_date]
    fold_prices = prices.loc[start_date:end_date]

    daily_returns = fold_prices.pct_change()
    positions = fold_signals.shift(1).fillna(0)
    strategy_returns = positions * daily_returns

    position_changes = positions.diff().abs()
    transaction_costs = position_changes * tc

    net_returns = (strategy_returns - transaction_costs).dropna()
    return net_returns


def compute_fold_metrics(returns):
    """Compute metrics for a single fold."""
    if len(returns) == 0:
        return {"sharpe": 0.0, "return_pct": 0.0, "max_dd_pct": 0.0, "n_days": 0}

    sharpe = annualized_sharpe(returns, risk_free_rate=0.0, periods_per_year=365)
    cum_ret = (1 + returns).prod() - 1
    max_dd = max_drawdown((1 + returns).cumprod())

    return {
        "sharpe": float(sharpe),
        "return_pct": float(cum_ret * 100),
        "max_dd_pct": float(max_dd * 100),
        "n_days": len(returns),
    }


def validate_strategy(name: str, strategy_func: Callable, prices: pd.Series) -> Dict:
    """Run walk-forward validation for a single strategy."""
    logger.info("")
    logger.info("="*70)
    logger.info(f"VALIDATING: {name}")
    logger.info("="*70)

    # Generate signals
    signals = strategy_func(prices)

    # Define folds
    folds = [
        {"name": "2018", "start": "2018-01-01", "end": "2018-12-31"},
        {"name": "2019", "start": "2019-01-01", "end": "2019-12-31"},
        {"name": "2020", "start": "2020-01-01", "end": "2020-12-31"},
        {"name": "2021", "start": "2021-01-01", "end": "2021-12-31"},
        {"name": "2022", "start": "2022-01-01", "end": "2022-12-31"},
        {"name": "2023", "start": "2023-01-01", "end": "2023-12-31"},
    ]

    for year in [2021, 2022, 2023]:
        for q in range(1, 5):
            if q == 1:
                start, end = f"{year}-01-01", f"{year}-03-31"
            elif q == 2:
                start, end = f"{year}-04-01", f"{year}-06-30"
            elif q == 3:
                start, end = f"{year}-07-01", f"{year}-09-30"
            else:
                start, end = f"{year}-10-01", f"{year}-12-31"
            folds.append({"name": f"{year}Q{q}", "start": start, "end": end})

    # Run validation
    fold_results = []
    for fold in folds:
        fold_returns = compute_fold_returns(signals, prices, fold['start'], fold['end'])
        metrics = compute_fold_metrics(fold_returns)
        fold_results.append({
            "fold": fold['name'],
            **metrics
        })

    # Aggregate statistics
    sharpes = [r['sharpe'] for r in fold_results]
    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes, ddof=1)
    min_sharpe = np.min(sharpes)
    max_sharpe = np.max(sharpes)
    positive_count = sum(1 for s in sharpes if s > 0)

    logger.info(f"\nResults: Mean Sharpe={mean_sharpe:.3f}, Min={min_sharpe:.3f}, Max={max_sharpe:.3f}")
    logger.info(f"Positive folds: {positive_count}/18")

    return {
        "strategy": name,
        "aggregate": {
            "mean_sharpe": float(mean_sharpe),
            "std_sharpe": float(std_sharpe),
            "min_sharpe": float(min_sharpe),
            "max_sharpe": float(max_sharpe),
            "positive_folds": positive_count,
            "total_folds": len(folds),
        },
        "folds": fold_results,
    }


def main():
    logger.info("="*70)
    logger.info("UNIFIED STRATEGY VALIDATION")
    logger.info("="*70)

    # Load data
    prices = load_prices()
    logger.info(f"Loaded {len(prices)} days of BTC prices (2017-2023)\n")

    # Define strategies to test
    strategies = [
        ("Pure Donchian(20/10)", strategy_pure_donchian),
        ("Conservative Donchian(30/15)", strategy_conservative_donchian),
        ("RSI Mean Reversion", strategy_rsi_mean_reversion),
        ("Bollinger Mean Reversion", strategy_bollinger_mean_reversion),
        ("SMA Crossover (50/200)", strategy_sma_crossover),
        ("Buy & Hold", strategy_buy_and_hold),
    ]

    # Validate all strategies
    all_results = []
    for name, func in strategies:
        try:
            result = validate_strategy(name, func, prices)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error validating {name}: {e}")
            continue

    # Summary comparison
    logger.info("")
    logger.info("="*70)
    logger.info("STRATEGY COMPARISON")
    logger.info("="*70)
    logger.info(f"{'Strategy':<30} {'Mean Sharpe':>12} {'Min':>8} {'Max':>8} {'Positive':>10}")
    logger.info("-"*70)

    for result in all_results:
        agg = result['aggregate']
        logger.info(f"{result['strategy']:<30} {agg['mean_sharpe']:>12.3f} {agg['min_sharpe']:>8.3f} {agg['max_sharpe']:>8.3f} {agg['positive_folds']:>3}/{agg['total_folds']:<3}")

    # Find best strategy
    logger.info("")
    logger.info("="*70)
    logger.info("BEST STRATEGY")
    logger.info("="*70)

    best = max(all_results, key=lambda x: x['aggregate']['mean_sharpe'])
    logger.info(f"Winner: {best['strategy']}")
    logger.info(f"Mean Sharpe: {best['aggregate']['mean_sharpe']:.3f}")
    logger.info(f"Min Sharpe: {best['aggregate']['min_sharpe']:.3f}")
    logger.info(f"Positive folds: {best['aggregate']['positive_folds']}/{best['aggregate']['total_folds']}")

    # Check if passes criteria
    passes_mean = best['aggregate']['mean_sharpe'] >= 1.0
    passes_min = best['aggregate']['min_sharpe'] > 0.0
    passes_positive = best['aggregate']['positive_folds'] >= 14  # 75%+ positive

    logger.info("")
    if passes_mean and passes_min and passes_positive:
        logger.info("✅ PASS: Best strategy meets deployment criteria")
    elif passes_mean and passes_min:
        logger.info("⚠️ PARTIAL: Best strategy viable but not all criteria met")
    else:
        logger.info("❌ FAIL: No strategy meets deployment criteria")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "strategies_tested": len(all_results),
        "all_results": all_results,
        "best_strategy": best,
        "criteria": {
            "mean_sharpe_gte_1.0": passes_mean,
            "min_sharpe_gt_0": passes_min,
            "positive_folds_gte_75pct": passes_positive,
        }
    }

    output_path = Path("results/validation/unified_strategy_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")
    logger.info("")
    logger.info("="*70)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
