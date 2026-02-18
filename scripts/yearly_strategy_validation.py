#!/usr/bin/env python3
"""
Yearly-Fold Strategy Validation

Test strategies with YEARLY folds instead of quarterly to reduce noise.
Crypto volatility is so high that quarterly metrics are too noisy.

Yearly validation provides better signal-to-noise ratio.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sparky.features.returns import annualized_sharpe
from sparky.models.simple_baselines import donchian_channel_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_prices():
    price_path = Path("data/raw/btc/ohlcv_hourly.parquet")
    prices = pd.read_parquet(price_path)
    prices_daily = prices["close"].resample("D").last()
    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)
    return prices_daily.loc["2017-01-01":"2023-12-31"]


def strategy_pure_donchian(prices):
    return donchian_channel_strategy(prices, entry_period=20, exit_period=10)


def strategy_multi_timeframe_ensemble(prices):
    """Multi-Timeframe Donchian Ensemble (20/40/60) - originally claimed Sharpe 1.624"""
    signals_20 = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
    signals_40 = donchian_channel_strategy(prices, entry_period=40, exit_period=20)
    signals_60 = donchian_channel_strategy(prices, entry_period=60, exit_period=30)

    # Ensemble: LONG if ANY timeframe signals LONG
    ensemble = ((signals_20 + signals_40 + signals_60) > 0).astype(int)
    return ensemble


def strategy_rsi_mean_reversion(prices):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    signals = pd.Series(0, index=prices.index, dtype=int)
    in_position = False

    for i in range(len(prices)):
        if pd.isna(rsi.iloc[i]):
            signals.iloc[i] = 0
            continue
        if not in_position:
            if rsi.iloc[i] < 30:
                in_position = True
                signals.iloc[i] = 1
            else:
                signals.iloc[i] = 0
        else:
            if rsi.iloc[i] > 70:
                in_position = False
                signals.iloc[i] = 0
            else:
                signals.iloc[i] = 1

    return signals


def strategy_bollinger_mean_reversion(prices, period=20, num_std=2.0):
    """Bollinger Band mean reversion: LONG at lower band, EXIT at upper band."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std

    signals = pd.Series(0, index=prices.index, dtype=int)
    in_position = False

    for i in range(len(prices)):
        if pd.isna(lower_band.iloc[i]) or pd.isna(upper_band.iloc[i]):
            signals.iloc[i] = 0
            continue

        if not in_position:
            if prices.iloc[i] < lower_band.iloc[i]:
                in_position = True
                signals.iloc[i] = 1
            else:
                signals.iloc[i] = 0
        else:
            if prices.iloc[i] > upper_band.iloc[i]:
                in_position = False
                signals.iloc[i] = 0
            else:
                signals.iloc[i] = 1

    return signals


def strategy_sma_crossover(prices, fast_period=50, slow_period=200):
    """SMA crossover: LONG when fast > slow, FLAT when fast < slow."""
    fast_sma = prices.rolling(window=fast_period).mean()
    slow_sma = prices.rolling(window=slow_period).mean()
    signals = (fast_sma > slow_sma).astype(int)
    signals = signals.fillna(0)
    return signals


def strategy_buy_and_hold(prices):
    return pd.Series(1, index=prices.index, dtype=int)


def compute_fold_metrics(signals, prices, start, end):
    fold_signals = signals.loc[start:end]
    fold_prices = prices.loc[start:end]

    daily_returns = fold_prices.pct_change()
    positions = fold_signals.shift(1).fillna(0)
    strategy_returns = positions * daily_returns

    position_changes = positions.diff().abs()
    transaction_costs = position_changes * 0.0026

    net_returns = (strategy_returns - transaction_costs).dropna()

    if len(net_returns) == 0:
        return {"sharpe": 0.0, "return_pct": 0.0}

    sharpe = annualized_sharpe(net_returns, risk_free_rate=0.0, periods_per_year=365)
    cum_ret = (1 + net_returns).prod() - 1

    return {"sharpe": float(sharpe), "return_pct": float(cum_ret * 100)}


def main():
    logger.info("=" * 70)
    logger.info("YEARLY-FOLD STRATEGY VALIDATION")
    logger.info("=" * 70)

    prices = load_prices()
    logger.info(f"Loaded {len(prices)} days of BTC prices (2017-2023)\n")

    # YEARLY folds only (6 folds instead of 18)
    folds = [
        {"name": "2018", "start": "2018-01-01", "end": "2018-12-31"},
        {"name": "2019", "start": "2019-01-01", "end": "2019-12-31"},
        {"name": "2020", "start": "2020-01-01", "end": "2020-12-31"},
        {"name": "2021", "start": "2021-01-01", "end": "2021-12-31"},
        {"name": "2022", "start": "2022-01-01", "end": "2022-12-31"},
        {"name": "2023", "start": "2023-01-01", "end": "2023-12-31"},
    ]

    strategies = [
        ("Multi-Timeframe Ensemble (20/40/60)", strategy_multi_timeframe_ensemble),
        ("Pure Donchian(20/10)", strategy_pure_donchian),
        ("Conservative Donchian(30/15)", lambda p: donchian_channel_strategy(p, entry_period=30, exit_period=15)),
        ("RSI Mean Reversion", strategy_rsi_mean_reversion),
        ("Bollinger Mean Reversion", lambda p: strategy_bollinger_mean_reversion(p)),
        ("SMA Crossover (50/200)", lambda p: strategy_sma_crossover(p)),
        ("Buy & Hold", strategy_buy_and_hold),
    ]

    all_results = {}

    for strat_name, strat_func in strategies:
        logger.info(f"\nValidating: {strat_name}")
        signals = strat_func(prices)

        fold_results = []
        for fold in folds:
            metrics = compute_fold_metrics(signals, prices, fold["start"], fold["end"])
            fold_results.append(
                {
                    "year": fold["name"],
                    "sharpe": metrics["sharpe"],
                    "return_pct": metrics["return_pct"],
                }
            )
            logger.info(f"  {fold['name']}: Sharpe={metrics['sharpe']:.3f}, Return={metrics['return_pct']:.1f}%")

        sharpes = [r["sharpe"] for r in fold_results]
        mean_sharpe = np.mean(sharpes)
        min_sharpe = np.min(sharpes)
        max_sharpe = np.max(sharpes)
        positive_count = sum(1 for s in sharpes if s > 0)

        logger.info(
            f"  AGGREGATE: Mean={mean_sharpe:.3f}, Min={min_sharpe:.3f}, Max={max_sharpe:.3f}, Positive={positive_count}/6"
        )

        all_results[strat_name] = {
            "mean_sharpe": mean_sharpe,
            "min_sharpe": min_sharpe,
            "max_sharpe": max_sharpe,
            "positive_count": positive_count,
            "folds": fold_results,
        }

    # Comparison
    logger.info("")
    logger.info("=" * 70)
    logger.info("YEARLY-FOLD COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Strategy':<25} {'Mean Sharpe':>12} {'Min':>8} {'Max':>8} {'Positive':>10}")
    logger.info("-" * 70)

    for name, result in all_results.items():
        logger.info(
            f"{name:<25} {result['mean_sharpe']:>12.3f} {result['min_sharpe']:>8.3f} {result['max_sharpe']:>8.3f} {result['positive_count']:>3}/6"
        )

    # Save
    output_path = Path("results/validation/yearly_strategy_comparison.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
