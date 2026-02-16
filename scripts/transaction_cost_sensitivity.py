#!/usr/bin/env python3
"""Transaction Cost Sensitivity Analysis

Test Multi-Timeframe Ensemble performance across different transaction cost scenarios:
- 0.10% (maker fees only, optimistic)
- 0.26% (Binance spot baseline)
- 0.50% (taker + slippage, conservative)

RBM Decision Gate: Sharpe must remain ≥ 1.4 at 0.5% costs to deploy.
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.features.returns import annualized_sharpe, max_drawdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_prices():
    price_path = Path("data/raw/btc/ohlcv_hourly.parquet")
    prices = pd.read_parquet(price_path)
    prices_daily = prices['close'].resample('D').last()
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


def compute_returns(signals, prices, tc):
    daily_returns = prices.pct_change()
    positions = signals.shift(1).fillna(0)
    strategy_returns = positions * daily_returns
    position_changes = positions.diff().abs()
    transaction_costs = position_changes * tc
    return (strategy_returns - transaction_costs).dropna()


def metrics(returns):
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
        "sharpe_rf0": float(sharpe_rf0),
        "sharpe_rf45": float(sharpe_rf),
        "return_pct": float(cum_ret * 100),
        "max_dd_pct": float(max_dd * 100),
        "win_rate": float(win_rate),
        "n_trades": int(n_trades),
    }


def main():
    logger.info("="*70)
    logger.info("TRANSACTION COST SENSITIVITY ANALYSIS")
    logger.info("Multi-Timeframe Donchian Ensemble (2017-2023)")
    logger.info("="*70)

    prices = load_prices()
    ensemble_signals = compute_ensemble_signals(prices)

    # Test multiple transaction cost scenarios
    tc_scenarios = [
        (0.001, "0.10% (Optimistic - Maker Only)"),
        (0.0026, "0.26% (Baseline - Binance Spot)"),
        (0.005, "0.50% (Conservative - Taker + Slippage)"),
    ]

    results = {}

    for tc, label in tc_scenarios:
        logger.info(f"\n{'='*70}")
        logger.info(f"Transaction Cost: {label}")
        logger.info(f"{'='*70}")

        returns = compute_returns(ensemble_signals, prices, tc=tc)
        m = metrics(returns)

        logger.info(f"  Sharpe (rf=0.0%): {m['sharpe_rf0']:.3f}")
        logger.info(f"  Sharpe (rf=4.5%): {m['sharpe_rf45']:.3f}")
        logger.info(f"  Return: {m['return_pct']:.2f}%")
        logger.info(f"  Max DD: {m['max_dd_pct']:.2f}%")
        logger.info(f"  Win Rate: {m['win_rate']*100:.1f}%")
        logger.info(f"  Trades: {m['n_trades']}")

        results[f"tc_{int(tc*10000)}bp"] = {
            "label": label,
            "tc_pct": tc * 100,
            "metrics": m
        }

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Transaction Cost Sensitivity")
    print("="*70)
    print(f"{'Cost':<30s} {'Sharpe (rf=0)':<15s} {'Sharpe (rf=4.5%)':<15s} {'Return %':<12s}")
    print("-"*70)

    for tc, label in tc_scenarios:
        key = f"tc_{int(tc*10000)}bp"
        m = results[key]["metrics"]
        print(f"{label:<30s} {m['sharpe_rf0']:<15.3f} {m['sharpe_rf45']:<15.3f} {m['return_pct']:<12.1f}")

    print("="*70)

    # Decision gate
    conservative_sharpe = results["tc_50bp"]["metrics"]["sharpe_rf0"]
    conservative_sharpe_rf45 = results["tc_50bp"]["metrics"]["sharpe_rf45"]

    print("\n" + "="*70)
    print("RBM DECISION GATE")
    print("="*70)
    print(f"Requirement: Sharpe ≥ 1.4 at 0.5% transaction costs")
    print(f"Result (rf=0.0%): {conservative_sharpe:.3f}")
    print(f"Result (rf=4.5%): {conservative_sharpe_rf45:.3f}")

    if conservative_sharpe >= 1.4:
        print(f"\n✅ PASS: Strategy remains profitable at conservative costs")
        print(f"   Sharpe degradation: 1.624 → {conservative_sharpe:.3f} (-{(1.624-conservative_sharpe)*100:.1f}%)")
    else:
        print(f"\n❌ FAIL: Strategy falls below threshold at conservative costs")
        print(f"   Sharpe degradation: 1.624 → {conservative_sharpe:.3f} (-{(1.624-conservative_sharpe)*100:.1f}%)")

    print("="*70)

    # Save results
    output_path = Path("results/validation/transaction_cost_sensitivity.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
