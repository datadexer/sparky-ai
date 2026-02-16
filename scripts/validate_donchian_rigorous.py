#!/usr/bin/env python3
"""RIGOROUS VALIDATION: Donchian(20/10) Strategy

RBM-mandated validation protocol:
1. Out-of-sample validation (2017-2023) - includes bear markets
2. Bootstrap confidence interval (1000 resamples)
3. Transaction cost sensitivity (0.1%, 0.26%, 0.5%)
4. Regime breakdown analysis
5. Monte Carlo simulation vs Buy & Hold

Success criteria:
- Sharpe ≥ 0.8 on 2017-2023 (bear + bull)
- Bootstrap 95% CI lower bound > 0.5
- Sharpe ≥ 0.95 even with 0.5% transaction costs
- Works in ALL regimes, not just bull markets
- Beats Buy & Hold in 80%+ of Monte Carlo simulations
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "src")
from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.features.regime_indicators import compute_volatility_regime
from sparky.features.returns import annualized_sharpe, max_drawdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_price_data(start_date, end_date):
    """Load BTC daily prices."""
    price_path = Path("data/raw/btc/ohlcv_hourly.parquet")
    if not price_path.exists():
        raise FileNotFoundError(f"Price data not found: {price_path}")

    prices = pd.read_parquet(price_path)
    prices_daily = prices['close'].resample('D').last()

    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)

    prices_daily = prices_daily.loc[start_date:end_date]
    logger.info(f"Loaded {len(prices_daily)} days: {prices_daily.index.min()} to {prices_daily.index.max()}")

    return prices_daily


def compute_strategy_returns(signals, prices, transaction_cost=0.0026):
    """Compute strategy returns with transaction costs."""
    common_dates = signals.index.intersection(prices.index)
    signals = signals.loc[common_dates]
    prices = prices.loc[common_dates]

    daily_returns = prices.pct_change()
    positions = signals.shift(1).fillna(0)

    # Strategy returns before costs
    strategy_returns = positions * daily_returns

    # Apply transaction costs on position changes
    position_changes = positions.diff().abs()
    transaction_costs = position_changes * transaction_cost
    strategy_returns = strategy_returns - transaction_costs

    strategy_returns = strategy_returns.dropna()
    return strategy_returns


def compute_metrics(returns, name="Strategy"):
    """Compute comprehensive performance metrics."""
    if len(returns) == 0:
        return {}

    cumulative_return = (1 + returns).prod() - 1
    sharpe = annualized_sharpe(returns, periods_per_year=365)
    cumulative_wealth = (1 + returns).cumprod()
    max_dd = max_drawdown(cumulative_wealth)
    win_rate = (returns > 0).sum() / len(returns)
    positions = (returns != 0).astype(int)
    n_trades = (positions.diff().abs()).sum()

    return {
        "total_return_pct": float(cumulative_return * 100),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd * 100),
        "win_rate": float(win_rate),
        "n_trades": int(n_trades),
        "n_days": len(returns),
    }


def bootstrap_confidence_interval(returns, n_resamples=1000, confidence_level=0.95):
    """Bootstrap confidence interval for Sharpe ratio."""
    logger.info(f"Computing bootstrap CI ({n_resamples} resamples)...")

    sharpe_samples = []
    returns_array = returns.values

    for i in range(n_resamples):
        # Resample with replacement
        sample = np.random.choice(returns_array, size=len(returns_array), replace=True)
        sharpe = annualized_sharpe(pd.Series(sample), periods_per_year=365)
        sharpe_samples.append(sharpe)

    sharpe_samples = np.array(sharpe_samples)

    # Compute percentiles for confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(sharpe_samples, alpha/2 * 100)
    upper = np.percentile(sharpe_samples, (1 - alpha/2) * 100)
    mean = np.mean(sharpe_samples)

    logger.info(f"Bootstrap Sharpe: mean={mean:.3f}, 95% CI=[{lower:.3f}, {upper:.3f}]")

    return {
        "mean": float(mean),
        "lower": float(lower),
        "upper": float(upper),
        "samples": sharpe_samples.tolist(),
    }


def monte_carlo_vs_buyhold(strategy_returns, market_returns, n_simulations=1000):
    """Monte Carlo simulation: Does strategy beat Buy & Hold?"""
    logger.info(f"Running Monte Carlo simulation ({n_simulations} trials)...")

    strategy_array = strategy_returns.values
    market_array = market_returns.values

    wins = 0

    for i in range(n_simulations):
        # Resample both strategy and market
        strategy_sample = np.random.choice(strategy_array, size=len(strategy_array), replace=True)
        market_sample = np.random.choice(market_array, size=len(market_array), replace=True)

        strategy_sharpe = annualized_sharpe(pd.Series(strategy_sample), periods_per_year=365)
        market_sharpe = annualized_sharpe(pd.Series(market_sample), periods_per_year=365)

        if strategy_sharpe > market_sharpe:
            wins += 1

    win_rate = wins / n_simulations

    logger.info(f"Monte Carlo: Strategy beats Buy & Hold in {wins}/{n_simulations} trials ({win_rate*100:.1f}%)")

    return {
        "win_rate": float(win_rate),
        "n_simulations": n_simulations,
        "wins": wins,
    }


def regime_breakdown_analysis(signals, prices, returns):
    """Analyze performance by volatility regime."""
    logger.info("Analyzing performance by volatility regime...")

    # Compute volatility regimes
    regimes = compute_volatility_regime(prices, window=30, frequency="1d")

    # Align with returns
    common_dates = regimes.index.intersection(returns.index)
    regimes = regimes.loc[common_dates]
    returns = returns.loc[common_dates]

    regime_results = {}

    for regime in ["low", "medium", "high"]:
        regime_mask = regimes == regime
        regime_returns = returns[regime_mask]

        if len(regime_returns) == 0:
            continue

        metrics = compute_metrics(regime_returns, name=f"Regime {regime}")
        regime_results[regime] = metrics

        logger.info(f"  {regime.upper()}: Sharpe={metrics['sharpe_ratio']:.3f}, Return={metrics['total_return_pct']:.2f}%, Days={metrics['n_days']}")

    return regime_results


def validate_period(prices, period_name, start_date, end_date, transaction_cost=0.0026):
    """Validate strategy on a specific period."""
    logger.info("=" * 80)
    logger.info(f"PERIOD: {period_name} ({start_date} to {end_date})")
    logger.info("=" * 80)

    period_prices = prices.loc[start_date:end_date]

    if len(period_prices) < 50:
        logger.warning(f"Insufficient data for {period_name}: {len(period_prices)} days")
        return None

    # Generate signals
    signals = donchian_channel_strategy(period_prices, entry_period=20, exit_period=10)

    # Compute returns
    strategy_returns = compute_strategy_returns(signals, period_prices, transaction_cost=transaction_cost)
    market_returns = period_prices.pct_change().dropna()

    # Metrics
    strategy_metrics = compute_metrics(strategy_returns, name="Donchian(20/10)")
    market_metrics = compute_metrics(market_returns, name="Buy & Hold")

    logger.info(f"Donchian Sharpe: {strategy_metrics['sharpe_ratio']:.3f}")
    logger.info(f"Buy & Hold Sharpe: {market_metrics['sharpe_ratio']:.3f}")
    logger.info(f"Delta: {strategy_metrics['sharpe_ratio'] - market_metrics['sharpe_ratio']:+.3f}")
    logger.info("")

    return {
        "period": period_name,
        "start": start_date,
        "end": end_date,
        "strategy": strategy_metrics,
        "buyhold": market_metrics,
        "delta_sharpe": float(strategy_metrics['sharpe_ratio'] - market_metrics['sharpe_ratio']),
    }


def main():
    """Main validation pipeline."""
    logger.info("=" * 80)
    logger.info("RIGOROUS VALIDATION: Donchian(20/10) Strategy")
    logger.info("=" * 80)
    logger.info("")

    # Load full history
    prices_full = load_price_data("2017-01-01", "2025-12-31")

    # ========================================================================
    # VALIDATION 1: Out-of-Sample Performance (2017-2023)
    # ========================================================================
    logger.info("=" * 80)
    logger.info("VALIDATION 1: OUT-OF-SAMPLE (2017-2023)")
    logger.info("=" * 80)

    # Full 2017-2023
    result_2017_2023 = validate_period(prices_full, "Full 2017-2023", "2017-01-01", "2023-12-31")

    # Bear markets
    result_2018_bear = validate_period(prices_full, "2018 Bear Market", "2018-01-01", "2018-12-31")
    result_2022_bear = validate_period(prices_full, "2022 Bear Market", "2022-01-01", "2022-12-31")

    # Sideways
    result_2023_sideways = validate_period(prices_full, "2023 Sideways", "2023-01-01", "2023-12-31")

    # Bull (for comparison)
    result_2024_2025_bull = validate_period(prices_full, "2024-2025 Bull (Original)", "2024-01-01", "2025-12-31")

    # ========================================================================
    # VALIDATION 2: Bootstrap Confidence Interval
    # ========================================================================
    logger.info("=" * 80)
    logger.info("VALIDATION 2: BOOTSTRAP CONFIDENCE INTERVAL")
    logger.info("=" * 80)

    prices_2017_2023 = prices_full.loc["2017-01-01":"2023-12-31"]
    signals_2017_2023 = donchian_channel_strategy(prices_2017_2023, entry_period=20, exit_period=10)
    returns_2017_2023 = compute_strategy_returns(signals_2017_2023, prices_2017_2023)

    bootstrap_ci = bootstrap_confidence_interval(returns_2017_2023, n_resamples=1000)

    # ========================================================================
    # VALIDATION 3: Transaction Cost Sensitivity
    # ========================================================================
    logger.info("=" * 80)
    logger.info("VALIDATION 3: TRANSACTION COST SENSITIVITY")
    logger.info("=" * 80)

    cost_sensitivity = {}
    for cost_pct in [0.001, 0.0026, 0.005]:
        returns_cost = compute_strategy_returns(signals_2017_2023, prices_2017_2023, transaction_cost=cost_pct)
        metrics_cost = compute_metrics(returns_cost)
        cost_sensitivity[f"{cost_pct*100:.2f}%"] = metrics_cost
        logger.info(f"Cost {cost_pct*100:.2f}%: Sharpe={metrics_cost['sharpe_ratio']:.3f}")

    # ========================================================================
    # VALIDATION 4: Regime Breakdown
    # ========================================================================
    logger.info("=" * 80)
    logger.info("VALIDATION 4: REGIME BREAKDOWN ANALYSIS")
    logger.info("=" * 80)

    regime_results = regime_breakdown_analysis(signals_2017_2023, prices_2017_2023, returns_2017_2023)

    # ========================================================================
    # VALIDATION 5: Monte Carlo vs Buy & Hold
    # ========================================================================
    logger.info("=" * 80)
    logger.info("VALIDATION 5: MONTE CARLO SIMULATION")
    logger.info("=" * 80)

    market_returns_2017_2023 = prices_2017_2023.pct_change().dropna()
    monte_carlo_results = monte_carlo_vs_buyhold(returns_2017_2023, market_returns_2017_2023, n_simulations=1000)

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    logger.info("=" * 80)
    logger.info("FINAL VERDICT")
    logger.info("=" * 80)

    # Success criteria
    criteria = {
        "out_of_sample_sharpe": result_2017_2023["strategy"]["sharpe_ratio"] >= 0.8,
        "bootstrap_ci_lower": bootstrap_ci["lower"] > 0.5,
        "high_cost_sharpe": cost_sensitivity["0.50%"]["sharpe_ratio"] >= 0.95,
        "regime_robust": all(r["sharpe_ratio"] > 0.0 for r in regime_results.values()),
        "monte_carlo_win_rate": monte_carlo_results["win_rate"] >= 0.8,
    }

    all_pass = all(criteria.values())

    print("\n" + "=" * 80)
    print("VALIDATION CRITERIA")
    print("=" * 80)
    print(f"1. Out-of-sample Sharpe ≥ 0.8:      {'✅ PASS' if criteria['out_of_sample_sharpe'] else '❌ FAIL'} ({result_2017_2023['strategy']['sharpe_ratio']:.3f})")
    print(f"2. Bootstrap CI lower > 0.5:        {'✅ PASS' if criteria['bootstrap_ci_lower'] else '❌ FAIL'} ({bootstrap_ci['lower']:.3f})")
    print(f"3. Sharpe @ 0.5% costs ≥ 0.95:      {'✅ PASS' if criteria['high_cost_sharpe'] else '❌ FAIL'} ({cost_sensitivity['0.50%']['sharpe_ratio']:.3f})")
    print(f"4. All regimes Sharpe > 0:          {'✅ PASS' if criteria['regime_robust'] else '❌ FAIL'}")
    print(f"5. Monte Carlo win rate ≥ 80%:      {'✅ PASS' if criteria['monte_carlo_win_rate'] else '❌ FAIL'} ({monte_carlo_results['win_rate']*100:.1f}%)")
    print("=" * 80)

    if all_pass:
        print("\n✅ VALIDATED: All criteria met - Strategy is ROBUST")
    else:
        print("\n❌ INVALIDATED: Failed one or more criteria")

    print("=" * 80)

    # Save results
    results = {
        "validation_timestamp": datetime.now().isoformat(),
        "out_of_sample_periods": {
            "2017_2023_full": result_2017_2023,
            "2018_bear": result_2018_bear,
            "2022_bear": result_2022_bear,
            "2023_sideways": result_2023_sideways,
            "2024_2025_bull": result_2024_2025_bull,
        },
        "bootstrap_ci": bootstrap_ci,
        "transaction_cost_sensitivity": cost_sensitivity,
        "regime_breakdown": regime_results,
        "monte_carlo": monte_carlo_results,
        "validation_criteria": criteria,
        "final_verdict": "VALIDATED" if all_pass else "INVALIDATED",
    }

    output_path = Path("results/validation/donchian_rigorous_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
