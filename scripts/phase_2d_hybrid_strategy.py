#!/usr/bin/env python3
"""PHASE 2D: HYBRID DONCHIAN + REGIME STRATEGY

RBM-mandated hybrid to fix Donchian's bear market weakness.

Strategy Logic:
1. Donchian(20/10) provides trend signals (LONG/FLAT)
2. Volatility regime provides risk context (HIGH/MEDIUM/LOW)
3. Hybrid position sizing combines both:
   - HIGH vol + FLAT → 0% (stay out during volatile downtrends - fixes 2022 bear)
   - LOW vol + LONG → 100% (full exposure in calm uptrends - captures 2024-2025 bull)
   - MEDIUM vol + LONG → 75% (moderate exposure)
   - Defensive 25% in uncertain conditions

SUCCESS CRITERIA (ALL must pass):
1. Out-of-sample Sharpe ≥ 1.4 (2017-2023)
2. Bear market Sharpe > Buy & Hold (2018: > -1.121, 2022: > -1.340)
3. 2023 sideways Sharpe > 1.5
4. Monte Carlo win rate ≥ 75%
5. Bootstrap CI lower bound > 0.7
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sparky.features.regime_indicators import compute_volatility_regime
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.simple_baselines import donchian_channel_strategy

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
    prices_daily = prices["close"].resample("D").last()

    if prices_daily.index.tz is not None:
        prices_daily.index = prices_daily.index.tz_localize(None)

    prices_daily = prices_daily.loc[start_date:end_date]
    return prices_daily


def compute_hybrid_position_sizing(
    donchian_signals: pd.Series,
    volatility_regimes: pd.Series,
) -> pd.Series:
    """Compute hybrid position sizes combining Donchian signals + volatility regimes.

    Position sizing matrix:

    | Regime | Donchian=LONG (1) | Donchian=FLAT (0) |
    |--------|-------------------|-------------------|
    | HIGH   | 50%               | 0%                | Defensive in volatile markets
    | MEDIUM | 75%               | 25%               | Moderate in normal vol
    | LOW    | 100%              | 25%               | Aggressive in calm markets

    Key insight: In HIGH volatility, if Donchian says FLAT (no breakout),
    stay completely OUT to avoid bear market whipsaws.

    Args:
        donchian_signals: Binary signals (1=LONG, 0=FLAT).
        volatility_regimes: Regime labels ("high", "medium", "low").

    Returns:
        Series of position sizes (0.0 to 1.0).
    """
    # Align indices
    common_idx = donchian_signals.index.intersection(volatility_regimes.index)
    donchian_signals = donchian_signals.loc[common_idx]
    volatility_regimes = volatility_regimes.loc[common_idx]

    # Position sizing matrix
    position_sizes = pd.Series(0.0, index=common_idx)

    for i in range(len(common_idx)):
        signal = donchian_signals.iloc[i]
        regime = volatility_regimes.iloc[i]

        if regime == "high":
            position_sizes.iloc[i] = 0.50 if signal == 1 else 0.0  # Stay out when FLAT in high vol
        elif regime == "medium":
            position_sizes.iloc[i] = 0.75 if signal == 1 else 0.25
        elif regime == "low":
            position_sizes.iloc[i] = 1.00 if signal == 1 else 0.25
        else:
            position_sizes.iloc[i] = 0.25  # Default defensive

    logger.info(
        f"Hybrid Position Sizing: mean={position_sizes.mean():.2f}, min={position_sizes.min():.2f}, max={position_sizes.max():.2f}"
    )

    return position_sizes


def compute_strategy_returns(signals, prices, position_sizes=None, transaction_cost=0.0026):
    """Compute strategy returns with optional position sizing and transaction costs."""
    common_dates = signals.index.intersection(prices.index)
    signals = signals.loc[common_dates]
    prices = prices.loc[common_dates]

    daily_returns = prices.pct_change()

    # Positions: lag by 1 day
    if position_sizes is not None:
        position_sizes = position_sizes.loc[common_dates]
        positions = (signals * position_sizes).shift(1).fillna(0)
    else:
        positions = signals.shift(1).fillna(0)

    # Strategy returns before costs
    strategy_returns = positions * daily_returns

    # Transaction costs on position changes
    position_changes = positions.diff().abs()
    transaction_costs = position_changes * transaction_cost
    strategy_returns = strategy_returns - transaction_costs

    strategy_returns = strategy_returns.dropna()
    return strategy_returns


def compute_metrics(returns, name="Strategy"):
    """Compute performance metrics."""
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


def bootstrap_ci(returns, n_resamples=1000):
    """Bootstrap confidence interval for Sharpe."""
    sharpe_samples = []
    returns_array = returns.values

    for i in range(n_resamples):
        sample = np.random.choice(returns_array, size=len(returns_array), replace=True)
        sharpe = annualized_sharpe(pd.Series(sample), periods_per_year=365)
        sharpe_samples.append(sharpe)

    sharpe_samples = np.array(sharpe_samples)
    lower = np.percentile(sharpe_samples, 2.5)
    upper = np.percentile(sharpe_samples, 97.5)
    mean = np.mean(sharpe_samples)

    return {"mean": float(mean), "lower": float(lower), "upper": float(upper)}


def monte_carlo_vs_buyhold(strategy_returns, market_returns, n_simulations=1000):
    """Monte Carlo: % of times strategy beats Buy & Hold."""
    strategy_array = strategy_returns.values
    market_array = market_returns.values

    wins = 0
    for i in range(n_simulations):
        strategy_sample = np.random.choice(strategy_array, size=len(strategy_array), replace=True)
        market_sample = np.random.choice(market_array, size=len(market_array), replace=True)

        strategy_sharpe = annualized_sharpe(pd.Series(strategy_sample), periods_per_year=365)
        market_sharpe = annualized_sharpe(pd.Series(market_sample), periods_per_year=365)

        if strategy_sharpe > market_sharpe:
            wins += 1

    win_rate = wins / n_simulations
    return {"win_rate": float(win_rate), "wins": wins, "n_simulations": n_simulations}


def validate_period(prices, period_name, start_date, end_date):
    """Validate hybrid strategy on a period."""
    logger.info("=" * 80)
    logger.info(f"VALIDATING: {period_name} ({start_date} to {end_date})")
    logger.info("=" * 80)

    period_prices = prices.loc[start_date:end_date]

    if len(period_prices) < 50:
        logger.warning(f"Insufficient data: {len(period_prices)} days")
        return None

    # Generate Donchian signals
    donchian_signals = donchian_channel_strategy(period_prices, entry_period=20, exit_period=10)

    # Compute volatility regimes
    vol_regimes = compute_volatility_regime(period_prices, window=30, frequency="1d")

    # Compute hybrid position sizes
    hybrid_positions = compute_hybrid_position_sizing(donchian_signals, vol_regimes)

    # Compute returns
    hybrid_returns = compute_strategy_returns(donchian_signals, period_prices, position_sizes=hybrid_positions)
    donchian_returns = compute_strategy_returns(donchian_signals, period_prices)  # Pure Donchian for comparison
    market_returns = period_prices.pct_change().dropna()

    # Metrics
    hybrid_metrics = compute_metrics(hybrid_returns, "Hybrid")
    donchian_metrics = compute_metrics(donchian_returns, "Donchian")
    market_metrics = compute_metrics(market_returns, "Buy & Hold")

    logger.info(
        f"Hybrid:     Sharpe={hybrid_metrics['sharpe_ratio']:.3f}, Return={hybrid_metrics['total_return_pct']:.2f}%"
    )
    logger.info(
        f"Donchian:   Sharpe={donchian_metrics['sharpe_ratio']:.3f}, Return={donchian_metrics['total_return_pct']:.2f}%"
    )
    logger.info(
        f"Buy & Hold: Sharpe={market_metrics['sharpe_ratio']:.3f}, Return={market_metrics['total_return_pct']:.2f}%"
    )
    logger.info("")

    return {
        "period": period_name,
        "start": start_date,
        "end": end_date,
        "hybrid": hybrid_metrics,
        "donchian": donchian_metrics,
        "buyhold": market_metrics,
        "hybrid_vs_donchian": float(hybrid_metrics["sharpe_ratio"] - donchian_metrics["sharpe_ratio"]),
        "hybrid_vs_buyhold": float(hybrid_metrics["sharpe_ratio"] - market_metrics["sharpe_ratio"]),
    }


def main():
    """Main validation pipeline."""
    logger.info("=" * 80)
    logger.info("PHASE 2D: HYBRID DONCHIAN + REGIME STRATEGY")
    logger.info("=" * 80)
    logger.info("")

    # Load full history
    prices = load_price_data("2017-01-01", "2025-12-31")

    # ========================================================================
    # CRITICAL PERIODS VALIDATION
    # ========================================================================

    # Out-of-sample 2017-2023
    result_2017_2023 = validate_period(prices, "Full 2017-2023", "2017-01-01", "2023-12-31")

    # Bear markets (critical test)
    result_2018_bear = validate_period(prices, "2018 Bear", "2018-01-01", "2018-12-31")
    result_2022_bear = validate_period(prices, "2022 Bear", "2022-01-01", "2022-12-31")

    # Sideways (critical test)
    result_2023_sideways = validate_period(prices, "2023 Sideways", "2023-01-01", "2023-12-31")

    # Bull (for reference)
    result_2024_2025 = validate_period(prices, "2024-2025 Bull", "2024-01-01", "2025-12-31")

    # ========================================================================
    # STATISTICAL VALIDATION ON 2017-2023
    # ========================================================================
    logger.info("=" * 80)
    logger.info("BOOTSTRAP & MONTE CARLO VALIDATION (2017-2023)")
    logger.info("=" * 80)

    prices_2017_2023 = prices.loc["2017-01-01":"2023-12-31"]
    donchian_signals_2017_2023 = donchian_channel_strategy(prices_2017_2023, entry_period=20, exit_period=10)
    vol_regimes_2017_2023 = compute_volatility_regime(prices_2017_2023, window=30, frequency="1d")
    hybrid_positions_2017_2023 = compute_hybrid_position_sizing(donchian_signals_2017_2023, vol_regimes_2017_2023)

    hybrid_returns_2017_2023 = compute_strategy_returns(
        donchian_signals_2017_2023, prices_2017_2023, position_sizes=hybrid_positions_2017_2023
    )
    market_returns_2017_2023 = prices_2017_2023.pct_change().dropna()

    # Bootstrap CI
    bootstrap_result = bootstrap_ci(hybrid_returns_2017_2023, n_resamples=1000)
    logger.info(
        f"Bootstrap Sharpe: mean={bootstrap_result['mean']:.3f}, 95% CI=[{bootstrap_result['lower']:.3f}, {bootstrap_result['upper']:.3f}]"
    )

    # Monte Carlo
    monte_carlo_result = monte_carlo_vs_buyhold(hybrid_returns_2017_2023, market_returns_2017_2023, n_simulations=1000)
    logger.info(
        f"Monte Carlo: Beats Buy & Hold in {monte_carlo_result['wins']}/1000 trials ({monte_carlo_result['win_rate'] * 100:.1f}%)"
    )
    logger.info("")

    # ========================================================================
    # SUCCESS CRITERIA EVALUATION
    # ========================================================================
    logger.info("=" * 80)
    logger.info("SUCCESS CRITERIA EVALUATION")
    logger.info("=" * 80)

    criteria = {
        "1_out_of_sample_sharpe": result_2017_2023["hybrid"]["sharpe_ratio"] >= 1.4,
        "2_bear_2018_protection": result_2018_bear["hybrid"]["sharpe_ratio"]
        > result_2018_bear["buyhold"]["sharpe_ratio"],
        "3_bear_2022_protection": result_2022_bear["hybrid"]["sharpe_ratio"]
        > result_2022_bear["buyhold"]["sharpe_ratio"],
        "4_sideways_2023_sharpe": result_2023_sideways["hybrid"]["sharpe_ratio"] > 1.5,
        "5_monte_carlo_win_rate": monte_carlo_result["win_rate"] >= 0.75,
        "6_bootstrap_ci_lower": bootstrap_result["lower"] > 0.7,
    }

    all_pass = all(criteria.values())

    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA (ALL MUST PASS)")
    print("=" * 80)
    print(
        f"1. Out-of-sample Sharpe ≥ 1.4:         {'✅ PASS' if criteria['1_out_of_sample_sharpe'] else '❌ FAIL'} ({result_2017_2023['hybrid']['sharpe_ratio']:.3f})"
    )
    print(
        f"2. 2018 Bear > Buy & Hold:             {'✅ PASS' if criteria['2_bear_2018_protection'] else '❌ FAIL'} ({result_2018_bear['hybrid']['sharpe_ratio']:.3f} vs {result_2018_bear['buyhold']['sharpe_ratio']:.3f})"
    )
    print(
        f"3. 2022 Bear > Buy & Hold:             {'✅ PASS' if criteria['3_bear_2022_protection'] else '❌ FAIL'} ({result_2022_bear['hybrid']['sharpe_ratio']:.3f} vs {result_2022_bear['buyhold']['sharpe_ratio']:.3f})"
    )
    print(
        f"4. 2023 Sideways Sharpe > 1.5:         {'✅ PASS' if criteria['4_sideways_2023_sharpe'] else '❌ FAIL'} ({result_2023_sideways['hybrid']['sharpe_ratio']:.3f})"
    )
    print(
        f"5. Monte Carlo win rate ≥ 75%:         {'✅ PASS' if criteria['5_monte_carlo_win_rate'] else '❌ FAIL'} ({monte_carlo_result['win_rate'] * 100:.1f}%)"
    )
    print(
        f"6. Bootstrap CI lower > 0.7:           {'✅ PASS' if criteria['6_bootstrap_ci_lower'] else '❌ FAIL'} ({bootstrap_result['lower']:.3f})"
    )
    print("=" * 80)

    if all_pass:
        print("\n✅ BREAKTHROUGH: All criteria met - DEPLOY TO PAPER TRADING")
    else:
        passed = sum(criteria.values())
        print(f"\n⚠️ PARTIAL SUCCESS: {passed}/6 criteria met - Proceed to Phase 2E explorations")

    print("=" * 80)

    # Save results
    results = {
        "validation_timestamp": datetime.now().isoformat(),
        "periods": {
            "2017_2023": result_2017_2023,
            "2018_bear": result_2018_bear,
            "2022_bear": result_2022_bear,
            "2023_sideways": result_2023_sideways,
            "2024_2025_bull": result_2024_2025,
        },
        "bootstrap_ci": bootstrap_result,
        "monte_carlo": monte_carlo_result,
        "success_criteria": criteria,
        "final_verdict": "BREAKTHROUGH" if all_pass else f"PARTIAL ({sum(criteria.values())}/6)",
    }

    output_path = Path("results/validation/phase_2d_hybrid_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
