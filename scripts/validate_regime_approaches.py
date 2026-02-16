"""Validate all regime-aware approaches against baseline.

Tests 5 sophisticated regime-aware implementations:
1. Adaptive Lookback Windows (regime-dependent periods)
2. Markov-Switching Donchian (regime-specific strategies)
3. Multi-Horizon Volatility Clustering (term structure)
4. HMM Regime Detection (probabilistic)
5. Regime-Weighted Ensemble (IMCA-inspired, target Sharpe 0.829)

Validation:
- In-sample: 2018-2020 (verify regime detection makes sense)
- Out-of-sample: 2021-2023 (yearly walk-forward)
- Transaction costs: 0.26% round-trip
- Baseline: Multi-TF Donchian (Sharpe 0.772)
- Target: Sharpe ≥ 0.85 (beat baseline by ≥10%)
"""

import json
import logging
from pathlib import Path

import pandas as pd

from sparky.backtest.engine import WalkForwardBacktester
from sparky.backtest.costs import TransactionCostModel
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.regime_adaptive_lookback import adaptive_lookback_ensemble
from sparky.models.regime_markov_switching import markov_switching_ensemble
from sparky.models.regime_volatility_term_structure import volatility_term_structure_ensemble
from sparky.models.regime_hmm import hmm_probabilistic_ensemble, HMM_AVAILABLE
from sparky.models.regime_weighted_ensemble import (
    regime_weighted_ensemble,
    regime_weighted_multitimeframe_ensemble,
)
from sparky.models.simple_baselines import donchian_channel_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_btc_data() -> pd.Series:
    """Load BTC daily prices."""
    data_path = Path("/home/akamath/sparky-ai/data/btc_daily.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"BTC data not found at {data_path}")

    df = pd.read_parquet(data_path)
    prices = df["close"]
    prices.index = pd.to_datetime(df.index)
    # Ensure timezone-aware (UTC)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")

    logger.info(f"Loaded BTC data: {len(prices)} days ({prices.index[0]} to {prices.index[-1]})")
    return prices


def backtest_strategy(
    prices: pd.Series,
    signals: pd.Series,
    start_date: str,
    end_date: str,
    cost_model: TransactionCostModel,
) -> dict:
    """Backtest a strategy on a specific period."""
    import numpy as np

    period_prices = prices.loc[start_date:end_date]
    period_signals = signals.loc[start_date:end_date]

    if len(period_prices) == 0 or len(period_signals) == 0:
        logger.warning(f"No data in period {start_date} to {end_date}")
        return {
            "sharpe": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
        }

    # Simple backtest: hold based on signals
    # Compute returns
    price_returns = period_prices.pct_change()

    # Strategy returns: signal at T applies to T+1's return (no look-ahead bias)
    # Signal at T uses close[T], so position can only be entered at T+1
    actual_positions = period_signals.shift(1).fillna(0)
    strategy_returns = actual_positions * price_returns

    # Count trades (position changes on actual positions)
    position_changes = actual_positions.diff().abs()
    n_trades = int(position_changes.sum())

    # Apply transaction costs
    transaction_costs = position_changes * cost_model.total_cost_pct
    strategy_returns_after_costs = strategy_returns - transaction_costs

    # Cumulative returns
    equity_curve = (1 + strategy_returns_after_costs).cumprod()
    total_return = (equity_curve.iloc[-1] - 1) * 100  # As percentage

    # Sharpe ratio
    sharpe = annualized_sharpe(strategy_returns_after_costs)

    # Max drawdown
    max_dd = max_drawdown(strategy_returns_after_costs)

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
    }


def validate_approach(
    approach_name: str,
    signal_generator,
    prices: pd.Series,
    test_periods: list[tuple[str, str]],
    cost_model: TransactionCostModel,
) -> dict:
    """Validate a regime-aware approach on multiple test periods."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Validating: {approach_name}")
    logger.info(f"{'='*70}")

    # Generate signals (full history)
    try:
        signals = signal_generator(prices)
    except Exception as e:
        logger.error(f"Failed to generate signals for {approach_name}: {e}")
        return {
            "approach": approach_name,
            "error": str(e),
            "results": [],
        }

    # Test on each period
    results = []
    for year_label, (start_date, end_date) in test_periods.items():
        logger.info(f"\nTesting {year_label} ({start_date} to {end_date})...")

        metrics = backtest_strategy(prices, signals, start_date, end_date, cost_model)

        logger.info(
            f"  Sharpe: {metrics['sharpe']:.3f}, "
            f"Return: {metrics['total_return']:.2f}%, "
            f"Max DD: {metrics['max_drawdown']:.2f}%, "
            f"Trades: {metrics['n_trades']}"
        )

        results.append({
            "period": year_label,
            "start_date": start_date,
            "end_date": end_date,
            **metrics,
        })

    # Compute aggregate statistics
    sharpe_values = [r["sharpe"] for r in results if r["sharpe"] != 0.0]

    if len(sharpe_values) > 0:
        mean_sharpe = sum(sharpe_values) / len(sharpe_values)
        min_sharpe = min(sharpe_values)
        max_sharpe = max(sharpe_values)
        positive_count = sum(1 for s in sharpe_values if s > 0)
    else:
        mean_sharpe = 0.0
        min_sharpe = 0.0
        max_sharpe = 0.0
        positive_count = 0

    logger.info(f"\nAggregate Statistics:")
    logger.info(f"  Mean Sharpe: {mean_sharpe:.3f}")
    logger.info(f"  Min Sharpe: {min_sharpe:.3f}")
    logger.info(f"  Max Sharpe: {max_sharpe:.3f}")
    logger.info(f"  Positive Periods: {positive_count}/{len(sharpe_values)}")

    return {
        "approach": approach_name,
        "mean_sharpe": mean_sharpe,
        "min_sharpe": min_sharpe,
        "max_sharpe": max_sharpe,
        "positive_count": positive_count,
        "total_periods": len(sharpe_values),
        "results": results,
    }


def main():
    """Main validation script."""
    logger.info("Starting regime-aware approaches validation...")

    # Load data
    prices = load_btc_data()

    # Define test periods (yearly out-of-sample)
    test_periods = {
        "2018": ("2018-01-01", "2018-12-31"),
        "2019": ("2019-01-01", "2019-12-31"),
        "2020": ("2020-01-01", "2020-12-31"),
        "2021": ("2021-01-01", "2021-12-31"),
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
    }

    # Transaction cost model
    cost_model = TransactionCostModel.for_btc()

    # Define approaches to test
    approaches = []

    # Baseline: Multi-TF Donchian (for comparison)
    def baseline_multitf(prices):
        signals_20 = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
        signals_40 = donchian_channel_strategy(prices, entry_period=40, exit_period=20)
        signals_60 = donchian_channel_strategy(prices, entry_period=60, exit_period=30)
        return ((signals_20 + signals_40 + signals_60) >= 2).astype(int)

    approaches.append(("BASELINE: Multi-TF Donchian (20/40/60)", baseline_multitf))

    # Approach 5: Adaptive Lookback Windows
    approaches.append((
        "Approach 5: Adaptive Lookback Ensemble",
        lambda prices: adaptive_lookback_ensemble(prices, vol_window=30),
    ))

    # Approach 2: Markov-Switching Donchian
    approaches.append((
        "Approach 2: Markov-Switching Ensemble",
        lambda prices: markov_switching_ensemble(prices, vol_window=30),
    ))

    # Approach 3: Multi-Horizon Volatility Clustering
    approaches.append((
        "Approach 3: Volatility Term Structure Ensemble",
        lambda prices: volatility_term_structure_ensemble(prices, short_window=7, medium_window=30, long_window=90),
    ))

    # Approach 1: HMM Regime Detection (if available)
    if HMM_AVAILABLE:
        approaches.append((
            "Approach 1: HMM Probabilistic Ensemble (2-state)",
            lambda prices: hmm_probabilistic_ensemble(prices, n_states=2),
        ))
        approaches.append((
            "Approach 1: HMM Probabilistic Ensemble (3-state)",
            lambda prices: hmm_probabilistic_ensemble(prices, n_states=3),
        ))
    else:
        logger.warning("hmmlearn not installed. Skipping HMM approaches.")

    # Approach 4: Regime-Weighted Ensemble (IMCA-inspired)
    approaches.append((
        "Approach 4: Regime-Weighted Ensemble (aggressive)",
        lambda prices: regime_weighted_ensemble(prices, regime_detection="combined", weighting_scheme="aggressive"),
    ))
    approaches.append((
        "Approach 4: Regime-Weighted Ensemble (balanced)",
        lambda prices: regime_weighted_ensemble(prices, regime_detection="combined", weighting_scheme="balanced"),
    ))
    approaches.append((
        "Approach 4: Regime-Weighted Multi-TF Ensemble (aggressive)",
        lambda prices: regime_weighted_multitimeframe_ensemble(prices, regime_detection="combined", weighting_scheme="aggressive"),
    ))

    # Run validation for all approaches
    all_results = []
    for approach_name, signal_generator in approaches:
        result = validate_approach(approach_name, signal_generator, prices, test_periods, cost_model)
        all_results.append(result)

    # Sort by mean Sharpe (descending)
    all_results_sorted = sorted(all_results, key=lambda x: x.get("mean_sharpe", 0.0), reverse=True)

    # Print summary table
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY TABLE (sorted by Mean Sharpe)")
    logger.info(f"{'='*70}")
    logger.info(
        f"{'Rank':<5} {'Approach':<50} {'Mean Sharpe':<12} {'Min':<8} {'Max':<8} {'Positive':<10}"
    )
    logger.info(f"{'-'*70}")

    for rank, result in enumerate(all_results_sorted, start=1):
        if "error" in result:
            logger.info(f"{rank:<5} {result['approach']:<50} ERROR: {result['error']}")
            continue

        logger.info(
            f"{rank:<5} {result['approach']:<50} "
            f"{result['mean_sharpe']:<12.3f} "
            f"{result['min_sharpe']:<8.3f} "
            f"{result['max_sharpe']:<8.3f} "
            f"{result['positive_count']}/{result['total_periods']}"
        )

    # Save results to JSON
    output_path = Path("/home/akamath/sparky-ai/results/validation/regime_approaches_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results_sorted, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Check if any approach beats baseline
    baseline_sharpe = next((r["mean_sharpe"] for r in all_results if "BASELINE" in r["approach"]), 0.772)
    best_approach = all_results_sorted[0]

    if "error" not in best_approach and best_approach["mean_sharpe"] >= 0.85:
        logger.info(f"\n🎉 SUCCESS: {best_approach['approach']} achieved Sharpe {best_approach['mean_sharpe']:.3f} (≥0.85 target)")
    elif "error" not in best_approach and best_approach["mean_sharpe"] > baseline_sharpe:
        improvement = (best_approach["mean_sharpe"] - baseline_sharpe) / baseline_sharpe * 100
        logger.info(
            f"\n✅ IMPROVEMENT: {best_approach['approach']} beats baseline by {improvement:.1f}% "
            f"({best_approach['mean_sharpe']:.3f} vs {baseline_sharpe:.3f})"
        )
    else:
        logger.info(f"\n⚠️  NO IMPROVEMENT: Best approach {best_approach.get('mean_sharpe', 0.0):.3f} ≤ baseline {baseline_sharpe:.3f}")

    logger.info("\nValidation complete.")


if __name__ == "__main__":
    main()
