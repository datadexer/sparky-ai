#!/usr/bin/env python3
"""
Validate Regime-Filtered Donchian Strategy

Tests if filtering out HIGH volatility periods fixes the 2022 catastrophic failure.

Hypothesis:
- Multi-Timeframe failed walk-forward with mean Sharpe 0.365
- 2022Q2 had Sharpe -3.534 (catastrophic whipsaw in high vol)
- Filtering HIGH vol should improve robustness

Target:
- Walk-forward mean Sharpe ≥ 1.0
- Min fold Sharpe > 0
- 2022 performance improved
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sparky.models.regime_filtered_donchian import regime_filtered_ensemble
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
    buyhold_returns = daily_returns.dropna()

    return net_returns, buyhold_returns


def compute_fold_metrics(returns, buyhold_returns):
    """Compute metrics for a single fold."""
    if len(returns) == 0:
        return {}

    sharpe = annualized_sharpe(returns, risk_free_rate=0.0, periods_per_year=365)
    buyhold_sharpe = annualized_sharpe(buyhold_returns, risk_free_rate=0.0, periods_per_year=365)

    cum_ret = (1 + returns).prod() - 1
    buyhold_cum_ret = (1 + buyhold_returns).prod() - 1

    max_dd = max_drawdown((1 + returns).cumprod())
    buyhold_max_dd = max_drawdown((1 + buyhold_returns).cumprod())

    win_rate = (returns > 0).sum() / len(returns)

    positions = (returns != 0).astype(int)
    n_trades = (positions.diff().abs()).sum()

    return {
        "sharpe": float(sharpe),
        "buyhold_sharpe": float(buyhold_sharpe),
        "return_pct": float(cum_ret * 100),
        "buyhold_return_pct": float(buyhold_cum_ret * 100),
        "max_dd_pct": float(max_dd * 100),
        "buyhold_max_dd_pct": float(buyhold_max_dd * 100),
        "win_rate": float(win_rate),
        "n_trades": int(n_trades),
        "n_days": len(returns),
    }


def main():
    logger.info("="*70)
    logger.info("VALIDATION: Regime-Filtered Donchian Ensemble")
    logger.info("="*70)

    # Load data
    prices = load_prices()
    logger.info(f"Loaded {len(prices)} days of BTC prices (2017-2023)")

    # Compute regime-filtered signals
    logger.info("\nComputing regime-filtered ensemble signals...")
    signals = regime_filtered_ensemble(prices, vol_window=30, filter_high_vol=True)
    logger.info(f"Generated {len(signals)} signals")

    # Define folds (same as walk-forward validation)
    folds = [
        {"name": "2018", "test_start": "2018-01-01", "test_end": "2018-12-31"},
        {"name": "2019", "test_start": "2019-01-01", "test_end": "2019-12-31"},
        {"name": "2020", "test_start": "2020-01-01", "test_end": "2020-12-31"},
        {"name": "2021", "test_start": "2021-01-01", "test_end": "2021-12-31"},
        {"name": "2022", "test_start": "2022-01-01", "test_end": "2022-12-31"},
        {"name": "2023", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    ]

    quarters = []
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
            quarters.append({"name": f"{year}Q{q}", "test_start": start, "test_end": end})

    all_folds = folds + quarters

    logger.info(f"\nRunning validation with {len(all_folds)} folds...")
    logger.info("")

    # Run validation
    fold_results = []
    for fold in all_folds:
        logger.info(f"Fold: {fold['name']} ({fold['test_start']} to {fold['test_end']})")

        fold_returns, buyhold_returns = compute_fold_returns(
            signals, prices, fold['test_start'], fold['test_end'], tc=0.0026
        )

        metrics = compute_fold_metrics(fold_returns, buyhold_returns)

        result = {
            "fold": fold['name'],
            "test_start": fold['test_start'],
            "test_end": fold['test_end'],
            **metrics
        }

        fold_results.append(result)

        logger.info(f"  Sharpe: {metrics['sharpe']:.3f} (vs BH: {metrics['buyhold_sharpe']:.3f})")
        logger.info(f"  Return: {metrics['return_pct']:.1f}% (vs BH: {metrics['buyhold_return_pct']:.1f}%)")
        logger.info("")

    # Aggregate statistics
    sharpes = [r['sharpe'] for r in fold_results]
    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes, ddof=1)
    min_sharpe = np.min(sharpes)
    max_sharpe = np.max(sharpes)

    positive_sharpe_count = sum(1 for s in sharpes if s > 0)
    high_sharpe_count = sum(1 for s in sharpes if s > 0.8)

    logger.info("="*70)
    logger.info("AGGREGATE STATISTICS")
    logger.info("="*70)
    logger.info(f"Mean Sharpe: {mean_sharpe:.3f}")
    logger.info(f"Std Sharpe: {std_sharpe:.3f}")
    logger.info(f"Min Sharpe: {min_sharpe:.3f}")
    logger.info(f"Max Sharpe: {max_sharpe:.3f}")
    logger.info(f"Folds with Sharpe > 0: {positive_sharpe_count}/{len(sharpes)}")
    logger.info(f"Folds with Sharpe > 0.8: {high_sharpe_count}/{len(sharpes)}")

    # Comparison to unfiltered
    logger.info("")
    logger.info("="*70)
    logger.info("COMPARISON TO UNFILTERED")
    logger.info("="*70)
    logger.info(f"Unfiltered mean Sharpe: 0.365")
    logger.info(f"Filtered mean Sharpe: {mean_sharpe:.3f}")
    improvement = mean_sharpe - 0.365
    logger.info(f"Improvement: {improvement:+.3f} ({improvement/0.365*100:+.1f}%)")

    # Success criteria
    logger.info("")
    logger.info("="*70)
    logger.info("SUCCESS CRITERIA")
    logger.info("="*70)

    criteria = []

    if mean_sharpe >= 1.0:
        logger.info(f"✅ Mean Sharpe ≥ 1.0: {mean_sharpe:.3f}")
        criteria.append(True)
    else:
        logger.info(f"❌ Mean Sharpe < 1.0: {mean_sharpe:.3f}")
        criteria.append(False)

    if min_sharpe > 0:
        logger.info(f"✅ Min Sharpe > 0: {min_sharpe:.3f}")
        criteria.append(True)
    else:
        logger.info(f"❌ Min Sharpe ≤ 0: {min_sharpe:.3f}")
        criteria.append(False)

    # Check 2022 improvement
    fold_2022 = next((r for r in fold_results if r['fold'] == '2022'), None)
    if fold_2022:
        sharpe_2022 = fold_2022['sharpe']
        if sharpe_2022 > -1.902:
            logger.info(f"✅ 2022 improved: {sharpe_2022:.3f} (vs unfiltered -1.902)")
            criteria.append(True)
        else:
            logger.info(f"❌ 2022 NOT improved: {sharpe_2022:.3f} (vs unfiltered -1.902)")
            criteria.append(False)

    # Overall verdict
    passed_count = sum(criteria)
    total_count = len(criteria)

    logger.info("")
    if all(criteria):
        logger.info(f"✅ PASS: {passed_count}/{total_count} criteria met")
        verdict = "PASS"
    elif passed_count >= 2:
        logger.info(f"⚠️ PARTIAL PASS: {passed_count}/{total_count} criteria met")
        verdict = "PARTIAL"
    else:
        logger.info(f"❌ FAIL: {passed_count}/{total_count} criteria met")
        verdict = "FAIL"

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "period": "2017-2023",
        "strategy": "Regime-Filtered Donchian Ensemble (20/40/60, filter HIGH vol)",
        "folds": fold_results,
        "aggregate": {
            "mean_sharpe": float(mean_sharpe),
            "std_sharpe": float(std_sharpe),
            "min_sharpe": float(min_sharpe),
            "max_sharpe": float(max_sharpe),
            "positive_sharpe_folds": positive_sharpe_count,
            "high_sharpe_folds": high_sharpe_count,
            "total_folds": len(fold_results),
        },
        "comparison_to_unfiltered": {
            "unfiltered_mean_sharpe": 0.365,
            "filtered_mean_sharpe": float(mean_sharpe),
            "improvement": float(improvement),
            "improvement_pct": float(improvement / 0.365 * 100),
        },
        "criteria": {
            "mean_sharpe_gte_1.0": criteria[0],
            "min_sharpe_gt_0": criteria[1],
            "sharpe_2022_improved": criteria[2] if len(criteria) > 2 else False,
        },
        "verdict": verdict,
    }

    output_path = Path("results/validation/regime_filtered_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info(f"Results saved to: {output_path}")
    logger.info("")
    logger.info("="*70)
    logger.info("REGIME-FILTERED VALIDATION COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
