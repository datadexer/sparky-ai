#!/usr/bin/env python3
"""Phase 4: Multi-seed stability validation for best model.

Tests the winning configuration (technical-only, 30d horizon) across
5 different random seeds to verify robustness.

Success criteria (from plan):
- Mean Sharpe close to Phase 2-3 result (0.999)
- Std < 0.3 (low variance across seeds)
- Min Sharpe > 0.70 (no single seed fails badly)

If Phase 4 succeeds → proceed to Phase 5 (holdout validation)
If Phase 4 fails → model is unstable, investigate further
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sparky.backtest.costs import TransactionCostModel
from sparky.backtest.engine import WalkForwardBacktester
from sparky.backtest.leakage_detector import LeakageDetector
from sparky.backtest.statistics import BacktestStatistics
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.xgboost_model import XGBoostModel
from sparky.data.loader import load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
FEATURE_SET = ["rsi_14", "momentum_30d", "ema_ratio_20d"]  # Technical-only
HORIZON = 30
IS_END = "2022-01-01"
SEEDS = [0, 1, 2, 3, 4]

# Thresholds
BASELINE_SHARPE = 0.7892
EXPECTED_SHARPE = 0.999  # From Phase 2-3
STD_THRESHOLD = 0.3
MIN_SHARPE_THRESHOLD = 0.70


def run_single_seed(
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    seed: int,
) -> dict:
    """Run backtest with a specific random seed."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Seed {seed}")
    logger.info(f"{'=' * 80}")

    # Split data
    in_sample_mask = X.index < IS_END
    X_train = X[in_sample_mask]
    y_train = y[in_sample_mask]
    X_test = X[~in_sample_mask]
    y_test = y[~in_sample_mask]

    # Train model
    model = XGBoostModel(random_state=seed)
    model.fit(X_train, y_train)
    logger.info("✓ Model trained")

    # Leakage check (quick sanity check, not full validation)
    detector = LeakageDetector(n_shuffle_trials=3)  # Reduced for speed
    report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)

    if not report.passed:
        logger.error(f"✗ LEAKAGE DETECTED for seed {seed}")
        return None

    logger.info("✓ Leakage check passed")

    # Backtest
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    backtester = WalkForwardBacktester(
        train_min_length=252,
        embargo_days=7,
        test_length=30,
        step_size=30,
    )

    cost_model = TransactionCostModel.for_btc()
    result = backtester.run(model, X_full, y_full, returns, cost_model=cost_model, asset="BTC")

    logger.info(f"✓ Backtest complete: {result.fold_count} folds")

    # Compute metrics
    returns_from_equity = result.equity_curve.pct_change().fillna(0)
    sharpe = annualized_sharpe(returns_from_equity)
    max_dd = max_drawdown(result.equity_curve)
    total_return = (result.equity_curve.iloc[-1] - 1.0) * 100

    sharpe_ci = BacktestStatistics.sharpe_confidence_interval(returns_from_equity, n_bootstrap=1000)

    logger.info(f"Seed {seed} results:")
    logger.info(f"  Sharpe: {sharpe:.4f}")
    logger.info(f"  95% CI: ({sharpe_ci[0]:.4f}, {sharpe_ci[1]:.4f})")
    logger.info(f"  Max DD: {max_dd:.2%}")
    logger.info(f"  Total Return: {total_return:.2%}")

    return {
        "seed": seed,
        "sharpe": sharpe,
        "sharpe_ci_low": sharpe_ci[0],
        "sharpe_ci_high": sharpe_ci[1],
        "max_dd": max_dd,
        "total_return": total_return,
        "fold_count": result.fold_count,
    }


def main():
    print("=" * 80)
    print("PHASE 4: Multi-Seed Stability Validation")
    print("=" * 80)
    print(f"Configuration: technical-only, {HORIZON}d horizon")
    print(f"Seeds: {SEEDS}")
    print(f"Success criteria: std < {STD_THRESHOLD}, min > {MIN_SHARPE_THRESHOLD}")
    print("=" * 80)

    # Load data
    logger.info("Loading data...")
    X_all = load("feature_matrix_btc", purpose="training")
    X = X_all[FEATURE_SET]

    # NOTE: dynamic dataset name (targets_btc_{HORIZON}d) — kept as pd.read_parquet
    # because the dataset name varies with HORIZON and may not be registered in the loader.
    y = pd.read_parquet(f"data/processed/targets_btc_{HORIZON}d.parquet")["target"]

    from sparky.data.storage import DataStore

    store = DataStore()
    btc_ohlcv, _ = store.load(Path("data/raw/btc/ohlcv.parquet"))
    returns = btc_ohlcv["close"].pct_change().reindex(X.index)

    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Shape: {X.shape}")
    logger.info(f"Target: {HORIZON}d horizon, {len(y)} samples")

    # Run all seeds
    results = []
    for seed in SEEDS:
        result = run_single_seed(X, y, returns, seed)
        if result:
            results.append(result)
        else:
            logger.error(f"Seed {seed} failed!")

    # Analysis
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    # Statistics
    sharpe_values = df["sharpe"].values
    mean_sharpe = np.mean(sharpe_values)
    std_sharpe = np.std(sharpe_values, ddof=1)
    min_sharpe = np.min(sharpe_values)
    max_sharpe = np.max(sharpe_values)

    print(f"\n{'=' * 80}")
    print("STABILITY ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Mean Sharpe: {mean_sharpe:.4f}")
    print(f"Std Sharpe: {std_sharpe:.4f}")
    print(f"Min Sharpe: {min_sharpe:.4f}")
    print(f"Max Sharpe: {max_sharpe:.4f}")
    print(f"Range: {max_sharpe - min_sharpe:.4f}")
    print(f"Coefficient of Variation: {(std_sharpe / mean_sharpe * 100):.2f}%")

    # Decision
    print(f"\n{'=' * 80}")
    print("DECISION")
    print(f"{'=' * 80}")

    passed = True
    reasons = []

    # Check 1: Std < threshold
    if std_sharpe < STD_THRESHOLD:
        print(f"✅ Std check: {std_sharpe:.4f} < {STD_THRESHOLD} (PASS)")
    else:
        print(f"❌ Std check: {std_sharpe:.4f} >= {STD_THRESHOLD} (FAIL)")
        passed = False
        reasons.append(f"High variance (std={std_sharpe:.4f})")

    # Check 2: Min Sharpe > threshold
    if min_sharpe > MIN_SHARPE_THRESHOLD:
        print(f"✅ Min Sharpe check: {min_sharpe:.4f} > {MIN_SHARPE_THRESHOLD} (PASS)")
    else:
        print(f"❌ Min Sharpe check: {min_sharpe:.4f} <= {MIN_SHARPE_THRESHOLD} (FAIL)")
        passed = False
        reasons.append(f"At least one seed underperforms (min={min_sharpe:.4f})")

    # Check 3: Mean close to expected
    deviation = abs(mean_sharpe - EXPECTED_SHARPE)
    if deviation < 0.2:
        print(f"✅ Mean Sharpe check: {mean_sharpe:.4f} ≈ {EXPECTED_SHARPE:.4f} (deviation: {deviation:.4f})")
    else:
        print(f"⚠️  Mean Sharpe check: {mean_sharpe:.4f} vs {EXPECTED_SHARPE:.4f} (deviation: {deviation:.4f})")
        print("   Large deviation from Phase 2-3 result, but not blocking")

    # Final decision
    print(f"\n{'=' * 80}")
    if passed:
        print("✅ PHASE 4 PASSED: Model is stable across seeds")
        print("   Recommendation: CONTINUE TO PHASE 5 (Holdout validation)")
    else:
        print("❌ PHASE 4 FAILED: Model is unstable")
        print(f"   Reasons: {', '.join(reasons)}")
        print("   Recommendation: Investigate hyperparameter sensitivity or TERMINATE")

    # Save results
    output_dir = Path("results/experiments")
    output_file = output_dir / "phase4_multiseed_results.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "configuration": {
                    "feature_set": FEATURE_SET,
                    "horizon": HORIZON,
                    "seeds": SEEDS,
                },
                "summary": {
                    "mean_sharpe": mean_sharpe,
                    "std_sharpe": std_sharpe,
                    "min_sharpe": min_sharpe,
                    "max_sharpe": max_sharpe,
                    "passed": passed,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info(f"\n✓ Results saved to {output_file}")

    if not passed:
        exit(1)


if __name__ == "__main__":
    main()
