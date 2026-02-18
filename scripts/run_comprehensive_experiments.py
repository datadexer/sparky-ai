#!/usr/bin/env python3
"""Phase 2-3: Comprehensive feature ablation + horizon optimization experiments.

Runs all combinations of:
- Feature sets: all, technical-only, onchain-only
- Horizons: 1d, 3d, 7d, 14d, 30d

This script is designed to run ALL experiments and save results to a JSON file
for later analysis. Each experiment includes leakage validation.

Total experiments: 3 feature sets × 5 horizons = 15 runs
Expected time: 4-6 hours (sequential) or 1-2 hours (parallel with Task tool)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from sparky.backtest.costs import TransactionCostModel
from sparky.backtest.engine import WalkForwardBacktester
from sparky.backtest.leakage_detector import LeakageDetector
from sparky.backtest.statistics import BacktestStatistics
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.xgboost_model import XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
HORIZONS = [1, 3, 7, 14, 30]
IS_END = "2022-01-01"
SEED = 42

# Feature groups
FEATURE_GROUPS = {
    "all": [
        "rsi_14",
        "momentum_30d",
        "ema_ratio_20d",
        "hash_ribbon_btc",
        "address_momentum_btc",
        "volume_momentum_btc",
    ],
    "technical": ["rsi_14", "momentum_30d", "ema_ratio_20d"],
    "onchain": ["hash_ribbon_btc", "address_momentum_btc", "volume_momentum_btc"],
}

# Baseline for comparison
BASELINE_SHARPE = 0.7892


def run_single_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    feature_set_name: str,
    horizon: int,
) -> dict:
    """Run a single experiment: train XGBoost, validate, backtest, return metrics.

    Returns None if leakage is detected or experiment fails.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Experiment: {feature_set_name} features, {horizon}d horizon")
    logger.info(f"{'=' * 80}")

    # Select features
    feature_cols = FEATURE_GROUPS[feature_set_name]
    X_subset = X[feature_cols]

    logger.info(f"Features: {list(X_subset.columns)}")
    logger.info(f"Shape: {X_subset.shape}")

    # Split data
    in_sample_mask = X_subset.index < IS_END
    X_train = X_subset[in_sample_mask]
    y_train = y[in_sample_mask]
    X_test = X_subset[~in_sample_mask]
    y_test = y[~in_sample_mask]

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train model
    try:
        model = XGBoostModel(random_state=SEED)
        model.fit(X_train, y_train)
        logger.info("✓ Model trained")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        return None

    # Leakage detection
    try:
        detector = LeakageDetector(n_shuffle_trials=10)
        report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)

        if not report.passed:
            logger.error(f"✗ LEAKAGE DETECTED: {[c.check_name for c in report.failed_checks]}")
            return {
                "feature_set": feature_set_name,
                "horizon": horizon,
                "status": "LEAKAGE_DETECTED",
                "failed_checks": [c.check_name for c in report.failed_checks],
            }

        logger.info("✓ Leakage checks passed")
    except Exception as e:
        logger.error(f"✗ Leakage detection failed: {e}")
        return None

    # Backtest
    try:
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
    except Exception as e:
        logger.error(f"✗ Backtest failed: {e}")
        return None

    # Compute metrics
    try:
        returns_from_equity = result.equity_curve.pct_change().fillna(0)

        sharpe = annualized_sharpe(returns_from_equity)
        max_dd = max_drawdown(result.equity_curve)
        total_return = (result.equity_curve.iloc[-1] - 1.0) * 100

        sharpe_ci = BacktestStatistics.sharpe_confidence_interval(returns_from_equity, n_bootstrap=1000)

        logger.info(f"Sharpe: {sharpe:.4f}, CI: ({sharpe_ci[0]:.4f}, {sharpe_ci[1]:.4f})")
        logger.info(f"Max DD: {max_dd:.2%}, Total Return: {total_return:.2%}")

    except Exception as e:
        logger.error(f"✗ Metrics computation failed: {e}")
        return None

    return {
        "feature_set": feature_set_name,
        "horizon": horizon,
        "status": "SUCCESS",
        "sharpe": sharpe,
        "sharpe_ci_low": sharpe_ci[0],
        "sharpe_ci_high": sharpe_ci[1],
        "max_dd": max_dd,
        "total_return": total_return,
        "delta_vs_baseline": sharpe - BASELINE_SHARPE,
        "fold_count": result.fold_count,
        "num_features": len(feature_cols),
    }


def main():
    print("=" * 80)
    print("PHASE 2-3: Comprehensive Experiments")
    print("=" * 80)
    print(f"Feature sets: {list(FEATURE_GROUPS.keys())}")
    print(f"Horizons: {HORIZONS}")
    print(f"Total experiments: {len(FEATURE_GROUPS) * len(HORIZONS)}")
    print("=" * 80)

    # Load base feature matrix
    logger.info("Loading feature matrix...")
    X_all = pd.read_parquet("data/processed/feature_matrix_btc.parquet")

    # Load returns for backtesting
    from sparky.data.storage import DataStore

    store = DataStore()
    btc_ohlcv, _ = store.load(Path("data/raw/btc/ohlcv.parquet"))
    returns = btc_ohlcv["close"].pct_change().reindex(X_all.index)

    logger.info(f"Feature matrix: {X_all.shape}")
    logger.info(f"Available features: {list(X_all.columns)}")

    # Run all experiments
    results = []
    total = len(FEATURE_GROUPS) * len(HORIZONS)
    current = 0

    for feature_set_name in FEATURE_GROUPS.keys():
        for horizon in HORIZONS:
            current += 1
            logger.info(f"\n[{current}/{total}] Starting: {feature_set_name} + {horizon}d")

            # Load targets for this horizon
            try:
                y = pd.read_parquet(f"data/processed/targets_btc_{horizon}d.parquet")["target"]
                logger.info(f"Loaded {len(y)} targets for {horizon}d horizon")
            except FileNotFoundError:
                logger.error(f"Target file not found for {horizon}d horizon")
                continue

            # Run experiment
            result = run_single_experiment(X_all, y, returns, feature_set_name, horizon)

            if result:
                results.append(result)
                logger.info(f"✓ Experiment complete: {result.get('status')}")
            else:
                logger.error("✗ Experiment failed")
                results.append(
                    {
                        "feature_set": feature_set_name,
                        "horizon": horizon,
                        "status": "FAILED",
                    }
                )

    # Save results
    output_dir = Path("results/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"phase2_3_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "baseline_sharpe": BASELINE_SHARPE,
                "experiments": results,
            },
            f,
            indent=2,
        )

    logger.info(f"\n✓ Results saved to {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r.get("status") == "SUCCESS"]
    leakage = [r for r in results if r.get("status") == "LEAKAGE_DETECTED"]
    failed = [r for r in results if r.get("status") == "FAILED"]

    print(f"Total experiments: {total}")
    print(f"  Successful: {len(successful)}")
    print(f"  Leakage detected: {len(leakage)}")
    print(f"  Failed: {len(failed)}")

    if successful:
        print(f"\n{'=' * 80}")
        print("TOP 5 PERFORMERS (by Sharpe)")
        print(f"{'=' * 80}")

        df = pd.DataFrame(successful)
        df_sorted = df.sort_values("sharpe", ascending=False).head(5)

        print(
            df_sorted[
                ["feature_set", "horizon", "sharpe", "sharpe_ci_low", "sharpe_ci_high", "max_dd", "delta_vs_baseline"]
            ].to_string(index=False)
        )

        # Check if any beat baseline
        best_sharpe = df["sharpe"].max()
        best_row = df.loc[df["sharpe"].idxmax()]

        print(f"\n{'=' * 80}")
        print("BEST RESULT")
        print(f"{'=' * 80}")
        print(f"Feature set: {best_row['feature_set']}")
        print(f"Horizon: {best_row['horizon']}d")
        print(f"Sharpe: {best_row['sharpe']:.4f}")
        print(f"95% CI: ({best_row['sharpe_ci_low']:.4f}, {best_row['sharpe_ci_high']:.4f})")
        print(f"Max DD: {best_row['max_dd']:.2%}")
        print(f"Delta vs Baseline: {best_row['delta_vs_baseline']:.4f}")

        # Decision logic
        print(f"\n{'=' * 80}")
        print("DECISION")
        print(f"{'=' * 80}")

        if best_sharpe < 0.50:
            print("❌ NO ALPHA DETECTED")
            print(f"   Best Sharpe: {best_sharpe:.4f} < 0.50 threshold")
            print("   Recommendation: TERMINATE project or pivot to ETH")
        elif best_sharpe >= 0.70:
            print("✅ POTENTIAL ALPHA DETECTED")
            print(f"   Best Sharpe: {best_sharpe:.4f} >= 0.70 threshold")
            print("   Recommendation: Continue to multi-seed validation (Phase 4)")
        else:
            print("⚠️  MARGINAL PERFORMANCE")
            print(f"   Best Sharpe: {best_sharpe:.4f} in [0.50, 0.70] range")
            print("   Recommendation: Pivot to ETH or different target variable")

    print("\n✓ Phase 2-3 complete")
    print(f"Results: {output_file}")


if __name__ == "__main__":
    main()
