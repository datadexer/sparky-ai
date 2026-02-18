#!/usr/bin/env python3
"""Phase 1: Re-run baseline models (XGBoost, LSTM) with leakage-free data.

After fixing the returns_1d leakage issue, we re-establish baseline performance
with honest results. Both models are trained on 7d horizon and validated with
LeakageDetector before results are considered valid.

Expected outcomes:
- Leakage detector PASSES for both models
- Sharpe ratios may be low or negative (that's OK — honest failure better than false success)
- Identify which model performs better as foundation for Phase 2+
"""

import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

from sparky.backtest.costs import TransactionCostModel
from sparky.backtest.engine import WalkForwardBacktester
from sparky.backtest.leakage_detector import LeakageDetector
from sparky.backtest.statistics import BacktestStatistics
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.lstm_model import LSTMModel
from sparky.models.xgboost_model import XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
HORIZON = 7  # days
IS_END = "2022-01-01"
SEED = 42
MLFLOW_EXPERIMENT = "phase3_baseline_models"

# Baseline comparison (from RESEARCH_LOG.md)
BASELINE_SHARPE = 0.7892


def run_model_with_validation(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    returns: pd.Series,
) -> dict:
    """Train model, run leakage detector, backtest, and return results.

    Returns None if leakage is detected (model is invalid).
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'=' * 80}")

    # 1. Train model
    logger.info(f"Training on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # 2. Run leakage detector (MANDATORY)
    logger.info("Running leakage detector...")
    detector = LeakageDetector(n_shuffle_trials=10)

    try:
        report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)

        logger.info("Leakage detection results:")
        for check in report.checks:
            status = "✅ PASS" if check.passed else "❌ FAIL"
            logger.info(f"  {check.check_name}: {status} — {check.detail}")

        if not report.passed:
            logger.error(f"❌ {model_name} FAILED leakage detection!")
            logger.error(f"Failed checks: {[c.check_name for c in report.failed_checks]}")
            return None

        logger.info(f"✅ {model_name} passed all leakage checks")

    except Exception as e:
        logger.error(f"Leakage detector error: {e}")
        return None

    # 3. Run walk-forward backtest
    logger.info("Running walk-forward backtest...")
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    backtester = WalkForwardBacktester(
        train_min_length=252,  # 1 year minimum
        embargo_days=7,
        test_length=30,
        step_size=30,
    )

    cost_model = TransactionCostModel.for_btc()

    result = backtester.run(model, X, y, returns, cost_model=cost_model, asset="BTC")

    logger.info(f"Backtest complete: {result.fold_count} folds")

    # 4. Compute statistics
    returns_from_equity = result.equity_curve.pct_change().fillna(0)

    sharpe = annualized_sharpe(returns_from_equity)
    max_dd = max_drawdown(result.equity_curve)
    total_return = (result.equity_curve.iloc[-1] - 1.0) * 100  # Convert to percentage

    # Bootstrap CI for Sharpe
    sharpe_ci = BacktestStatistics.sharpe_confidence_interval(returns_from_equity, n_bootstrap=1000)

    logger.info(f"\n{model_name} Results:")
    logger.info(f"  Sharpe: {sharpe:.4f}")
    logger.info(f"  Sharpe 95% CI: ({sharpe_ci[0]:.4f}, {sharpe_ci[1]:.4f})")
    logger.info(f"  Max DD: {max_dd:.2%}")
    logger.info(f"  Total Return: {total_return:.2%}")
    logger.info(f"  Delta vs Baseline: {sharpe - BASELINE_SHARPE:.4f}")

    return {
        "model_name": model_name,
        "sharpe": sharpe,
        "sharpe_ci_low": sharpe_ci[0],
        "sharpe_ci_high": sharpe_ci[1],
        "max_dd": max_dd,
        "total_return": total_return,
        "fold_count": result.fold_count,
        "leakage_passed": True,
        "backtest_result": result,
    }


def main():
    print("=" * 80)
    print("PHASE 1: Baseline Models with Leakage-Free Data")
    print("=" * 80)

    # Load data
    logger.info("Loading data...")
    X = pd.read_parquet("data/processed/feature_matrix_btc.parquet")
    y = pd.read_parquet(f"data/processed/targets_btc_{HORIZON}d.parquet")["target"]

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Target ({HORIZON}d horizon): {y.shape}")

    # Load returns for backtesting
    from sparky.data.storage import DataStore

    store = DataStore()
    btc_ohlcv, _ = store.load(Path("data/raw/btc/ohlcv.parquet"))
    returns = btc_ohlcv["close"].pct_change()
    returns = returns.reindex(X.index)

    # Split data
    in_sample_mask = X.index < IS_END
    X_train = X[in_sample_mask]
    y_train = y[in_sample_mask]
    X_test = X[~in_sample_mask]
    y_test = y[~in_sample_mask]

    logger.info("\nData split:")
    logger.info(f"  Train: {len(X_train)} samples (up to {IS_END})")
    logger.info(f"  Test: {len(X_test)} samples (after {IS_END})")

    # Set up MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = {}

    # 1. XGBoost
    xgb_model = XGBoostModel(random_state=SEED)
    xgb_result = run_model_with_validation(xgb_model, "XGBoost", X_train, y_train, X_test, y_test, returns)
    if xgb_result:
        results["XGBoost"] = xgb_result
    else:
        logger.error("XGBoost failed validation — not logging to MLflow")

    # 2. LSTM
    lstm_model = LSTMModel(
        window_length=10,
        max_epochs=50,
        patience=10,
        random_state=SEED,
    )
    lstm_result = run_model_with_validation(lstm_model, "LSTM", X_train, y_train, X_test, y_test, returns)
    if lstm_result:
        results["LSTM"] = lstm_result
    else:
        logger.error("LSTM failed validation — not logging to MLflow")

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    if not results:
        print("❌ No models passed validation!")
        sys.exit(1)

    comparison_df = pd.DataFrame(
        {
            name: {
                "Sharpe": res["sharpe"],
                "Sharpe CI Low": res["sharpe_ci_low"],
                "Sharpe CI High": res["sharpe_ci_high"],
                "Max DD": res["max_dd"],
                "Total Return": res["total_return"],
                "Delta vs Baseline": res["sharpe"] - BASELINE_SHARPE,
            }
            for name, res in results.items()
        }
    ).T

    print("\n" + comparison_df.to_string())

    # Determine winner
    winner = max(results.items(), key=lambda x: x[1]["sharpe"])
    print(f"\n🏆 Winner: {winner[0]} (Sharpe: {winner[1]['sharpe']:.4f})")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    best_sharpe = winner[1]["sharpe"]
    if best_sharpe < 0:
        print("❌ Both models have negative Sharpe — NO ALPHA detected")
        print("   Recommendation: Consider pivoting strategy or terminating project")
    elif best_sharpe < 0.50:
        print("⚠️  Weak performance (Sharpe < 0.50)")
        print("   Recommendation: Proceed cautiously, but alpha is marginal")
    elif best_sharpe < BASELINE_SHARPE:
        print("⚠️  Underperforms baseline (BuyAndHold)")
        print("   Recommendation: Not worth deploying unless improved in Phase 2+")
    else:
        print("✅ Beats baseline! Proceed to Phase 2 (feature ablation)")

    print("\n✅ Phase 1 complete — all models validated with leakage-free data")
    sys.exit(0)


if __name__ == "__main__":
    main()
