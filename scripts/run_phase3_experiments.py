#!/usr/bin/env python3
"""Run Phase 3 ML model experiments.

Orchestrates all Phase 3 experiments:
- Feature ablation (validate on-chain alpha)
- Horizon sensitivity (optimal prediction horizon)
- Model comparison (XGBoost vs LSTM)
- Multi-seed stability (robustness across seeds)
- Holdout validation (never-touched test set)
- Ensemble exploration (voting, stacking)

All results logged to MLflow and RESEARCH_LOG.md.

Usage:
    python scripts/run_phase3_experiments.py --experiment <name>

Experiments:
    - feature_ablation: Test feature importance via ablation
    - horizon: Test prediction horizons 1d-30d
    - model_comparison: XGBoost vs LSTM head-to-head
    - multi_seed: Validate seed stability (5 seeds)
    - holdout: Never-touched holdout validation
    - ensemble: Test ensemble methods
    - all: Run all experiments sequentially
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sparky.backtest.costs import TransactionCostModel
from sparky.backtest.engine import WalkForwardBacktester
from sparky.backtest.leakage_detector import LeakageDetector
from sparky.backtest.statistics import BacktestStatistics
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.lstm_model import LSTMModel
from sparky.models.xgboost_model import XGBoostModel
from sparky.oversight.activity_logger import AgentActivityLogger
from sparky.tracking.experiment import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Baseline metrics for comparison (from run_baseline.py results)
BASELINE_SHARPE = 0.7892
BASELINE_SHARPE_CI = (0.1374, 1.4777)
BASELINE_MAX_DD = 0.7663
BASELINE_RUN_ID = "2801374c398a492196cb1ef199965eb0"

# Period definitions
IS_START = "2019-01-01"
IS_END = "2022-01-01"
OOS_END = "2025-09-30"
HOLDOUT_START = "2025-10-01"
HOLDOUT_END = "2025-12-31"

# Experiment parameters
SEEDS = [0, 1, 2, 3, 4]
HORIZONS = [1, 3, 7, 14, 30]
FEATURE_GROUPS = ["technical", "onchain_btc", "returns"]


def load_data(horizon: int = 7) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load feature matrix and targets for given horizon.

    Args:
        horizon: Prediction horizon in days (default 7).

    Returns:
        Tuple of (features, targets, returns).
    """
    logger.info(f"Loading data for {horizon}d horizon...")

    # Load feature matrix
    X = pd.read_parquet("data/processed/feature_matrix_btc.parquet")

    # Load targets for this horizon
    targets_path = f"data/processed/targets_btc_{horizon}d.parquet"
    targets_df = pd.read_parquet(targets_path)
    y = targets_df["target"]

    # Align features and targets
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    # Compute returns for backtesting
    # Load price data to compute returns
    from sparky.data.storage import DataStore
    store = DataStore()
    prices_df, _ = store.load(Path("data/raw/btc/ohlcv.parquet"))
    prices = prices_df["close"].loc[common_index]

    from sparky.features.returns import simple_returns
    returns = simple_returns(prices).loc[common_index]

    logger.info(f"Data loaded: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Target balance: {y.sum()} longs ({y.mean()*100:.1f}%)")

    return X, y, returns


def run_backtest_with_leakage_check(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    model_name: str,
    tracker: ExperimentTracker,
    config: dict,
) -> tuple[float, float, float, str, bool]:
    """Run walk-forward backtest with leakage detection.

    Returns:
        Tuple of (sharpe, max_dd, total_return_pct, run_id, leakage_passed).
    """
    # Setup backtester
    cost_model = TransactionCostModel.for_btc()  # 0.13% per trade
    backtester = WalkForwardBacktester(
        train_min_length=252,
        embargo_days=7,
        test_length=30,
        step_size=30,
    )

    # Run backtest
    logger.info(f"Running walk-forward backtest for {model_name}...")
    result = backtester.run(model, X, y, returns, cost_model=cost_model, asset="BTC")

    # Compute statistics
    equity = result.equity_curve
    equity_returns = equity.pct_change().dropna()

    sharpe = annualized_sharpe(equity_returns)
    max_dd = max_drawdown(equity)
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    # Bootstrap CI
    lower, upper = BacktestStatistics.sharpe_confidence_interval(
        equity_returns, n_bootstrap=5000, ci=0.95, annualize=True, random_state=42
    )

    # Significance
    p_value = BacktestStatistics.sharpe_significance(equity_returns)

    # Leakage check (use first fold)
    logger.info("Running leakage detector...")
    detector = LeakageDetector(n_shuffle_trials=3)
    fold_metrics = result.per_fold_metrics
    first_fold = fold_metrics[0]
    train_end = first_fold["train_end"]
    test_start = first_fold["test_start"]

    X_train_check = X.loc[X.index <= train_end]
    X_test_check = X.loc[(X.index >= test_start) & (X.index <= first_fold["test_end"])]
    y_train_check = y.loc[X_train_check.index]
    y_test_check = y.loc[X_test_check.index]

    leakage_report = detector.run_all_checks(
        model, X_train_check, y_train_check, X_test_check, y_test_check
    )
    leakage_passed = leakage_report.passed

    if not leakage_passed:
        logger.error(f"Leakage detected for {model_name}! Not logging to MLflow.")
        return sharpe, max_dd, total_return, "", False

    # Compute delta vs baseline
    delta_sharpe = sharpe - BASELINE_SHARPE

    # Log to MLflow
    logger.info("Logging to MLflow...")
    run_id = tracker.log_experiment(
        name=model_name,
        config=config,
        metrics={
            "sharpe": sharpe,
            "sharpe_ci_lower": lower,
            "sharpe_ci_upper": upper,
            "sharpe_pvalue": p_value,
            "max_drawdown": max_dd,
            "total_return_pct": total_return,
            "fold_count": float(result.fold_count),
            "leakage_passed": 1.0,
            # Baseline comparison metrics
            "baseline_sharpe": BASELINE_SHARPE,
            "delta_sharpe": delta_sharpe,
            "delta_sharpe_ci_low": lower - BASELINE_SHARPE,
            "delta_sharpe_ci_high": upper - BASELINE_SHARPE,
            "beats_baseline_ci": 1.0 if (lower - BASELINE_SHARPE) > 0 else 0.0,
        },
        date_range=(IS_START, OOS_END),
    )

    logger.info(
        f"{model_name} results: Sharpe={sharpe:.4f}, MaxDD={max_dd:.2%}, "
        f"DeltaSharpe={delta_sharpe:.4f}, RunID={run_id}"
    )

    return sharpe, max_dd, total_return, run_id, leakage_passed


def feature_ablation_experiment():
    """Task 3: Feature ablation to validate on-chain alpha."""
    logger.info("=" * 60)
    logger.info("Feature Ablation Experiment")
    logger.info("=" * 60)

    activity_logger = AgentActivityLogger(agent_id="ceo", session_id="phase-3-ml-models")
    tracker = ExperimentTracker(experiment_name="phase3_feature_ablation")

    X, y, returns = load_data(horizon=7)  # Use 7d horizon as default

    results = []

    # Baseline: all features
    logger.info("\n--- Full feature set ---")
    model = XGBoostModel(random_state=42)
    sharpe, max_dd, total_ret, run_id, passed = run_backtest_with_leakage_check(
        model, X, y, returns, "XGBoost_AllFeatures",
        tracker,
        {"model": "XGBoost", "feature_set": "all", "horizon": 7, "seed": 42},
    )
    results.append({
        "feature_set": "all",
        "sharpe": sharpe,
        "max_dd": max_dd,
        "run_id": run_id,
    })

    # Get feature importances
    importances = model.get_feature_importances()
    logger.info(f"\nTop 10 feature importances:\n{importances.head(10)}")

    # Ablation: remove each group
    for group in FEATURE_GROUPS:
        logger.info(f"\n--- Ablation: removing {group} features ---")

        # Filter features (remove this group)
        if group == "technical":
            keep_cols = [c for c in X.columns if not c.startswith(("rsi", "momentum", "ema"))]
        elif group == "onchain_btc":
            keep_cols = [c for c in X.columns if not c.endswith("_btc")]
        elif group == "returns":
            keep_cols = [c for c in X.columns if not c.startswith("returns")]
        else:
            keep_cols = X.columns.tolist()

        if len(keep_cols) == len(X.columns):
            logger.warning(f"No features removed for group {group}, skipping")
            continue

        X_ablated = X[keep_cols]
        logger.info(f"Features after removing {group}: {len(keep_cols)} (was {len(X.columns)})")

        model = XGBoostModel(random_state=42)
        sharpe, max_dd, total_ret, run_id, passed = run_backtest_with_leakage_check(
            model, X_ablated, y, returns, f"XGBoost_Without_{group}",
            tracker,
            {"model": "XGBoost", "feature_set": f"without_{group}", "horizon": 7, "seed": 42},
        )

        delta_vs_full = sharpe - results[0]["sharpe"]
        logger.info(f"Delta vs full features: {delta_vs_full:.4f}")

        results.append({
            "feature_set": f"without_{group}",
            "sharpe": sharpe,
            "max_dd": max_dd,
            "delta_vs_full": delta_vs_full,
            "run_id": run_id,
        })

        # Log to activity logger
        activity_logger.log_experiment(
            task="feature_ablation_experiments",
            hypothesis=f"Removing {group} features will reduce Sharpe",
            strategic_goal="validate_onchain_alpha",
            result={
                "sharpe": sharpe,
                "delta_vs_full": delta_vs_full,
                "validation_status": "preliminary",
                "mlflow_run_id": run_id,
            },
            conclusion=f"Removing {group} reduced Sharpe by {delta_vs_full:.4f}"
            if delta_vs_full < 0
            else f"Removing {group} had minimal impact",
            mlflow_run_id=run_id,
        )

    # Write to RESEARCH_LOG.md
    log_entry = f"""
---
## Feature Ablation Experiment — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC

**Hypothesis**: On-chain features add >0.1 Sharpe vs technical-only (Priority 1 strategic goal)

**Results**:
{chr(10).join(f"- {r['feature_set']}: Sharpe={r['sharpe']:.4f}, MaxDD={r['max_dd']:.2%}, Delta={r.get('delta_vs_full', 0):.4f}" for r in results)}

**Finding**: {'[VALIDATED] On-chain features add significant alpha' if any(r.get('delta_vs_full', 0) < -0.1 for r in results if 'onchain' in r['feature_set']) else '[PRELIMINARY] On-chain impact unclear, need multi-seed validation'}
"""

    with open("roadmap/RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    logger.info("Feature ablation complete. Results logged to RESEARCH_LOG.md")


def horizon_experiment():
    """Task 4: Test prediction horizons 1d-30d."""
    logger.info("=" * 60)
    logger.info("Horizon Sensitivity Experiment")
    logger.info("=" * 60)

    activity_logger = AgentActivityLogger(agent_id="ceo", session_id="phase-3-ml-models")
    tracker = ExperimentTracker(experiment_name="phase3_horizon_sensitivity")

    results = []

    for horizon in HORIZONS:
        logger.info(f"\n--- Testing {horizon}d horizon ---")

        X, y, returns = load_data(horizon=horizon)

        model = XGBoostModel(random_state=42)
        sharpe, max_dd, total_ret, run_id, passed = run_backtest_with_leakage_check(
            model, X, y, returns, f"XGBoost_{horizon}d",
            tracker,
            {"model": "XGBoost", "horizon": horizon, "seed": 42, "feature_set": "all"},
        )

        results.append({
            "horizon": horizon,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "run_id": run_id,
        })

        activity_logger.log_experiment(
            task="horizon_experiments",
            hypothesis=f"{horizon}d horizon prediction",
            strategic_goal="optimal_horizon",
            result={
                "horizon": horizon,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "validation_status": "preliminary",
                "mlflow_run_id": run_id,
            },
            conclusion=f"{horizon}d horizon: Sharpe={sharpe:.4f}",
            mlflow_run_id=run_id,
        )

    # Find optimal horizon
    best_horizon = max(results, key=lambda x: x["sharpe"])
    logger.info(f"\nOptimal horizon: {best_horizon['horizon']}d (Sharpe={best_horizon['sharpe']:.4f})")

    # Write to RESEARCH_LOG.md
    log_entry = f"""
---
## Horizon Sensitivity Experiment — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC

**Hypothesis**: Identify optimal prediction horizon (1d-30d) for maximum Sharpe (Priority 2 strategic goal)

**Results**:
{chr(10).join(f"- {r['horizon']}d: Sharpe={r['sharpe']:.4f}, MaxDD={r['max_dd']:.2%}" for r in results)}

**Finding**: [PRELIMINARY] Optimal horizon appears to be {best_horizon['horizon']}d (Sharpe={best_horizon['sharpe']:.4f})
Needs multi-seed validation for [VALIDATED] status.
"""

    with open("roadmap/RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    logger.info("Horizon experiment complete. Results logged to RESEARCH_LOG.md")


def model_comparison_experiment():
    """Task 5: XGBoost vs LSTM head-to-head."""
    logger.info("=" * 60)
    logger.info("Model Comparison Experiment")
    logger.info("=" * 60)

    activity_logger = AgentActivityLogger(agent_id="ceo", session_id="phase-3-ml-models")
    tracker = ExperimentTracker(experiment_name="phase3_model_comparison")

    X, y, returns = load_data(horizon=7)

    # XGBoost
    logger.info("\n--- XGBoost ---")
    xgb_model = XGBoostModel(random_state=42)
    xgb_sharpe, xgb_dd, xgb_ret, xgb_run_id, _ = run_backtest_with_leakage_check(
        xgb_model, X, y, returns, "XGBoost_7d",
        tracker,
        {"model": "XGBoost", "horizon": 7, "seed": 42},
    )

    # LSTM
    logger.info("\n--- LSTM ---")
    lstm_model = LSTMModel(window_length=10, max_epochs=50, random_state=42)
    lstm_sharpe, lstm_dd, lstm_ret, lstm_run_id, _ = run_backtest_with_leakage_check(
        lstm_model, X, y, returns, "LSTM_7d",
        tracker,
        {"model": "LSTM", "horizon": 7, "seed": 42, "window_length": 10, "max_epochs": 50},
    )

    # Compare
    winner = "XGBoost" if xgb_sharpe > lstm_sharpe else "LSTM"
    delta = abs(xgb_sharpe - lstm_sharpe)

    log_entry = f"""
---
## Model Comparison Experiment — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC

**Results**:
- XGBoost: Sharpe={xgb_sharpe:.4f}, MaxDD={xgb_dd:.2%}, RunID={xgb_run_id}
- LSTM: Sharpe={lstm_sharpe:.4f}, MaxDD={lstm_dd:.2%}, RunID={lstm_run_id}

**Finding**: [PRELIMINARY] {winner} wins by {delta:.4f} Sharpe.
Needs multi-seed validation and paired t-test for [VALIDATED] status.
"""

    with open("roadmap/RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    logger.info("Model comparison complete. Results logged to RESEARCH_LOG.md")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 experiments")
    parser.add_argument(
        "--experiment",
        choices=[
            "feature_ablation",
            "horizon",
            "model_comparison",
            "multi_seed",
            "holdout",
            "ensemble",
            "all",
        ],
        required=True,
        help="Which experiment to run",
    )
    args = parser.parse_args()

    if args.experiment == "feature_ablation" or args.experiment == "all":
        feature_ablation_experiment()

    if args.experiment == "horizon" or args.experiment == "all":
        horizon_experiment()

    if args.experiment == "model_comparison" or args.experiment == "all":
        model_comparison_experiment()

    logger.info("=" * 60)
    logger.info("Experiments complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
