#!/usr/bin/env python3
"""Run BuyAndHold baseline on BTC and log results.

Loads real BTC price data, runs walk-forward backtest with BuyAndHold,
computes statistics, checks for leakage, and logs to MLflow.

Usage:
    python scripts/run_baseline.py
"""

import logging
from pathlib import Path

import pandas as pd

from sparky.backtest.costs import TransactionCostModel
from sparky.backtest.engine import WalkForwardBacktester
from sparky.backtest.leakage_detector import LeakageDetector
from sparky.backtest.statistics import BacktestStatistics
from sparky.data.storage import DataStore
from sparky.features.returns import annualized_sharpe, max_drawdown, simple_returns
from sparky.features.technical import ema, momentum, rsi
from sparky.models.baselines import BuyAndHold
from sparky.tracking.experiment import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Period definitions
IS_START = "2019-01-01"
IS_END = "2022-01-01"
OOS_END = "2025-12-31"

DATA_PATH = Path("data/raw/btc/ohlcv.parquet")


def main():
    # 1. Load BTC price data
    logger.info("Loading BTC OHLCV data...")
    store = DataStore()
    df, meta = store.load(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows, range: {df.index.min().date()} to {df.index.max().date()}")

    # Filter to full period (IS + OOS)
    df = df.loc[IS_START:OOS_END]
    logger.info(f"Filtered to {len(df)} rows ({IS_START} to {OOS_END})")

    prices = df["close"]

    # 2. Compute features
    logger.info("Computing features...")
    mom = momentum(prices, period=30)
    rsi_14 = rsi(prices, period=14)
    ema_ratio = prices / ema(prices, 20) - 1

    X = pd.DataFrame(
        {"momentum": mom, "rsi": rsi_14, "ema_ratio": ema_ratio},
        index=df.index,
    ).dropna()

    # 3. Compute returns and target
    returns = simple_returns(prices).loc[X.index]
    y = (returns.shift(-1) > 0).astype(int).loc[X.index].fillna(0).astype(int)

    logger.info(f"Feature matrix: {X.shape[0]} rows, {X.shape[1]} features")

    # 4. Run walk-forward backtest
    logger.info("Running walk-forward backtest (BuyAndHold + BTC costs)...")
    model = BuyAndHold()
    cost_model = TransactionCostModel.for_btc()
    backtester = WalkForwardBacktester(
        train_min_length=252,
        embargo_days=7,
        test_length=30,
        step_size=30,
    )

    result = backtester.run(model, X, y, returns, cost_model=cost_model, asset="BTC")
    logger.info(f"Completed {result.fold_count} folds")

    # 5. Compute overall statistics
    equity = result.equity_curve
    equity_returns = equity.pct_change().dropna()

    full_sharpe = annualized_sharpe(equity_returns)
    full_drawdown = max_drawdown(equity)
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    # OOS-only statistics
    oos_start = pd.Timestamp(IS_END, tz="UTC")
    oos_equity = equity.loc[equity.index >= oos_start]
    if len(oos_equity) > 1:
        oos_returns = oos_equity.pct_change().dropna()
        oos_sharpe = annualized_sharpe(oos_returns)
        oos_drawdown = max_drawdown(oos_equity)
        oos_total_return = (oos_equity.iloc[-1] / oos_equity.iloc[0] - 1) * 100
    else:
        oos_sharpe = 0.0
        oos_drawdown = 0.0
        oos_total_return = 0.0

    # Bootstrap CI
    lower, upper = BacktestStatistics.sharpe_confidence_interval(
        equity_returns,
        n_bootstrap=5000,
        ci=0.95,
        annualize=True,
        random_state=42,
    )

    # Significance
    p_value = BacktestStatistics.sharpe_significance(equity_returns)

    # 6. Leakage check
    logger.info("Running leakage detector...")
    detector = LeakageDetector(n_shuffle_trials=3)
    # Use first fold's train/test split for leakage check
    fold_metrics = result.per_fold_metrics
    first_fold = fold_metrics[0]
    train_end = first_fold["train_end"]
    test_start = first_fold["test_start"]

    X_train_check = X.loc[X.index <= train_end]
    X_test_check = X.loc[(X.index >= test_start) & (X.index <= first_fold["test_end"])]
    y_train_check = y.loc[X_train_check.index]
    y_test_check = y.loc[X_test_check.index]

    report = detector.run_all_checks(
        model,
        X_train_check,
        y_train_check,
        X_test_check,
        y_test_check,
    )
    leakage_passed = report.passed
    logger.info(f"Leakage check: {'PASSED' if leakage_passed else 'FAILED'}")

    # 7. Log to MLflow
    logger.info("Logging to MLflow...")
    tracker = ExperimentTracker(experiment_name="sparky_baselines")
    run_id = tracker.log_experiment(
        name="BuyAndHold_BTC",
        config={
            "model": "BuyAndHold",
            "asset": "BTC",
            "cost_model": "TransactionCostModel.for_btc()",
            "train_min_length": 252,
            "embargo_days": 7,
            "test_length": 30,
            "step_size": 30,
            "is_period": f"{IS_START} to {IS_END}",
            "oos_period": f"{IS_END} to {OOS_END}",
        },
        metrics={
            "sharpe_full": full_sharpe,
            "sharpe_oos": oos_sharpe,
            "sharpe_ci_lower": lower,
            "sharpe_ci_upper": upper,
            "sharpe_pvalue": p_value,
            "max_drawdown_full": full_drawdown,
            "max_drawdown_oos": oos_drawdown,
            "total_return_pct": total_return,
            "total_return_oos_pct": oos_total_return,
            "fold_count": float(result.fold_count),
            "leakage_passed": 1.0 if leakage_passed else 0.0,
        },
        date_range=(IS_START, OOS_END),
    )

    # 8. Print summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS — BuyAndHold BTC")
    print("=" * 60)
    print(f"Period:              {IS_START} to {OOS_END}")
    print(f"In-sample:           {IS_START} to {IS_END}")
    print(f"Out-of-sample:       {IS_END} to {OOS_END}")
    print(f"Data rows:           {len(X)}")
    print(f"Walk-forward folds:  {result.fold_count}")
    print()
    print(f"Sharpe (full):       {full_sharpe:.4f}")
    print(f"Sharpe (OOS):        {oos_sharpe:.4f}")
    print(f"95% CI:              ({lower:.4f}, {upper:.4f})")
    print(f"Sharpe p-value:      {p_value:.6f}")
    print(f"Max drawdown (full): {full_drawdown:.2%}")
    print(f"Max drawdown (OOS):  {oos_drawdown:.2%}")
    print(f"Total return:        {total_return:.2f}%")
    print(f"Total return (OOS):  {oos_total_return:.2f}%")
    print()
    print(f"Leakage check:       {'PASSED' if leakage_passed else 'FAILED'}")
    print(f"MLflow run ID:       {run_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
