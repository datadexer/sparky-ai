#!/usr/bin/env python3
"""CRITICAL VALIDATION: Holdout Test

Tests the best Phase 2-3 configuration (technical-only, 30d) on never-touched
holdout data (2025-10-01 to 2025-12-31).

This is THE smoking gun for overfitting. Multi-seed validation cannot catch
overfitting to the train/test split - only holdout can.

Expected:
- If result is genuine: holdout Sharpe should be 0.7-1.0
- If result is overfitting: holdout Sharpe < 0.5 despite multi-seed stability

Run this BEFORE multi-seed validation to save compute on potentially bogus results.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sparky.backtest.costs import TransactionCostModel
from sparky.data.loader import load
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.xgboost_model import XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("HOLDOUT VALIDATION TEST")
    logger.info("=" * 80)
    logger.info("Configuration: Technical-only features, 30d horizon")
    logger.info("Training: 2019-01-15 to 2024-12-31")
    logger.info("Holdout: 2025-01-01 to 2025-12-31 (FULL YEAR)")
    logger.info("=" * 80)

    # Load data
    logger.info("Loading data...")
    X = load("feature_matrix_btc", purpose="training")
    targets_df = load("targets_btc_30d", purpose="training")
    y = targets_df["target"]

    # Technical-only features
    technical_features = ["rsi_14", "momentum_30d", "ema_ratio_20d"]
    X_technical = X[technical_features]

    # Align
    common_index = X_technical.index.intersection(y.index)
    X_technical = X_technical.loc[common_index]
    y = y.loc[common_index]

    # Load prices for returns
    from sparky.data.storage import DataStore

    store = DataStore()
    prices_df, _ = store.load(Path("data/raw/btc/ohlcv.parquet"))
    prices = prices_df["close"].loc[common_index]
    returns = prices.pct_change().fillna(0)

    logger.info(f"Total samples: {len(X_technical)}")
    logger.info(f"Features: {list(X_technical.columns)}")
    logger.info(f"Date range: {X_technical.index[0]} to {X_technical.index[-1]}")

    # Split into train+test vs holdout
    holdout_start = pd.Timestamp("2025-01-01", tz="UTC")

    train_test_mask = X_technical.index < holdout_start
    holdout_mask = X_technical.index >= holdout_start

    X_train_test = X_technical[train_test_mask]
    y_train_test = y[train_test_mask]
    returns_train_test = returns[train_test_mask]

    X_holdout = X_technical[holdout_mask]
    y_holdout = y[holdout_mask]
    returns_holdout = returns[holdout_mask]

    logger.info(f"Train+Test samples: {len(X_train_test)} ({X_train_test.index[0]} to {X_train_test.index[-1]})")
    logger.info(f"Holdout samples: {len(X_holdout)} ({X_holdout.index[0]} to {X_holdout.index[-1]})")

    # Train model on ALL train+test data
    logger.info("\nTraining XGBoost on all train+test data...")
    model = XGBoostModel(random_state=0)
    model.fit(X_train_test, y_train_test)
    logger.info("✓ Model trained")

    # Predict on holdout
    logger.info("\nPredicting on holdout...")
    predictions = model.predict(X_holdout)
    logger.info(f"Holdout predictions: {predictions.sum()} longs / {len(predictions)} total ({predictions.mean():.1%})")

    # Log prediction distribution with counts
    predictions_array = predictions.values if hasattr(predictions, "values") else predictions
    long_count = int((predictions_array == 1).sum())
    short_count = len(predictions_array) - long_count
    long_pct = long_count / len(predictions_array) * 100

    logger.info("\nPrediction Distribution (Holdout Period):")
    logger.info(f"  Long (1):  {long_count:4d} days ({long_pct:5.1f}%)")
    logger.info(f"  Short (0): {short_count:4d} days ({100 - long_pct:5.1f}%)")

    # Compute holdout performance
    logger.info("\nComputing holdout performance...")

    # Transaction costs setup
    cost_model = TransactionCostModel.for_btc()

    # Run baseline (BuyAndHold) on same holdout for comparison
    logger.info("\nRunning baseline (BuyAndHold) on same holdout period...")
    from sparky.models.baselines import BuyAndHold

    baseline_model = BuyAndHold()
    baseline_predictions = baseline_model.predict(X_holdout)  # All 1s
    baseline_positions = baseline_predictions
    baseline_strategy_returns = baseline_positions * returns_holdout

    # Apply costs
    baseline_costs = np.abs(np.diff(baseline_positions, prepend=0)) * cost_model.compute_cost(1.0, "BTC")
    baseline_returns_after_costs = baseline_strategy_returns - baseline_costs

    # Compute metrics
    baseline_equity = (1 + baseline_returns_after_costs).cumprod()
    baseline_sharpe = annualized_sharpe(baseline_returns_after_costs, periods_per_year=365)
    baseline_dd = max_drawdown(baseline_equity)
    baseline_total_return = (baseline_equity.iloc[-1] - 1) * 100

    logger.info("Baseline (BuyAndHold) on holdout:")
    logger.info(f"  Sharpe: {baseline_sharpe:.4f}")
    logger.info(f"  Max DD: {baseline_dd:.2%}")
    logger.info(f"  Total Return: {baseline_total_return:.2f}%")

    # Simple equity curve computation for model
    positions = predictions  # 0 or 1
    strategy_returns = positions * returns_holdout

    # Transaction costs for model
    position_changes = np.abs(np.diff(positions, prepend=0))
    costs = position_changes * cost_model.compute_cost(1.0, "BTC")  # 0.13% per trade
    strategy_returns_after_costs = strategy_returns - costs

    # Metrics
    equity_curve = (1 + strategy_returns_after_costs).cumprod()
    total_return = (equity_curve.iloc[-1] - 1) * 100

    holdout_sharpe = annualized_sharpe(strategy_returns_after_costs, periods_per_year=365)
    holdout_dd = max_drawdown(equity_curve)

    # Number of trades
    num_trades = int(position_changes.sum())

    logger.info("\n" + "=" * 80)
    logger.info("HOLDOUT RESULTS")
    logger.info("=" * 80)
    logger.info(f"Sharpe Ratio: {holdout_sharpe:.4f}")
    logger.info(f"Max Drawdown: {holdout_dd:.2%}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Trades: {num_trades}")
    logger.info("=" * 80)

    # Compare to Phase 2-3 results
    phase23_sharpe = 0.999
    delta = holdout_sharpe - phase23_sharpe

    logger.info("\nCOMPARISON TO PHASE 2-3:")
    logger.info(f"Phase 2-3 Sharpe (train+test): {phase23_sharpe:.4f}")
    logger.info(f"Holdout Sharpe (never seen): {holdout_sharpe:.4f}")
    logger.info(f"Delta: {delta:.4f}")

    logger.info("\nCOMPARISON TO BASELINE (BuyAndHold on same holdout period):")
    logger.info(f"Baseline Sharpe (holdout): {baseline_sharpe:.4f}")
    logger.info(f"Model Sharpe (holdout): {holdout_sharpe:.4f}")
    logger.info(f"Delta: {holdout_sharpe - baseline_sharpe:+.4f}")

    # Verdict
    logger.info("\n" + "=" * 80)
    if holdout_sharpe >= 0.7:
        logger.info("✅ VALIDATION PASSED")
        logger.info("Holdout Sharpe >= 0.7 suggests result is GENUINE, not overfitting")
        logger.info("This would be an extraordinary finding (Sharpe ~1.0 on crypto)")
        verdict = "PASS"
    elif holdout_sharpe >= 0.4:
        logger.info("⚠️ VALIDATION BORDERLINE")
        logger.info("Holdout Sharpe 0.4-0.7 suggests some alpha but weaker than train/test")
        logger.info("Possible degradation or lucky train/test split")
        verdict = "BORDERLINE"
    else:
        logger.info("❌ VALIDATION FAILED")
        logger.info("Holdout Sharpe < 0.4 indicates OVERFITTING despite multi-seed stability")
        logger.info("Result is NOT real alpha - likely overfit to train/test split")
        verdict = "FAIL"
    logger.info("=" * 80)

    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "features": technical_features,
            "horizon": 30,
            "seed": 0,
        },
        "data_split": {
            "train_test_samples": len(X_train_test),
            "train_test_period": f"{X_train_test.index[0]} to {X_train_test.index[-1]}",
            "holdout_samples": len(X_holdout),
            "holdout_period": f"{X_holdout.index[0]} to {X_holdout.index[-1]}",
        },
        "prediction_distribution": {
            "long_count": int(long_count),
            "short_count": int(short_count),
            "long_pct": float(long_pct),
            "short_pct": float(100 - long_pct),
        },
        "results": {
            "holdout_sharpe": float(holdout_sharpe),
            "holdout_max_dd": float(holdout_dd),
            "holdout_total_return_pct": float(total_return),
            "num_trades": int(num_trades),
            "phase23_sharpe": float(phase23_sharpe),
            "delta": float(delta),
            "verdict": verdict,
        },
        "baseline": {
            "baseline_sharpe": float(baseline_sharpe),
            "baseline_max_dd": float(baseline_dd),
            "baseline_total_return_pct": float(baseline_total_return),
            "model_vs_baseline_delta": float(holdout_sharpe - baseline_sharpe),
        },
    }

    import json

    output_path = "results/experiments/holdout_validation_1year.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Append to RESEARCH_LOG.md
    log_entry = f"""
---
## VALIDATION 1: Holdout Test — {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC

**Configuration**: Technical-only (RSI, Momentum, EMA), 30d horizon, seed=0

**Data Split**:
- Training: 2019-01-15 to 2024-12-31 ({len(X_train_test)} samples)
- Holdout: 2025-01-01 to 2025-12-31 ({len(X_holdout)} samples, FULL YEAR)

**Results**:
- Holdout Sharpe: {holdout_sharpe:.4f}
- Max Drawdown: {holdout_dd:.2%}
- Total Return: {total_return:.2f}%
- Trades: {num_trades}

**Comparison**:
- Phase 2-3 Sharpe (train+test): {phase23_sharpe:.4f}
- Holdout Sharpe (never seen): {holdout_sharpe:.4f}
- Delta: {delta:.4f}

**Verdict**: [{verdict}]
{"✅ Holdout validates Phase 2-3 finding. Result appears GENUINE." if verdict == "PASS" else "❌ Holdout FAILS to replicate Phase 2-3. Result is OVERFITTING." if verdict == "FAIL" else "⚠️ Holdout shows degradation. Possible lucky split or marginal alpha."}
"""

    with open("roadmap/RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    logger.info("Results logged to roadmap/RESEARCH_LOG.md")

    return results


if __name__ == "__main__":
    results = main()
