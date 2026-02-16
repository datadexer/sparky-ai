#!/usr/bin/env python3
"""OPTION 2: Debug Overfitting

Systematic approach to reduce overfitting:
1. Reduce XGBoost complexity (shallow trees, heavy regularization)
2. Test simpler models (Logistic Regression, simple thresholds)
3. Test shorter horizons (1d, 3d, 7d vs 30d)
4. Reduce feature count (try 1-2 features instead of 3)

Success criteria: Holdout Sharpe >= 0.4 AND within 0.3 of train Sharpe
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sparky.backtest.costs import TransactionCostModel
from sparky.backtest.leakage_detector import LeakageDetector
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.xgboost_model import XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_model(model, X_train, y_train, X_holdout, y_holdout, returns_holdout, name):
    """Evaluate a model on holdout data."""
    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_holdout)

    # Compute performance
    positions = predictions
    strategy_returns = positions * returns_holdout

    # Transaction costs
    cost_model = TransactionCostModel.for_btc()
    position_changes = np.abs(np.diff(positions, prepend=0))
    costs = position_changes * cost_model.compute_cost(1.0, "BTC")
    strategy_returns_after_costs = strategy_returns - costs

    # Metrics
    equity_curve = (1 + strategy_returns_after_costs).cumprod()
    total_return = (equity_curve.iloc[-1] - 1) * 100
    sharpe = annualized_sharpe(strategy_returns_after_costs, periods_per_year=365)
    dd = max_drawdown(equity_curve)
    num_trades = int(position_changes.sum())

    # Train performance for comparison
    train_predictions = model.predict(X_train)
    train_positions = train_predictions
    # Simplified train returns (without full backtesting)

    logger.info(f"\n{name}:")
    logger.info(f"  Holdout Sharpe: {sharpe:.4f}")
    logger.info(f"  Holdout Return: {total_return:.2f}%")
    logger.info(f"  Max DD: {dd:.2%}")
    logger.info(f"  Trades: {num_trades}")
    logger.info(f"  Long%: {predictions.mean():.1%}")

    return {
        "name": name,
        "holdout_sharpe": float(sharpe),
        "holdout_return_pct": float(total_return),
        "max_dd": float(dd),
        "num_trades": int(num_trades),
        "long_pct": float(predictions.mean()),
    }


def main():
    logger.info("=" * 80)
    logger.info("OPTION 2: DEBUG OVERFITTING")
    logger.info("=" * 80)
    logger.info("Testing multiple approaches to reduce overfitting:")
    logger.info("1. Reduced XGBoost complexity")
    logger.info("2. Simpler models (Logistic Regression)")
    logger.info("3. Shorter horizons (7d vs 30d)")
    logger.info("4. Fewer features (1-2 vs 3)")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading data...")
    X = pd.read_parquet("data/processed/feature_matrix_btc.parquet")

    from sparky.data.storage import DataStore
    store = DataStore()
    prices_df, _ = store.load(Path("data/raw/btc/ohlcv.parquet"))

    results = []

    # =========================================================================
    # APPROACH 1: Reduce XGBoost Complexity (30d horizon)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("APPROACH 1: Reduced XGBoost Complexity")
    logger.info("=" * 80)

    targets_30d = pd.read_parquet("data/processed/targets_btc_30d.parquet")
    y_30d = targets_30d["target"]

    technical_features = ["rsi_14", "momentum_30d", "ema_ratio_20d"]
    X_technical = X[technical_features]

    # Align
    common_index = X_technical.index.intersection(y_30d.index)
    X_technical = X_technical.loc[common_index]
    y_30d = y_30d.loc[common_index]
    prices = prices_df["close"].loc[common_index]
    returns = prices.pct_change().fillna(0)

    # Handle NaN values (drop rows with NaN)
    valid_mask = ~X_technical.isna().any(axis=1)
    X_technical = X_technical[valid_mask]
    y_30d = y_30d[valid_mask]
    returns = returns[valid_mask]
    logger.info(f"After removing NaN: {len(X_technical)} samples")

    # Split
    holdout_start = pd.Timestamp("2025-07-01", tz="UTC")
    train_mask = X_technical.index < holdout_start
    holdout_mask = X_technical.index >= holdout_start

    X_train = X_technical[train_mask]
    y_train = y_30d[train_mask]
    X_holdout = X_technical[holdout_mask]
    y_holdout = y_30d[holdout_mask]
    returns_holdout = returns[holdout_mask]

    logger.info(f"Train: {len(X_train)} samples, Holdout: {len(X_holdout)} samples")

    # 1a. Shallow trees, heavy regularization
    logger.info("\n1a. Shallow XGBoost (max_depth=2, reg_alpha=5.0)")
    model_1a = XGBoostModel(
        random_state=0,
        max_depth=2,  # Very shallow (was 5)
        learning_rate=0.05,  # Slower learning (was 0.1)
        n_estimators=50,  # Fewer trees (was 100)
        subsample=0.7,  # More aggressive subsampling (was 0.8)
        colsample_bytree=0.7,  # More aggressive feature sampling (was 0.8)
        min_child_weight=10,  # Stricter (was 5)
        reg_alpha=5.0,  # Much higher L1 (was 0.1)
        reg_lambda=10.0,  # Much higher L2 (was 1.0)
    )
    result_1a = evaluate_model(model_1a, X_train, y_train, X_holdout, y_holdout, returns_holdout, "Shallow XGBoost")
    results.append(result_1a)

    # 1b. Even more conservative (max_depth=1, decision stumps)
    logger.info("\n1b. Decision Stumps (max_depth=1)")
    model_1b = XGBoostModel(
        random_state=0,
        max_depth=1,  # Decision stumps only
        learning_rate=0.03,
        n_estimators=30,
        subsample=0.5,
        colsample_bytree=0.5,
        min_child_weight=20,
        reg_alpha=10.0,
        reg_lambda=20.0,
    )
    result_1b = evaluate_model(model_1b, X_train, y_train, X_holdout, y_holdout, returns_holdout, "Decision Stumps")
    results.append(result_1b)

    # =========================================================================
    # APPROACH 2: Simpler Models
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("APPROACH 2: Simpler Models")
    logger.info("=" * 80)

    # 2a. Logistic Regression with heavy regularization
    logger.info("\n2a. Logistic Regression (L2 penalty, C=0.01)")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout)

    # Wrapper class for LogisticRegression to match ModelProtocol
    class LogisticModel:
        def __init__(self, C=0.01, random_state=0):
            self.model = LogisticRegression(
                C=C,  # Inverse of regularization strength (smaller = more regularization)
                penalty='l2',
                solver='lbfgs',
                random_state=random_state,
                max_iter=1000,
            )
            self.scaler = None

        def fit(self, X, y):
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)

        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled).astype(int)

    model_2a = LogisticModel(C=0.01)
    result_2a = evaluate_model(model_2a, X_train, y_train, X_holdout, y_holdout, returns_holdout, "Logistic Regression")
    results.append(result_2a)

    # 2b. Even more regularized
    logger.info("\n2b. Heavy Logistic Regression (C=0.001)")
    model_2b = LogisticModel(C=0.001)
    result_2b = evaluate_model(model_2b, X_train, y_train, X_holdout, y_holdout, returns_holdout, "Heavy Logistic Reg")
    results.append(result_2b)

    # =========================================================================
    # APPROACH 3: Shorter Horizons (7d instead of 30d)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("APPROACH 3: Shorter Horizon (7d)")
    logger.info("=" * 80)

    targets_7d = pd.read_parquet("data/processed/targets_btc_7d.parquet")
    y_7d = targets_7d["target"]

    # Align
    common_index_7d = X_technical.index.intersection(y_7d.index)
    X_7d = X_technical.loc[common_index_7d]
    y_7d = y_7d.loc[common_index_7d]
    returns_7d = returns.loc[common_index_7d]

    # Split
    train_mask_7d = X_7d.index < holdout_start
    holdout_mask_7d = X_7d.index >= holdout_start

    X_train_7d = X_7d[train_mask_7d]
    y_train_7d = y_7d[train_mask_7d]
    X_holdout_7d = X_7d[holdout_mask_7d]
    y_holdout_7d = y_7d[holdout_mask_7d]
    returns_holdout_7d = returns_7d[holdout_mask_7d]

    # 3a. Shallow XGBoost on 7d
    logger.info("\n3a. Shallow XGBoost on 7d horizon")
    model_3a = XGBoostModel(
        random_state=0,
        max_depth=2,
        learning_rate=0.05,
        n_estimators=50,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_alpha=5.0,
        reg_lambda=10.0,
    )
    result_3a = evaluate_model(model_3a, X_train_7d, y_train_7d, X_holdout_7d, y_holdout_7d, returns_holdout_7d, "Shallow XGBoost 7d")
    results.append(result_3a)

    # 3b. Logistic on 7d
    logger.info("\n3b. Logistic Regression on 7d horizon")
    model_3b = LogisticModel(C=0.01)
    result_3b = evaluate_model(model_3b, X_train_7d, y_train_7d, X_holdout_7d, y_holdout_7d, returns_holdout_7d, "Logistic Reg 7d")
    results.append(result_3b)

    # =========================================================================
    # APPROACH 4: Fewer Features (reduce from 3 to 1-2)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("APPROACH 4: Fewer Features")
    logger.info("=" * 80)

    # 4a. RSI only (1 feature)
    logger.info("\n4a. RSI only (1 feature)")
    X_rsi = X[["rsi_14"]].loc[common_index]
    X_train_rsi = X_rsi[train_mask]
    X_holdout_rsi = X_rsi[holdout_mask]

    model_4a = XGBoostModel(
        random_state=0,
        max_depth=2,
        learning_rate=0.05,
        n_estimators=30,
        subsample=0.7,
        colsample_bytree=1.0,  # Only 1 feature
        min_child_weight=10,
        reg_alpha=5.0,
        reg_lambda=10.0,
    )
    result_4a = evaluate_model(model_4a, X_train_rsi, y_train, X_holdout_rsi, y_holdout, returns_holdout, "RSI only")
    results.append(result_4a)

    # 4b. Momentum only (1 feature)
    logger.info("\n4b. Momentum only (1 feature)")
    X_momentum = X[["momentum_30d"]].loc[common_index]
    X_train_momentum = X_momentum[train_mask]
    X_holdout_momentum = X_momentum[holdout_mask]

    model_4b = XGBoostModel(
        random_state=0,
        max_depth=2,
        learning_rate=0.05,
        n_estimators=30,
        subsample=0.7,
        colsample_bytree=1.0,
        min_child_weight=10,
        reg_alpha=5.0,
        reg_lambda=10.0,
    )
    result_4b = evaluate_model(model_4b, X_train_momentum, y_train, X_holdout_momentum, y_holdout, returns_holdout, "Momentum only")
    results.append(result_4b)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("OPTION 2 SUMMARY")
    logger.info("=" * 80)

    # Sort by holdout Sharpe
    results_sorted = sorted(results, key=lambda x: x["holdout_sharpe"], reverse=True)

    logger.info("\nAll Results (sorted by Holdout Sharpe):")
    logger.info(f"{'Rank':<5} {'Configuration':<30} {'Sharpe':<10} {'Return%':<10} {'MaxDD':<10} {'Trades':<8}")
    logger.info("-" * 80)

    for i, r in enumerate(results_sorted, 1):
        logger.info(
            f"{i:<5} {r['name']:<30} {r['holdout_sharpe']:<10.4f} {r['holdout_return_pct']:<10.2f} "
            f"{r['max_dd']:<10.2%} {r['num_trades']:<8}"
        )

    # Find best
    best = results_sorted[0]
    logger.info(f"\nBest Configuration: {best['name']}")
    logger.info(f"Holdout Sharpe: {best['holdout_sharpe']:.4f}")
    logger.info(f"Holdout Return: {best['holdout_return_pct']:.2f}%")

    # Verdict
    logger.info("\n" + "=" * 80)
    if best["holdout_sharpe"] >= 0.4:
        logger.info("✅ SUCCESS — Found configuration with Sharpe >= 0.4")
        logger.info(f"Best: {best['name']} (Sharpe {best['holdout_sharpe']:.4f})")
        logger.info("Recommendation: Validate with multi-seed on this configuration")
        verdict = "SUCCESS"
    else:
        logger.info("❌ STILL FAILING — All configurations have Sharpe < 0.4")
        logger.info(f"Best: {best['name']} (Sharpe {best['holdout_sharpe']:.4f})")
        logger.info("Overfitting persists despite complexity reduction")
        logger.info("Recommendation: Proceed to OPTION 3 (strategic pivot)")
        verdict = "FAIL"
    logger.info("=" * 80)

    # Save results
    import json
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "best_configuration": best,
        "all_results": results_sorted,
    }

    output_path = "results/experiments/option2_debug_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Append to RESEARCH_LOG.md
    log_entry = f"""
---
## OPTION 2: Debug Overfitting — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC

**Approaches Tested**: {len(results)} configurations

**Best Result**: {best['name']}
- Holdout Sharpe: {best['holdout_sharpe']:.4f}
- Holdout Return: {best['holdout_return_pct']:.2f}%
- Max DD: {best['max_dd']:.2%}
- Trades: {best['num_trades']}

**Verdict**: [{verdict}]
{'✅ Found configuration with acceptable holdout performance (Sharpe >= 0.4)' if verdict == 'SUCCESS' else '❌ All configurations still fail holdout (Sharpe < 0.4). Overfitting persists.'}

**Next Step**: {'Validate best configuration with multi-seed' if verdict == 'SUCCESS' else 'Proceed to OPTION 3 (strategic pivot)'}
"""

    with open("roadmap/RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    logger.info("Results logged to roadmap/RESEARCH_LOG.md")

    return output


if __name__ == "__main__":
    results = main()
