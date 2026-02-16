#!/usr/bin/env python3
"""OPTION 2: Debug Overfitting (Simplified)

Focus on the most promising approaches:
1. Reduce complexity (shallow XGBoost)
2. Simpler model (Logistic Regression)
3. Shorter horizon (7d vs 30d)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sparky.backtest.costs import TransactionCostModel
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.xgboost_model import XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_experiment(name, model, X_train, y_train, X_holdout, y_holdout, returns_holdout):
    """Run single experiment and return results."""
    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_holdout)

    # Evaluate
    positions = predictions
    strategy_returns = positions * returns_holdout

    # Costs
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

    logger.info(f"{name:40} Sharpe: {sharpe:7.4f}  Return: {total_return:7.2f}%  Trades: {num_trades:3}")

    return {
        "name": name,
        "sharpe": float(sharpe),
        "return_pct": float(total_return),
        "max_dd": float(dd),
        "num_trades": int(num_trades),
    }


def main():
    logger.info("=" * 90)
    logger.info("OPTION 2: DEBUG OVERFITTING (Simplified)")
    logger.info("=" * 90)

    # Load data
    logger.info("\nLoading data...")
    X = pd.read_parquet("data/processed/feature_matrix_btc.parquet")

    from sparky.data.storage import DataStore
    store = DataStore()
    prices_df, _ = store.load(Path("data/raw/btc/ohlcv.parquet"))

    technical_features = ["rsi_14", "momentum_30d", "ema_ratio_20d"]

    results = []

    # =========================================================================
    # Test on 30d horizon
    # =========================================================================
    logger.info("\n" + "=" * 90)
    logger.info("TESTING: 30d Horizon")
    logger.info("=" * 90)

    targets_30d = pd.read_parquet("data/processed/targets_btc_30d.parquet")["target"]
    X_30d = X[technical_features]

    # Align and clean
    common_idx = X_30d.index.intersection(targets_30d.index)
    X_30d = X_30d.loc[common_idx]
    y_30d = targets_30d.loc[common_idx]
    returns_30d = prices_df["close"].loc[common_idx].pct_change().fillna(0)

    # Remove NaN
    valid = ~X_30d.isna().any(axis=1)
    X_30d = X_30d[valid]
    y_30d = y_30d[valid]
    returns_30d = returns_30d[valid]

    # Split (recreate masks after NaN removal)
    holdout_start = pd.Timestamp("2025-07-01", tz="UTC")
    train_idx = X_30d.index < holdout_start
    holdout_idx = X_30d.index >= holdout_start

    X_train_30 = X_30d[train_idx]
    y_train_30 = y_30d[train_idx]
    X_hold_30 = X_30d[holdout_idx]
    y_hold_30 = y_30d[holdout_idx]
    ret_hold_30 = returns_30d[holdout_idx]

    logger.info(f"Train: {len(X_train_30)}, Holdout: {len(X_hold_30)}")

    # 1. Original XGBoost (for baseline)
    logger.info("\nConfiguration                            Sharpe   Return   Trades")
    logger.info("-" * 90)

    model = XGBoostModel(random_state=0)  # Default params
    results.append(run_experiment("1. Original XGBoost (30d)", model, X_train_30, y_train_30, X_hold_30, y_hold_30, ret_hold_30))

    # 2. Shallow XGBoost
    model = XGBoostModel(
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
    results.append(run_experiment("2. Shallow XGBoost (30d)", model, X_train_30, y_train_30, X_hold_30, y_hold_30, ret_hold_30))

    # 3. Logistic Regression
    class LogisticModel:
        def __init__(self, C=0.01):
            self.model = LogisticRegression(C=C, max_iter=1000, random_state=0)
            self.scaler = None

        def fit(self, X, y):
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)

        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled).astype(int)

    model = LogisticModel(C=0.01)
    results.append(run_experiment("3. Logistic Regression (30d)", model, X_train_30, y_train_30, X_hold_30, y_hold_30, ret_hold_30))

    # =========================================================================
    # Test on 7d horizon
    # =========================================================================
    logger.info("\n" + "=" * 90)
    logger.info("TESTING: 7d Horizon")
    logger.info("=" * 90)

    targets_7d = pd.read_parquet("data/processed/targets_btc_7d.parquet")["target"]
    X_7d = X[technical_features]

    # Align and clean
    common_idx = X_7d.index.intersection(targets_7d.index)
    X_7d = X_7d.loc[common_idx]
    y_7d = targets_7d.loc[common_idx]
    returns_7d = prices_df["close"].loc[common_idx].pct_change().fillna(0)

    # Remove NaN
    valid = ~X_7d.isna().any(axis=1)
    X_7d = X_7d[valid]
    y_7d = y_7d[valid]
    returns_7d = returns_7d[valid]

    # Split
    train_idx = X_7d.index < holdout_start
    holdout_idx = X_7d.index >= holdout_start

    X_train_7 = X_7d[train_idx]
    y_train_7 = y_7d[train_idx]
    X_hold_7 = X_7d[holdout_idx]
    y_hold_7 = y_7d[holdout_idx]
    ret_hold_7 = returns_7d[holdout_idx]

    logger.info(f"Train: {len(X_train_7)}, Holdout: {len(X_hold_7)}")
    logger.info("\nConfiguration                            Sharpe   Return   Trades")
    logger.info("-" * 90)

    # 4. Original XGBoost on 7d
    model = XGBoostModel(random_state=0)
    results.append(run_experiment("4. Original XGBoost (7d)", model, X_train_7, y_train_7, X_hold_7, y_hold_7, ret_hold_7))

    # 5. Shallow XGBoost on 7d
    model = XGBoostModel(
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
    results.append(run_experiment("5. Shallow XGBoost (7d)", model, X_train_7, y_train_7, X_hold_7, y_hold_7, ret_hold_7))

    # 6. Logistic on 7d
    model = LogisticModel(C=0.01)
    results.append(run_experiment("6. Logistic Regression (7d)", model, X_train_7, y_train_7, X_hold_7, y_hold_7, ret_hold_7))

    # =========================================================================
    # Test on 1d horizon (shortest)
    # =========================================================================
    logger.info("\n" + "=" * 90)
    logger.info("TESTING: 1d Horizon (Shortest)")
    logger.info("=" * 90)

    targets_1d = pd.read_parquet("data/processed/targets_btc_1d.parquet")["target"]
    X_1d = X[technical_features]

    # Align and clean
    common_idx = X_1d.index.intersection(targets_1d.index)
    X_1d = X_1d.loc[common_idx]
    y_1d = targets_1d.loc[common_idx]
    returns_1d = prices_df["close"].loc[common_idx].pct_change().fillna(0)

    # Remove NaN
    valid = ~X_1d.isna().any(axis=1)
    X_1d = X_1d[valid]
    y_1d = y_1d[valid]
    returns_1d = returns_1d[valid]

    # Split
    train_idx = X_1d.index < holdout_start
    holdout_idx = X_1d.index >= holdout_start

    X_train_1 = X_1d[train_idx]
    y_train_1 = y_1d[train_idx]
    X_hold_1 = X_1d[holdout_idx]
    y_hold_1 = y_1d[holdout_idx]
    ret_hold_1 = returns_1d[holdout_idx]

    logger.info(f"Train: {len(X_train_1)}, Holdout: {len(X_hold_1)}")
    logger.info("\nConfiguration                            Sharpe   Return   Trades")
    logger.info("-" * 90)

    # 7. Shallow XGBoost on 1d
    model = XGBoostModel(
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
    results.append(run_experiment("7. Shallow XGBoost (1d)", model, X_train_1, y_train_1, X_hold_1, y_hold_1, ret_hold_1))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 90)
    logger.info("SUMMARY")
    logger.info("=" * 90)

    # Sort by Sharpe
    results_sorted = sorted(results, key=lambda x: x["sharpe"], reverse=True)

    logger.info(f"\n{'Rank':<6} {'Configuration':<40} {'Sharpe':<10} {'Return%':<10} {'Trades':<8}")
    logger.info("-" * 90)
    for i, r in enumerate(results_sorted, 1):
        logger.info(f"{i:<6} {r['name']:<40} {r['sharpe']:<10.4f} {r['return_pct']:<10.2f} {r['num_trades']:<8}")

    best = results_sorted[0]
    logger.info(f"\n✨ Best: {best['name']} (Sharpe {best['sharpe']:.4f})")

    # Verdict
    logger.info("\n" + "=" * 90)
    if best["sharpe"] >= 0.4:
        logger.info("✅ SUCCESS — Found configuration with Sharpe >= 0.4")
        logger.info("Recommendation: Validate with multi-seed")
        verdict = "SUCCESS"
    else:
        logger.info("❌ STILL FAILING — All configurations have Sharpe < 0.4")
        logger.info("Overfitting persists despite complexity reduction")
        logger.info("Recommendation: Proceed to OPTION 3 (strategic pivot)")
        verdict = "FAIL"
    logger.info("=" * 90)

    # Save
    import json
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "best_configuration": best,
        "all_results": results_sorted,
    }

    with open("results/experiments/option2_debug_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Log to RESEARCH_LOG
    log_entry = f"""
---
## OPTION 2: Debug Overfitting — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC

**Configurations Tested**: {len(results)}

**Best**: {best['name']}
- Sharpe: {best['sharpe']:.4f}
- Return: {best['return_pct']:.2f}%
- Trades: {best['num_trades']}

**Verdict**: [{verdict}]

**Next**: {'Validate with multi-seed' if verdict == 'SUCCESS' else 'Proceed to OPTION 3 (strategic pivot)'}
"""

    with open("roadmap/RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    logger.info(f"\nResults saved to results/experiments/option2_debug_results.json")

    return output


if __name__ == "__main__":
    main()
