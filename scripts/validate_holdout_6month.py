#!/usr/bin/env python3
"""OPTION 1: Expand Holdout to 6 Months

Test if 3-month holdout (Oct-Dec 2025) was too short or unlucky.
Expand to 6 months (Jul-Dec 2025) for more robust test.

If this still fails (Sharpe < 0.4), overfitting is confirmed.
If this improves (Sharpe >= 0.4), Oct-Dec may have been unlucky period.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sparky.backtest.costs import TransactionCostModel
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
    logger.info("OPTION 1: 6-MONTH HOLDOUT TEST")
    logger.info("=" * 80)
    logger.info("Configuration: Technical-only features, 30d horizon")
    logger.info("Training: 2019-01-01 to 2025-06-30")
    logger.info("Holdout: 2025-07-01 to 2025-12-31 (6 MONTHS)")
    logger.info("=" * 80)

    # Load data
    logger.info("Loading data...")
    X = pd.read_parquet("data/processed/feature_matrix_btc.parquet")
    targets_df = pd.read_parquet("data/processed/targets_btc_30d.parquet")
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
    logger.info(f"Date range: {X_technical.index[0]} to {X_technical.index[-1]}")

    # Split into train vs 6-month holdout
    holdout_start = pd.Timestamp("2025-07-01", tz="UTC")

    train_mask = X_technical.index < holdout_start
    holdout_mask = X_technical.index >= holdout_start

    X_train = X_technical[train_mask]
    y_train = y[train_mask]

    X_holdout = X_technical[holdout_mask]
    y_holdout = y[holdout_mask]
    returns_holdout = returns[holdout_mask]

    logger.info(f"Train samples: {len(X_train)} ({X_train.index[0]} to {X_train.index[-1]})")
    logger.info(f"Holdout samples: {len(X_holdout)} ({X_holdout.index[0]} to {X_holdout.index[-1]})")

    # Train model on ALL training data
    logger.info("\nTraining XGBoost on all training data...")
    model = XGBoostModel(random_state=0)
    model.fit(X_train, y_train)
    logger.info("✓ Model trained")

    # Predict on holdout
    logger.info("\nPredicting on 6-month holdout...")
    predictions = model.predict(X_holdout)
    logger.info(f"Holdout predictions: {predictions.sum()} longs / {len(predictions)} total ({predictions.mean():.1%})")

    # Compute holdout performance
    logger.info("\nComputing holdout performance...")

    # Simple equity curve computation
    positions = predictions  # 0 or 1
    strategy_returns = positions * returns_holdout

    # Transaction costs
    cost_model = TransactionCostModel.for_btc()
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
    logger.info("6-MONTH HOLDOUT RESULTS")
    logger.info("=" * 80)
    logger.info(f"Sharpe Ratio: {holdout_sharpe:.4f}")
    logger.info(f"Max Drawdown: {holdout_dd:.2%}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Trades: {num_trades}")
    logger.info("=" * 80)

    # Compare to 3-month and Phase 2-3
    logger.info("\nCOMPARISON:")
    logger.info(f"Phase 2-3 Sharpe (train+test): 0.999")
    logger.info(f"3-month holdout (Oct-Dec): -1.477")
    logger.info(f"6-month holdout (Jul-Dec): {holdout_sharpe:.4f}")

    delta_3m = holdout_sharpe - (-1.477)
    delta_phase23 = holdout_sharpe - 0.999

    logger.info(f"Delta vs 3-month: {delta_3m:+.4f}")
    logger.info(f"Delta vs Phase 2-3: {delta_phase23:+.4f}")

    # Verdict
    logger.info("\n" + "=" * 80)
    if holdout_sharpe >= 0.7:
        logger.info("✅ IMPROVED — 6-month holdout validates result")
        logger.info("Oct-Dec 2025 may have been unlucky period")
        logger.info("Result shows some generalization capability")
        verdict = "PASS"
    elif holdout_sharpe >= 0.4:
        logger.info("⚠️ BORDERLINE — Some improvement but still weak")
        logger.info("Suggests partial alpha but degraded from train/test")
        logger.info("May proceed with caution to OPTION 2 (debug overfitting)")
        verdict = "BORDERLINE"
    else:
        logger.info("❌ OVERFITTING CONFIRMED")
        logger.info("6-month holdout also fails (Sharpe < 0.4)")
        logger.info("Result is NOT real alpha — overfitting to train period")
        logger.info("Recommendation: Proceed to OPTION 2 (debug) or OPTION 4 (terminate)")
        verdict = "FAIL"
    logger.info("=" * 80)

    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "features": technical_features,
            "horizon": 30,
            "seed": 0,
            "holdout_months": 6,
        },
        "data_split": {
            "train_samples": len(X_train),
            "train_period": f"{X_train.index[0]} to {X_train.index[-1]}",
            "holdout_samples": len(X_holdout),
            "holdout_period": f"{X_holdout.index[0]} to {X_holdout.index[-1]}",
        },
        "results": {
            "holdout_sharpe_6m": float(holdout_sharpe),
            "holdout_sharpe_3m": -1.477,
            "phase23_sharpe": 0.999,
            "holdout_max_dd": float(holdout_dd),
            "holdout_total_return_pct": float(total_return),
            "num_trades": int(num_trades),
            "delta_vs_3m": float(delta_3m),
            "delta_vs_phase23": float(delta_phase23),
            "verdict": verdict,
        },
    }

    import json
    output_path = "results/experiments/holdout_6month_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Append to RESEARCH_LOG.md
    log_entry = f"""
---
## OPTION 1: 6-Month Holdout Test — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC

**Configuration**: Technical-only (RSI, Momentum, EMA), 30d horizon, seed=0

**Data Split**:
- Training: 2019-01-15 to 2025-06-30 ({len(X_train)} samples)
- Holdout: 2025-07-01 to 2025-12-31 ({len(X_holdout)} samples, 6 MONTHS)

**Results**:
- 6-month holdout Sharpe: {holdout_sharpe:.4f}
- Max Drawdown: {holdout_dd:.2%}
- Total Return: {total_return:.2f}%
- Trades: {num_trades}

**Comparison**:
- Phase 2-3 (train+test): Sharpe 0.999
- 3-month holdout (Oct-Dec): Sharpe -1.477
- 6-month holdout (Jul-Dec): Sharpe {holdout_sharpe:.4f}

**Verdict**: [{verdict}]
{'✅ 6-month holdout improved significantly. Oct-Dec may have been unlucky.' if verdict == 'PASS' else '⚠️ Some improvement but still weak (Sharpe 0.4-0.7). Proceed with caution.' if verdict == 'BORDERLINE' else '❌ Overfitting confirmed. 6-month holdout also fails.'}

**Next Step**: {'Proceed to multi-seed validation on 6-month holdout' if verdict == 'PASS' else 'Proceed to OPTION 2 (debug overfitting)' if verdict in ['BORDERLINE', 'FAIL'] else 'Unknown'}
"""

    with open("roadmap/RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    logger.info("Results logged to roadmap/RESEARCH_LOG.md")

    return results


if __name__ == "__main__":
    results = main()
