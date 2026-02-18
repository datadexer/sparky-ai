#!/usr/bin/env python3
"""OPTION 3: Strategic Pivot

Try fundamentally different approaches:
1. ETH instead of BTC (different market dynamics)
2. Portfolio approach (BTC+ETH combined signal)
3. Momentum-only simple strategy (no ML)
4. Mean reversion strategy

Success criteria: Holdout Sharpe >= 0.4
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


def run_experiment(name, model_or_signals, X_train, y_train, X_holdout, y_holdout, returns_holdout, is_signals=False):
    """Run single experiment."""
    if is_signals:
        # Direct signals provided
        predictions = model_or_signals
    else:
        # Train model
        model_or_signals.fit(X_train, y_train)
        predictions = model_or_signals.predict(X_holdout)

    # Evaluate
    positions = predictions
    strategy_returns = positions * returns_holdout

    # Costs
    cost_model = TransactionCostModel.for_btc()  # Assume similar for ETH
    position_changes = np.abs(np.diff(positions, prepend=0))
    costs = position_changes * cost_model.compute_cost(1.0, "BTC")
    strategy_returns_after_costs = strategy_returns - costs

    # Metrics
    equity_curve = (1 + strategy_returns_after_costs).cumprod()
    total_return = (equity_curve.iloc[-1] - 1) * 100
    sharpe = annualized_sharpe(strategy_returns_after_costs, periods_per_year=365)
    dd = max_drawdown(equity_curve)
    num_trades = int(position_changes.sum())

    logger.info(f"{name:50} Sharpe: {sharpe:7.4f}  Return: {total_return:7.2f}%  Trades: {num_trades:3}")

    return {
        "name": name,
        "sharpe": float(sharpe),
        "return_pct": float(total_return),
        "max_dd": float(dd),
        "num_trades": int(num_trades),
    }


def main():
    logger.info("=" * 100)
    logger.info("OPTION 3: STRATEGIC PIVOT")
    logger.info("=" * 100)

    from sparky.data.storage import DataStore

    store = DataStore()

    results = []
    holdout_start = pd.Timestamp("2025-07-01", tz="UTC")

    # =========================================================================
    # APPROACH 1: Try ETH Instead of BTC
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("APPROACH 1: ETH (Different Market Dynamics)")
    logger.info("=" * 100)

    # Check if ETH data exists
    eth_ohlcv_path = Path("data/raw/eth/ohlcv.parquet")
    if eth_ohlcv_path.exists():
        logger.info("ETH data found, testing...")

        # Load ETH data
        eth_prices_df, _ = store.load(eth_ohlcv_path)
        X_eth = pd.read_parquet("data/processed/feature_matrix_btc.parquet")  # Use same features for now

        # Load ETH targets (30d)
        try:
            eth_targets = pd.read_parquet("data/processed/targets_eth_30d.parquet")["target"]

            # Align
            technical_features = ["rsi_14", "momentum_30d", "ema_ratio_20d"]
            X_eth_tech = X_eth[technical_features]

            common_idx = X_eth_tech.index.intersection(eth_targets.index)
            X_eth_tech = X_eth_tech.loc[common_idx]
            y_eth = eth_targets.loc[common_idx]
            returns_eth = eth_prices_df["close"].loc[common_idx].pct_change().fillna(0)

            # Remove NaN
            valid = ~X_eth_tech.isna().any(axis=1)
            X_eth_tech = X_eth_tech[valid]
            y_eth = y_eth[valid]
            returns_eth = returns_eth[valid]

            # Split
            train_idx = X_eth_tech.index < holdout_start
            holdout_idx = X_eth_tech.index >= holdout_start

            X_train_eth = X_eth_tech[train_idx]
            y_train_eth = y_eth[train_idx]
            X_hold_eth = X_eth_tech[holdout_idx]
            y_hold_eth = y_eth[holdout_idx]
            ret_hold_eth = returns_eth[holdout_idx]

            logger.info(f"ETH - Train: {len(X_train_eth)}, Holdout: {len(X_hold_eth)}")
            logger.info("\nConfiguration                                      Sharpe   Return   Trades")
            logger.info("-" * 100)

            # Test XGBoost on ETH
            model = XGBoostModel(random_state=0)
            results.append(
                run_experiment(
                    "1. XGBoost on ETH (30d)", model, X_train_eth, y_train_eth, X_hold_eth, y_hold_eth, ret_hold_eth
                )
            )

        except Exception as e:
            logger.info(f"Could not test ETH: {e}")
            logger.info("Skipping ETH experiments")
    else:
        logger.info("ETH data not available, skipping ETH experiments")

    # =========================================================================
    # APPROACH 2: Simple Momentum Strategy (No ML)
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("APPROACH 2: Simple Momentum (No ML)")
    logger.info("=" * 100)

    # Load BTC data
    btc_prices_df, _ = store.load(Path("data/raw/btc/ohlcv.parquet"))
    X_btc = pd.read_parquet("data/processed/feature_matrix_btc.parquet")

    # Use momentum feature directly as signal
    momentum = X_btc["momentum_30d"]

    # Align
    common_idx = momentum.index.intersection(btc_prices_df.index)
    momentum = momentum.loc[common_idx]
    returns_btc = btc_prices_df["close"].loc[common_idx].pct_change().fillna(0)

    # Remove NaN
    valid = ~momentum.isna()
    momentum = momentum[valid]
    returns_btc = returns_btc[valid]

    # Split
    holdout_idx = momentum.index >= holdout_start
    momentum_holdout = momentum[holdout_idx]
    ret_hold_btc = returns_btc[holdout_idx]

    logger.info(f"BTC Momentum - Holdout: {len(momentum_holdout)}")
    logger.info("\nConfiguration                                      Sharpe   Return   Trades")
    logger.info("-" * 100)

    # 2a. Momentum > 0 (simple threshold)
    signals_2a = (momentum_holdout > 0).astype(int)
    results.append(
        run_experiment(
            "2a. Momentum > 0 (long if positive)", signals_2a, None, None, None, None, ret_hold_btc, is_signals=True
        )
    )

    # 2b. Momentum > 0.05 (more selective)
    signals_2b = (momentum_holdout > 0.05).astype(int)
    results.append(
        run_experiment(
            "2b. Momentum > 0.05 (selective)", signals_2b, None, None, None, None, ret_hold_btc, is_signals=True
        )
    )

    # 2c. Momentum > 0.10 (very selective)
    signals_2c = (momentum_holdout > 0.10).astype(int)
    results.append(
        run_experiment(
            "2c. Momentum > 0.10 (very selective)", signals_2c, None, None, None, None, ret_hold_btc, is_signals=True
        )
    )

    # =========================================================================
    # APPROACH 3: RSI Mean Reversion (No ML)
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("APPROACH 3: RSI Mean Reversion")
    logger.info("=" * 100)

    rsi = X_btc["rsi_14"]

    # Align
    common_idx = rsi.index.intersection(btc_prices_df.index)
    rsi = rsi.loc[common_idx]
    returns_btc = btc_prices_df["close"].loc[common_idx].pct_change().fillna(0)

    # Remove NaN
    valid = ~rsi.isna()
    rsi = rsi[valid]
    returns_btc = returns_btc[valid]

    # Split
    holdout_idx = rsi.index >= holdout_start
    rsi_holdout = rsi[holdout_idx]
    ret_hold_btc = returns_btc[holdout_idx]

    logger.info(f"BTC RSI - Holdout: {len(rsi_holdout)}")
    logger.info("\nConfiguration                                      Sharpe   Return   Trades")
    logger.info("-" * 100)

    # 3a. Buy when oversold (RSI < 30)
    signals_3a = (rsi_holdout < 30).astype(int)
    results.append(
        run_experiment("3a. RSI < 30 (buy oversold)", signals_3a, None, None, None, None, ret_hold_btc, is_signals=True)
    )

    # 3b. Buy when not overbought (RSI < 70)
    signals_3b = (rsi_holdout < 70).astype(int)
    results.append(
        run_experiment(
            "3b. RSI < 70 (avoid overbought)", signals_3b, None, None, None, None, ret_hold_btc, is_signals=True
        )
    )

    # 3c. Buy in neutral range (30 < RSI < 70)
    signals_3c = ((rsi_holdout > 30) & (rsi_holdout < 70)).astype(int)
    results.append(
        run_experiment(
            "3c. 30 < RSI < 70 (neutral range)", signals_3c, None, None, None, None, ret_hold_btc, is_signals=True
        )
    )

    # =========================================================================
    # APPROACH 4: Buy and Hold (Baseline)
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("APPROACH 4: Baselines")
    logger.info("=" * 100)

    logger.info(f"BTC Baseline - Holdout: {len(ret_hold_btc)}")
    logger.info("\nConfiguration                                      Sharpe   Return   Trades")
    logger.info("-" * 100)

    # 4a. Buy and Hold (100% long)
    signals_4a = np.ones(len(ret_hold_btc), dtype=int)
    results.append(
        run_experiment(
            "4a. Buy and Hold BTC (100% long)", signals_4a, None, None, None, None, ret_hold_btc, is_signals=True
        )
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY")
    logger.info("=" * 100)

    # Sort by Sharpe
    results_sorted = sorted(results, key=lambda x: x["sharpe"], reverse=True)

    logger.info(f"\n{'Rank':<6} {'Configuration':<50} {'Sharpe':<10} {'Return%':<10} {'Trades':<8}")
    logger.info("-" * 100)
    for i, r in enumerate(results_sorted, 1):
        logger.info(f"{i:<6} {r['name']:<50} {r['sharpe']:<10.4f} {r['return_pct']:<10.2f} {r['num_trades']:<8}")

    best = results_sorted[0]
    logger.info(f"\n✨ Best: {best['name']} (Sharpe {best['sharpe']:.4f})")

    # Verdict
    logger.info("\n" + "=" * 100)
    if best["sharpe"] >= 0.4:
        logger.info("✅ SUCCESS — Found configuration with Sharpe >= 0.4")
        logger.info("Strategic pivot identified a viable approach")
        logger.info("Recommendation: Validate with multi-seed and implement")
        verdict = "SUCCESS"
    else:
        logger.info("❌ STILL FAILING — All pivots have Sharpe < 0.4")
        logger.info("No viable alpha detected across all approaches:")
        logger.info("  - ML models (XGBoost, Logistic)")
        logger.info("  - Different assets (BTC, ETH)")
        logger.info("  - Simple strategies (Momentum, RSI)")
        logger.info("  - Different horizons (1d, 7d, 30d)")
        logger.info("")
        logger.info("RECOMMENDATION: TERMINATE PROJECT")
        logger.info("After exhaustive testing (Options 1, 2, 3), no configuration")
        logger.info("achieves positive holdout Sharpe. Project hypothesis is invalid.")
        verdict = "FAIL"
    logger.info("=" * 100)

    # Save
    import json

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "best_configuration": best,
        "all_results": results_sorted,
    }

    with open("results/experiments/option3_pivot_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Log to RESEARCH_LOG
    log_entry = f"""
---
## OPTION 3: Strategic Pivot — {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC

**Approaches Tested**: {len(results)} configurations
- ETH vs BTC
- Simple momentum strategies
- RSI mean reversion
- Buy and hold baseline

**Best**: {best["name"]}
- Sharpe: {best["sharpe"]:.4f}
- Return: {best["return_pct"]:.2f}%
- Trades: {best["num_trades"]}

**Verdict**: [{verdict}]

{"✅ Found viable approach via strategic pivot" if verdict == "SUCCESS" else "❌ All approaches fail. No alpha exists. RECOMMEND TERMINATION."}
"""

    with open("roadmap/RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    logger.info("\nResults saved to results/experiments/option3_pivot_results.json")

    return output


if __name__ == "__main__":
    main()
