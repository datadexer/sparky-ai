#!/usr/bin/env python3
"""Prove the look-ahead bias bug in option3_strategic_pivot.py

The script claims Sharpe 2.556 for "momentum > 0.05" strategy.
This is FALSE due to look-ahead bias.

Bug: Line 157 computes returns = prices.pct_change()
     Line 178 computes signals = (momentum > 0.05)
     Line 44 (run_experiment) computes strategy_returns = signals * returns

This uses close[T] in momentum[T] to predict returns[T] which ENDS at close[T]!
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sparky.backtest.costs import TransactionCostModel
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.data.storage import DataStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def compute_sharpe(positions, returns, cost_model):
    """Compute Sharpe with transaction costs."""
    strategy_returns = positions * returns

    # Apply costs
    position_changes = np.abs(np.diff(positions, prepend=0))
    costs = position_changes * cost_model.compute_cost(1.0, "BTC")
    strategy_returns_after_costs = strategy_returns - costs

    # Metrics
    equity_curve = (1 + strategy_returns_after_costs).cumprod()
    total_return = (equity_curve.iloc[-1] - 1) * 100
    sharpe = annualized_sharpe(strategy_returns_after_costs, periods_per_year=365)
    dd = max_drawdown(equity_curve)
    num_trades = int(position_changes.sum())

    return sharpe, total_return, dd, num_trades

def main():
    logger.info("=" * 100)
    logger.info("PROVING THE LOOK-AHEAD BIAS BUG")
    logger.info("=" * 100)

    # Load data (same as option3_strategic_pivot.py)
    store = DataStore()
    btc_prices_df, _ = store.load(Path("data/raw/btc/ohlcv.parquet"))
    X_btc = pd.read_parquet("data/processed/feature_matrix_btc.parquet")

    # Use momentum feature directly as signal
    momentum = X_btc["momentum_30d"]

    # Align
    common_idx = momentum.index.intersection(btc_prices_df.index)
    momentum = momentum.loc[common_idx]
    prices = btc_prices_df["close"].loc[common_idx]

    # Remove NaN
    valid = ~momentum.isna()
    momentum = momentum[valid]
    prices = prices[valid]

    # Split (same as option3_strategic_pivot.py line 79)
    holdout_start = pd.Timestamp("2025-07-01", tz="UTC")
    holdout_idx = momentum.index >= holdout_start
    momentum_holdout = momentum[holdout_idx]
    prices_holdout = prices[holdout_idx]

    logger.info(f"Holdout period: {momentum_holdout.index[0]} to {momentum_holdout.index[-1]}")
    logger.info(f"Holdout samples: {len(momentum_holdout)}")

    # Generate signals (same as option3_strategic_pivot.py line 178)
    signals = (momentum_holdout > 0.05).astype(int)

    logger.info(f"\nSignal distribution:")
    logger.info(f"  Long (1):  {signals.sum()} days ({signals.mean():.1%})")
    logger.info(f"  Short (0): {(signals == 0).sum()} days ({(1-signals.mean()):.1%})")

    # Transaction costs
    cost_model = TransactionCostModel.for_btc()

    # =========================================================================
    # BUGGY APPROACH (what option3_strategic_pivot.py does)
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("BUGGY APPROACH (what the script does)")
    logger.info("=" * 100)

    # Line 157: returns_btc = btc_prices_df["close"].loc[common_idx].pct_change().fillna(0)
    returns_buggy = prices_holdout.pct_change().fillna(0)

    logger.info(f"\nReturns definition: returns[T] = (close[T] - close[T-1]) / close[T-1]")
    logger.info(f"This is the return FROM close[T-1] TO close[T]")
    logger.info(f"\nMomentum definition: momentum[T] = (close[T] - close[T-30]) / close[T-30]")
    logger.info(f"This USES close[T]")
    logger.info(f"\nStrategy: position[T] = 1 if momentum[T] > 0.05")
    logger.info(f"          return[T] = position[T] * returns[T]")
    logger.info(f"\n⚠️  BUG: Momentum uses close[T] to predict returns that END at close[T]!")

    sharpe_buggy, ret_buggy, dd_buggy, trades_buggy = compute_sharpe(signals, returns_buggy, cost_model)

    logger.info(f"\nBUGGY RESULTS:")
    logger.info(f"  Sharpe: {sharpe_buggy:.4f}")
    logger.info(f"  Return: {ret_buggy:.2f}%")
    logger.info(f"  Max DD: {dd_buggy:.2%}")
    logger.info(f"  Trades: {trades_buggy}")

    # =========================================================================
    # CORRECT APPROACH (shift returns forward)
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("CORRECT APPROACH (fix the bug)")
    logger.info("=" * 100)

    # Use forward returns: returns[T] should be from close[T] to close[T+1]
    returns_correct = prices_holdout.pct_change().shift(-1).fillna(0)

    logger.info(f"\nCorrected returns definition: returns[T] = (close[T+1] - close[T]) / close[T]")
    logger.info(f"This is the return FROM close[T] TO close[T+1]")
    logger.info(f"\nMomentum definition: momentum[T] = (close[T] - close[T-30]) / close[T-30]")
    logger.info(f"This USES close[T]")
    logger.info(f"\nStrategy: position[T] = 1 if momentum[T] > 0.05")
    logger.info(f"          return[T] = position[T] * forward_returns[T]")
    logger.info(f"\n✓ CORRECT: Momentum uses close[T] to predict NEXT period's return!")

    sharpe_correct, ret_correct, dd_correct, trades_correct = compute_sharpe(signals, returns_correct, cost_model)

    logger.info(f"\nCORRECT RESULTS:")
    logger.info(f"  Sharpe: {sharpe_correct:.4f}")
    logger.info(f"  Return: {ret_correct:.2f}%")
    logger.info(f"  Max DD: {dd_correct:.2%}")
    logger.info(f"  Trades: {trades_correct}")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("COMPARISON")
    logger.info("=" * 100)

    logger.info(f"\nBUGGY (with look-ahead bias):")
    logger.info(f"  Sharpe: {sharpe_buggy:.4f} ← THIS IS THE CLAIMED 2.556!")
    logger.info(f"  Return: {ret_buggy:.2f}%")

    logger.info(f"\nCORRECT (without look-ahead bias):")
    logger.info(f"  Sharpe: {sharpe_correct:.4f} ← TRUE PERFORMANCE")
    logger.info(f"  Return: {ret_correct:.2f}%")

    logger.info(f"\nDifference:")
    logger.info(f"  Sharpe degradation: {sharpe_buggy - sharpe_correct:.4f}")
    logger.info(f"  Return degradation: {ret_buggy - ret_correct:.2f}%")

    logger.info("\n" + "=" * 100)
    logger.info("VERDICT")
    logger.info("=" * 100)

    if abs(sharpe_buggy - 2.556) < 0.01:
        logger.info("\n✓ CONFIRMED: The buggy approach reproduces the claimed Sharpe 2.556")

    if sharpe_correct < 1.0:
        logger.info(f"\n❌ BUSTED: The correct Sharpe is {sharpe_correct:.4f}, NOT 2.556!")
        logger.info(f"   The claimed result is FALSE due to look-ahead bias.")
    elif sharpe_correct < 2.0:
        logger.info(f"\n⚠️  DEGRADED: The correct Sharpe is {sharpe_correct:.4f}, significantly lower than 2.556")
        logger.info(f"   The claimed result is INFLATED by {sharpe_buggy - sharpe_correct:.2f} due to look-ahead bias.")
    else:
        logger.info(f"\n✓ VALIDATED: The correct Sharpe is {sharpe_correct:.4f}, similar to the claim")
        logger.info(f"   The result appears genuine (though still needs proper validation).")

    logger.info("\n" + "=" * 100)

    # Save detailed comparison
    results = {
        "buggy": {
            "sharpe": float(sharpe_buggy),
            "return": float(ret_buggy),
            "max_dd": float(dd_buggy),
            "trades": int(trades_buggy),
        },
        "correct": {
            "sharpe": float(sharpe_correct),
            "return": float(ret_correct),
            "max_dd": float(dd_correct),
            "trades": int(trades_correct),
        },
        "degradation": {
            "sharpe": float(sharpe_buggy - sharpe_correct),
            "return": float(ret_buggy - ret_correct),
        }
    }

    import json
    with open("results/experiments/bug_proof.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Detailed results saved to results/experiments/bug_proof.json")

if __name__ == "__main__":
    main()
