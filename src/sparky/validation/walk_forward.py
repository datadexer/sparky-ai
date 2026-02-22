"""Expanding-window walk-forward validation for rule-based strategies."""

import pandas as pd

from sparky.backtest.rule_based import net_ret
from sparky.tracking.metrics import compute_all_metrics

__all__ = ["run_walk_forward", "walk_forward_summary"]


def run_walk_forward(
    prices, signal_fn, cost_frac=0.0015, n_folds=5, min_train_periods=365, embargo_periods=30, ppy=365, n_trials=1
):
    """Expanding-window walk-forward for rule-based strategies.

    signal_fn(prices_slice) -> positions Series. Expanding train window,
    evaluate on each test fold. Returns fold metrics + aggregate stats.
    """
    n = len(prices)
    fold_size = (n - min_train_periods) // n_folds
    if fold_size < 30:
        raise ValueError(f"Fold size {fold_size} too small (need ≥30). Reduce n_folds or min_train_periods.")

    folds = []
    all_test_returns = []

    for i in range(n_folds):
        train_end_idx = min_train_periods + i * fold_size - 1
        test_start_idx = train_end_idx + 1 + embargo_periods
        test_end_idx = min(train_end_idx + fold_size, n - 1) if i < n_folds - 1 else n - 1

        if test_start_idx >= test_end_idx:
            continue

        test_prices = prices.iloc[test_start_idx : test_end_idx + 1]
        # Generate positions using full data up to train_end (signal_fn sees train data)
        train_prices = prices.iloc[: train_end_idx + 1]
        positions = signal_fn(train_prices)
        # Reindex positions to test period (forward-fill from last known position)
        full_idx = positions.index.append(test_prices.index)
        test_positions = positions.reindex(full_idx).ffill().reindex(test_prices.index).fillna(0)

        test_ret = net_ret(test_prices, test_positions, cost_frac)
        if len(test_ret) < 10:
            continue

        m = compute_all_metrics(test_ret, n_trials=n_trials, periods_per_year=ppy)
        n_trades = int((test_positions.diff().abs().fillna(0) > 0.01).sum())

        folds.append(
            {
                "fold": i,
                "sharpe": m["sharpe"],
                "mean_return": m["mean_return"],
                "max_drawdown": m["max_drawdown"],
                "n_trades": n_trades,
                "n_obs": len(test_ret),
                "train_end": str(prices.index[train_end_idx].date()),
                "test_start": str(test_prices.index[0].date()),
                "test_end": str(test_prices.index[-1].date()),
            }
        )
        all_test_returns.append(test_ret)

    if not folds:
        raise ValueError("No valid folds produced")

    combined = pd.concat(all_test_returns)
    agg_metrics = compute_all_metrics(combined, n_trials=n_trials, periods_per_year=ppy)
    fold_sharpes = [f["sharpe"] for f in folds]
    max_train_end = min_train_periods + (n_folds - 1) * fold_size - 1
    is_prices = prices.iloc[: max_train_end + 1]
    is_positions = signal_fn(is_prices)
    is_sharpe = compute_all_metrics(
        net_ret(is_prices, is_positions, cost_frac), n_trials=n_trials, periods_per_year=ppy
    )["sharpe"]

    return {
        "folds": folds,
        "aggregate_sharpe": agg_metrics["sharpe"],
        "retention_ratio": agg_metrics["sharpe"] / is_sharpe if is_sharpe != 0 else 0.0,
        "all_folds_positive": all(s > 0 for s in fold_sharpes),
        "n_folds": len(folds),
    }


def walk_forward_summary(wf_result):
    """Human-readable summary of walk-forward results."""
    lines = [
        f"Walk-Forward: {wf_result['n_folds']} folds, "
        f"aggregate Sharpe {wf_result['aggregate_sharpe']:.3f}, "
        f"retention {wf_result['retention_ratio']:.1%}"
    ]
    for f in wf_result["folds"]:
        lines.append(
            f"  Fold {f['fold']}: S={f['sharpe']:.3f}, "
            f"DD={f['max_drawdown']:.1%}, trades={f['n_trades']}, "
            f"{f['test_start']} to {f['test_end']}"
        )
    lines.append(f"  All folds positive: {wf_result['all_folds_positive']}")
    return "\n".join(lines)
