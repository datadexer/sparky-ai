"""Rule-based backtest utilities for non-ML strategies.

Promoted from bin/infra/sweep_utils.py — same logic, proper module location.
Supports continuous positions [-1.0 to 1.0] and fractional sizing.
"""

import pandas as pd

from sparky.tracking.metrics import compute_all_metrics

__all__ = ["net_ret", "subperiod_analysis", "compute_strategy_metrics"]

SUB_PERIODS = [
    ("full", None),
    ("2017+", "2017-01-01"),
    ("2020+", "2020-01-01"),
]


def net_ret(
    prices: pd.Series,
    positions: pd.Series,
    cost_frac: float,
) -> pd.Series:
    """Compute net returns after transaction costs.

    Args:
        prices: Price series with DatetimeIndex.
        positions: Position series [-1.0 to 1.0] with DatetimeIndex.
        cost_frac: Transaction cost as fraction (e.g. 0.003 for 30 bps).

    Returns:
        Net return series (lagged positions * price returns - costs).
    """
    if not isinstance(prices.index, pd.DatetimeIndex) or not isinstance(positions.index, pd.DatetimeIndex):
        raise ValueError("prices and positions must have DatetimeIndex")
    pr = prices.pct_change()
    lp = positions.shift(1).fillna(0)
    costs = lp.diff().abs().fillna(0) * cost_frac
    return (lp * pr - costs).dropna()


def subperiod_analysis(
    prices: pd.Series,
    positions: pd.Series,
    cost_frac: float,
    periods_per_year: int = 365,
) -> dict[str, dict]:
    """Compute metrics across standard sub-periods.

    Returns dict keyed by period label ("full", "2017+", "2020+") with
    sharpe, max_drawdown, annual_return, n_trades, win_rate, bh_sharpe.
    """
    ret = net_ret(prices, positions, cost_frac)
    bh_ret = prices.pct_change().dropna()
    out = {}
    for label, start in SUB_PERIODS:
        r = ret if start is None else ret[ret.index >= start]
        b = bh_ret if start is None else bh_ret[bh_ret.index >= start]
        if len(r) < 30:
            continue
        m = compute_all_metrics(r, n_trials=1, periods_per_year=periods_per_year)
        bh_m = compute_all_metrics(b, n_trials=1, periods_per_year=periods_per_year)
        p_slice = positions[positions.index >= start] if start else positions
        out[label] = {
            "sharpe": round(m["sharpe"], 4),
            "max_drawdown": round(m["max_drawdown"], 4),
            "annual_return": round(m["mean_return"] * periods_per_year, 4),
            "n_trades": int((p_slice.diff().abs().fillna(0) > 0.01).sum()),
            "win_rate": round(m["win_rate"], 4),
            "bh_sharpe": round(bh_m["sharpe"], 4),
        }
    return out


def compute_strategy_metrics(
    prices: pd.Series,
    positions: pd.Series,
    cost_frac: float,
    n_trials: int = 1,
    periods_per_year: int = 365,
) -> dict | None:
    """Compute full strategy metrics, returning None if <5 trades.

    Args:
        prices: Price series with DatetimeIndex.
        positions: Position series [-1.0 to 1.0] with DatetimeIndex.
        cost_frac: Transaction cost as fraction.
        n_trials: Total configs tested (for DSR).
        periods_per_year: Annualization factor.

    Returns:
        Metrics dict from compute_all_metrics + n_trades, or None if <5 trades.
    """
    n_trades = int((positions.diff().abs().fillna(0) > 0.01).sum())
    if n_trades < 5:
        return None
    ret = net_ret(prices, positions, cost_frac)
    m = compute_all_metrics(ret, n_trials=n_trials, periods_per_year=periods_per_year)
    m["n_trades"] = n_trades
    return m
