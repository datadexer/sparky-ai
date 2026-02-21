"""CPCV validation wrapper — callback interface for rule-based strategies."""

import numpy as np

from sparky.backtest.cpcv import cpcv_paths
from sparky.backtest.rule_based import net_ret

__all__ = ["run_cpcv", "probability_of_overfitting"]


def run_cpcv(prices, positions_func, n_groups=12, k_test=2, purge_window=5, embargo_window=5, cost_frac=0.0015):
    """Run CPCV on a rule-based strategy via callback interface.

    positions_func(prices) is called once to generate positions.
    Returns are computed via net_ret, then passed to cpcv_paths.
    k_test is accepted for API compatibility but cpcv_paths always uses n_groups//2.
    """
    positions = positions_func(prices)
    returns = net_ret(prices, positions, cost_frac)
    result = cpcv_paths(returns.values, n_groups=n_groups, purge_days=purge_window, ppy=365)
    return result


def probability_of_overfitting(path_sharpes):
    """Fraction of path Sharpes below zero (PBO)."""
    arr = np.asarray(path_sharpes, dtype=float)
    if len(arr) == 0:
        return 1.0
    return float(np.mean(arr < 0))
