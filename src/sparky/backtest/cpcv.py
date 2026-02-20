"""Combinatorial Purged Cross-Validation (CPCV).

For fixed-parameter strategies: no training. Tests time-stability by evaluating
on all C(N, N//2) path combinations of data groups.

Reference: de Prado, "Advances in Financial Machine Learning" (2018), Ch. 12.
"""

from itertools import combinations
from math import comb

import numpy as np


def cpcv_paths(returns, n_groups=6, purge_days=5, ppy=365):
    """Run CPCV on a return series.

    Splits returns into n_groups contiguous blocks, purges boundary observations,
    then evaluates annualized Sharpe on all C(n_groups, n_groups//2) path
    combinations of test blocks.

    Args:
        returns: 1-D array-like of period returns.
        n_groups: Number of contiguous blocks (default 6 → 20 paths).
        purge_days: Observations to remove at each block boundary.
        ppy: Periods per year for annualization.

    Returns:
        dict with keys: paths (list of dicts), pbo, sharpe_distribution,
        n_paths, median_path_sharpe, mean_path_sharpe.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < n_groups * (purge_days + 2):
        raise ValueError(f"Too few observations ({n}) for {n_groups} groups with {purge_days}-day purge")

    block_size = n // n_groups
    blocks = []
    for i in range(n_groups):
        start = i * block_size
        end = start + block_size if i < n_groups - 1 else n
        # Purge boundaries (skip first purge_days of each non-first block)
        if i > 0:
            start += purge_days
        if start < end:
            blocks.append(returns[start:end])
        else:
            blocks.append(np.array([]))

    test_size = n_groups // 2
    combos = list(combinations(range(n_groups), test_size))

    paths = []
    sharpes = []
    for combo in combos:
        # Concatenate test blocks in chronological order
        test_returns = np.concatenate([blocks[i] for i in sorted(combo)])
        if len(test_returns) < 10:
            continue
        std = np.std(test_returns, ddof=1)
        if std > 0:
            sr = (np.mean(test_returns) / std) * np.sqrt(ppy)
        else:
            sr = 0.0
        sharpes.append(sr)
        paths.append({"test_groups": sorted(combo), "sharpe": float(sr), "n_obs": len(test_returns)})

    sharpes = np.array(sharpes)
    pbo = float(np.mean(sharpes < 0)) if len(sharpes) > 0 else 1.0

    return {
        "paths": paths,
        "pbo": pbo,
        "sharpe_distribution": sharpes.tolist(),
        "n_paths": len(paths),
        "n_expected_paths": comb(n_groups, test_size),
        "median_path_sharpe": float(np.median(sharpes)) if len(sharpes) > 0 else 0.0,
        "mean_path_sharpe": float(np.mean(sharpes)) if len(sharpes) > 0 else 0.0,
    }
