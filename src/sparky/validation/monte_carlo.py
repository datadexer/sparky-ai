"""Monte Carlo permutation tests for strategy validation."""

import numpy as np
import pandas as pd

from sparky.backtest.rule_based import net_ret
from sparky.tracking.metrics import compute_all_metrics

__all__ = ["permutation_test", "block_permutation_test"]


def _sharpe(prices, positions, cost_frac):
    """Compute annualized Sharpe from prices + positions."""
    r = net_ret(prices, positions, cost_frac)
    if len(r) < 2:
        return 0.0
    return compute_all_metrics(r, n_trials=1, periods_per_year=365)["sharpe"]


def permutation_test(prices, positions, n_permutations=1000, cost_frac=0.0015, seed=42):
    """Circular-shift permutation test.

    Shifts positions by a random offset, recomputes net_ret + Sharpe.
    p_value = fraction of permutation Sharpes >= observed Sharpe.
    """
    rng = np.random.default_rng(seed)
    observed_sharpe = _sharpe(prices, positions, cost_frac)

    n = len(positions)
    perm_sharpes = []
    pos_values = positions.values.copy()

    for _ in range(n_permutations):
        offset = rng.integers(1, n)
        shifted = pd.Series(np.roll(pos_values, offset), index=positions.index)
        s = _sharpe(prices, shifted, cost_frac)
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = float(np.mean(perm_sharpes >= observed_sharpe))

    return {
        "observed_sharpe": observed_sharpe,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "perm_sharpe_mean": float(np.mean(perm_sharpes)),
        "perm_sharpe_std": float(np.std(perm_sharpes)),
    }


def block_permutation_test(prices, positions, n_permutations=1000, block_size=None, cost_frac=0.0015, seed=42):
    """Block permutation test — shuffles blocks to preserve autocorrelation.

    Default block_size = int(sqrt(N)).
    """
    rng = np.random.default_rng(seed)
    observed_sharpe = _sharpe(prices, positions, cost_frac)

    n = len(positions)
    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))

    pos_values = positions.values.copy()
    n_blocks = n // block_size
    remainder = n - n_blocks * block_size

    perm_sharpes = []
    for _ in range(n_permutations):
        blocks = [pos_values[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
        if remainder > 0:
            blocks.append(pos_values[n_blocks * block_size :])
        rng.shuffle(blocks)
        shuffled = np.concatenate(blocks)[:n]
        shifted = pd.Series(shuffled, index=positions.index)
        s = _sharpe(prices, shifted, cost_frac)
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = float(np.mean(perm_sharpes >= observed_sharpe))

    return {
        "observed_sharpe": observed_sharpe,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "block_size": block_size,
        "perm_sharpe_mean": float(np.mean(perm_sharpes)),
        "perm_sharpe_std": float(np.std(perm_sharpes)),
    }
