"""Audit tests for block_bootstrap_monte_carlo — edge cases and properties."""

import numpy as np
import pandas as pd

from sparky.backtest.statistics import BacktestStatistics


def _make_returns(n=500, seed=42):
    rng = np.random.RandomState(seed)
    r = pd.Series(rng.randn(n) * 0.02 + 0.001)
    for i in range(1, n):
        r.iloc[i] += 0.3 * r.iloc[i - 1]
    market = pd.Series(rng.randn(n) * 0.03 + 0.0005)
    return r, market


def test_deterministic_seeding():
    """Same global seed → same output."""
    s, m = _make_returns()
    np.random.seed(123)
    r1 = BacktestStatistics.block_bootstrap_monte_carlo(s, m, n_simulations=50, block_size=10)
    np.random.seed(123)
    r2 = BacktestStatistics.block_bootstrap_monte_carlo(s, m, n_simulations=50, block_size=10)
    assert r1["wins"] == r2["wins"]
    assert r1["losses"] == r2["losses"]


def test_block_size_exceeds_series():
    """block_size > n should not crash — clamped by auto-sizing or handled."""
    s = pd.Series(np.random.RandomState(42).randn(20) * 0.01)
    m = pd.Series(np.random.RandomState(43).randn(20) * 0.01)
    # Auto-sizing: sqrt(20) ≈ 4, clamped to 5
    result = BacktestStatistics.block_bootstrap_monte_carlo(s, m, n_simulations=10)
    assert result["wins"] + result["ties"] + result["losses"] == 10
    assert result["block_size"] == 5  # max(5, min(sqrt(20)=4, 50)) → 5


def test_very_short_series():
    """n=10, block_size auto-clamped to 5."""
    rng = np.random.RandomState(99)
    s = pd.Series(rng.randn(10) * 0.02 + 0.005)
    m = pd.Series(rng.randn(10) * 0.02)
    result = BacktestStatistics.block_bootstrap_monte_carlo(s, m, n_simulations=20)
    assert result["block_size"] == 5
    assert result["wins"] + result["ties"] + result["losses"] == 20


def test_output_distribution_shape():
    """Win/tie/loss counts are non-negative and sum to n_simulations."""
    s, m = _make_returns(200, seed=7)
    result = BacktestStatistics.block_bootstrap_monte_carlo(s, m, n_simulations=500, block_size=8)
    assert result["wins"] >= 0
    assert result["ties"] >= 0
    assert result["losses"] >= 0
    assert result["wins"] + result["ties"] + result["losses"] == 500
    assert 0 <= result["win_rate"] <= 1


def test_strong_signal_high_win_rate():
    """Obviously profitable strategy should win most simulations."""
    rng = np.random.RandomState(42)
    n = 500
    # Strategy: consistent +0.5% daily (huge edge)
    s = pd.Series(rng.randn(n) * 0.005 + 0.005)
    # Market: roughly zero mean
    m = pd.Series(rng.randn(n) * 0.02)
    result = BacktestStatistics.block_bootstrap_monte_carlo(s, m, n_simulations=200, block_size=10)
    assert result["win_rate"] > 0.8
