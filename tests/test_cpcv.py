"""Tests for Combinatorial Purged Cross-Validation."""

import numpy as np
import pytest

from sparky.backtest.cpcv import cpcv_paths


def test_n_paths_6_groups():
    """C(6,3) = 20 paths."""
    rng = np.random.RandomState(42)
    returns = rng.randn(600) * 0.01 + 0.001
    result = cpcv_paths(returns, n_groups=6, purge_days=2, ppy=365)
    assert result["n_expected_paths"] == 20
    assert result["n_paths"] == 20
    assert len(result["paths"]) == 20
    assert len(result["sharpe_distribution"]) == 20


def test_n_paths_4_groups():
    """C(4,2) = 6 paths."""
    rng = np.random.RandomState(42)
    returns = rng.randn(400) * 0.01
    result = cpcv_paths(returns, n_groups=4, purge_days=2, ppy=365)
    assert result["n_expected_paths"] == 6
    assert result["n_paths"] == 6


def test_strong_signal_low_pbo():
    """All-positive returns → PBO ≈ 0."""
    rng = np.random.RandomState(42)
    returns = np.abs(rng.randn(600)) * 0.01 + 0.005  # strictly positive mean
    result = cpcv_paths(returns, n_groups=6, purge_days=2, ppy=365)
    assert result["pbo"] < 0.1
    assert result["median_path_sharpe"] > 0


def test_noise_higher_pbo():
    """Pure noise → PBO should be substantial (> 0.2)."""
    rng = np.random.RandomState(42)
    returns = rng.randn(600) * 0.01  # zero mean
    result = cpcv_paths(returns, n_groups=6, purge_days=2, ppy=365)
    assert result["pbo"] > 0.2


def test_purge_removes_observations():
    """Purged result has fewer total observations than unpurged."""
    rng = np.random.RandomState(42)
    returns = rng.randn(600) * 0.01 + 0.001
    r_purged = cpcv_paths(returns, n_groups=6, purge_days=10, ppy=365)
    r_nopurge = cpcv_paths(returns, n_groups=6, purge_days=0, ppy=365)
    # Each path's n_obs should be smaller with purge
    purged_obs = sum(p["n_obs"] for p in r_purged["paths"])
    nopurge_obs = sum(p["n_obs"] for p in r_nopurge["paths"])
    assert purged_obs < nopurge_obs


def test_too_few_observations():
    """Should raise ValueError for tiny series."""
    returns = np.array([0.01, 0.02, 0.03])
    with pytest.raises(ValueError, match="Too few observations"):
        cpcv_paths(returns, n_groups=6, purge_days=5)


def test_path_sharpes_are_finite():
    """All path Sharpes should be finite numbers."""
    rng = np.random.RandomState(42)
    returns = rng.randn(1000) * 0.01 + 0.0005
    result = cpcv_paths(returns, n_groups=6, purge_days=3, ppy=365)
    for s in result["sharpe_distribution"]:
        assert np.isfinite(s)
