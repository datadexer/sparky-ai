"""Tests for CPCV validation wrapper."""

import numpy as np
import pandas as pd

from sparky.validation.cpcv import probability_of_overfitting, run_cpcv


def _make_prices(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    returns = rng.normal(0.0003, 0.02, n)
    prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    return prices


def _trending_positions(prices):
    """Simple momentum: long when 20d MA > 50d MA."""
    ma20 = prices.rolling(20).mean()
    ma50 = prices.rolling(50).mean()
    pos = (ma20 > ma50).astype(float)
    pos[:50] = 0.0
    return pos


class TestRunCpcv:
    def test_returns_expected_keys(self):
        prices = _make_prices()
        result = run_cpcv(prices, _trending_positions, n_groups=6, cost_frac=0.0015)
        assert "pbo" in result
        assert "n_paths" in result
        assert "median_path_sharpe" in result
        assert "sharpe_distribution" in result
        assert 0.0 <= result["pbo"] <= 1.0

    def test_more_groups(self):
        prices = _make_prices(n=2000)
        result = run_cpcv(prices, _trending_positions, n_groups=12, cost_frac=0.0015)
        assert result["n_paths"] > 0


class TestProbabilityOfOverfitting:
    def test_all_positive(self):
        assert probability_of_overfitting([1.0, 0.5, 2.0]) == 0.0

    def test_all_negative(self):
        assert probability_of_overfitting([-1.0, -0.5]) == 1.0

    def test_mixed(self):
        pbo = probability_of_overfitting([1.0, -0.5, 0.5, -1.0])
        assert abs(pbo - 0.5) < 1e-10

    def test_empty(self):
        assert probability_of_overfitting([]) == 1.0
