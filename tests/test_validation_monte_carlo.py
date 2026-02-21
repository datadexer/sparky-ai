"""Tests for Monte Carlo permutation tests."""

import numpy as np
import pandas as pd

from sparky.validation.monte_carlo import block_permutation_test, permutation_test


def _make_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    returns = rng.normal(0.0005, 0.02, n)
    prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    return prices


class TestPermutationTest:
    def test_returns_expected_keys(self):
        prices = _make_data()
        positions = pd.Series(1.0, index=prices.index)  # always long
        result = permutation_test(prices, positions, n_permutations=50, seed=42)
        assert "observed_sharpe" in result
        assert "p_value" in result
        assert "n_permutations" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_random_positions_high_pvalue(self):
        prices = _make_data(n=500)
        rng = np.random.default_rng(99)
        positions = pd.Series(rng.choice([0.0, 1.0], 500), index=prices.index)
        result = permutation_test(prices, positions, n_permutations=100, seed=42)
        # Random positions should generally not be significant
        assert result["p_value"] > 0.01


class TestBlockPermutationTest:
    def test_returns_expected_keys(self):
        prices = _make_data()
        positions = pd.Series(1.0, index=prices.index)
        result = block_permutation_test(prices, positions, n_permutations=50, seed=42)
        assert "observed_sharpe" in result
        assert "p_value" in result
        assert "block_size" in result
        assert result["block_size"] > 0

    def test_custom_block_size(self):
        prices = _make_data()
        positions = pd.Series(1.0, index=prices.index)
        result = block_permutation_test(prices, positions, n_permutations=50, block_size=20, seed=42)
        assert result["block_size"] == 20
