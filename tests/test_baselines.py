"""Tests for baseline trading strategies.

Tests for BuyAndHold, SimpleMomentum, EqualWeight, and compute_equal_weight_equity.
"""

import numpy as np
import pandas as pd
import pytest

from sparky.models.baselines import (
    BuyAndHold,
    SimpleMomentum,
    EqualWeight,
    compute_equal_weight_equity,
)


class TestBuyAndHold:
    """Tests for BuyAndHold baseline strategy."""

    def test_predict_always_returns_ones(self):
        """BuyAndHold.predict should always return all 1s."""
        model = BuyAndHold()

        # Create test DataFrame with DatetimeIndex
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        X = pd.DataFrame({
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }, index=dates)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert np.all(predictions == 1)
        assert predictions.dtype == int

    def test_predict_empty_dataframe(self):
        """BuyAndHold.predict should handle empty DataFrame."""
        model = BuyAndHold()
        X = pd.DataFrame()

        predictions = model.predict(X)

        assert len(predictions) == 0

    def test_fit_is_noop(self):
        """BuyAndHold.fit should not error (no-op)."""
        model = BuyAndHold()

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        X = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)
        y = pd.Series([1, 0, 1, 0, 1], index=dates)

        # Should not raise any errors
        model.fit(X, y)

        # Verify predict still works after fit
        predictions = model.predict(X)
        assert np.all(predictions == 1)


class TestSimpleMomentum:
    """Tests for SimpleMomentum baseline strategy."""

    def test_predict_with_momentum_column(self):
        """SimpleMomentum should use momentum column when available."""
        model = SimpleMomentum(period=30, momentum_col="momentum")

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        X = pd.DataFrame({
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "momentum": [0.05, -0.02, 0.03, -0.01, 0.04, 0.02, -0.03, 0.01, -0.02, 0.03],
        }, index=dates)

        predictions = model.predict(X)

        # Should be 1 when momentum > 0, 0 otherwise
        expected = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
        assert np.array_equal(predictions, expected)

    def test_predict_positive_momentum_returns_one(self):
        """SimpleMomentum should return 1 for positive momentum."""
        model = SimpleMomentum()

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        X = pd.DataFrame({
            "momentum": [0.05, 0.10, 0.02, 0.001, 0.15],
        }, index=dates)

        predictions = model.predict(X)

        assert np.all(predictions == 1)

    def test_predict_negative_momentum_returns_zero(self):
        """SimpleMomentum should return 0 for negative momentum."""
        model = SimpleMomentum()

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        X = pd.DataFrame({
            "momentum": [-0.05, -0.10, -0.02, -0.001, -0.15],
        }, index=dates)

        predictions = model.predict(X)

        assert np.all(predictions == 0)

    def test_predict_zero_momentum_returns_zero(self):
        """SimpleMomentum should return 0 for zero momentum."""
        model = SimpleMomentum()

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        X = pd.DataFrame({
            "momentum": [0.0, 0.0, 0.0],
        }, index=dates)

        predictions = model.predict(X)

        assert np.all(predictions == 0)

    def test_fallback_to_close_column(self):
        """SimpleMomentum should fall back to close column if momentum not available."""
        model = SimpleMomentum(period=3)

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        # Create close prices with clear trends
        X = pd.DataFrame({
            "close": [100, 103, 106, 109, 112, 115, 118, 115, 112, 109],
        }, index=dates)

        predictions = model.predict(X)

        # First 3 will be 0 (NaN from pct_change)
        # Then momentum is computed as pct_change(3)
        # Index 3: (109-100)/100 = 0.09 > 0 → 1
        # Index 4: (112-103)/103 = 0.0874 > 0 → 1
        # Index 5: (115-106)/106 = 0.0849 > 0 → 1
        # Index 6: (118-109)/109 = 0.0826 > 0 → 1
        # Index 7: (115-112)/112 = 0.0268 > 0 → 1
        # Index 8: (112-115)/115 = -0.0261 < 0 → 0
        # Index 9: (109-118)/118 = -0.0763 < 0 → 0

        expected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
        assert np.array_equal(predictions, expected)

    def test_handles_nan_momentum_as_zero(self):
        """SimpleMomentum should handle NaN momentum as 0 (flat)."""
        model = SimpleMomentum()

        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        X = pd.DataFrame({
            "momentum": [np.nan, 0.05, -0.02, np.nan, 0.03, np.nan, -0.01, 0.04],
        }, index=dates)

        predictions = model.predict(X)

        # NaN → 0, positive → 1, negative → 0
        expected = np.array([0, 1, 0, 0, 1, 0, 0, 1])
        assert np.array_equal(predictions, expected)

    def test_fit_is_noop(self):
        """SimpleMomentum.fit should not error (no-op)."""
        model = SimpleMomentum()

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        X = pd.DataFrame({
            "momentum": [0.05, -0.02, 0.03, -0.01, 0.04],
        }, index=dates)
        y = pd.Series([1, 0, 1, 0, 1], index=dates)

        # Should not raise any errors
        model.fit(X, y)

        # Verify predict still works after fit
        predictions = model.predict(X)
        expected = np.array([1, 0, 1, 0, 1])
        assert np.array_equal(predictions, expected)

    def test_custom_momentum_column_name(self):
        """SimpleMomentum should respect custom momentum_col parameter."""
        model = SimpleMomentum(momentum_col="custom_mom")

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        X = pd.DataFrame({
            "custom_mom": [0.05, -0.02, 0.03, -0.01, 0.04],
            "momentum": [0.10, 0.10, 0.10, 0.10, 0.10],  # Should be ignored
        }, index=dates)

        predictions = model.predict(X)

        # Should use custom_mom, not momentum
        expected = np.array([1, 0, 1, 0, 1])
        assert np.array_equal(predictions, expected)


class TestEqualWeight:
    """Tests for EqualWeight baseline strategy."""

    def test_predict_always_returns_ones(self):
        """EqualWeight.predict should always return all 1s."""
        model = EqualWeight()

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        X = pd.DataFrame({
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }, index=dates)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert np.all(predictions == 1)
        assert predictions.dtype == int

    def test_predict_empty_dataframe(self):
        """EqualWeight.predict should handle empty DataFrame."""
        model = EqualWeight()
        X = pd.DataFrame()

        predictions = model.predict(X)

        assert len(predictions) == 0

    def test_fit_is_noop(self):
        """EqualWeight.fit should not error (no-op)."""
        model = EqualWeight()

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        X = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)
        y = pd.Series([1, 0, 1, 0, 1], index=dates)

        # Should not raise any errors
        model.fit(X, y)

        # Verify predict still works after fit
        predictions = model.predict(X)
        assert np.all(predictions == 1)


class TestComputeEqualWeightEquity:
    """Tests for compute_equal_weight_equity function."""

    def test_correctly_averages_two_equity_curves(self):
        """compute_equal_weight_equity should correctly average two equity curves 50/50."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")

        btc_equity = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=dates)
        eth_equity = pd.Series([1.0, 1.2, 1.4, 1.6, 1.8], index=dates)

        portfolio = compute_equal_weight_equity(btc_equity, eth_equity)

        # Expected: 50% * btc + 50% * eth
        expected = pd.Series([
            0.5 * 1.0 + 0.5 * 1.0,  # 1.0
            0.5 * 1.1 + 0.5 * 1.2,  # 1.15
            0.5 * 1.2 + 0.5 * 1.4,  # 1.3
            0.5 * 1.3 + 0.5 * 1.6,  # 1.45
            0.5 * 1.4 + 0.5 * 1.8,  # 1.6
        ], index=dates, name="portfolio")

        pd.testing.assert_series_equal(portfolio, expected)

    def test_handles_different_date_ranges(self):
        """compute_equal_weight_equity should handle different date ranges by aligning."""
        btc_dates = pd.date_range("2024-01-01", periods=7, freq="D")
        eth_dates = pd.date_range("2024-01-03", periods=7, freq="D")

        btc_equity = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], index=btc_dates)
        eth_equity = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], index=eth_dates)

        portfolio = compute_equal_weight_equity(btc_equity, eth_equity)

        # Should only include overlapping dates (2024-01-03 to 2024-01-07)
        expected_dates = pd.date_range("2024-01-03", periods=5, freq="D")
        assert len(portfolio) == 5
        pd.testing.assert_index_equal(portfolio.index, expected_dates)

        # Verify values for overlapping period
        # BTC at 2024-01-03 onwards: 1.2, 1.3, 1.4, 1.5, 1.6
        # ETH at 2024-01-03 onwards: 1.0, 1.1, 1.2, 1.3, 1.4
        expected_values = [
            0.5 * 1.2 + 0.5 * 1.0,  # 1.1
            0.5 * 1.3 + 0.5 * 1.1,  # 1.2
            0.5 * 1.4 + 0.5 * 1.2,  # 1.3
            0.5 * 1.5 + 0.5 * 1.3,  # 1.4
            0.5 * 1.6 + 0.5 * 1.4,  # 1.5
        ]
        np.testing.assert_array_almost_equal(portfolio.values, expected_values)

    def test_handles_no_overlap(self):
        """compute_equal_weight_equity should return empty series when no overlap."""
        btc_dates = pd.date_range("2024-01-01", periods=5, freq="D")
        eth_dates = pd.date_range("2024-02-01", periods=5, freq="D")

        btc_equity = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=btc_dates)
        eth_equity = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=eth_dates)

        portfolio = compute_equal_weight_equity(btc_equity, eth_equity)

        assert len(portfolio) == 0

    def test_handles_nan_values(self):
        """compute_equal_weight_equity should drop NaN values."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")

        btc_equity = pd.Series([1.0, 1.1, np.nan, 1.3, 1.4], index=dates)
        eth_equity = pd.Series([1.0, 1.2, 1.4, np.nan, 1.8], index=dates)

        portfolio = compute_equal_weight_equity(btc_equity, eth_equity)

        # Only dates without NaN in either series should remain
        # 2024-01-01: both valid → 0.5*1.0 + 0.5*1.0 = 1.0
        # 2024-01-02: both valid → 0.5*1.1 + 0.5*1.2 = 1.15
        # 2024-01-03: btc has NaN → dropped
        # 2024-01-04: eth has NaN → dropped
        # 2024-01-05: both valid → 0.5*1.4 + 0.5*1.8 = 1.6

        expected_dates = pd.DatetimeIndex([dates[0], dates[1], dates[4]])
        expected_values = [1.0, 1.15, 1.6]

        assert len(portfolio) == 3
        pd.testing.assert_index_equal(portfolio.index, expected_dates)
        np.testing.assert_array_almost_equal(portfolio.values, expected_values)

    def test_equal_weight_starts_at_one(self):
        """compute_equal_weight_equity should start at 1.0 when both inputs start at 1.0."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")

        # Both start at 1.0 and grow differently
        btc_equity = pd.Series([1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45], index=dates)
        eth_equity = pd.Series([1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.21, 1.24, 1.27], index=dates)

        portfolio = compute_equal_weight_equity(btc_equity, eth_equity)

        # First value should be 1.0
        assert portfolio.iloc[0] == 1.0

        # All subsequent values should be weighted average
        for i in range(len(portfolio)):
            expected = 0.5 * btc_equity.iloc[i] + 0.5 * eth_equity.iloc[i]
            assert abs(portfolio.iloc[i] - expected) < 1e-10
