"""Tests for technical indicator calculations.

CRITICAL: RSI must use Wilder's smoothing (not SMA).
CRITICAL: Momentum sign convention — positive = price went UP = bullish.

Cross-validation against pandas_ta is in tests/test_cross_validation.py.
"""

import numpy as np
import pandas as pd
import pytest

from sparky.features.technical import ema, macd, momentum, rsi


class TestEMA:
    """Test Exponential Moving Average."""

    def test_ema_converges_to_value(self):
        """EMA of a constant series should equal that constant."""
        series = pd.Series([50.0] * 20)
        result = ema(series, span=10)
        assert result.iloc[-1] == pytest.approx(50.0)

    def test_ema_weights_recent_more(self):
        """EMA should be closer to recent values than old ones."""
        series = pd.Series([10.0] * 10 + [20.0] * 10)
        result = ema(series, span=5)
        # After 10 values of 20, EMA should be close to 20
        assert result.iloc[-1] > 19.0

    def test_ema_span_1_equals_original(self):
        """EMA with span=1 should equal the original series (alpha=1)."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(series, span=1)
        pd.testing.assert_series_equal(result, series)


class TestRSI:
    """Test RSI with Wilder's smoothing."""

    def test_rsi_all_gains(self):
        """Monotonically increasing prices should give RSI near 100."""
        prices = pd.Series([float(i) for i in range(1, 50)])
        result = rsi(prices, period=14)
        # After sufficient warmup, RSI should be near 100
        assert result.iloc[-1] > 95

    def test_rsi_all_losses(self):
        """Monotonically decreasing prices should give RSI near 0."""
        prices = pd.Series([float(50 - i) for i in range(49)])
        result = rsi(prices, period=14)
        assert result.iloc[-1] < 5

    def test_rsi_range(self):
        """RSI should always be between 0 and 100."""
        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.randn(200)) + 100)
        result = rsi(prices, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_first_period_nan(self):
        """First `period` values should be NaN."""
        prices = pd.Series(range(1, 31), dtype=float)
        result = rsi(prices, period=14)
        assert result.iloc[:14].isna().all()
        assert result.iloc[14:].notna().all()

    def test_rsi_known_value(self):
        """RSI of balanced gains/losses should be near 50."""
        # Alternating +1, -1 changes
        prices = pd.Series([100.0])
        for i in range(30):
            if i % 2 == 0:
                prices = pd.concat([prices, pd.Series([prices.iloc[-1] + 1.0])], ignore_index=True)
            else:
                prices = pd.concat([prices, pd.Series([prices.iloc[-1] - 1.0])], ignore_index=True)
        result = rsi(prices, period=14)
        # With equal up/down moves, RSI should be near 50
        assert 40 < result.iloc[-1] < 60


class TestMACD:
    """Test MACD calculation."""

    def test_macd_trend(self):
        """Uptrending prices should give positive MACD."""
        prices = pd.Series([float(i) for i in range(100)])
        macd_line, signal_line, histogram = macd(prices)
        # In a strong uptrend, MACD should be positive
        assert macd_line.iloc[-1] > 0

    def test_macd_components(self):
        """Histogram should equal MACD - Signal."""
        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        macd_line, signal_line, histogram = macd(prices)
        expected_hist = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected_hist)

    def test_macd_constant_prices(self):
        """Constant prices should give MACD near 0."""
        prices = pd.Series([100.0] * 50)
        macd_line, signal_line, histogram = macd(prices)
        assert macd_line.iloc[-1] == pytest.approx(0.0)


class TestMomentum:
    """Test momentum calculation with sign convention.

    v1 had a sign inversion bug here. These tests explicitly validate
    the sign convention: positive momentum = price went UP = bullish.
    """

    def test_positive_momentum(self):
        """Prices going up should give positive momentum."""
        prices = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0])
        result = momentum(prices, period=2)
        # momentum at index 2: (110-100)/100 = 0.1
        assert result.iloc[2] == pytest.approx(0.1)
        # momentum at index 4: (120-110)/110 = 0.0909
        assert result.iloc[4] == pytest.approx(10.0 / 110.0)

    def test_negative_momentum(self):
        """Prices going down should give negative momentum."""
        prices = pd.Series([120.0, 115.0, 110.0, 105.0, 100.0])
        result = momentum(prices, period=2)
        # momentum at index 2: (110-120)/120 = -0.0833
        assert result.iloc[2] < 0

    def test_momentum_sign_convention(self):
        """CRITICAL: Verify sign convention explicitly.

        Bull market -> positive momentum -> bullish signal.
        This is where v1 had the bug.
        """
        # Bull market
        bull_prices = pd.Series([100.0 + i * 2.0 for i in range(40)])
        bull_mom = momentum(bull_prices, period=30)
        assert bull_mom.iloc[-1] > 0, "Bull market must give positive momentum"

        # Bear market
        bear_prices = pd.Series([200.0 - i * 2.0 for i in range(40)])
        bear_mom = momentum(bear_prices, period=30)
        assert bear_mom.iloc[-1] < 0, "Bear market must give negative momentum"

    def test_momentum_first_period_nan(self):
        """First `period` values should be NaN."""
        prices = pd.Series(range(1, 41), dtype=float)
        result = momentum(prices, period=30)
        assert result.iloc[:30].isna().all()
        assert result.iloc[30:].notna().all()
