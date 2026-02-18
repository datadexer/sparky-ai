"""Dedicated sign convention validation tests.

This file exists because v1 had a sign inversion bug that went
undetected for weeks, invalidating all results. These tests
explicitly validate every sign-sensitive scenario.

Test every scenario:
- Bull market (prices trending up) -> positive momentum -> positive signal
- Bear market (prices trending down) -> negative momentum -> negative signal
- A model that always predicts "up" in a bull market -> positive returns
- A model that always predicts "down" in a bull market -> negative returns
- Verify: sign(prediction) * sign(actual_return) > 0 means CORRECT prediction
"""

import numpy as np
import pandas as pd

from sparky.features.returns import log_returns, simple_returns
from sparky.features.technical import momentum, rsi


class TestMomentumSignConvention:
    """Validate that momentum sign matches market direction."""

    def test_bull_market_positive_momentum(self):
        """Bull market (prices trending up) -> positive momentum."""
        prices = pd.Series([100.0 + i * 3.0 for i in range(50)])
        mom = momentum(prices, period=20)
        # All valid momentum values should be positive
        valid = mom.dropna()
        assert (valid > 0).all(), "Bull market must produce positive momentum"

    def test_bear_market_negative_momentum(self):
        """Bear market (prices trending down) -> negative momentum."""
        prices = pd.Series([200.0 - i * 3.0 for i in range(50)])
        mom = momentum(prices, period=20)
        valid = mom.dropna()
        assert (valid < 0).all(), "Bear market must produce negative momentum"

    def test_sideways_market_near_zero(self):
        """Sideways market should give near-zero momentum."""
        prices = pd.Series([100.0] * 50)
        mom = momentum(prices, period=20)
        valid = mom.dropna()
        assert (valid.abs() < 0.001).all(), "Flat market should have near-zero momentum"


class TestReturnSignConvention:
    """Validate that returns correctly reflect price changes."""

    def test_price_increase_positive_return(self):
        """Price going up must give positive return."""
        prices = pd.Series([100.0, 110.0])
        sr = simple_returns(prices)
        lr = log_returns(prices)
        assert sr.iloc[1] > 0, "Price increase must give positive simple return"
        assert lr.iloc[1] > 0, "Price increase must give positive log return"

    def test_price_decrease_negative_return(self):
        """Price going down must give negative return."""
        prices = pd.Series([100.0, 90.0])
        sr = simple_returns(prices)
        lr = log_returns(prices)
        assert sr.iloc[1] < 0, "Price decrease must give negative simple return"
        assert lr.iloc[1] < 0, "Price decrease must give negative log return"


class TestRSISignConvention:
    """Validate RSI direction in trending markets."""

    def test_bull_market_high_rsi(self):
        """Strong uptrend should produce RSI > 70."""
        prices = pd.Series([100.0 + i * 5.0 for i in range(30)])
        r = rsi(prices, period=14)
        assert r.iloc[-1] > 70, "Strong uptrend must produce high RSI"

    def test_bear_market_low_rsi(self):
        """Strong downtrend should produce RSI < 30."""
        prices = pd.Series([200.0 - i * 5.0 for i in range(30)])
        r = rsi(prices, period=14)
        assert r.iloc[-1] < 30, "Strong downtrend must produce low RSI"


class TestPredictionCorrectness:
    """Validate the fundamental prediction correctness formula.

    sign(prediction) * sign(actual_return) > 0 means CORRECT prediction.
    """

    def test_correct_up_prediction(self):
        """Predict UP in bull market = correct."""
        prediction = 1  # Predicts up
        actual_return = 0.05  # Price went up 5%
        assert np.sign(prediction) * np.sign(actual_return) > 0

    def test_correct_down_prediction(self):
        """Predict DOWN in bear market = correct."""
        prediction = -1  # Predicts down
        actual_return = -0.05  # Price went down 5%
        assert np.sign(prediction) * np.sign(actual_return) > 0

    def test_incorrect_up_prediction(self):
        """Predict UP in bear market = incorrect."""
        prediction = 1  # Predicts up
        actual_return = -0.05  # Price went down 5%
        assert np.sign(prediction) * np.sign(actual_return) < 0

    def test_incorrect_down_prediction(self):
        """Predict DOWN in bull market = incorrect."""
        prediction = -1  # Predicts down
        actual_return = 0.05  # Price went up 5%
        assert np.sign(prediction) * np.sign(actual_return) < 0

    def test_always_up_in_bull_market(self):
        """Model that always predicts 'up' in a bull market -> positive returns."""
        # Simulate: bull market prices
        prices = pd.Series([100.0 + i * 2.0 for i in range(50)])
        returns = simple_returns(prices).dropna()

        # Model always predicts +1 (up)
        predictions = pd.Series([1] * len(returns))

        # Strategy returns = prediction * actual_return
        strategy_returns = predictions * returns

        # In a bull market, always predicting up should give positive cumulative return
        assert strategy_returns.sum() > 0, "Always-up prediction in bull market must yield positive returns"

    def test_always_down_in_bull_market(self):
        """Model that always predicts 'down' in a bull market -> negative returns."""
        prices = pd.Series([100.0 + i * 2.0 for i in range(50)])
        returns = simple_returns(prices).dropna()

        predictions = pd.Series([-1] * len(returns))
        strategy_returns = predictions * returns

        assert strategy_returns.sum() < 0, "Always-down prediction in bull market must yield negative returns"
