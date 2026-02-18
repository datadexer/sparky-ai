"""Tests for financial returns calculations.

Each function is tested with hand-calculated expected values
to catch bugs like the v1 sign inversion.

Test cases from CODEBASE_PLAN.md:
- Simple returns: prices [100, 110, 99, 110] -> [NaN, 0.1, -0.1, 0.11111...]
- Log returns: same prices -> [NaN, ln(1.1), ln(0.9), ln(110/99)]
- Sharpe: known return series, verify sqrt(365) annualization (crypto 24/7)
- Max drawdown: prices [100, 120, 90, 110, 80] -> MDD = (120-80)/120 = 33.33%
- Realized vol: returns with std=0.02 -> annualized = 0.02*sqrt(365)
"""

import numpy as np
import pandas as pd
import pytest

from sparky.features.returns import (
    annualized_sharpe,
    log_returns,
    max_drawdown,
    realized_volatility,
    simple_returns,
)


class TestSimpleReturns:
    """Test simple (arithmetic) returns."""

    def test_basic_returns(self):
        """Hand-calculated: [100, 110, 99, 110] -> [NaN, 0.1, -0.1, 0.11111...]"""
        prices = pd.Series([100.0, 110.0, 99.0, 110.0])
        result = simple_returns(prices)

        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(0.1)
        assert result.iloc[2] == pytest.approx(-0.1)
        assert result.iloc[3] == pytest.approx(110.0 / 99.0 - 1.0)

    def test_constant_prices(self):
        """Constant prices should give zero returns."""
        prices = pd.Series([100.0, 100.0, 100.0])
        result = simple_returns(prices)
        assert result.iloc[1] == pytest.approx(0.0)
        assert result.iloc[2] == pytest.approx(0.0)

    def test_single_price(self):
        """Single price should give NaN."""
        prices = pd.Series([100.0])
        result = simple_returns(prices)
        assert len(result) == 1
        assert pd.isna(result.iloc[0])


class TestLogReturns:
    """Test logarithmic returns."""

    def test_basic_log_returns(self):
        """Hand-calculated: [100, 110, 99, 110] -> [NaN, ln(1.1), ln(0.9), ln(110/99)]"""
        prices = pd.Series([100.0, 110.0, 99.0, 110.0])
        result = log_returns(prices)

        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(np.log(1.1))
        assert result.iloc[2] == pytest.approx(np.log(99.0 / 110.0))
        assert result.iloc[3] == pytest.approx(np.log(110.0 / 99.0))

    def test_log_returns_sum_property(self):
        """Sum of log returns should equal log of total return."""
        prices = pd.Series([100.0, 120.0, 90.0, 150.0])
        lr = log_returns(prices).dropna()
        total_log_return = lr.sum()
        expected = np.log(150.0 / 100.0)
        assert total_log_return == pytest.approx(expected)


class TestAnnualizedSharpe:
    """Test annualized Sharpe ratio calculation."""

    def test_positive_sharpe(self):
        """Known positive returns should give positive Sharpe."""
        # Daily returns with mean ~0.1% and std ~1%
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sharpe = annualized_sharpe(returns)
        # With mean=0.001 and std=0.01, Sharpe ~ 0.001/0.01 * sqrt(252) ~ 1.59
        assert sharpe > 0

    def test_zero_std_returns_zero(self):
        """Constant returns (zero std) should return 0."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        assert annualized_sharpe(returns) == 0.0

    def test_annualization_factor(self):
        """Verify sqrt(365) annualization factor (crypto 24/7)."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
        daily_sharpe = returns.mean() / returns.std(ddof=1)
        annual_sharpe = annualized_sharpe(returns)
        assert annual_sharpe == pytest.approx(daily_sharpe * np.sqrt(365), rel=0.01)

    def test_empty_returns(self):
        """Empty series should return 0."""
        assert annualized_sharpe(pd.Series([], dtype=float)) == 0.0

    def test_single_return(self):
        """Single return (can't compute std) should return 0."""
        assert annualized_sharpe(pd.Series([0.01])) == 0.0


class TestMaxDrawdown:
    """Test maximum drawdown calculation."""

    def test_known_drawdown(self):
        """Hand-calculated: [100, 120, 90, 110, 80] -> MDD = (120-80)/120 = 33.33%"""
        prices = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0])
        mdd = max_drawdown(prices)
        assert mdd == pytest.approx(1.0 / 3.0, rel=0.001)  # 33.33%

    def test_monotonically_increasing(self):
        """No drawdown for strictly increasing prices."""
        prices = pd.Series([100.0, 110.0, 120.0, 130.0])
        assert max_drawdown(prices) == pytest.approx(0.0)

    def test_monotonically_decreasing(self):
        """Full drawdown for strictly decreasing prices."""
        prices = pd.Series([100.0, 80.0, 60.0, 40.0])
        mdd = max_drawdown(prices)
        assert mdd == pytest.approx(0.6)  # (100-40)/100

    def test_empty_series(self):
        """Empty series should return 0."""
        assert max_drawdown(pd.Series([], dtype=float)) == 0.0


class TestRealizedVolatility:
    """Test annualized realized volatility."""

    def test_known_volatility(self):
        """Returns with std=0.02 -> annualized = 0.02*sqrt(365) (crypto 24/7)."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0, 0.02, 10000))
        vol = realized_volatility(returns)
        expected = 0.02 * np.sqrt(365)
        assert vol == pytest.approx(expected, rel=0.05)  # 5% tolerance for sampling

    def test_zero_volatility(self):
        """Constant returns should give near-zero vol."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        assert realized_volatility(returns) == pytest.approx(0.0)

    def test_empty_returns(self):
        """Empty series should return 0."""
        assert realized_volatility(pd.Series([], dtype=float)) == 0.0

    def test_single_return(self):
        """Single return should return 0."""
        assert realized_volatility(pd.Series([0.01])) == 0.0
