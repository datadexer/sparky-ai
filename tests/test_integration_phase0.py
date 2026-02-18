"""Phase 0 integration tests — cross-validation between returns.py and technical.py.

Verifies that the foundational calculation modules produce consistent,
compatible outputs when wired together.
"""

import numpy as np
import pandas as pd
import pytest

from sparky.features.returns import (
    annualized_sharpe,
    log_returns,
    realized_volatility,
    simple_returns,
)
from sparky.features.technical import ema, macd, momentum, rsi


@pytest.fixture
def synthetic_prices():
    """Generate synthetic daily prices with known statistical properties."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    # Geometric Brownian motion: drift + noise
    log_rets = np.random.normal(0.0005, 0.02, 500)
    prices = 100 * np.exp(np.cumsum(log_rets))
    return pd.Series(prices, index=dates, name="close")


class TestReturnsInternalConsistency:
    """Verify returns.py functions are self-consistent."""

    def test_annualized_sharpe_matches_manual_calculation(self, synthetic_prices):
        """annualized_sharpe matches mean/std * sqrt(252) manually."""
        rets = simple_returns(synthetic_prices).dropna()

        # Manual calculation
        mean_r = rets.mean()
        std_r = rets.std(ddof=1)
        manual_sharpe = (mean_r / std_r) * np.sqrt(252)

        computed_sharpe = annualized_sharpe(rets)

        assert computed_sharpe == pytest.approx(manual_sharpe, rel=1e-10)

    def test_simple_returns_to_realized_volatility(self, synthetic_prices):
        """realized_volatility equals std(returns) * sqrt(252)."""
        rets = simple_returns(synthetic_prices).dropna()

        manual_vol = float(rets.std(ddof=1) * np.sqrt(252))
        computed_vol = realized_volatility(rets)

        assert computed_vol == pytest.approx(manual_vol, rel=1e-10)

    def test_log_returns_close_to_simple_returns(self, synthetic_prices):
        """For small returns, log returns approximate simple returns."""
        s_rets = simple_returns(synthetic_prices).dropna()
        l_rets = log_returns(synthetic_prices).dropna()

        # For daily returns ~2%, the approximation should be very close
        assert np.allclose(s_rets.values, l_rets.values, atol=0.005)


class TestTechnicalInternalConsistency:
    """Verify technical.py functions are self-consistent."""

    def test_rsi_output_range(self, synthetic_prices):
        """RSI values are always in [0, 100] after warmup."""
        rsi_values = rsi(synthetic_prices, period=14).dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
        assert len(rsi_values) > 0

    def test_macd_uses_ema_internally(self, synthetic_prices):
        """MACD line equals fast EMA minus slow EMA."""
        macd_line, _, _ = macd(synthetic_prices, fast_period=12, slow_period=26)
        ema_fast = ema(synthetic_prices, 12)
        ema_slow = ema(synthetic_prices, 26)

        expected = ema_fast - ema_slow
        # Compare only where both are valid
        valid = macd_line.dropna().index.intersection(expected.dropna().index)
        assert len(valid) > 0
        assert np.allclose(
            macd_line.loc[valid].values,
            expected.loc[valid].values,
            equal_nan=True,
        )

    def test_momentum_sign_convention(self, synthetic_prices):
        """Positive momentum means price went up."""
        mom = momentum(synthetic_prices, period=30).dropna()
        # Check a specific point
        idx = mom.index[0]
        offset = synthetic_prices.index.get_loc(idx)
        expected = (synthetic_prices.iloc[offset] - synthetic_prices.iloc[offset - 30]) / synthetic_prices.iloc[
            offset - 30
        ]
        assert mom.iloc[0] == pytest.approx(expected, rel=1e-10)


class TestCrossModuleIntegration:
    """Verify returns and technical modules work together."""

    def test_returns_feed_into_sharpe(self, synthetic_prices):
        """simple_returns -> annualized_sharpe produces finite, non-zero result."""
        rets = simple_returns(synthetic_prices)
        sharpe = annualized_sharpe(rets)
        assert np.isfinite(sharpe)
        assert sharpe != 0.0

    def test_momentum_and_simple_returns_agree_on_direction(self, synthetic_prices):
        """Over a period, momentum sign should match cumulative return sign."""
        period = 30
        mom = momentum(synthetic_prices, period=period).dropna()
        rets = simple_returns(synthetic_prices).dropna()

        # For each momentum point, check the sign matches cumulative return
        # over the same period (they should be identical by construction)
        for i in range(min(10, len(mom))):
            idx = mom.index[i]
            pos = synthetic_prices.index.get_loc(idx)
            cum_ret = (synthetic_prices.iloc[pos] / synthetic_prices.iloc[pos - period]) - 1
            assert np.sign(mom.iloc[i]) == np.sign(cum_ret)
