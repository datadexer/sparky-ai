"""Cross-validation of our implementations against pandas_ta.

If our code and pandas_ta diverge, OUR code is wrong (pandas_ta is the reference).

Tests:
- Generate 1000 random price points (random walk with drift)
- Compute RSI, EMA, MACD using our code AND pandas_ta
- Assert maximum absolute difference < threshold
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest

from sparky.features.technical import ema, macd, momentum, rsi


@pytest.fixture
def random_prices():
    """Generate 1000 random price points (random walk with drift)."""
    np.random.seed(42)
    # Random walk with slight upward drift
    log_returns = np.random.normal(0.0005, 0.02, 1000)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, name="close")


class TestRSICrossValidation:
    """Cross-validate our RSI against pandas_ta."""

    def test_rsi_matches_pandas_ta(self, random_prices):
        """Our RSI must match pandas_ta RSI within 0.1 points."""
        our_rsi = rsi(random_prices, period=14)
        ta_rsi = ta.rsi(random_prices, length=14)

        # Align on valid (non-NaN) range
        valid_mask = our_rsi.notna() & ta_rsi.notna()
        our_valid = our_rsi[valid_mask]
        ta_valid = ta_rsi[valid_mask]

        assert len(our_valid) > 900, "Should have >900 valid RSI values"

        max_diff = (our_valid - ta_valid).abs().max()
        assert max_diff < 0.1, \
            f"RSI max difference {max_diff:.4f} exceeds 0.1 threshold"

    def test_rsi_different_periods(self, random_prices):
        """Test RSI cross-validation with different periods."""
        for period in [7, 14, 21]:
            our_rsi = rsi(random_prices, period=period)
            ta_rsi = ta.rsi(random_prices, length=period)

            valid_mask = our_rsi.notna() & ta_rsi.notna()
            max_diff = (our_rsi[valid_mask] - ta_rsi[valid_mask]).abs().max()
            assert max_diff < 0.1, \
                f"RSI(period={period}) max diff {max_diff:.4f} exceeds 0.1"


class TestEMACrossValidation:
    """Cross-validate our EMA against pandas_ta."""

    def test_ema_matches_pandas_ta(self, random_prices):
        """Our EMA must match pandas_ta EMA within 0.01."""
        for span in [10, 20, 50]:
            our_ema = ema(random_prices, span=span)
            ta_ema = ta.ema(random_prices, length=span)

            valid_mask = our_ema.notna() & ta_ema.notna()
            our_valid = our_ema[valid_mask]
            ta_valid = ta_ema[valid_mask]

            # Normalize by price level to get relative difference
            max_diff = (our_valid - ta_valid).abs().max()
            assert max_diff < 0.01, \
                f"EMA(span={span}) max diff {max_diff:.6f} exceeds 0.01"


class TestMACDCrossValidation:
    """Cross-validate our MACD against pandas_ta."""

    def test_macd_matches_pandas_ta(self, random_prices):
        """Our MACD components must match pandas_ta."""
        our_macd, our_signal, our_hist = macd(random_prices)
        ta_result = ta.macd(random_prices)

        # pandas_ta returns a DataFrame with columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        ta_macd = ta_result.iloc[:, 0]  # MACD line
        ta_signal = ta_result.iloc[:, 2]  # Signal line
        ta_hist = ta_result.iloc[:, 1]  # Histogram

        # Check MACD line
        valid = our_macd.notna() & ta_macd.notna()
        max_diff = (our_macd[valid] - ta_macd[valid]).abs().max()
        assert max_diff < 0.01, f"MACD line max diff {max_diff:.6f} exceeds 0.01"

        # Check signal line
        valid = our_signal.notna() & ta_signal.notna()
        max_diff = (our_signal[valid] - ta_signal[valid]).abs().max()
        assert max_diff < 0.01, f"Signal line max diff {max_diff:.6f} exceeds 0.01"


class TestMomentumCrossValidation:
    """Cross-validate our momentum against pandas_ta."""

    def test_momentum_matches_pandas_ta(self, random_prices):
        """Our momentum must match pandas_ta ROC."""
        for period in [10, 20, 30]:
            our_mom = momentum(random_prices, period=period)
            ta_mom = ta.roc(random_prices, length=period)

            if ta_mom is None:
                pytest.skip(f"pandas_ta.roc not available for period={period}")

            valid = our_mom.notna() & ta_mom.notna()
            # pandas_ta ROC is in percentage, our momentum is in fraction
            ta_mom_fraction = ta_mom[valid] / 100.0
            our_valid = our_mom[valid]

            max_diff = (our_valid - ta_mom_fraction).abs().max()
            assert max_diff < 0.001, \
                f"Momentum(period={period}) max diff {max_diff:.6f} exceeds 0.001"
