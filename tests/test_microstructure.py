"""Tests for market microstructure features."""

import numpy as np
import pandas as pd
import pytest

from sparky.features.microstructure import (
    bid_ask_imbalance_proxy,
    candle_body_ratio,
    consecutive_candles,
    high_low_ratio,
    intraday_momentum_reversal,
    lower_wick_ratio,
    overnight_gap,
    tick_direction_ratio,
    upper_wick_ratio,
)


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": 40000 + np.random.randn(50) * 100,
            "high": 40100 + np.random.randn(50) * 100,
            "low": 39900 + np.random.randn(50) * 100,
            "close": 40000 + np.random.randn(50) * 100,
            "volume": 1000 + np.random.randn(50) * 50,
        },
        index=dates,
    )


def test_tick_direction_ratio(sample_ohlcv):
    """Test tick direction ratio calculation."""
    df = sample_ohlcv.copy()

    # Create known pattern: 20 up, 4 down
    df["open"] = 40000.0
    df.loc[df.index[:20], "close"] = 40100.0  # Up ticks
    df.loc[df.index[20:24], "close"] = 39900.0  # Down ticks
    df.loc[df.index[24:], "close"] = 40100.0  # Up ticks again

    ratio = tick_direction_ratio(df, window=24)

    assert isinstance(ratio, pd.Series)
    assert len(ratio) == len(df)

    # After 24 candles (20 up + 4 down), ratio should be 20/24 ≈ 0.833
    assert 0.82 < ratio.iloc[23] < 0.85


def test_candle_body_ratio():
    """Test candle body ratio calculation."""
    df = pd.DataFrame(
        {
            "open": [100, 100, 100],
            "high": [110, 110, 110],
            "low": [90, 90, 90],
            "close": [108, 100, 92],  # Strong up, doji, strong down
        }
    )

    ratio = candle_body_ratio(df)

    # Strong up: body=8, range=20 → ratio=0.4
    assert abs(ratio.iloc[0] - 0.4) < 0.01

    # Doji: body=0, range=20 → ratio=0
    assert abs(ratio.iloc[1] - 0.0) < 0.01

    # Strong down: body=8, range=20 → ratio=0.4
    assert abs(ratio.iloc[2] - 0.4) < 0.01


def test_upper_wick_ratio():
    """Test upper wick ratio calculation."""
    df = pd.DataFrame(
        {
            "open": [100, 100],
            "high": [110, 110],
            "low": [90, 90],
            "close": [105, 108],
        }
    )

    ratio = upper_wick_ratio(df)

    # First: upper_wick=5 (110-105), range=20 → ratio=0.25
    assert abs(ratio.iloc[0] - 0.25) < 0.01

    # Second: upper_wick=2 (110-108), range=20 → ratio=0.10
    assert abs(ratio.iloc[1] - 0.10) < 0.01


def test_lower_wick_ratio():
    """Test lower wick ratio calculation."""
    df = pd.DataFrame(
        {
            "open": [100, 100],
            "high": [110, 110],
            "low": [90, 90],
            "close": [95, 92],
        }
    )

    ratio = lower_wick_ratio(df)

    # First: lower_wick=5 (95-90), range=20 → ratio=0.25
    assert abs(ratio.iloc[0] - 0.25) < 0.01

    # Second: lower_wick=2 (92-90), range=20 → ratio=0.10
    assert abs(ratio.iloc[1] - 0.10) < 0.01


def test_consecutive_candles():
    """Test consecutive candle counting."""
    df = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100],
            "close": [105, 110, 115, 95, 90, 110],  # 3 up, 2 down, 1 up
        }
    )

    green = consecutive_candles(df, direction="green")
    red = consecutive_candles(df, direction="red")

    # Green count should be [1, 2, 3, 0, 0, 1]
    assert list(green) == [1, 2, 3, 0, 0, 1]

    # Red count should be [0, 0, 0, 1, 2, 0]
    assert list(red) == [0, 0, 0, 1, 2, 0]


def test_high_low_ratio(sample_ohlcv):
    """Test high/low ratio calculation."""
    df = sample_ohlcv.copy()

    ratio = high_low_ratio(df, window=20)

    assert isinstance(ratio, pd.Series)
    assert len(ratio) == len(df)

    # All values should be > 1 (high > low)
    assert (ratio.dropna() > 1.0).all()

    # Typical BTC hourly: 1.01 - 1.10
    assert (ratio.dropna() < 1.2).all()


def test_bid_ask_imbalance_proxy():
    """Test bid/ask imbalance proxy."""
    df = pd.DataFrame(
        {
            "high": [110, 110, 110],
            "low": [90, 90, 90],
            "close": [90, 100, 110],  # Near low, middle, near high
        }
    )

    imbalance = bid_ask_imbalance_proxy(df)

    # Near low (high sell pressure): imbalance ≈ 1.0
    assert abs(imbalance.iloc[0] - 1.0) < 0.01

    # Middle (balanced): imbalance ≈ 0.5
    assert abs(imbalance.iloc[1] - 0.5) < 0.01

    # Near high (low sell pressure): imbalance ≈ 0.0
    assert abs(imbalance.iloc[2] - 0.0) < 0.01


def test_intraday_momentum_reversal():
    """Test intraday momentum reversal detection."""
    df = pd.DataFrame(
        {
            "open": [100, 100, 100, 100],
            "close": [105, 110, 95, 90],  # Up, up, down, down
        }
    )

    reversal = intraday_momentum_reversal(df)

    # Pattern: [0, 0, 1, 0] — reversal at index 2
    assert reversal.iloc[0] == 0  # First candle
    assert reversal.iloc[1] == 0  # Continuation (up -> up)
    assert reversal.iloc[2] == 1  # Reversal (up -> down)
    assert reversal.iloc[3] == 0  # Continuation (down -> down)


def test_overnight_gap():
    """Test overnight gap calculation."""
    df = pd.DataFrame(
        {
            "open": [100, 105, 95, 100],
            "close": [102, 104, 94, 101],
        }
    )

    gap = overnight_gap(df)

    # Gap 0: No gap (first candle)
    assert gap.iloc[0] == 0

    # Gap 1: (105 - 102) / 102 ≈ 0.0294
    assert abs(gap.iloc[1] - 0.0294) < 0.001

    # Gap 2: (95 - 104) / 104 ≈ -0.0865
    assert abs(gap.iloc[2] - (-0.0865)) < 0.001

    # Gap 3: (100 - 94) / 94 ≈ 0.0638
    assert abs(gap.iloc[3] - 0.0638) < 0.001


def test_zero_range_handling():
    """Test features handle zero-range candles (high=low)."""
    df = pd.DataFrame(
        {
            "open": [100, 100],
            "high": [100, 110],
            "low": [100, 90],
            "close": [100, 100],
        }
    )

    # Should not crash on zero-range candle
    body = candle_body_ratio(df)
    upper = upper_wick_ratio(df)
    lower = lower_wick_ratio(df)

    assert body.iloc[0] == 0  # Zero range → zero ratio
    assert not np.isnan(body.iloc[1])
    assert not np.isnan(upper.iloc[0])
    assert not np.isnan(lower.iloc[0])
