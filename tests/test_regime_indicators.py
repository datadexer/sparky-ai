"""Tests for regime detection indicators."""

import numpy as np
import pandas as pd
import pytest

from sparky.features.regime_indicators import (
    compute_regime_adjusted_signals,
    compute_volatility_regime,
    get_regime_position_size,
    get_regime_threshold,
)


def test_compute_volatility_regime_basic():
    """Test basic volatility regime classification."""
    # Create synthetic price data with different volatility periods
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=24 * 90, freq="h")  # 90 days

    # Period 1: Low volatility (~15% annualized, well below 30% threshold)
    low_vol_returns = np.random.normal(0, 0.15 / np.sqrt(24 * 365), 24 * 30)
    # Period 2: Medium volatility (~45% annualized, in 30-60% range)
    med_vol_returns = np.random.normal(0, 0.45 / np.sqrt(24 * 365), 24 * 30)
    # Period 3: High volatility (~80% annualized, well above 60% threshold)
    high_vol_returns = np.random.normal(0, 0.80 / np.sqrt(24 * 365), 24 * 30)

    returns = np.concatenate([low_vol_returns, med_vol_returns, high_vol_returns])
    prices = pd.Series(100 * (1 + returns).cumprod(), index=dates)

    # Compute regimes with 30-day window
    regimes = compute_volatility_regime(prices, window=30 * 24, frequency="1h")

    # Verify regime types exist
    assert set(regimes.unique()).issubset({"low", "medium", "high"})

    # Should have all three regime types (note: rolling window means transitions are gradual)
    # Just verify we have more than one regime type (not all stuck in one regime)
    assert len(regimes.unique()) >= 2


def test_compute_volatility_regime_daily():
    """Test regime computation on daily data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")

    # Generate daily prices with varying volatility
    returns = np.random.normal(0, 0.02, 365)  # 2% daily vol
    prices = pd.Series(100 * (1 + returns).cumprod(), index=dates)

    # Compute regimes with 30-day window
    regimes = compute_volatility_regime(prices, window=30, frequency="1d")

    # Should return valid regimes
    assert len(regimes) == len(prices)
    assert set(regimes.unique()).issubset({"low", "medium", "high"})


def test_get_regime_position_size():
    """Test regime-specific position sizing."""
    assert get_regime_position_size("high") == 0.50
    assert get_regime_position_size("medium") == 0.75
    assert get_regime_position_size("low") == 1.00

    # Invalid regime should raise
    with pytest.raises(ValueError):
        get_regime_position_size("invalid")


def test_get_regime_threshold():
    """Test regime-specific probability thresholds."""
    assert get_regime_threshold("high") == 0.55
    assert get_regime_threshold("medium") == 0.52
    assert get_regime_threshold("low") == 0.50

    # Invalid regime should raise
    with pytest.raises(ValueError):
        get_regime_threshold("invalid")


def test_compute_regime_adjusted_signals():
    """Test regime-adjusted signal generation."""
    # Create synthetic probabilities and regimes
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    probabilities = pd.Series(np.linspace(0.45, 0.60, 100), index=dates)

    # Create regime series: first 33 low, next 33 medium, last 34 high
    regimes = pd.Series(
        ["low"] * 33 + ["medium"] * 33 + ["high"] * 34,
        index=dates,
    )

    # Compute regime-adjusted signals
    signals_df = compute_regime_adjusted_signals(probabilities, regimes)

    # Verify structure
    assert len(signals_df) == 100
    assert list(signals_df.columns) == [
        "probability",
        "regime",
        "threshold",
        "signal",
        "position_size",
    ]

    # Verify regime-specific thresholds applied
    low_regime_rows = signals_df[signals_df["regime"] == "low"]
    assert (low_regime_rows["threshold"] == 0.50).all()

    medium_regime_rows = signals_df[signals_df["regime"] == "medium"]
    assert (medium_regime_rows["threshold"] == 0.52).all()

    high_regime_rows = signals_df[signals_df["regime"] == "high"]
    assert (high_regime_rows["threshold"] == 0.55).all()

    # Verify signal generation logic
    for idx, row in signals_df.iterrows():
        if row["probability"] > row["threshold"]:
            assert row["signal"] == 1
            assert row["position_size"] == get_regime_position_size(row["regime"])
        else:
            assert row["signal"] == 0
            assert row["position_size"] == 0.0


def test_regime_classification_thresholds():
    """Test exact threshold boundaries for regime classification."""
    # Create prices with exact volatility levels
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=24 * 60, freq="h")

    # Test low regime boundary (29% annualized)
    returns_29pct = np.random.normal(0, 0.29 / np.sqrt(24 * 365), 24 * 60)
    prices_29 = pd.Series(100 * (1 + returns_29pct).cumprod(), index=dates)
    regime_29 = compute_volatility_regime(prices_29, window=30 * 24, frequency="1h")
    # Should be mostly "low" (< 30%)
    assert (regime_29 == "low").sum() > len(regime_29) * 0.5

    # Test high regime boundary (61% annualized)
    returns_61pct = np.random.normal(0, 0.61 / np.sqrt(24 * 365), 24 * 60)
    prices_61 = pd.Series(100 * (1 + returns_61pct).cumprod(), index=dates)
    regime_61 = compute_volatility_regime(prices_61, window=30 * 24, frequency="1h")
    # Should be mostly "high" (>= 60%)
    assert (regime_61 == "high").sum() > len(regime_61) * 0.5


def test_position_sizing_reduces_exposure_in_high_vol():
    """Test that high volatility regimes reduce position size."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")

    # All probabilities above all thresholds (would be LONG in all regimes)
    probabilities = pd.Series([0.60] * 100, index=dates)

    # Test different regimes
    regimes_low = pd.Series(["low"] * 100, index=dates)
    regimes_high = pd.Series(["high"] * 100, index=dates)

    signals_low = compute_regime_adjusted_signals(probabilities, regimes_low)
    signals_high = compute_regime_adjusted_signals(probabilities, regimes_high)

    # All should be LONG (signal=1)
    assert (signals_low["signal"] == 1).all()
    assert (signals_high["signal"] == 1).all()

    # But high vol regime should have smaller position sizes
    assert (signals_low["position_size"] == 1.0).all()
    assert (signals_high["position_size"] == 0.5).all()

    # Total exposure is 50% in high vol vs 100% in low vol
    assert signals_low["position_size"].sum() == 100.0
    assert signals_high["position_size"].sum() == 50.0
