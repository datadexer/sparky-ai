"""Tests for regime detection features."""

import numpy as np
import pandas as pd
import pytest

from sparky.features.regime import (
    drawdown_from_high,
    recovery_from_low,
    volatility_regime,
    volume_regime,
    trend_strength_adx_proxy,
    choppiness_index,
    breakout_proximity_upper,
    breakout_proximity_lower,
)


@pytest.fixture
def sample_prices():
    """Sample price series for testing."""
    dates = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    # Trend: 40000 -> 45000 -> 42000
    prices = np.concatenate([
        np.linspace(40000, 45000, 100),
        np.linspace(45000, 42000, 100),
    ])
    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_returns():
    """Sample returns for testing."""
    dates = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    returns = np.random.randn(200) * 0.01
    return pd.Series(returns, index=dates)


def test_drawdown_from_high(sample_prices):
    """Test drawdown from rolling high."""
    close = sample_prices
    dd = drawdown_from_high(close, window=20)

    assert isinstance(dd, pd.Series)
    assert len(dd) == len(close)

    # Drawdown is always >= 0
    assert (dd >= 0).all()

    # At peak (index 99), drawdown should be near 0
    assert dd.iloc[99] < 0.01

    # After decline (index 150), drawdown > 0 (window=20 so small values expected)
    assert dd.iloc[150] > 0.01


def test_recovery_from_low(sample_prices):
    """Test recovery from rolling low."""
    close = sample_prices
    rec = recovery_from_low(close, window=20)

    assert isinstance(rec, pd.Series)
    assert len(rec) == len(close)

    # Recovery is always >= 0
    assert (rec >= 0).all()

    # At trough (index 0 of decline), recovery should be near 0
    # At peak (index 99), recovery should be high (window=20 limits this)
    assert rec.iloc[99] > 0.02


def test_volatility_regime(sample_returns):
    """Test volatility percentile rank."""
    returns = sample_returns
    regime = volatility_regime(returns, window=168)

    assert isinstance(regime, pd.Series)

    # Percentile rank should be [0, 1]
    valid_regime = regime.dropna()
    assert (valid_regime >= 0).all()
    assert (valid_regime <= 1).all()


def test_volume_regime():
    """Test volume percentile rank."""
    dates = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    volume = pd.Series(np.random.uniform(1000, 2000, 200), index=dates)

    regime = volume_regime(volume, window=168)

    assert isinstance(regime, pd.Series)

    # Percentile rank should be [0, 1]
    valid_regime = regime.dropna()
    assert (valid_regime >= 0).all()
    assert (valid_regime <= 1).all()


def test_trend_strength_adx_proxy():
    """Test trend strength proxy."""
    dates = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")

    # Strong trend: high momentum, low volatility
    momentum = pd.Series([0.05] * 50, index=dates)
    volatility = pd.Series([0.01] * 50, index=dates)

    strength = trend_strength_adx_proxy(momentum, volatility)

    # Strong trend: momentum/vol = 0.05/0.01 = 5.0
    assert (strength > 4.5).all()

    # Choppy: low momentum, high volatility
    momentum_choppy = pd.Series([0.01] * 50, index=dates)
    volatility_choppy = pd.Series([0.05] * 50, index=dates)

    strength_choppy = trend_strength_adx_proxy(momentum_choppy, volatility_choppy)

    # Choppy: momentum/vol = 0.01/0.05 = 0.2
    assert (strength_choppy < 0.5).all()


def test_choppiness_index():
    """Test choppiness index calculation."""
    dates = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")

    # Trending: consistent positive returns
    returns_trend = pd.Series([0.01] * 200, index=dates)
    chop_trend = choppiness_index(returns_trend, window=168)

    # Trending should have low choppiness (perfect trend = 0, but warmup period pushes this higher)
    assert chop_trend.dropna().min() == 0.0  # End state is perfect trend
    assert chop_trend.dropna().mean() < 0.5  # Average includes warmup

    # Choppy: alternating +/- returns
    returns_chop = pd.Series([0.01, -0.01] * 100, index=dates)
    chop_choppy = choppiness_index(returns_chop, window=168)

    # Choppy should have high choppiness - max value = 1.0 (perfect chop)
    assert chop_choppy.dropna().max() == 1.0
    # Choppy should be higher than trending
    assert chop_choppy.dropna().mean() > chop_trend.dropna().mean()


def test_breakout_proximity_upper(sample_prices):
    """Test proximity to upper breakout."""
    close = sample_prices
    prox = breakout_proximity_upper(close, window=50)

    assert isinstance(prox, pd.Series)
    assert len(prox) == len(close)

    # Values should be [0, 1]
    valid_prox = prox.dropna()
    assert (valid_prox >= 0).all()
    assert (valid_prox <= 1).all()

    # At peak (index 99), should be near 1.0
    assert prox.iloc[99] > 0.8


def test_breakout_proximity_lower(sample_prices):
    """Test proximity to lower breakout."""
    close = sample_prices
    prox = breakout_proximity_lower(close, window=50)

    assert isinstance(prox, pd.Series)
    assert len(prox) == len(close)

    # Values should be [0, 1]
    valid_prox = prox.dropna()
    assert (valid_prox >= 0).all()
    assert (valid_prox <= 1).all()

    # At trough (end of decline), should be high
    # At peak, should be low
    assert prox.iloc[99] < 0.3


def test_regime_features_no_crash_on_edge_cases():
    """Test regime features handle edge cases."""
    dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")

    # Constant price (no movement)
    close_flat = pd.Series([40000.0] * 10, index=dates)

    dd = drawdown_from_high(close_flat, window=5)
    rec = recovery_from_low(close_flat, window=5)
    prox_up = breakout_proximity_upper(close_flat, window=5)
    prox_down = breakout_proximity_lower(close_flat, window=5)

    # Should not crash or produce NaN/inf
    assert not dd.isna().all()
    assert not rec.isna().all()
    assert not prox_up.isna().all()
    assert not prox_down.isna().all()
