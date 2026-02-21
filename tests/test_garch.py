"""Tests for GARCH volatility module."""

import numpy as np
import pandas as pd
import pytest

from sparky.features.garch import (
    ewma_volatility,
    fit_garch,
    garch_parameter_stability,
    rolling_garch_forecast,
)


@pytest.fixture
def vol_clustering_returns():
    """Returns with vol clustering: alternating low/high variance regimes."""
    np.random.seed(42)
    n = 500
    returns = np.empty(n)
    returns[:250] = np.random.normal(0, 0.01, 250)  # low vol
    returns[250:] = np.random.normal(0, 0.04, 250)  # high vol
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(returns, index=idx)


@pytest.fixture
def short_returns():
    """Short white noise series."""
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    return pd.Series(np.random.normal(0, 0.01, 100), index=idx)


def test_fit_garch_synthetic_vol_clustering(vol_clustering_returns):
    result = fit_garch(vol_clustering_returns)
    assert hasattr(result, "params")
    assert "omega" in result.params
    assert "alpha[1]" in result.params
    assert "beta[1]" in result.params


def test_fit_garch_nonconvergence_white_noise(short_returns):
    # Should still return a result even on short series
    result = fit_garch(short_returns)
    assert hasattr(result, "params")


def test_ewma_volatility_matches_pandas(short_returns):
    span = 20
    result = ewma_volatility(short_returns, span=span)
    expected = short_returns.ewm(span=span).std()
    pd.testing.assert_series_equal(result, expected)


def test_rolling_garch_forecast_shape(vol_clustering_returns):
    window = 252
    forecasts = rolling_garch_forecast(vol_clustering_returns, window=window, refit_every=50)
    assert len(forecasts) == len(vol_clustering_returns)
    # First `window` values should be NaN
    assert forecasts.iloc[:window].isna().all()
    # After window, should have values
    assert forecasts.iloc[window:].notna().any()


def test_rolling_garch_forecast_no_lookahead(vol_clustering_returns):
    window = 252
    forecasts = rolling_garch_forecast(vol_clustering_returns, window=window, refit_every=50)
    # Forecast at index i should only depend on data[:i]
    # Verify by checking that truncating future data gives same forecast
    mid = 350
    truncated = vol_clustering_returns.iloc[:mid]
    truncated_forecasts = rolling_garch_forecast(truncated, window=window, refit_every=50)
    # Forecasts up to mid should be close (may differ slightly due to refit alignment)
    valid_full = forecasts.iloc[window:mid].dropna()
    valid_trunc = truncated_forecasts.iloc[window:mid].dropna()
    common = valid_full.index.intersection(valid_trunc.index)
    if len(common) > 0:
        np.testing.assert_allclose(
            valid_full.loc[common].values,
            valid_trunc.loc[common].values,
            rtol=1e-6,
        )


def test_garch_parameter_stability_output_structure(vol_clustering_returns):
    df = garch_parameter_stability(vol_clustering_returns, window=252, step=63)
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        for col in ["omega", "alpha", "beta", "persistence"]:
            assert col in df.columns


def test_garch_parameter_stability_empty_returns():
    # Too-short series
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    short = pd.Series(np.random.normal(0, 0.01, 10), index=idx)
    df = garch_parameter_stability(short, window=252, step=63)
    assert df.empty
