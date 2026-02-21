"""Tests for volatility targeting module."""

import numpy as np
import pandas as pd

from sparky.features.vol_targeting import apply_vol_targeting, vol_target_position_size


def _make_series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx)


def test_high_vol_reduces_position():
    forecast_vol = _make_series([0.80] * 10)  # 80% vol
    pos = vol_target_position_size(forecast_vol, target_vol=0.20, max_leverage=1.0)
    assert (pos == 0.25).all()


def test_low_vol_increases_position():
    forecast_vol = _make_series([0.05] * 10)  # 5% vol
    pos = vol_target_position_size(forecast_vol, target_vol=0.20, max_leverage=1.0)
    # 0.20/0.05 = 4.0, but capped at 1.0
    assert (pos == 1.0).all()


def test_max_leverage_cap():
    forecast_vol = _make_series([0.01] * 10)
    pos = vol_target_position_size(forecast_vol, target_vol=0.20, max_leverage=2.0)
    assert (pos <= 2.0).all()


def test_min_position_floor():
    forecast_vol = _make_series([10.0] * 10)  # absurdly high vol
    pos = vol_target_position_size(forecast_vol, target_vol=0.20, min_position=0.05)
    assert (pos >= 0.05).all()


def test_smoothing_reduces_turnover():
    np.random.seed(42)
    vols = np.random.uniform(0.10, 0.50, 100)
    forecast_vol = _make_series(vols)

    pos_raw = vol_target_position_size(forecast_vol, target_vol=0.20, smoothing=1)
    pos_smooth = vol_target_position_size(forecast_vol, target_vol=0.20, smoothing=10)

    turnover_raw = pos_raw.diff().abs().sum()
    turnover_smooth = pos_smooth.diff().abs().sum()
    assert turnover_smooth < turnover_raw


def test_apply_vol_targeting_scales_returns():
    returns = _make_series([0.01, -0.02, 0.005, 0.03, -0.01])
    forecast_vol = _make_series([0.40, 0.40, 0.40, 0.40, 0.40])
    result = apply_vol_targeting(returns, forecast_vol, target_vol=0.20)
    # position = 0.20/0.40 = 0.5
    expected = returns * 0.5
    pd.testing.assert_series_equal(result, expected)


def test_no_lookahead():
    np.random.seed(42)
    returns = _make_series(np.random.normal(0, 0.02, 50))
    forecast_vol = _make_series(np.random.uniform(0.10, 0.50, 50))

    full_result = apply_vol_targeting(returns, forecast_vol, target_vol=0.20)
    # Truncate to first 30 and verify same values
    trunc_result = apply_vol_targeting(returns.iloc[:30], forecast_vol.iloc[:30], target_vol=0.20)
    pd.testing.assert_series_equal(full_result.iloc[:30], trunc_result)
