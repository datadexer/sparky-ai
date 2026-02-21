"""Tests for funding rate features module."""

import numpy as np
import pandas as pd

from sparky.features.funding_rate_features import (
    funding_rate_carry_signal,
    funding_rate_regime,
    funding_rate_rolling_avg,
    funding_rate_zscore,
)


def _make_funding(values, start="2020-01-01", freq="8h"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)


def test_rolling_avg_matches_pandas_native():
    np.random.seed(42)
    fr = _make_funding(np.random.normal(0.0001, 0.0005, 200))
    result = funding_rate_rolling_avg(fr, windows=[8])
    expected = fr.rolling(window=8, min_periods=8).mean()
    pd.testing.assert_series_equal(result["fr_avg_8"], expected, check_names=False)


def test_rolling_avg_default_windows():
    fr = _make_funding(np.random.normal(0, 0.001, 200))
    result = funding_rate_rolling_avg(fr)
    assert list(result.columns) == ["fr_avg_8", "fr_avg_24", "fr_avg_168"]


def test_rolling_avg_custom_windows():
    fr = _make_funding(np.random.normal(0, 0.001, 100))
    result = funding_rate_rolling_avg(fr, windows=[5, 10])
    assert list(result.columns) == ["fr_avg_5", "fr_avg_10"]


def test_zscore_zero_mean_unit_variance():
    np.random.seed(42)
    n = 2000
    fr = _make_funding(np.random.normal(0.0001, 0.0005, n))
    z = funding_rate_zscore(fr, window=720)
    valid = z.dropna()
    assert abs(valid.mean()) < 0.15
    assert abs(valid.std() - 1.0) < 0.3


def test_zscore_handles_zero_std():
    fr = _make_funding([0.001] * 800)  # constant → std = 0
    z = funding_rate_zscore(fr, window=720)
    valid_after_warmup = z.iloc[720:]
    assert valid_after_warmup.isna().all()


def test_regime_classification_positive():
    fr = _make_funding([0.001] * 100)  # clearly positive
    regime = funding_rate_regime(fr, positive_threshold=0.0001, window=10)
    assert (regime.iloc[10:] == "positive").all()


def test_regime_classification_negative():
    fr = _make_funding([-0.001] * 100)
    regime = funding_rate_regime(fr, positive_threshold=0.0001, window=10)
    assert (regime.iloc[10:] == "negative").all()


def test_regime_classification_neutral():
    fr = _make_funding([0.00001] * 100)  # below threshold
    regime = funding_rate_regime(fr, positive_threshold=0.0001, window=10)
    assert (regime.iloc[10:] == "neutral").all()


def test_carry_signal_threshold():
    # 0.065% = 0.00065 in fraction; make rates above that
    high_rates = [0.001] * 100  # 0.1% > 0.065%
    low_rates = [0.0001] * 100  # 0.01% < 0.065%
    fr_high = _make_funding(high_rates)
    fr_low = _make_funding(low_rates)

    sig_high = funding_rate_carry_signal(fr_high, cost_threshold_pct=0.065, window=10)
    sig_low = funding_rate_carry_signal(fr_low, cost_threshold_pct=0.065, window=10)

    assert (sig_high.iloc[10:] == 1.0).all()
    assert (sig_low.iloc[10:] == 0.0).all()


def test_carry_signal_binary():
    np.random.seed(42)
    fr = _make_funding(np.random.normal(0.0005, 0.001, 200))
    sig = funding_rate_carry_signal(fr)
    assert set(sig.unique()).issubset({0.0, 1.0})
