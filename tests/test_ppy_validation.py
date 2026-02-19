"""Tests for PPY validation guard in compute_all_metrics."""

import numpy as np
import pandas as pd
import pytest

from sparky.tracking.metrics import compute_all_metrics, validate_periods_per_year


class TestNumpyArrayHighPPY:
    def test_raises_for_numpy_array_with_high_ppy(self):
        returns = np.random.randn(500) * 0.01
        with pytest.raises(ValueError, match="no DatetimeIndex"):
            compute_all_metrics(returns, periods_per_year=2190)

    def test_raises_for_numpy_array_ppy_4380(self):
        returns = np.random.randn(500) * 0.01
        with pytest.raises(ValueError, match="no DatetimeIndex"):
            compute_all_metrics(returns, periods_per_year=4380)

    def test_numpy_array_daily_ppy_passes(self):
        returns = np.random.randn(500) * 0.01
        m = compute_all_metrics(returns, periods_per_year=365)
        assert "sharpe" in m

    def test_numpy_array_8h_ppy_passes(self):
        returns = np.random.randn(500) * 0.01
        m = compute_all_metrics(returns, periods_per_year=1095)
        assert "sharpe" in m

    def test_strict_false_warns_but_computes(self):
        returns = np.random.randn(500) * 0.01
        m = compute_all_metrics(returns, periods_per_year=2190, strict_ppy=False)
        assert "sharpe" in m


class TestPandasSeriesValidation:
    def test_8h_series_correct_ppy_passes(self):
        idx = pd.date_range("2020-01-01", periods=1000, freq="8h", tz="UTC")
        returns = pd.Series(np.random.randn(1000) * 0.01, index=idx)
        m = compute_all_metrics(returns, periods_per_year=1095)
        assert "sharpe" in m

    def test_8h_series_wrong_ppy_raises(self):
        idx = pd.date_range("2020-01-01", periods=1000, freq="8h", tz="UTC")
        returns = pd.Series(np.random.randn(1000) * 0.01, index=idx)
        with pytest.raises(ValueError, match="wrong ppy"):
            compute_all_metrics(returns, periods_per_year=2190)

    def test_4h_series_correct_ppy_passes(self):
        idx = pd.date_range("2020-01-01", periods=1000, freq="4h", tz="UTC")
        returns = pd.Series(np.random.randn(1000) * 0.01, index=idx)
        m = compute_all_metrics(returns, periods_per_year=2190)
        assert "sharpe" in m

    def test_daily_series_correct_ppy_passes(self):
        idx = pd.date_range("2020-01-01", periods=500, freq="D", tz="UTC")
        returns = pd.Series(np.random.randn(500) * 0.01, index=idx)
        m = compute_all_metrics(returns, periods_per_year=365)
        assert "sharpe" in m


class TestValidatePeriodsPerYear:
    def test_invalid_ppy_rejected(self):
        with pytest.raises(ValueError, match="not a standard"):
            validate_periods_per_year(np.array([0.01]), 252)

    def test_strict_default_true(self):
        with pytest.raises(ValueError, match="no DatetimeIndex"):
            validate_periods_per_year(np.array([0.01] * 100), 2190)

    def test_strict_false_no_raise(self):
        validate_periods_per_year(np.array([0.01] * 100), 2190, strict=False)

    def test_low_ppy_numpy_passes(self):
        validate_periods_per_year(np.array([0.01] * 100), 365)
        validate_periods_per_year(np.array([0.01] * 100), 1095)
