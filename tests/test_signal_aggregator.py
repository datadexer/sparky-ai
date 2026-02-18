"""Tests for HourlyToDailyAggregator signal aggregation."""

import numpy as np
import pandas as pd
import pytest

from sparky.models.signal_aggregator import HourlyToDailyAggregator


@pytest.fixture
def hourly_probas():
    """Generate 3 days of synthetic hourly P(up) predictions."""
    dates = pd.date_range("2024-01-01", periods=72, freq="h")
    # Day 1: mostly bullish (mean ~0.6)
    # Day 2: mostly bearish (mean ~0.4)
    # Day 3: neutral (mean ~0.5)
    rng = np.random.default_rng(42)
    values = np.concatenate(
        [
            rng.normal(0.6, 0.05, 24),
            rng.normal(0.4, 0.05, 24),
            rng.normal(0.5, 0.05, 24),
        ]
    )
    values = np.clip(values, 0, 1)
    return pd.Series(values, index=dates, name="proba")


@pytest.fixture
def hourly_features(hourly_probas):
    """Generate matching hourly features with volatility column."""
    n = len(hourly_probas)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "realized_vol_24h": rng.normal(0.02, 0.005, n),
            "momentum_4h": rng.normal(0, 0.01, n),
        },
        index=hourly_probas.index,
    )


class TestHourlyToDailyAggregator:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            HourlyToDailyAggregator(method="invalid")

    def test_mean_aggregation_shape(self, hourly_probas):
        agg = HourlyToDailyAggregator(method="mean")
        result = agg.aggregate(hourly_probas)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"daily_proba", "signal", "n_hours", "std"}
        assert len(result) == 3  # 3 days

    def test_mean_aggregation_signals(self, hourly_probas):
        agg = HourlyToDailyAggregator(method="mean", threshold=0.5)
        result = agg.aggregate(hourly_probas)
        # Day 1 (mean ~0.6) should be LONG (1)
        assert result.iloc[0]["signal"] == 1
        # Day 2 (mean ~0.4) should be SHORT (0)
        assert result.iloc[1]["signal"] == 0

    def test_mean_requires_20_hours(self):
        """Days with < 20 hours should have signal = 0."""
        dates = pd.date_range("2024-01-01", periods=15, freq="h")
        probas = pd.Series(np.full(15, 0.8), index=dates)
        agg = HourlyToDailyAggregator(method="mean")
        result = agg.aggregate(probas)
        assert result.iloc[0]["signal"] == 0  # Only 15 hours

    def test_mean_n_hours_count(self, hourly_probas):
        agg = HourlyToDailyAggregator(method="mean")
        result = agg.aggregate(hourly_probas)
        assert all(result["n_hours"] == 24)

    def test_weighted_aggregation(self, hourly_probas):
        agg = HourlyToDailyAggregator(method="weighted", ema_halflife=6)
        result = agg.aggregate(hourly_probas)
        assert len(result) == 3
        assert "daily_proba" in result.columns
        assert "signal" in result.columns

    def test_weighted_recency_bias(self):
        """Weighted method should favor later hours more."""
        dates = pd.date_range("2024-01-01", periods=24, freq="h")
        # First 12 hours bearish, last 12 hours bullish
        values = np.concatenate([np.full(12, 0.3), np.full(12, 0.7)])
        probas = pd.Series(values, index=dates)

        # Mean would give 0.5, weighted should give > 0.5 (recency bias to bullish)
        agg_weighted = HourlyToDailyAggregator(method="weighted", ema_halflife=6)
        result = agg_weighted.aggregate(probas)
        assert result.iloc[0]["daily_proba"] > 0.5

    def test_regime_aggregation(self, hourly_probas, hourly_features):
        agg = HourlyToDailyAggregator(method="regime")
        result = agg.aggregate(hourly_probas, hourly_features)
        assert len(result) == 3
        assert "signal" in result.columns

    def test_regime_requires_features(self, hourly_probas):
        agg = HourlyToDailyAggregator(method="regime")
        with pytest.raises(ValueError, match="hourly_features required"):
            agg.aggregate(hourly_probas)

    def test_timezone_preserved_utc(self):
        """Output index should preserve UTC timezone."""
        dates = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        probas = pd.Series(np.full(24, 0.6), index=dates)
        agg = HourlyToDailyAggregator(method="mean")
        result = agg.aggregate(probas)
        assert str(result.index.tz) == "UTC"

    def test_threshold_customization(self, hourly_probas):
        """Different thresholds should produce different signals."""
        agg_low = HourlyToDailyAggregator(method="mean", threshold=0.3)
        agg_high = HourlyToDailyAggregator(method="mean", threshold=0.7)
        result_low = agg_low.aggregate(hourly_probas)
        result_high = agg_high.aggregate(hourly_probas)
        # Low threshold should produce more LONG signals
        assert result_low["signal"].sum() >= result_high["signal"].sum()
