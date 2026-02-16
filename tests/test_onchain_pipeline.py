"""
Tests for on-chain feature engineering pipeline (scripts/prepare_onchain_features.py).

Verifies:
- Output file structure and columns
- Look-ahead bias prevention (1-day shift before hourly alignment)
- Reasonable value ranges
- NaN handling (warmup period only)

These tests require local data files and are skipped in CI.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

ONCHAIN_PATH = Path('data/processed/onchain_features_hourly.parquet')
COINMETRICS_PATH = Path('data/raw/onchain/coinmetrics_btc_daily.parquet')
BTC_HOURLY_PATH = Path('data/raw/btc/ohlcv_hourly_max_coverage.parquet')
_has_onchain_data = ONCHAIN_PATH.exists()

pytestmark = pytest.mark.skipif(
    not _has_onchain_data,
    reason="On-chain features data not available (requires local data files)"
)


@pytest.fixture
def onchain_features():
    """Load on-chain features hourly data."""
    return pd.read_parquet(ONCHAIN_PATH)


@pytest.fixture
def coinmetrics_daily():
    """Load CoinMetrics daily data."""
    return pd.read_parquet(COINMETRICS_PATH)


@pytest.fixture
def btc_hourly():
    """Load BTC hourly OHLCV data."""
    return pd.read_parquet(BTC_HOURLY_PATH)


def test_output_exists():
    """Verify output file exists."""
    assert ONCHAIN_PATH.exists(), "Output file not found"


def test_correct_columns(onchain_features):
    """Verify all 8 expected features are present."""
    expected_columns = [
        'mvrv_ratio',
        'mvrv_zscore',
        'active_addresses_change_7d',
        'hash_rate_change_30d',
        'exchange_net_flow_7d',
        'fee_ratio_change_7d',
        'tx_count_change_7d',
        'nvt_ratio'
    ]
    assert list(onchain_features.columns) == expected_columns


def test_index_is_datetime_utc(onchain_features):
    """Verify index is UTC DatetimeIndex."""
    assert isinstance(onchain_features.index, pd.DatetimeIndex)
    assert onchain_features.index.tz is not None
    assert str(onchain_features.index.tz) == 'UTC'


def test_hourly_frequency(onchain_features):
    """Verify data is at hourly frequency."""
    # Check that most intervals are 1 hour (allowing for gaps)
    intervals = onchain_features.index.to_series().diff().dropna()
    most_common_interval = intervals.mode()[0]
    assert most_common_interval == pd.Timedelta(hours=1)


def test_no_look_ahead_bias(onchain_features, coinmetrics_daily, btc_hourly):
    """
    Verify that on-chain features are shifted by 1 day before hourly alignment.

    Strategy:
    1. Pick a date where CoinMetrics has data
    2. Check that hourly features on that date match CoinMetrics data from the PREVIOUS day
    3. This confirms the 1-day shift is applied correctly
    """
    # Pick a date in the middle of the range with good data
    test_date = pd.Timestamp('2020-06-15', tz='UTC')

    # Get hourly features for this date (any hour will do, they're all the same)
    hourly_value = onchain_features.loc[test_date, 'mvrv_ratio']

    # Get CoinMetrics value for the PREVIOUS day (due to 1-day shift)
    prev_day = test_date - pd.Timedelta(days=1)
    daily_value = coinmetrics_daily.loc[prev_day, 'CapMVRVCur']

    # They should match
    assert np.isclose(hourly_value, daily_value, rtol=1e-5), \
        f"Look-ahead bias detected! Hourly features on {test_date} should match CoinMetrics from {prev_day}"


def test_forward_fill_within_day(onchain_features):
    """
    Verify that all hours within a day have the same feature values.

    Since daily features are forward-filled to hourly, all 24 hours
    in a given day should have identical values.
    """
    # Pick a random day with data
    test_date = pd.Timestamp('2021-03-10', tz='UTC')
    day_start = test_date
    day_end = test_date + pd.Timedelta(hours=23)

    # Get all hours for this day
    day_data = onchain_features.loc[day_start:day_end]

    # All values should be identical within the day
    for col in day_data.columns:
        if day_data[col].notna().all():  # Skip if there are NaNs
            unique_values = day_data[col].nunique()
            assert unique_values == 1, \
                f"{col} has {unique_values} unique values within day {test_date}, expected 1"


def test_value_ranges(onchain_features):
    """Verify features have reasonable value ranges."""
    # MVRV ratio: typically 0.5 to 5.0
    mvrv = onchain_features['mvrv_ratio'].dropna()
    assert (mvrv >= 0).all(), "MVRV ratio should be positive"
    assert (mvrv < 10).all(), "MVRV ratio should be < 10 (sanity check)"

    # MVRV Z-score: typically -3 to 5
    mvrv_z = onchain_features['mvrv_zscore'].dropna()
    assert (mvrv_z >= -5).all(), "MVRV Z-score should be >= -5"
    assert (mvrv_z <= 10).all(), "MVRV Z-score should be <= 10"

    # Active addresses change: typically -50% to +100%
    aa_change = onchain_features['active_addresses_change_7d'].dropna()
    assert (aa_change >= -0.8).all(), "Active address change should be >= -80%"
    assert (aa_change <= 2.0).all(), "Active address change should be <= 200%"

    # Hash rate change: typically -60% to +200%
    hr_change = onchain_features['hash_rate_change_30d'].dropna()
    assert (hr_change >= -0.8).all(), "Hash rate change should be >= -80%"
    assert (hr_change <= 3.0).all(), "Hash rate change should be <= 300%"

    # Exchange net flow: can be very large (BTC units)
    # Just check it's not all zeros or all NaN
    net_flow = onchain_features['exchange_net_flow_7d'].dropna()
    assert len(net_flow) > 0, "Exchange net flow should have valid data"
    assert net_flow.std() > 0, "Exchange net flow should vary"

    # Fee change: can spike during congestion
    fee_change = onchain_features['fee_ratio_change_7d'].dropna()
    assert (fee_change >= -1.0).all(), "Fee change should be >= -100%"

    # Transaction count change: typically -60% to +150%
    tx_change = onchain_features['tx_count_change_7d'].dropna()
    assert (tx_change >= -0.8).all(), "TX count change should be >= -80%"
    assert (tx_change <= 3.0).all(), "TX count change should be <= 300%"

    # NVT ratio: typically 20 to 200
    nvt = onchain_features['nvt_ratio'].dropna()
    assert (nvt > 0).all(), "NVT ratio should be positive"
    assert (nvt < 500).all(), "NVT ratio should be < 500 (sanity check)"


def test_nan_only_in_warmup(onchain_features):
    """
    Verify NaNs only occur in warmup period.

    NaNs should only appear:
    1. Before CoinMetrics data starts (2017-01-01)
    2. In warmup windows for rolling calculations (e.g., first 365 days for mvrv_zscore)
    """
    # Get first valid index for each feature
    first_valid = {}
    for col in onchain_features.columns:
        first_valid[col] = onchain_features[col].first_valid_index()

    # All features should have valid data starting no later than 2018
    # (allowing for 365-day warmup for mvrv_zscore)
    for col, idx in first_valid.items():
        assert idx is not None, f"{col} has no valid data"
        assert idx < pd.Timestamp('2018-02-01', tz='UTC'), \
            f"{col} first valid data at {idx}, expected before 2018-02-01"

    # After warmup period, there should be no NaNs
    # Check data from 2018 onwards (well past any warmup)
    recent_data = onchain_features.loc['2018-01-01':]
    for col in recent_data.columns:
        nan_count = recent_data[col].isna().sum()
        assert nan_count == 0, f"{col} has {nan_count} NaNs after warmup period"


def test_mvrv_zscore_warmup(onchain_features):
    """
    Verify MVRV Z-score has appropriate warmup period.

    MVRV Z-score uses 365-day rolling window with min_periods=30,
    so it starts producing values after 30 days.
    """
    mvrv_z = onchain_features['mvrv_zscore']

    # First valid data should be roughly 30 days after CoinMetrics starts (2017-01-01)
    # (due to min_periods=30 in the rolling window)
    first_valid = mvrv_z.first_valid_index()
    coinmetrics_start = pd.Timestamp('2017-01-01', tz='UTC')

    # Should start within ~30 days (allowing for 1-day shift)
    expected_start = coinmetrics_start + pd.Timedelta(days=30)
    time_diff = abs((first_valid - expected_start).total_seconds() / 86400)

    assert time_diff < 5, \
        f"MVRV Z-score first valid at {first_valid}, expected ~{expected_start} (30d min_periods)"


def test_alignment_with_btc_hourly(onchain_features, btc_hourly):
    """Verify on-chain features align with BTC hourly index."""
    # Indexes should match
    assert len(onchain_features) == len(btc_hourly), \
        "On-chain features and BTC hourly should have same length"

    assert (onchain_features.index == btc_hourly.index).all(), \
        "On-chain features and BTC hourly should have identical indexes"


def test_no_duplicate_index(onchain_features):
    """Verify no duplicate timestamps in index."""
    assert not onchain_features.index.duplicated().any(), \
        "Found duplicate timestamps in index"


def test_monotonic_increasing_index(onchain_features):
    """Verify index is sorted in ascending order."""
    assert onchain_features.index.is_monotonic_increasing, \
        "Index should be monotonically increasing"


def test_feature_correlation_sanity(onchain_features):
    """
    Sanity check: related features should be somewhat correlated.

    Example: active_addresses_change_7d and tx_count_change_7d should
    have positive correlation (more users -> more transactions).
    """
    # Drop NaNs for correlation calculation
    data = onchain_features[['active_addresses_change_7d', 'tx_count_change_7d']].dropna()

    if len(data) > 100:  # Need enough data points
        corr = data.corr().loc['active_addresses_change_7d', 'tx_count_change_7d']
        assert corr > 0.1, \
            f"Active addresses and TX count changes should be positively correlated (got {corr:.3f})"


def test_data_coverage(onchain_features):
    """Verify we have reasonable data coverage."""
    total_rows = len(onchain_features)

    # Each feature should have at least 60% valid data
    # (accounting for ~30% warmup period before 2017)
    for col in onchain_features.columns:
        valid_count = onchain_features[col].notna().sum()
        coverage = valid_count / total_rows

        assert coverage >= 0.60, \
            f"{col} has only {coverage:.1%} coverage, expected >= 60%"
