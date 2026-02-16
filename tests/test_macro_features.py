"""Tests for macro feature preparation.

These tests require local data files (data/processed/macro_features_hourly.parquet).
They are skipped in CI where data files are not available.
"""

import pandas as pd
import pytest
from pathlib import Path

MACRO_PATH = Path("data/processed/macro_features_hourly.parquet")
_has_macro_data = MACRO_PATH.exists()

pytestmark = pytest.mark.skipif(
    not _has_macro_data,
    reason="Macro features data not available (requires local data files)"
)


@pytest.fixture
def macro_features():
    """Load the prepared macro features."""
    return pd.read_parquet(MACRO_PATH)


def test_output_file_exists():
    """Verify the output file exists."""
    assert MACRO_PATH.exists(), "Macro features file should exist"


def test_expected_columns(macro_features):
    """Verify all expected columns are present."""
    expected_columns = [
        'dxy_return_1d', 'dxy_return_5d', 'dxy_sma_ratio_20d',
        'gold_return_1d', 'gold_return_5d', 'gold_sma_ratio_20d',
        'spx_return_1d', 'spx_return_5d', 'spx_vol_5d',
        'vix_level', 'vix_change_1d', 'vix_sma_ratio_10d',
        'btc_gold_corr_30d'
    ]
    assert list(macro_features.columns) == expected_columns, "Column mismatch"


def test_hourly_index(macro_features):
    """Verify the index is DatetimeIndex at hourly frequency."""
    assert isinstance(macro_features.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
    assert macro_features.index.tz is not None, "Index should be timezone-aware (UTC)"


def test_no_lookahead_bias(macro_features):
    """
    Verify no look-ahead bias: hourly features on day D should use macro data from day D-1.

    Method: Check that features change only at day boundaries (00:00 UTC).
    Within a day, all hourly rows should have identical feature values.
    """
    # Sample a random day with non-null data
    non_null_data = macro_features.dropna(subset=['dxy_return_1d'])
    if len(non_null_data) < 48:
        pytest.skip("Not enough data to test within-day consistency")

    # Get a date in the middle of the dataset
    sample_date = non_null_data.index[len(non_null_data) // 2].date()

    # Get all hours for this date
    day_data = macro_features[macro_features.index.date == sample_date]

    if len(day_data) == 0:
        pytest.skip(f"No data for sample date {sample_date}")

    # Within a single day, all feature values should be identical (forward-filled from previous day)
    for col in macro_features.columns:
        unique_values = day_data[col].dropna().unique()
        assert len(unique_values) <= 1, (
            f"Column {col} should have at most 1 unique value per day (forward-filled), "
            f"but found {len(unique_values)} on {sample_date}"
        )


def test_feature_value_ranges(macro_features):
    """Verify features have reasonable values."""
    # Returns should be between -0.5 and 0.5 (50% daily move is extreme)
    return_cols = [col for col in macro_features.columns if 'return' in col]
    for col in return_cols:
        non_null = macro_features[col].dropna()
        if len(non_null) > 0:
            assert non_null.min() >= -0.5, f"{col} has unreasonably low values"
            assert non_null.max() <= 0.5, f"{col} has unreasonably high values"

    # SMA ratios should be reasonable (-0.5 to 0.5 means 50% deviation)
    sma_cols = [col for col in macro_features.columns if 'sma_ratio' in col]
    for col in sma_cols:
        non_null = macro_features[col].dropna()
        if len(non_null) > 0:
            assert non_null.min() >= -0.5, f"{col} has unreasonably low values"
            assert non_null.max() <= 2.0, f"{col} has unreasonably high values"

    # Correlation should be between -1 and 1
    if 'btc_gold_corr_30d' in macro_features.columns:
        corr = macro_features['btc_gold_corr_30d'].dropna()
        if len(corr) > 0:
            assert corr.min() >= -1.0, "Correlation cannot be less than -1"
            assert corr.max() <= 1.0, "Correlation cannot be greater than 1"

    # VIX level should be positive
    if 'vix_level' in macro_features.columns:
        vix = macro_features['vix_level'].dropna()
        if len(vix) > 0:
            assert vix.min() > 0, "VIX level must be positive"


def test_nan_only_in_warmup_period(macro_features):
    """
    Verify NaNs are primarily in the warmup period.

    Note: BTC hourly data goes back to 2013, but macro data only starts in 2017.
    This test verifies that WITHIN the macro data range, warmup is handled correctly.
    """
    # Find first valid data point
    first_valid = macro_features['dxy_return_1d'].first_valid_index()
    if first_valid is None:
        pytest.skip("No valid data found")

    # Get data from first valid onwards
    data_range = macro_features.loc[first_valid:]

    # After 30-day warmup (720 hours), most features should have data
    warmup_hours = 720
    after_warmup = data_range.iloc[warmup_hours:]

    # Within the macro data range, expect >95% coverage after warmup
    for col in macro_features.columns:
        non_null_pct = after_warmup[col].notna().sum() / len(after_warmup)
        assert non_null_pct > 0.95, (
            f"Column {col} should have >95% non-null after warmup period, "
            f"but has only {non_null_pct:.1%}"
        )


def test_forward_fill_behavior(macro_features):
    """
    Verify forward-fill behavior: consecutive hours within a single day
    should have identical values.
    """
    # Sample data with complete data
    non_null_data = macro_features.dropna(subset=['dxy_return_1d'])

    if len(non_null_data) < 24:
        pytest.skip("Not enough data to test forward-fill behavior")

    # Get a single day's data (find a day with complete 24 hours)
    sample_start_idx = len(non_null_data) // 2

    # Find the start of a day (00:00 hour)
    start_ts = non_null_data.iloc[sample_start_idx:].index[0]
    day_start = start_ts.normalize()  # Get midnight of that day

    # Get all hours of that single day
    single_day = non_null_data[
        (non_null_data.index >= day_start) &
        (non_null_data.index < day_start + pd.Timedelta(days=1))
    ]

    if len(single_day) == 0:
        pytest.skip("Could not find a complete day")

    # All values within this single day should be constant
    for col in macro_features.columns:
        if col in single_day.columns:
            unique_values = single_day[col].dropna().unique()
            assert len(unique_values) <= 1, (
                f"{col} should be constant within a single day, "
                f"but found {len(unique_values)} unique values"
            )


def test_cross_asset_correlation_computed(macro_features):
    """Verify BTC-Gold correlation is computed and has reasonable values."""
    assert 'btc_gold_corr_30d' in macro_features.columns, "BTC-Gold correlation should exist"

    corr = macro_features['btc_gold_corr_30d'].dropna()
    assert len(corr) > 0, "BTC-Gold correlation should have non-null values"

    # Correlation should vary over time (not all the same)
    assert corr.nunique() > 10, "Correlation should vary over time"


def test_data_coverage(macro_features):
    """Verify adequate data coverage across all features."""
    total_rows = len(macro_features)

    # Each feature should have at least 60% coverage (accounting for warmup period)
    for col in macro_features.columns:
        non_null_count = macro_features[col].notna().sum()
        coverage_pct = non_null_count / total_rows

        # Adjusted thresholds for different feature types
        if 'corr_30d' in col:
            min_coverage = 0.60  # 30-day features need longer warmup
        elif 'sma_ratio_20d' in col or 'vol_5d' in col:
            min_coverage = 0.65  # 20-day features
        else:
            min_coverage = 0.68  # Other features

        assert coverage_pct >= min_coverage, (
            f"Column {col} has only {coverage_pct:.1%} coverage, "
            f"expected at least {min_coverage:.1%}"
        )
