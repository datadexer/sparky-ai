"""Tests for data quality checks."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sparky.data.quality import DataQualityChecker


@pytest.fixture
def checker():
    """DataQualityChecker instance."""
    return DataQualityChecker()


@pytest.fixture
def clean_df():
    """Clean DataFrame with no issues."""
    dates = pd.date_range("2026-02-10", periods=10, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "close": [100.0, 110.0, 105.0, 115.0, 108.0, 120.0, 118.0, 125.0, 122.0, 130.0],
            "volume": [1000, 1200, 900, 1500, 1100, 1300, 1400, 1600, 1200, 1800],
        },
        index=dates,
    )


@pytest.fixture
def df_with_nulls():
    """DataFrame with null values."""
    dates = pd.date_range("2026-02-10", periods=10, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "close": [100.0, np.nan, 105.0, np.nan, 108.0, 120.0, np.nan, 125.0, 122.0, 130.0],
            "volume": [1000, 1200, np.nan, 1500, 1100, np.nan, 1400, 1600, 1200, 1800],
        },
        index=dates,
    )


@pytest.fixture
def df_with_gaps():
    """DataFrame with date gaps."""
    # Create dates with a 5-day gap
    dates1 = pd.date_range("2026-02-01", periods=5, freq="D", tz="UTC")
    dates2 = pd.date_range("2026-02-10", periods=5, freq="D", tz="UTC")
    dates = dates1.union(dates2)
    return pd.DataFrame(
        {
            "close": [100.0, 110.0, 105.0, 115.0, 108.0, 120.0, 118.0, 125.0, 122.0, 130.0],
            "volume": [1000, 1200, 900, 1500, 1100, 1300, 1400, 1600, 1200, 1800],
        },
        index=dates,
    )


@pytest.fixture
def df_out_of_range():
    """DataFrame with out-of-range values."""
    dates = pd.date_range("2026-02-10", periods=10, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "close": [100.0, -10.0, 105.0, 115.0, 108.0, 120.0, 118.0, 500000.0, 122.0, 130.0],
            "volume": [1000, 1200, 900, 1500, 1100, 1300, 1400, 1600, 1200, 1800],
        },
        index=dates,
    )


@pytest.fixture
def stale_df():
    """DataFrame with stale data (old dates)."""
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "close": [100.0, 110.0, 105.0, 115.0, 108.0, 120.0, 118.0, 125.0, 122.0, 130.0],
            "volume": [1000, 1200, 900, 1500, 1100, 1300, 1400, 1600, 1200, 1800],
        },
        index=dates,
    )


class TestCheckCompleteness:
    def test_detects_nulls(self, checker, df_with_nulls):
        """Test that check_completeness detects null values."""
        result = checker.check_completeness(df_with_nulls)

        assert result["check"] == "completeness"
        assert result["total_rows"] == 10
        assert result["null_counts"]["close"] == 3
        assert result["null_counts"]["volume"] == 2
        assert result["null_pct"]["close"] == 30.0
        assert result["null_pct"]["volume"] == 20.0
        assert result["pass"] is False  # >5% nulls should fail

    def test_detects_gaps(self, checker, df_with_gaps):
        """Test that check_completeness detects date gaps."""
        result = checker.check_completeness(df_with_gaps, max_gap_days=3)

        assert result["check"] == "completeness"
        assert len(result["gaps"]) == 1
        assert result["gaps"][0]["gap_days"] == 5
        assert "2026-02-10" in result["gaps"][0]["gap_end"]
        assert result["pass"] is False  # Gaps should fail

    def test_passes_with_clean_data(self, checker, clean_df):
        """Test that check_completeness passes with clean data."""
        result = checker.check_completeness(clean_df)

        assert result["check"] == "completeness"
        assert result["total_rows"] == 10
        assert all(count == 0 for count in result["null_counts"].values())
        assert all(pct == 0.0 for pct in result["null_pct"].values())
        assert len(result["gaps"]) == 0
        assert result["pass"] is True

    def test_fails_empty_dataframe(self, checker):
        """Test that check_completeness fails on empty DataFrame."""
        df = pd.DataFrame()
        result = checker.check_completeness(df)

        assert result["total_rows"] == 0
        assert result["pass"] is False

    def test_small_nulls_pass(self, checker):
        """Test that <5% nulls pass the check."""
        dates = pd.date_range("2026-02-10", periods=100, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "close": [100.0] * 95 + [np.nan] * 5,  # 5% nulls
                "volume": [1000] * 100,
            },
            index=dates,
        )
        result = checker.check_completeness(df)

        assert result["null_pct"]["close"] == 5.0
        assert result["pass"] is True  # Exactly 5% is okay


class TestCheckRange:
    def test_detects_out_of_range_values(self, checker, df_out_of_range):
        """Test that check_range detects out-of-range values."""
        result = checker.check_range(df_out_of_range, "close", min_val=0.0, max_val=10000.0)

        assert result["check"] == "range"
        assert result["column"] == "close"
        assert result["min_val"] == 0.0
        assert result["max_val"] == 10000.0
        assert result["actual_min"] == -10.0
        assert result["actual_max"] == 500000.0
        assert result["out_of_range_count"] == 2  # -10.0 and 500000.0
        assert result["pass"] is False

    def test_passes_with_in_range_values(self, checker, clean_df):
        """Test that check_range passes with in-range values."""
        result = checker.check_range(clean_df, "close", min_val=0.0, max_val=200.0)

        assert result["check"] == "range"
        assert result["column"] == "close"
        assert result["actual_min"] == 100.0
        assert result["actual_max"] == 130.0
        assert result["out_of_range_count"] == 0
        assert result["pass"] is True

    def test_only_min_val(self, checker, df_out_of_range):
        """Test check_range with only min_val specified."""
        result = checker.check_range(df_out_of_range, "close", min_val=0.0)

        assert result["out_of_range_count"] == 1  # Only -10.0
        assert result["pass"] is False

    def test_only_max_val(self, checker, df_out_of_range):
        """Test check_range with only max_val specified."""
        result = checker.check_range(df_out_of_range, "close", max_val=10000.0)

        assert result["out_of_range_count"] == 1  # Only 500000.0
        assert result["pass"] is False

    def test_missing_column(self, checker, clean_df):
        """Test check_range with missing column."""
        result = checker.check_range(clean_df, "nonexistent", min_val=0.0)

        assert result["pass"] is False
        assert "error" in result
        assert "not found" in result["error"]

    def test_ignores_nulls(self, checker, df_with_nulls):
        """Test that check_range ignores null values."""
        result = checker.check_range(df_with_nulls, "close", min_val=0.0, max_val=200.0)

        # Should only check non-null values
        assert result["out_of_range_count"] == 0
        assert result["pass"] is True


class TestCheckStaleness:
    def test_detects_stale_data(self, checker, stale_df):
        """Test that check_staleness detects stale data."""
        result = checker.check_staleness(stale_df, max_stale_days=2)

        assert result["check"] == "staleness"
        assert result["max_stale_days"] == 2
        assert "2026-01-10" in result["last_date"]
        assert result["days_old"] > 30  # Much older than 2 days
        assert result["pass"] is False

    def test_passes_with_fresh_data(self, checker, clean_df):
        """Test that check_staleness passes with fresh data."""
        result = checker.check_staleness(clean_df, max_stale_days=10)

        assert result["check"] == "staleness"
        assert result["max_stale_days"] == 10
        assert "2026-02-19" in result["last_date"]
        assert result["days_old"] <= 10
        assert result["pass"] is True

    def test_fails_empty_dataframe(self, checker):
        """Test that check_staleness fails on empty DataFrame."""
        df = pd.DataFrame()
        result = checker.check_staleness(df)

        assert result["pass"] is False
        assert result["last_date"] is None
        assert result["days_old"] is None

    def test_fails_non_datetime_index(self, checker):
        """Test that check_staleness fails without DatetimeIndex."""
        df = pd.DataFrame({"close": [100.0, 110.0]})
        result = checker.check_staleness(df)

        assert result["pass"] is False

    def test_handles_naive_datetime(self, checker):
        """Test that check_staleness handles naive (non-timezone-aware) datetimes."""
        # Create DataFrame with naive datetime index
        dates = pd.date_range("2026-02-14", periods=5, freq="D")  # No tz
        df = pd.DataFrame({"close": [100.0, 110.0, 105.0, 115.0, 108.0]}, index=dates)

        result = checker.check_staleness(df, max_stale_days=2)

        # Should localize to UTC and check
        assert result["check"] == "staleness"
        assert result["last_date"] is not None
        assert "days_old" in result


class TestCrossValidatePrice:
    def test_detects_divergent_prices(self, checker):
        """Test that cross_validate_price detects divergent prices."""
        dates = pd.date_range("2026-02-10", periods=10, freq="D", tz="UTC")
        ccxt_df = pd.DataFrame(
            {"close": [100.0, 110.0, 105.0, 115.0, 108.0, 120.0, 118.0, 125.0, 122.0, 130.0]},
            index=dates,
        )
        # Reference prices differ significantly (10% off)
        reference_df = pd.DataFrame(
            {"PriceUSD": [110.0, 121.0, 115.5, 126.5, 118.8, 132.0, 129.8, 137.5, 134.2, 143.0]},
            index=dates,
        )

        result = checker.cross_validate_price(
            ccxt_df,
            reference_df,
            max_pct_diff=0.02,  # 2% threshold
        )

        assert result["check"] == "cross_validate_price"
        assert result["max_pct_diff_threshold"] == 0.02
        assert result["dates_compared"] == 10
        assert result["mean_pct_diff"] > 0.02  # Should exceed threshold
        assert result["pass"] is False

    def test_passes_with_matching_prices(self, checker):
        """Test that cross_validate_price passes with matching prices."""
        dates = pd.date_range("2026-02-10", periods=10, freq="D", tz="UTC")
        ccxt_df = pd.DataFrame(
            {"close": [100.0, 110.0, 105.0, 115.0, 108.0, 120.0, 118.0, 125.0, 122.0, 130.0]},
            index=dates,
        )
        # Reference prices are very close (within 0.5%)
        reference_df = pd.DataFrame(
            {"PriceUSD": [100.2, 110.3, 105.1, 115.4, 108.3, 120.5, 118.2, 125.3, 122.4, 130.2]},
            index=dates,
        )

        result = checker.cross_validate_price(
            ccxt_df,
            reference_df,
            max_pct_diff=0.02,  # 2% threshold
        )

        assert result["check"] == "cross_validate_price"
        assert result["dates_compared"] == 10
        assert result["mean_pct_diff"] <= 0.02
        assert result["max_pct_diff_observed"] <= 0.02
        assert result["pass"] is True

    def test_missing_columns(self, checker, clean_df):
        """Test cross_validate_price with missing columns."""
        result = checker.cross_validate_price(clean_df, clean_df, price_col="nonexistent")

        assert result["pass"] is False
        assert "error" in result

    def test_no_overlapping_dates(self, checker):
        """Test cross_validate_price with no overlapping dates."""
        ccxt_dates = pd.date_range("2026-02-01", periods=5, freq="D", tz="UTC")
        ref_dates = pd.date_range("2026-02-10", periods=5, freq="D", tz="UTC")

        ccxt_df = pd.DataFrame({"close": [100.0, 110.0, 105.0, 115.0, 108.0]}, index=ccxt_dates)
        reference_df = pd.DataFrame({"PriceUSD": [120.0, 118.0, 125.0, 122.0, 130.0]}, index=ref_dates)

        result = checker.cross_validate_price(ccxt_df, reference_df)

        assert result["dates_compared"] == 0
        assert result["pass"] is False

    def test_custom_columns(self, checker):
        """Test cross_validate_price with custom column names."""
        dates = pd.date_range("2026-02-10", periods=5, freq="D", tz="UTC")
        df1 = pd.DataFrame({"price": [100.0, 110.0, 105.0, 115.0, 108.0]}, index=dates)
        df2 = pd.DataFrame({"value": [100.5, 110.5, 105.5, 115.5, 108.5]}, index=dates)

        result = checker.cross_validate_price(df1, df2, price_col="price", reference_col="value", max_pct_diff=0.01)

        assert result["dates_compared"] == 5
        assert result["pass"] is True


class TestRunAllChecks:
    def test_returns_complete_report(self, checker, clean_df):
        """Test that run_all_checks returns a complete report."""
        report = checker.run_all_checks(clean_df, asset="btc", source="binance", price_range=(50.0, 200.0))

        assert report["asset"] == "btc"
        assert report["source"] == "binance"
        assert "checked_at" in report
        assert "checks" in report
        assert "completeness" in report["checks"]
        assert "staleness" in report["checks"]
        assert "price_range" in report["checks"]
        assert report["overall_pass"] is True

    def test_overall_pass_false_on_failures(self, checker, df_with_nulls):
        """Test that overall_pass is False when any check fails."""
        report = checker.run_all_checks(df_with_nulls, asset="eth", source="coinbase")

        assert report["overall_pass"] is False
        assert report["checks"]["completeness"]["pass"] is False

    def test_price_range_check_optional(self, checker, clean_df):
        """Test that price range check is optional."""
        report = checker.run_all_checks(clean_df, asset="btc", source="binance")

        # Should not include price_range check when not specified
        assert "price_range" not in report["checks"]
        assert "completeness" in report["checks"]
        assert "staleness" in report["checks"]

    def test_price_range_included_when_specified(self, checker, df_out_of_range):
        """Test that price range check is included when specified."""
        report = checker.run_all_checks(df_out_of_range, asset="btc", source="binance", price_range=(0.0, 10000.0))

        assert "price_range" in report["checks"]
        assert report["checks"]["price_range"]["pass"] is False
        assert report["overall_pass"] is False

    def test_timestamp_format(self, checker, clean_df):
        """Test that checked_at timestamp is in ISO format."""
        report = checker.run_all_checks(clean_df, asset="btc", source="binance")

        # Should be parseable as ISO datetime
        checked_at = datetime.fromisoformat(report["checked_at"])
        assert isinstance(checked_at, datetime)


class TestSaveReport:
    def test_saves_json_to_disk(self, checker, tmp_path, monkeypatch):
        """Test that save_report writes JSON to disk."""
        # Monkeypatch the QUALITY_REPORTS_DIR to use tmp_path
        monkeypatch.setattr("sparky.data.quality.QUALITY_REPORTS_DIR", tmp_path)

        report = {
            "asset": "btc",
            "source": "binance",
            "checked_at": datetime.now().isoformat(),
            "checks": {"completeness": {"pass": True}},
            "overall_pass": True,
        }

        saved_path = checker.save_report(report, "test_report.json")

        assert saved_path.exists()
        assert saved_path == tmp_path / "test_report.json"

        # Verify contents
        with open(saved_path) as f:
            loaded = pd.read_json(f)
            # Just verify it's valid JSON and has expected keys

        # Better verification: reload and compare
        import json

        with open(saved_path) as f:
            loaded = json.load(f)

        assert loaded["asset"] == "btc"
        assert loaded["source"] == "binance"
        assert loaded["overall_pass"] is True

    def test_creates_directory_if_missing(self, checker, tmp_path, monkeypatch):
        """Test that save_report creates directory if it doesn't exist."""
        reports_dir = tmp_path / "quality_reports"
        monkeypatch.setattr("sparky.data.quality.QUALITY_REPORTS_DIR", reports_dir)

        report = {"asset": "eth", "overall_pass": True}

        saved_path = checker.save_report(report, "test.json")

        assert reports_dir.exists()
        assert saved_path.exists()

    def test_report_is_valid_json(self, checker, clean_df, tmp_path, monkeypatch):
        """Test that saved report is valid JSON with all fields."""
        monkeypatch.setattr("sparky.data.quality.QUALITY_REPORTS_DIR", tmp_path)

        # Run actual checks to get a real report
        report = checker.run_all_checks(clean_df, asset="btc", source="binance", price_range=(50.0, 200.0))

        saved_path = checker.save_report(report, "full_report.json")

        # Load and verify
        import json

        with open(saved_path) as f:
            loaded = json.load(f)

        assert loaded["asset"] == "btc"
        assert loaded["source"] == "binance"
        assert "checked_at" in loaded
        assert "completeness" in loaded["checks"]
        assert "staleness" in loaded["checks"]
        assert "price_range" in loaded["checks"]
