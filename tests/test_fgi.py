"""Tests for Fear & Greed Index fetcher and features."""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from sparky.data.fgi import fetch_fgi, load_fgi
from sparky.features.fgi_features import fgi_extreme_signal, fgi_exposure_adjustment

# --- Sample API response data ---
SAMPLE_API_DATA = {
    "data": [
        {"value": "73", "value_classification": "Greed", "timestamp": "1700006400"},
        {"value": "50", "value_classification": "Neutral", "timestamp": "1699920000"},
        {"value": "25", "value_classification": "Fear", "timestamp": "1699833600"},
        {"value": "10", "value_classification": "Extreme Fear", "timestamp": "1699747200"},
        {"value": "90", "value_classification": "Extreme Greed", "timestamp": "1699660800"},
    ]
}


# ─── Fetcher tests ─────────────────────────────────────────


class TestFetchFgi:
    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_parses_response(self, mock_get):
        """Mock API response is parsed into correct DataFrame schema."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = SAMPLE_API_DATA
        mock_get.return_value = mock_resp

        df = fetch_fgi()

        assert len(df) == 5
        assert list(df.columns) == ["fgi", "fgi_class"]
        assert df["fgi"].dtype in (int, "int64")
        assert df["fgi_class"].dtype == object
        assert df.index.is_monotonic_increasing
        assert str(df.index.tz) == "UTC"

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_cache_hit(self, mock_get, tmp_path):
        """Recent cache file prevents API call and returns cached data."""
        cache_file = tmp_path / "fgi.parquet"
        sample_df = pd.DataFrame(
            {"fgi": [50], "fgi_class": ["Neutral"]},
            index=pd.DatetimeIndex([datetime(2024, 1, 1, tzinfo=timezone.utc)], name="date"),
        )
        sample_df.to_parquet(cache_file)

        df = fetch_fgi(cache_path=cache_file)

        mock_get.assert_not_called()
        assert len(df) == 1
        assert df.iloc[0]["fgi"] == 50
        assert str(df.index.tz) == "UTC"

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_cache_miss_old_file(self, mock_get, tmp_path):
        """Expired cache file triggers API call."""
        cache_file = tmp_path / "fgi.parquet"
        sample_df = pd.DataFrame(
            {"fgi": [50], "fgi_class": ["Neutral"]},
            index=pd.DatetimeIndex([datetime(2024, 1, 1, tzinfo=timezone.utc)], name="date"),
        )
        sample_df.to_parquet(cache_file)
        old_time = datetime.now(tz=timezone.utc).timestamp() - 25 * 3600
        os.utime(cache_file, (old_time, old_time))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = SAMPLE_API_DATA
        mock_get.return_value = mock_resp

        df = fetch_fgi(cache_path=cache_file)

        mock_get.assert_called_once()
        assert len(df) == 5

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_http_error(self, mock_get):
        """HTTP error raises RuntimeError."""
        mock_get.side_effect = requests.exceptions.HTTPError("500 Server Error")

        with pytest.raises(RuntimeError, match="FGI API request failed"):
            fetch_fgi()

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_connection_error(self, mock_get):
        """Network failure raises RuntimeError."""
        mock_get.side_effect = requests.exceptions.ConnectionError("DNS failure")

        with pytest.raises(RuntimeError, match="FGI API request failed"):
            fetch_fgi()

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_missing_data_key(self, mock_get):
        """Response without 'data' key raises ValueError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"error": "something"}
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="missing 'data' key"):
            fetch_fgi()

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_non_json_response(self, mock_get):
        """Non-JSON response raises ValueError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("No JSON object could be decoded")
        mock_resp.text = "<html>Error</html>"
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="non-JSON response"):
            fetch_fgi()

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_saves_cache(self, mock_get, tmp_path):
        """Fetched data is saved to cache_path."""
        cache_file = tmp_path / "fgi.parquet"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = SAMPLE_API_DATA
        mock_get.return_value = mock_resp

        fetch_fgi(cache_path=cache_file)

        assert cache_file.exists()
        loaded = pd.read_parquet(cache_file)
        assert len(loaded) == 5

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_empty_data(self, mock_get):
        """Empty data list raises ValueError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="zero usable records"):
            fetch_fgi()


class TestLoadFgi:
    def test_load_fgi_bad_schema(self, tmp_path):
        """Parquet with wrong columns raises ValueError."""
        bad_df = pd.DataFrame(
            {"wrong_col": [1]},
            index=pd.DatetimeIndex([datetime(2024, 1, 1, tzinfo=timezone.utc)], name="date"),
        )
        path = tmp_path / "bad.parquet"
        bad_df.to_parquet(path)

        with pytest.raises(ValueError, match="unexpected schema"):
            load_fgi(path)

    def test_load_fgi_bad_index(self, tmp_path):
        """Parquet without DatetimeIndex raises ValueError."""
        bad_df = pd.DataFrame({"fgi": [50], "fgi_class": ["Neutral"]})
        path = tmp_path / "bad.parquet"
        bad_df.to_parquet(path)

        with pytest.raises(ValueError, match="DatetimeIndex"):
            load_fgi(path)


# ─── Feature tests ──────────────────────────────────────────


class TestFgiExtremeSignal:
    def test_basic_signal_with_lag(self):
        """Signal reflects previous bar's FGI due to shift(1)."""
        fgi = pd.Series([10, 50, 90, 20, 80])
        signal = fgi_extreme_signal(fgi)
        # Lagged: [NaN, 10, 50, 90, 20]
        # Expected: [NaN, 1.0, 0.0, -1.0, 1.0]
        assert pd.isna(signal.iloc[0])
        expected_rest = pd.Series([1.0, 0.0, -1.0, 1.0], index=[1, 2, 3, 4])
        pd.testing.assert_series_equal(
            signal.iloc[1:].reset_index(drop=True),
            expected_rest.reset_index(drop=True),
            check_names=False,
        )

    def test_boundary_values(self):
        """Values exactly at thresholds are included (after lag)."""
        # Input:  [50, 20, 21, 79, 80]
        # Lagged: [NaN, 50, 20, 21, 79]
        # Signal: [NaN, 0.0, 1.0, 0.0, 0.0]
        fgi = pd.Series([50, 20, 21, 79, 80])
        signal = fgi_extreme_signal(fgi)
        assert pd.isna(signal.iloc[0])
        expected_rest = pd.Series([0.0, 1.0, 0.0, 0.0], index=[1, 2, 3, 4])
        pd.testing.assert_series_equal(
            signal.iloc[1:].reset_index(drop=True),
            expected_rest.reset_index(drop=True),
            check_names=False,
        )

    def test_custom_thresholds(self):
        """Custom thresholds work correctly with lag."""
        # Input:  [50, 10, 30, 50, 70, 90]
        # Lagged: [NaN, 50, 10, 30, 50, 70]
        # Signal: [NaN, 0.0, 1.0, 1.0, 0.0, -1.0]
        fgi = pd.Series([50, 10, 30, 50, 70, 90])
        signal = fgi_extreme_signal(fgi, fear_threshold=30, greed_threshold=70)
        assert pd.isna(signal.iloc[0])
        expected_rest = pd.Series([0.0, 1.0, 1.0, 0.0, -1.0], index=[1, 2, 3, 4, 5])
        pd.testing.assert_series_equal(
            signal.iloc[1:].reset_index(drop=True),
            expected_rest.reset_index(drop=True),
            check_names=False,
        )

    def test_empty_series(self):
        """Empty input returns empty output."""
        fgi = pd.Series([], dtype=int)
        signal = fgi_extreme_signal(fgi)
        assert len(signal) == 0

    def test_invalid_thresholds(self):
        """fear_threshold >= greed_threshold raises ValueError."""
        fgi = pd.Series([50])
        with pytest.raises(ValueError, match="fear_threshold.*must be <"):
            fgi_extreme_signal(fgi, fear_threshold=80, greed_threshold=20)
        with pytest.raises(ValueError, match="fear_threshold.*must be <"):
            fgi_extreme_signal(fgi, fear_threshold=50, greed_threshold=50)

    def test_nan_propagation(self):
        """NaN in input fgi propagates as NaN in output."""
        fgi = pd.Series([10, float("nan"), 50])
        signal = fgi_extreme_signal(fgi)
        # Lagged: [NaN, 10, NaN]
        # Signal: [NaN, 1.0, NaN]
        assert pd.isna(signal.iloc[0])
        assert signal.iloc[1] == 1.0
        assert pd.isna(signal.iloc[2])


class TestFgiExposureAdjustment:
    def test_basic_adjustment_with_lag(self):
        """Adjustment reflects previous bar's FGI due to shift(1)."""
        fgi = pd.Series([10, 50, 90, 20, 80])
        adj = fgi_exposure_adjustment(fgi)
        # Lagged: [NaN, 10, 50, 90, 20]
        # Adj:    [NaN, 1.25, 1.0, 0.75, 1.25]
        assert pd.isna(adj.iloc[0])
        expected_rest = pd.Series([1.25, 1.0, 0.75, 1.25], index=[1, 2, 3, 4])
        pd.testing.assert_series_equal(
            adj.iloc[1:].reset_index(drop=True),
            expected_rest.reset_index(drop=True),
            check_names=False,
        )

    def test_boundary_values(self):
        """Values exactly at thresholds are included (after lag)."""
        # Input:  [50, 20, 21, 79, 80]
        # Lagged: [NaN, 50, 20, 21, 79]
        # Adj:    [NaN, 1.0, 1.25, 1.0, 1.0]
        fgi = pd.Series([50, 20, 21, 79, 80])
        adj = fgi_exposure_adjustment(fgi)
        assert pd.isna(adj.iloc[0])
        expected_rest = pd.Series([1.0, 1.25, 1.0, 1.0], index=[1, 2, 3, 4])
        pd.testing.assert_series_equal(
            adj.iloc[1:].reset_index(drop=True),
            expected_rest.reset_index(drop=True),
            check_names=False,
        )

    def test_custom_adjustment(self):
        """Custom adjustment factor works correctly with lag."""
        fgi = pd.Series([50, 10, 50, 90])
        adj = fgi_exposure_adjustment(fgi, adjustment=0.5)
        # Lagged: [NaN, 50, 10, 50]
        # Adj:    [NaN, 1.0, 1.5, 1.0]
        assert pd.isna(adj.iloc[0])
        expected_rest = pd.Series([1.0, 1.5, 1.0], index=[1, 2, 3])
        pd.testing.assert_series_equal(
            adj.iloc[1:].reset_index(drop=True),
            expected_rest.reset_index(drop=True),
            check_names=False,
        )

    def test_empty_series(self):
        """Empty input returns empty output."""
        fgi = pd.Series([], dtype=int)
        adj = fgi_exposure_adjustment(fgi)
        assert len(adj) == 0

    def test_invalid_thresholds(self):
        """fear_threshold >= greed_threshold raises ValueError."""
        fgi = pd.Series([50])
        with pytest.raises(ValueError, match="fear_threshold.*must be <"):
            fgi_exposure_adjustment(fgi, fear_threshold=80, greed_threshold=20)

    def test_invalid_adjustment(self):
        """adjustment outside (0, 1) raises ValueError."""
        fgi = pd.Series([50])
        with pytest.raises(ValueError, match="adjustment must be in"):
            fgi_exposure_adjustment(fgi, adjustment=0.0)
        with pytest.raises(ValueError, match="adjustment must be in"):
            fgi_exposure_adjustment(fgi, adjustment=1.0)
        with pytest.raises(ValueError, match="adjustment must be in"):
            fgi_exposure_adjustment(fgi, adjustment=1.5)
