"""Tests for Fear & Greed Index fetcher and features."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sparky.data.fgi import fetch_fgi
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
        mock_resp.json.return_value = SAMPLE_API_DATA
        mock_get.return_value = mock_resp

        df = fetch_fgi()

        assert len(df) == 5
        assert list(df.columns) == ["fgi", "fgi_class"]
        assert df["fgi"].dtype in (int, "int64")
        assert df["fgi_class"].dtype == object
        # Should be sorted ascending
        assert df.index.is_monotonic_increasing
        # Index should be UTC
        assert str(df.index.tz) == "UTC"

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_cache_hit(self, mock_get, tmp_path):
        """Recent cache file prevents API call."""
        cache_file = tmp_path / "fgi.parquet"
        # Create a fresh cache file
        sample_df = pd.DataFrame(
            {"fgi": [50], "fgi_class": ["Neutral"]},
            index=pd.DatetimeIndex(
                [datetime(2024, 1, 1, tzinfo=timezone.utc)], name="date"
            ),
        )
        sample_df.to_parquet(cache_file)

        df = fetch_fgi(cache_path=cache_file)

        mock_get.assert_not_called()
        assert len(df) == 1

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_cache_miss_old_file(self, mock_get, tmp_path):
        """Expired cache file triggers API call."""
        import os

        cache_file = tmp_path / "fgi.parquet"
        sample_df = pd.DataFrame(
            {"fgi": [50], "fgi_class": ["Neutral"]},
            index=pd.DatetimeIndex(
                [datetime(2024, 1, 1, tzinfo=timezone.utc)], name="date"
            ),
        )
        sample_df.to_parquet(cache_file)
        # Set mtime to 25 hours ago
        old_time = datetime.now(tz=timezone.utc).timestamp() - 25 * 3600
        os.utime(cache_file, (old_time, old_time))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_API_DATA
        mock_get.return_value = mock_resp

        df = fetch_fgi(cache_path=cache_file)

        mock_get.assert_called_once()
        assert len(df) == 5

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_http_error(self, mock_get):
        """Non-200 status raises RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        with pytest.raises(RuntimeError, match="status 500"):
            fetch_fgi()

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_missing_data_key(self, mock_get):
        """Response without 'data' key raises ValueError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"error": "something"}
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="missing 'data' key"):
            fetch_fgi()

    @patch("sparky.data.fgi.requests.get")
    def test_fetch_fgi_saves_cache(self, mock_get, tmp_path):
        """Fetched data is saved to cache_path."""
        cache_file = tmp_path / "fgi.parquet"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_API_DATA
        mock_get.return_value = mock_resp

        fetch_fgi(cache_path=cache_file)

        assert cache_file.exists()
        loaded = pd.read_parquet(cache_file)
        assert len(loaded) == 5


# ─── Feature tests ──────────────────────────────────────────


class TestFgiExtremeSignal:
    def test_basic_signal(self):
        """Verify signal values for known inputs."""
        fgi = pd.Series([10, 50, 90, 20, 80])
        signal = fgi_extreme_signal(fgi)
        expected = pd.Series([1.0, 0.0, -1.0, 1.0, -1.0])
        pd.testing.assert_series_equal(signal, expected, check_names=False)

    def test_boundary_values(self):
        """Values exactly at thresholds are included."""
        fgi = pd.Series([20, 21, 79, 80])
        signal = fgi_extreme_signal(fgi)
        expected = pd.Series([1.0, 0.0, 0.0, -1.0])
        pd.testing.assert_series_equal(signal, expected, check_names=False)

    def test_custom_thresholds(self):
        """Custom thresholds work correctly."""
        fgi = pd.Series([10, 30, 50, 70, 90])
        signal = fgi_extreme_signal(fgi, fear_threshold=30, greed_threshold=70)
        expected = pd.Series([1.0, 1.0, 0.0, -1.0, -1.0])
        pd.testing.assert_series_equal(signal, expected, check_names=False)

    def test_empty_series(self):
        """Empty input returns empty output."""
        fgi = pd.Series([], dtype=int)
        signal = fgi_extreme_signal(fgi)
        assert len(signal) == 0


class TestFgiExposureAdjustment:
    def test_basic_adjustment(self):
        """Verify adjustment values for known inputs."""
        fgi = pd.Series([10, 50, 90, 20, 80])
        adj = fgi_exposure_adjustment(fgi)
        expected = pd.Series([1.25, 1.0, 0.75, 1.25, 0.75])
        pd.testing.assert_series_equal(adj, expected, check_names=False)

    def test_boundary_values(self):
        """Values exactly at thresholds are included."""
        fgi = pd.Series([20, 21, 79, 80])
        adj = fgi_exposure_adjustment(fgi)
        expected = pd.Series([1.25, 1.0, 1.0, 0.75])
        pd.testing.assert_series_equal(adj, expected, check_names=False)

    def test_custom_adjustment(self):
        """Custom adjustment factor works correctly."""
        fgi = pd.Series([10, 50, 90])
        adj = fgi_exposure_adjustment(fgi, adjustment=0.5)
        expected = pd.Series([1.5, 1.0, 0.5])
        pd.testing.assert_series_equal(adj, expected, check_names=False)

    def test_empty_series(self):
        """Empty input returns empty output."""
        fgi = pd.Series([], dtype=int)
        adj = fgi_exposure_adjustment(fgi)
        assert len(adj) == 0
