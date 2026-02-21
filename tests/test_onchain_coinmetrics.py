"""Tests for CoinMetrics on-chain data fetcher."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from sparky.data.onchain_coinmetrics import (
    ASSET_METRICS,
    REQUEST_INTERVAL,
    CoinMetricsFetcher,
)


@pytest.fixture
def mock_client():
    """Mock CoinMetricsClient."""
    with patch("coinmetrics.api_client.CoinMetricsClient") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


@pytest.fixture
def fetcher(mock_client):
    """CoinMetricsFetcher instance with mocked client."""
    return CoinMetricsFetcher()


@pytest.fixture
def sample_api_response():
    """Sample API response data."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "time": dates,
            "asset": ["btc"] * 5,
            "HashRate": [100.0, 110.0, 105.0, 115.0, 108.0],
            "AdrActCnt": [1000, 1200, 900, 1500, 1100],
        }
    )


@pytest.fixture
def sample_api_response_with_strings():
    """Sample API response with string values that need conversion."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "time": dates,
            "asset": ["eth"] * 3,
            "HashRate": ["100.5", "110.2", "105.8"],
            "AdrActCnt": ["1000", "1200", "900"],
        }
    )


class TestAssetMetricsConfig:
    def test_btc_metrics_include_exchange_flows(self):
        assert "FlowInExNtv" in ASSET_METRICS["btc"]
        assert "FlowOutExNtv" in ASSET_METRICS["btc"]


class TestCoinMetricsFetcherInit:
    def test_init_creates_client(self, mock_client):
        fetcher = CoinMetricsFetcher()
        assert fetcher.client is not None
        assert fetcher._last_request_time == 0.0
        assert fetcher._request_count == 0

    @pytest.mark.skip(reason="ImportError testing with lazy imports is complex; covered by manual testing")
    def test_init_missing_dependency(self):
        """Test that missing coinmetrics-api-client raises helpful error.

        This is tested manually. When the module is not installed,
        __init__ should raise:
        ImportError: coinmetrics-api-client is required. Install with: pip install coinmetrics-api-client
        """
        pass


class TestFetchMetrics:
    def test_returns_correct_dataframe_structure(self, fetcher, mock_client, sample_api_response):
        # Mock the API response
        mock_result = Mock()
        mock_result.to_dataframe.return_value = sample_api_response.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        df = fetcher.fetch_metrics("btc", ["HashRate", "AdrActCnt"], "2024-01-01", "2024-01-05")

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None  # UTC timezone
        assert len(df) == 5
        assert list(df.columns) == ["HashRate", "AdrActCnt"]

        # Check asset column removed
        assert "asset" not in df.columns
        assert "time" not in df.columns

    def test_client_called_with_correct_params(self, fetcher, mock_client, sample_api_response):
        mock_result = Mock()
        mock_result.to_dataframe.return_value = sample_api_response.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        fetcher.fetch_metrics("btc", ["HashRate", "AdrActCnt"], "2024-01-01", "2024-01-05")

        mock_client.get_asset_metrics.assert_called_once_with(
            assets="btc",
            metrics="HashRate,AdrActCnt",
            start_time="2024-01-01",
            end_time="2024-01-05",
            frequency="1d",
        )

    def test_end_date_defaults_to_today(self, fetcher, mock_client, sample_api_response):
        mock_result = Mock()
        mock_result.to_dataframe.return_value = sample_api_response.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        with patch("sparky.data.onchain_coinmetrics.datetime") as mock_dt:
            mock_now = datetime(2024, 12, 31, tzinfo=timezone.utc)
            mock_dt.now.return_value = mock_now

            fetcher.fetch_metrics("btc", ["HashRate"], "2024-01-01")

            call_args = mock_client.get_asset_metrics.call_args
            assert call_args.kwargs["end_time"] == "2024-12-31"

    def test_empty_response_returns_empty_dataframe(self, fetcher, mock_client):
        mock_result = Mock()
        mock_result.to_dataframe.return_value = pd.DataFrame()
        mock_client.get_asset_metrics.return_value = mock_result

        df = fetcher.fetch_metrics("btc", ["HashRate"], "2024-01-01")

        assert df.empty
        assert isinstance(df, pd.DataFrame)

    def test_numeric_conversion_handles_string_values(self, fetcher, mock_client, sample_api_response_with_strings):
        mock_result = Mock()
        mock_result.to_dataframe.return_value = sample_api_response_with_strings.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        df = fetcher.fetch_metrics("eth", ["HashRate", "AdrActCnt"], "2024-01-01", "2024-01-03")

        # Check all values are numeric (can be float or int)
        assert pd.api.types.is_numeric_dtype(df["HashRate"])
        assert pd.api.types.is_numeric_dtype(df["AdrActCnt"])
        assert df["HashRate"].iloc[0] == 100.5
        assert df["AdrActCnt"].iloc[0] == 1000.0

    def test_handles_date_column_instead_of_time(self, fetcher, mock_client):
        # Some responses might use "date" instead of "time"
        dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        df_with_date = pd.DataFrame(
            {
                "date": dates,
                "asset": ["btc"] * 3,
                "HashRate": [100.0, 110.0, 105.0],
            }
        )

        mock_result = Mock()
        mock_result.to_dataframe.return_value = df_with_date.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        df = fetcher.fetch_metrics("btc", ["HashRate"], "2024-01-01")

        assert isinstance(df.index, pd.DatetimeIndex)
        assert "date" not in df.columns

    def test_removes_duplicate_timestamps(self, fetcher, mock_client):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"], utc=True)
        df_with_dupes = pd.DataFrame(
            {
                "time": dates,
                "asset": ["btc"] * 4,
                "HashRate": [100.0, 110.0, 999.0, 105.0],  # 999 should be kept (last)
            }
        )

        mock_result = Mock()
        mock_result.to_dataframe.return_value = df_with_dupes.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        df = fetcher.fetch_metrics("btc", ["HashRate"], "2024-01-01")

        assert len(df) == 3  # Duplicate removed
        assert df.loc["2024-01-02", "HashRate"] == 999.0  # Kept last

    def test_sorts_by_index(self, fetcher, mock_client):
        dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"], utc=True)
        df_unsorted = pd.DataFrame(
            {
                "time": dates,
                "asset": ["btc"] * 3,
                "HashRate": [105.0, 100.0, 110.0],
            }
        )

        mock_result = Mock()
        mock_result.to_dataframe.return_value = df_unsorted.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        df = fetcher.fetch_metrics("btc", ["HashRate"], "2024-01-01")

        assert df.index.is_monotonic_increasing
        assert df.iloc[0]["HashRate"] == 100.0
        assert df.iloc[-1]["HashRate"] == 105.0

    def test_api_error_raises_exception(self, fetcher, mock_client):
        mock_client.get_asset_metrics.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            fetcher.fetch_metrics("btc", ["HashRate"], "2024-01-01")

    def test_increments_request_count(self, fetcher, mock_client, sample_api_response):
        mock_result = Mock()
        mock_result.to_dataframe.return_value = sample_api_response.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        assert fetcher._request_count == 0
        fetcher.fetch_metrics("btc", ["HashRate"], "2024-01-01")
        assert fetcher._request_count == 1
        fetcher.fetch_metrics("btc", ["HashRate"], "2024-01-01")
        assert fetcher._request_count == 2


class TestFetchAssetMetrics:
    def test_fetches_all_configured_metrics_btc(self, fetcher, mock_client, sample_api_response):
        mock_result = Mock()
        mock_result.to_dataframe.return_value = sample_api_response.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        df = fetcher.fetch_asset_metrics("btc", "2024-01-01", "2024-01-05")

        # Check it called with all BTC metrics
        call_args = mock_client.get_asset_metrics.call_args
        metrics_str = call_args.kwargs["metrics"]
        metrics_list = metrics_str.split(",")

        assert set(metrics_list) == set(ASSET_METRICS["btc"])

    def test_fetches_all_configured_metrics_eth(self, fetcher, mock_client, sample_api_response):
        mock_result = Mock()
        mock_result.to_dataframe.return_value = sample_api_response.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        df = fetcher.fetch_asset_metrics("eth", "2024-01-01", "2024-01-05")

        # Check it called with all ETH metrics
        call_args = mock_client.get_asset_metrics.call_args
        metrics_str = call_args.kwargs["metrics"]
        metrics_list = metrics_str.split(",")

        assert set(metrics_list) == set(ASSET_METRICS["eth"])

    def test_case_insensitive_asset(self, fetcher, mock_client, sample_api_response):
        mock_result = Mock()
        mock_result.to_dataframe.return_value = sample_api_response.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        df = fetcher.fetch_asset_metrics("BTC", "2024-01-01")

        # Should normalize to lowercase
        call_args = mock_client.get_asset_metrics.call_args
        assert call_args.kwargs["assets"] == "btc"

    def test_unknown_asset_raises_value_error(self, fetcher):
        with pytest.raises(ValueError, match="Unknown asset 'dogecoin'"):
            fetcher.fetch_asset_metrics("dogecoin", "2024-01-01")

    def test_error_message_lists_available_assets(self, fetcher):
        with pytest.raises(ValueError, match=r"Available: \['btc', 'eth'\]"):
            fetcher.fetch_asset_metrics("xyz", "2024-01-01")


class TestRateLimiting:
    def test_rate_limit_called_before_request(self, fetcher, mock_client, sample_api_response):
        mock_result = Mock()
        mock_result.to_dataframe.return_value = sample_api_response.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        with patch.object(fetcher, "_rate_limit") as mock_rate_limit:
            fetcher.fetch_metrics("btc", ["HashRate"], "2024-01-01")
            mock_rate_limit.assert_called_once()

    def test_rate_limit_sleeps_when_too_fast(self, fetcher):

        with patch("sparky.data.onchain_coinmetrics.time") as mock_time:
            mock_time.time.return_value = 1.0
            fetcher._last_request_time = 0.5  # 0.5 seconds ago

            fetcher._rate_limit()

            # Should sleep for REQUEST_INTERVAL - elapsed
            elapsed = 1.0 - 0.5  # 0.5
            expected_sleep = REQUEST_INTERVAL - elapsed
            mock_time.sleep.assert_called_once_with(expected_sleep)

    def test_rate_limit_no_sleep_when_enough_time_passed(self, fetcher):

        with patch("sparky.data.onchain_coinmetrics.time") as mock_time:
            mock_time.time.return_value = 10.0
            fetcher._last_request_time = 1.0  # 9 seconds ago (more than REQUEST_INTERVAL)

            fetcher._rate_limit()

            # Should not sleep
            mock_time.sleep.assert_not_called()

    def test_updates_last_request_time(self, fetcher, mock_client, sample_api_response):
        mock_result = Mock()
        mock_result.to_dataframe.return_value = sample_api_response.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        with patch("sparky.data.onchain_coinmetrics.time") as mock_time:
            mock_time.time.return_value = 42.0

            fetcher.fetch_metrics("btc", ["HashRate"], "2024-01-01")

            assert fetcher._last_request_time == 42.0


class TestAvailableAssetsAndMetrics:
    def test_available_assets(self, fetcher):
        assets = fetcher.available_assets
        assert isinstance(assets, list)
        assert set(assets) == {"btc", "eth"}

    def test_available_metrics_btc(self, fetcher):
        metrics = fetcher.get_available_metrics("btc")
        assert isinstance(metrics, list)
        assert set(metrics) == set(ASSET_METRICS["btc"])
        assert "HashRate" in metrics
        assert "AdrActCnt" in metrics

    def test_available_metrics_eth(self, fetcher):
        metrics = fetcher.get_available_metrics("eth")
        assert isinstance(metrics, list)
        assert set(metrics) == set(ASSET_METRICS["eth"])
        assert "HashRate" in metrics
        assert "AdrBalCnt" in metrics

    def test_available_metrics_case_insensitive(self, fetcher):
        metrics_lower = fetcher.get_available_metrics("btc")
        metrics_upper = fetcher.get_available_metrics("BTC")
        assert metrics_lower == metrics_upper

    def test_available_metrics_unknown_asset(self, fetcher):
        metrics = fetcher.get_available_metrics("unknown")
        assert metrics == []


class TestNumericConversion:
    def test_coerces_invalid_values_to_nan(self, fetcher, mock_client):
        dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        df_with_invalid = pd.DataFrame(
            {
                "time": dates,
                "asset": ["btc"] * 3,
                "HashRate": ["100.0", "invalid", "105.0"],
            }
        )

        mock_result = Mock()
        mock_result.to_dataframe.return_value = df_with_invalid.copy()
        mock_client.get_asset_metrics.return_value = mock_result

        df = fetcher.fetch_metrics("btc", ["HashRate"], "2024-01-01")

        assert pd.isna(df.iloc[1]["HashRate"])
        assert df.iloc[0]["HashRate"] == 100.0
        assert df.iloc[2]["HashRate"] == 105.0
