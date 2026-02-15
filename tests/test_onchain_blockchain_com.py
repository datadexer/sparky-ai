"""Tests for Blockchain.com on-chain data fetcher."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from sparky.data.onchain_blockchain_com import (
    BASE_URL,
    METRIC_ENDPOINTS,
    BlockchainComFetcher,
)


@pytest.fixture
def fetcher():
    """Create a BlockchainComFetcher instance."""
    return BlockchainComFetcher()


@pytest.fixture
def sample_response_data():
    """Sample API response data."""
    return {
        "values": [
            {"x": 1704067200, "y": 100.5},  # 2024-01-01 00:00:00 UTC
            {"x": 1704153600, "y": 105.2},  # 2024-01-02 00:00:00 UTC
            {"x": 1704240000, "y": 110.8},  # 2024-01-03 00:00:00 UTC
        ]
    }


@pytest.fixture
def empty_response_data():
    """Empty API response data."""
    return {"values": []}


class TestFetchMetric:
    def test_returns_correct_dataframe_structure(self, fetcher, sample_response_data):
        """Test fetch_metric returns DataFrame with correct structure."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = sample_response_data
            mock_get.return_value = mock_response

            df = fetcher.fetch_metric("hash_rate", timespan="5years")

            assert isinstance(df, pd.DataFrame)
            assert df.index.name == "date"
            assert list(df.columns) == ["hash_rate"]
            assert len(df) == 3
            assert df.index.tz == timezone.utc
            assert df["hash_rate"].iloc[0] == 100.5
            assert df["hash_rate"].iloc[1] == 105.2
            assert df["hash_rate"].iloc[2] == 110.8

    def test_constructs_correct_api_url_with_timespan(self, fetcher, sample_response_data):
        """Test that correct API URL is constructed with timespan parameter."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = sample_response_data
            mock_get.return_value = mock_response

            fetcher.fetch_metric("active_addresses", timespan="2years")

            expected_url = f"{BASE_URL}/charts/n-unique-addresses"
            expected_params = {"format": "json", "sampled": "false", "timespan": "2years"}

            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[0][0] == expected_url
            assert call_args[1]["params"] == expected_params
            assert call_args[1]["timeout"] == 30

    def test_constructs_correct_api_url_with_start_date(self, fetcher, sample_response_data):
        """Test that correct API URL is constructed with start_date parameter."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = sample_response_data
            mock_get.return_value = mock_response

            fetcher.fetch_metric("transaction_count", start_date="2024-01-01")

            expected_url = f"{BASE_URL}/charts/n-transactions"
            expected_params = {"format": "json", "sampled": "false", "start": "2024-01-01"}

            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[0][0] == expected_url
            assert call_args[1]["params"] == expected_params

    def test_unknown_metric_raises_value_error(self, fetcher):
        """Test that unknown metric raises ValueError with helpful message."""
        with pytest.raises(ValueError) as excinfo:
            fetcher.fetch_metric("unknown_metric")

        assert "Unknown metric 'unknown_metric'" in str(excinfo.value)
        assert "Available:" in str(excinfo.value)

    def test_empty_response_returns_empty_dataframe(self, fetcher, empty_response_data):
        """Test that empty API response returns empty DataFrame."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = empty_response_data
            mock_get.return_value = mock_response

            df = fetcher.fetch_metric("hash_rate")

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0

    def test_invalid_entries_are_skipped(self, fetcher):
        """Test that invalid entries in response are skipped."""
        invalid_data = {
            "values": [
                {"x": 1704067200, "y": 100.5},  # Valid
                {"x": "invalid", "y": 105.2},  # Invalid timestamp
                {"x": 1704240000, "y": None},  # Invalid value
                {"missing": "keys"},  # Missing keys
                {"x": 1704326400, "y": 120.3},  # Valid
            ]
        }

        with patch.object(fetcher.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = invalid_data
            mock_get.return_value = mock_response

            df = fetcher.fetch_metric("hash_rate")

            assert len(df) == 2  # Only 2 valid entries
            assert df["hash_rate"].iloc[0] == 100.5
            assert df["hash_rate"].iloc[1] == 120.3

    def test_duplicate_timestamps_keep_last(self, fetcher):
        """Test that duplicate timestamps keep the last value."""
        duplicate_data = {
            "values": [
                {"x": 1704067200, "y": 100.5},
                {"x": 1704067200, "y": 200.0},  # Duplicate - should be kept
                {"x": 1704153600, "y": 105.2},
            ]
        }

        with patch.object(fetcher.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = duplicate_data
            mock_get.return_value = mock_response

            df = fetcher.fetch_metric("hash_rate")

            assert len(df) == 2  # Duplicate removed
            # First timestamp should have the last value
            first_date = pd.Timestamp(1704067200, unit="s", tz="UTC")
            assert df.loc[first_date, "hash_rate"] == 200.0

    def test_dataframe_is_sorted_by_date(self, fetcher):
        """Test that returned DataFrame is sorted by date."""
        unsorted_data = {
            "values": [
                {"x": 1704240000, "y": 110.8},  # 2024-01-03
                {"x": 1704067200, "y": 100.5},  # 2024-01-01
                {"x": 1704153600, "y": 105.2},  # 2024-01-02
            ]
        }

        with patch.object(fetcher.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = unsorted_data
            mock_get.return_value = mock_response

            df = fetcher.fetch_metric("hash_rate")

            # Check that dates are in ascending order
            assert df.index[0] < df.index[1] < df.index[2]
            assert df["hash_rate"].iloc[0] == 100.5
            assert df["hash_rate"].iloc[1] == 105.2
            assert df["hash_rate"].iloc[2] == 110.8

    def test_request_exception_is_raised(self, fetcher):
        """Test that request exceptions are properly raised."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")

            with pytest.raises(requests.RequestException):
                fetcher.fetch_metric("hash_rate")

    def test_http_error_is_raised(self, fetcher):
        """Test that HTTP errors are raised via raise_for_status."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = requests.HTTPError("404")
            mock_get.return_value = mock_response

            with pytest.raises(requests.HTTPError):
                fetcher.fetch_metric("hash_rate")


class TestFetchAllMetrics:
    def test_merges_multiple_metrics_correctly(self, fetcher):
        """Test that fetch_all_metrics merges multiple metrics into one DataFrame."""
        hash_rate_data = {
            "values": [
                {"x": 1704067200, "y": 100.5},
                {"x": 1704153600, "y": 105.2},
            ]
        }
        addresses_data = {
            "values": [
                {"x": 1704067200, "y": 50000},
                {"x": 1704153600, "y": 52000},
            ]
        }

        def mock_get_side_effect(url, **kwargs):
            mock_response = MagicMock()
            if "hash-rate" in url:
                mock_response.json.return_value = hash_rate_data
            elif "n-unique-addresses" in url:
                mock_response.json.return_value = addresses_data
            else:
                mock_response.json.return_value = {"values": []}
            return mock_response

        with patch.object(fetcher.session, "get", side_effect=mock_get_side_effect):
            df = fetcher.fetch_all_metrics(
                metrics=["hash_rate", "active_addresses"], timespan="1year"
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "hash_rate" in df.columns
            assert "active_addresses" in df.columns
            assert df["hash_rate"].iloc[0] == 100.5
            assert df["active_addresses"].iloc[0] == 50000

    def test_handles_missing_dates_with_outer_join(self, fetcher):
        """Test that metrics with different dates are joined with outer join."""
        hash_rate_data = {
            "values": [
                {"x": 1704067200, "y": 100.5},  # 2024-01-01
                {"x": 1704153600, "y": 105.2},  # 2024-01-02
            ]
        }
        addresses_data = {
            "values": [
                {"x": 1704153600, "y": 50000},  # 2024-01-02
                {"x": 1704240000, "y": 52000},  # 2024-01-03
            ]
        }

        def mock_get_side_effect(url, **kwargs):
            mock_response = MagicMock()
            if "hash-rate" in url:
                mock_response.json.return_value = hash_rate_data
            elif "n-unique-addresses" in url:
                mock_response.json.return_value = addresses_data
            else:
                mock_response.json.return_value = {"values": []}
            return mock_response

        with patch.object(fetcher.session, "get", side_effect=mock_get_side_effect):
            df = fetcher.fetch_all_metrics(
                metrics=["hash_rate", "active_addresses"], timespan="1year"
            )

            # Should have 3 rows (outer join)
            assert len(df) == 3
            # 2024-01-01: hash_rate exists, active_addresses is NaN
            first_date = pd.Timestamp(1704067200, unit="s", tz="UTC")
            assert df.loc[first_date, "hash_rate"] == 100.5
            assert pd.isna(df.loc[first_date, "active_addresses"])

    def test_defaults_to_all_metrics_when_none_specified(self, fetcher):
        """Test that all metrics are fetched when metrics parameter is None."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"values": [{"x": 1704067200, "y": 100}]}
            mock_get.return_value = mock_response

            df = fetcher.fetch_all_metrics(timespan="1year")

            # Should have called once for each metric in METRIC_ENDPOINTS
            assert mock_get.call_count == len(METRIC_ENDPOINTS)

    def test_continues_on_individual_metric_failure(self, fetcher):
        """Test that fetch_all_metrics continues when individual metrics fail."""

        def mock_get_side_effect(url, **kwargs):
            mock_response = MagicMock()
            if "hash-rate" in url:
                # First metric succeeds
                mock_response.json.return_value = {
                    "values": [{"x": 1704067200, "y": 100.5}]
                }
            elif "n-unique-addresses" in url:
                # Second metric fails
                raise requests.RequestException("Network error")
            elif "n-transactions" in url:
                # Third metric succeeds
                mock_response.json.return_value = {
                    "values": [{"x": 1704067200, "y": 50000}]
                }
            else:
                mock_response.json.return_value = {"values": []}
            return mock_response

        with patch.object(fetcher.session, "get", side_effect=mock_get_side_effect):
            df = fetcher.fetch_all_metrics(
                metrics=["hash_rate", "active_addresses", "transaction_count"]
            )

            # Should have 2 metrics (hash_rate and transaction_count)
            assert len(df.columns) == 2
            assert "hash_rate" in df.columns
            assert "transaction_count" in df.columns
            assert "active_addresses" not in df.columns

    def test_returns_empty_dataframe_when_all_metrics_fail(self, fetcher):
        """Test that empty DataFrame is returned when all metrics fail."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")

            df = fetcher.fetch_all_metrics(metrics=["hash_rate", "active_addresses"])

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0

    def test_returns_empty_dataframe_when_all_responses_empty(self, fetcher):
        """Test that empty DataFrame is returned when all responses are empty."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"values": []}
            mock_get.return_value = mock_response

            df = fetcher.fetch_all_metrics(metrics=["hash_rate", "active_addresses"])

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0


class TestAvailableMetrics:
    def test_returns_correct_list_of_metrics(self, fetcher):
        """Test that available_metrics returns the correct list."""
        metrics = fetcher.available_metrics

        assert isinstance(metrics, list)
        assert len(metrics) == len(METRIC_ENDPOINTS)
        assert "hash_rate" in metrics
        assert "active_addresses" in metrics
        assert "transaction_count" in metrics
        assert "transfer_volume_usd" in metrics
        assert "miner_revenue" in metrics
        assert "total_fees_btc" in metrics
        assert "mempool_size" in metrics
        assert "mempool_count" in metrics


class TestRateLimiting:
    def test_rate_limiting_is_applied(self, fetcher, sample_response_data):
        """Test that rate limiting delay is applied between requests."""
        with patch.object(fetcher.session, "get") as mock_get, patch(
            "sparky.data.onchain_blockchain_com.time.sleep"
        ) as mock_sleep:
            mock_response = MagicMock()
            mock_response.json.return_value = sample_response_data
            mock_get.return_value = mock_response

            # Make two requests
            fetcher.fetch_metric("hash_rate")
            fetcher.fetch_metric("active_addresses")

            # Sleep should have been called once (before second request)
            assert mock_sleep.call_count >= 1

    def test_request_count_increments(self, fetcher, sample_response_data):
        """Test that request count is incremented."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = sample_response_data
            mock_get.return_value = mock_response

            initial_count = fetcher._request_count
            fetcher.fetch_metric("hash_rate")

            assert fetcher._request_count == initial_count + 1


class TestMetricEndpoints:
    def test_all_metric_endpoints_defined(self):
        """Test that METRIC_ENDPOINTS contains expected mappings."""
        expected_metrics = {
            "hash_rate",
            "active_addresses",
            "transaction_count",
            "transfer_volume_usd",
            "miner_revenue",
            "total_fees_btc",
            "mempool_size",
            "mempool_count",
        }

        assert set(METRIC_ENDPOINTS.keys()) == expected_metrics

    def test_metric_endpoint_values_are_strings(self):
        """Test that all endpoint values are non-empty strings."""
        for key, value in METRIC_ENDPOINTS.items():
            assert isinstance(value, str)
            assert len(value) > 0
