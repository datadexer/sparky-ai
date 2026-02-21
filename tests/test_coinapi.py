"""Tests for CoinAPI derivatives data fetcher."""

from datetime import datetime as dt, timezone
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sparky.data.coinapi import CoinAPIFetcher, sync_coinapi


class TestCoinAPIFetcherInit:
    def test_key_from_env(self, monkeypatch):
        monkeypatch.setenv("COINAPI_KEY", "test-key-123")  # pragma: allowlist secret
        fetcher = CoinAPIFetcher()
        assert fetcher.api_key == "test-key-123"  # pragma: allowlist secret
        assert fetcher.session.headers["X-CoinAPI-Key"] == "test-key-123"  # pragma: allowlist secret

    def test_key_from_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("COINAPI_KEY", raising=False)
        key_file = tmp_path / "coinapi_key.txt"
        key_file.write_text("file-key-456\n")
        with patch("sparky.data.coinapi.Path", return_value=key_file):
            fetcher = CoinAPIFetcher()
        assert fetcher.api_key == "file-key-456"  # pragma: allowlist secret

    def test_key_from_arg(self):
        fetcher = CoinAPIFetcher(api_key="direct-key")
        assert fetcher.api_key == "direct-key"  # pragma: allowlist secret

    def test_missing_key_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("COINAPI_KEY", raising=False)
        key_file = tmp_path / "coinapi_key.txt"  # does not exist
        with patch("sparky.data.coinapi.Path", return_value=key_file):
            with pytest.raises(ValueError, match="CoinAPI key required"):
                CoinAPIFetcher()


class TestDiscoverSymbols:
    def test_returns_list(self):
        fetcher = CoinAPIFetcher(api_key="test")
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"symbol_id": "BINANCE_PERP_BTC_USDT", "metric_id": "FUNDING_RATE"},
            {"symbol_id": "BINANCE_PERP_ETH_USDT", "metric_id": "FUNDING_RATE"},
        ]
        fetcher.session.get = Mock(return_value=mock_resp)

        result = fetcher.discover_symbols("BINANCE", "FUNDING_RATE")
        assert len(result) == 2
        assert result[0]["symbol_id"] == "BINANCE_PERP_BTC_USDT"

    def test_404_returns_empty(self):
        fetcher = CoinAPIFetcher(api_key="test")
        mock_resp = Mock()
        mock_resp.status_code = 404
        mock_resp.json.return_value = []
        fetcher.session.get = Mock(return_value=mock_resp)

        result = fetcher.discover_symbols("NONEXIST", "FUNDING_RATE")
        assert result == []


class TestFetchMetricHistory:
    def test_correct_dataframe(self):
        fetcher = CoinAPIFetcher(api_key="test")
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"time_period_start": "2024-01-01T00:00:00Z", "value_decimal": "0.0001"},
            {"time_period_start": "2024-01-01T08:00:00Z", "value_decimal": "0.0002"},
            {"time_period_start": "2024-01-01T16:00:00Z", "value_decimal": "-0.0001"},
        ]
        fetcher.session.get = Mock(return_value=mock_resp)

        df = fetcher.fetch_metric_history("FUNDING_RATE", "BINANCE_PERP_BTC_USDT", "2024-01-01")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"
        assert "value" in df.columns
        assert df["value"].iloc[0] == 0.0001

    def test_dedup_by_index(self):
        fetcher = CoinAPIFetcher(api_key="test")
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"time_period_start": "2024-01-01T00:00:00Z", "value_decimal": "0.0001"},
            {"time_period_start": "2024-01-01T00:00:00Z", "value_decimal": "0.0003"},
        ]
        fetcher.session.get = Mock(return_value=mock_resp)

        df = fetcher.fetch_metric_history("FUNDING_RATE", "SYM", "2024-01-01")
        assert len(df) == 1
        assert df["value"].iloc[0] == 0.0003

    def test_empty_response(self):
        fetcher = CoinAPIFetcher(api_key="test")
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        fetcher.session.get = Mock(return_value=mock_resp)

        df = fetcher.fetch_metric_history("FUNDING_RATE", "SYM", "2024-01-01")
        assert df.empty

    def test_credit_tracking(self):
        fetcher = CoinAPIFetcher(api_key="test")
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"time_period_start": f"2024-01-01T{i:02d}:00:00Z", "value_decimal": "0.0001"} for i in range(24)
        ]
        fetcher.session.get = Mock(return_value=mock_resp)

        fetcher.fetch_metric_history("FUNDING_RATE", "SYM", "2024-01-01")
        assert fetcher.credits_used > 0

    def test_time_end_param_passed(self):
        fetcher = CoinAPIFetcher(api_key="test")
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        fetcher.session.get = Mock(return_value=mock_resp)

        fetcher.fetch_metric_history("FUNDING_RATE", "SYM", "2024-01-01", time_end="2024-02-01")
        call_kwargs = fetcher.session.get.call_args
        params = call_kwargs[1].get("params") or call_kwargs.kwargs.get("params", {})
        assert params["time_end"] == "2024-02-01"

    def test_skips_invalid_records(self):
        fetcher = CoinAPIFetcher(api_key="test")
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"time_period_start": "2024-01-01T00:00:00Z", "value_decimal": "0.0001"},
            {"time_period_start": "2024-01-01T08:00:00Z", "value_decimal": "not_a_number"},
            {"time_period_start": None, "value_decimal": "0.0002"},
            {"time_period_start": "2024-01-01T16:00:00Z", "value_decimal": "0.0003"},
        ]
        fetcher.session.get = Mock(return_value=mock_resp)

        df = fetcher.fetch_metric_history("FUNDING_RATE", "SYM", "2024-01-01")
        assert len(df) == 2


class TestErrorHandling:
    def test_429_retry(self):
        fetcher = CoinAPIFetcher(api_key="test")
        resp_429 = Mock()
        resp_429.status_code = 429
        resp_200 = Mock()
        resp_200.status_code = 200
        resp_200.json.return_value = [
            {"time_period_start": "2024-01-01T00:00:00Z", "value_decimal": "0.0001"},
        ]
        fetcher.session.get = Mock(side_effect=[resp_429, resp_200])

        with patch("time.sleep"):
            df = fetcher.fetch_metric_history("FUNDING_RATE", "SYM", "2024-01-01")
        assert len(df) == 1

    def test_402_raises(self):
        fetcher = CoinAPIFetcher(api_key="test")
        mock_resp = Mock()
        mock_resp.status_code = 402
        fetcher.session.get = Mock(return_value=mock_resp)

        with pytest.raises(RuntimeError, match="credits exhausted"):
            fetcher.fetch_metric_history("FUNDING_RATE", "SYM", "2024-01-01")

    def test_404_returns_empty(self):
        fetcher = CoinAPIFetcher(api_key="test")
        mock_resp = Mock()
        mock_resp.status_code = 404
        mock_resp.json.return_value = []
        fetcher.session.get = Mock(return_value=mock_resp)

        df = fetcher.fetch_metric_history("FUNDING_RATE", "SYM", "2024-01-01")
        assert df.empty

    def test_generic_4xx_returns_empty(self):
        fetcher = CoinAPIFetcher(api_key="test")
        mock_resp = Mock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = []
        fetcher.session.get = Mock(return_value=mock_resp)

        df = fetcher.fetch_metric_history("FUNDING_RATE", "SYM", "2024-01-01")
        assert df.empty


class TestSyncCoinAPI:
    @patch("sparky.data.coinapi.DataStore")
    @patch("sparky.data.coinapi.CoinAPIFetcher")
    def test_saves_to_correct_path(self, MockFetcher, MockStore):
        mock_df = pd.DataFrame(
            {"value": [0.0001, 0.0002]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"),
        )
        mock_instance = MockFetcher.return_value
        mock_instance.fetch_metric_history.return_value = mock_df
        mock_instance.credits_used = 0.01
        MockStore.return_value.get_last_timestamp.return_value = None

        targets = [
            {
                "metric_id": "FUNDING_RATE",
                "symbol_id": "BINANCE_PERP_BTC_USDT",
                "asset": "btc",
                "parquet_name": "funding_rate_btc_binance",
                "column_name": "funding_rate",
            }
        ]

        results = sync_coinapi(targets)
        assert results["funding_rate_btc_binance"] == 2
        MockStore.return_value.append.assert_called_once()
        call_args = MockStore.return_value.append.call_args
        assert "funding_rate_btc_binance" in str(call_args)

    @patch("sparky.data.coinapi.DataStore")
    @patch("sparky.data.coinapi.CoinAPIFetcher")
    def test_incremental_fetch(self, MockFetcher, MockStore):
        MockStore.return_value.get_last_timestamp.return_value = dt(2024, 6, 1, tzinfo=timezone.utc)
        mock_instance = MockFetcher.return_value
        mock_instance.fetch_metric_history.return_value = pd.DataFrame()
        mock_instance.credits_used = 0.0

        targets = [
            {
                "metric_id": "OPEN_INTEREST",
                "symbol_id": "SYM",
                "asset": "btc",
                "parquet_name": "oi_btc",
            }
        ]

        sync_coinapi(targets)
        start_arg = mock_instance.fetch_metric_history.call_args[0][2]
        assert "2024-06-01" in start_arg

    @patch("sparky.data.coinapi.DataStore")
    @patch("sparky.data.coinapi.CoinAPIFetcher")
    def test_handles_failure_gracefully(self, MockFetcher, MockStore):
        mock_instance = MockFetcher.return_value
        mock_instance.fetch_metric_history.side_effect = Exception("network error")
        mock_instance.credits_used = 0.0
        MockStore.return_value.get_last_timestamp.return_value = None

        targets = [
            {
                "metric_id": "FUNDING_RATE",
                "symbol_id": "SYM",
                "asset": "btc",
                "parquet_name": "test",
            }
        ]

        results = sync_coinapi(targets)
        assert results["test"] == 0

    @patch("sparky.data.coinapi.DataStore")
    @patch("sparky.data.coinapi.CoinAPIFetcher")
    def test_runtime_error_breaks_loop(self, MockFetcher, MockStore):
        mock_instance = MockFetcher.return_value
        mock_instance.fetch_metric_history.side_effect = RuntimeError("CoinAPI credits exhausted (402)")
        mock_instance.credits_used = 0.0
        MockStore.return_value.get_last_timestamp.return_value = None

        targets = [
            {"metric_id": "FUNDING_RATE", "symbol_id": "SYM1", "asset": "btc", "parquet_name": "t1"},
            {"metric_id": "FUNDING_RATE", "symbol_id": "SYM2", "asset": "btc", "parquet_name": "t2"},
        ]

        results = sync_coinapi(targets)
        # RuntimeError breaks the loop, so second target is never attempted
        mock_instance.fetch_metric_history.assert_called_once()
        assert "t2" not in results

    @patch("sparky.data.coinapi.DataStore")
    @patch("sparky.data.coinapi.CoinAPIFetcher")
    def test_empty_fetch_skips_append(self, MockFetcher, MockStore):
        mock_instance = MockFetcher.return_value
        mock_instance.fetch_metric_history.return_value = pd.DataFrame()
        mock_instance.credits_used = 0.0
        MockStore.return_value.get_last_timestamp.return_value = None

        targets = [
            {
                "metric_id": "FUNDING_RATE",
                "symbol_id": "SYM",
                "asset": "btc",
                "parquet_name": "empty_test",
            }
        ]

        results = sync_coinapi(targets)
        assert results["empty_test"] == 0
        MockStore.return_value.append.assert_not_called()

    @patch("sparky.data.coinapi.DataStore")
    @patch("sparky.data.coinapi.CoinAPIFetcher")
    def test_column_rename(self, MockFetcher, MockStore):
        mock_df = pd.DataFrame(
            {"value": [0.0001]},
            index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
        )
        mock_instance = MockFetcher.return_value
        mock_instance.fetch_metric_history.return_value = mock_df
        mock_instance.credits_used = 0.0
        MockStore.return_value.get_last_timestamp.return_value = None

        targets = [
            {
                "metric_id": "FUNDING_RATE",
                "symbol_id": "SYM",
                "asset": "btc",
                "parquet_name": "fr_test",
                "column_name": "funding_rate",
            }
        ]

        sync_coinapi(targets)
        appended_df = MockStore.return_value.append.call_args[0][0]
        assert "funding_rate" in appended_df.columns
        assert "value" not in appended_df.columns
