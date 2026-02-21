"""Tests for CCXT-backed funding rate data fetcher."""

from datetime import timezone
from unittest.mock import Mock, patch

import ccxt
import pandas as pd
import pytest

from sparky.data.funding_rate import FundingRateFetcher, sync_funding_rates


class TestFactoryMethods:
    def test_binance_factory(self):
        with patch.object(FundingRateFetcher, "_create_exchange", return_value=Mock()):
            f = FundingRateFetcher.binance("BTC")
            assert f.exchange_id == "binance"
            assert f.symbol == "BTC/USDT:USDT"
            assert f.granularity == "8h"

    def test_hyperliquid_factory(self):
        with patch.object(FundingRateFetcher, "_create_exchange", return_value=Mock()):
            f = FundingRateFetcher.hyperliquid("BTC")
            assert f.exchange_id == "hyperliquid"
            assert f.symbol == "BTC/USDC:USDC"
            assert f.granularity == "1h"

    def test_coinbase_intl_factory(self):
        with patch.object(FundingRateFetcher, "_create_exchange", return_value=Mock()):
            f = FundingRateFetcher.coinbase_intl("ETH")
            assert f.exchange_id == "coinbaseinternational"
            assert f.symbol == "ETH/USDC:USDC"
            assert f.granularity == "1h"

    def test_binance_default_asset(self):
        with patch.object(FundingRateFetcher, "_create_exchange", return_value=Mock()):
            f = FundingRateFetcher.binance()
            assert f.symbol == "BTC/USDT:USDT"

    def test_binanceusdm_factory(self):
        with patch.object(FundingRateFetcher, "_create_exchange", return_value=Mock()):
            f = FundingRateFetcher.binanceusdm("BTC")
            assert f.exchange_id == "binanceusdm"
            assert f.symbol == "BTC/USDT:USDT"
            assert f.granularity == "8h"

    def test_okx_factory(self):
        with patch.object(FundingRateFetcher, "_create_exchange", return_value=Mock()):
            f = FundingRateFetcher.okx("BTC")
            assert f.exchange_id == "okx"
            assert f.symbol == "BTC/USDT:USDT"
            assert f.granularity == "8h"

    def test_bybit_factory(self):
        with patch.object(FundingRateFetcher, "_create_exchange", return_value=Mock()):
            f = FundingRateFetcher.bybit("ETH")
            assert f.exchange_id == "bybit"
            assert f.symbol == "ETH/USDT:USDT"
            assert f.granularity == "8h"


class TestFetchFundingRates:
    @pytest.fixture
    def fetcher(self):
        mock_exchange = Mock()
        with patch.object(FundingRateFetcher, "_create_exchange", return_value=mock_exchange):
            f = FundingRateFetcher("binance", "BTC/USDT:USDT", "8h")
            yield f

    def test_returns_correct_dataframe(self, fetcher):
        fetcher.exchange.fetch_funding_rate_history.side_effect = [
            [
                {"timestamp": 1704067200000, "fundingRate": 0.0001},
                {"timestamp": 1704096000000, "fundingRate": 0.0002},
            ],
            [],
        ]

        df = fetcher.fetch_funding_rates("2024-01-01", "2024-01-02")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "funding_rate" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz == timezone.utc
        assert df["funding_rate"].iloc[0] == 0.0001
        assert df["funding_rate"].iloc[1] == 0.0002

    def test_pagination_multiple_pages(self, fetcher):
        page1 = [{"timestamp": 1704067200000 + i * 28800000, "fundingRate": 0.0001} for i in range(3)]
        page2 = [{"timestamp": 1704067200000 + (3 + i) * 28800000, "fundingRate": 0.0002} for i in range(2)]

        fetcher.exchange.fetch_funding_rate_history.side_effect = [page1, page2, []]

        df = fetcher.fetch_funding_rates("2024-01-01", "2024-01-03")

        assert len(df) == 5
        assert fetcher.exchange.fetch_funding_rate_history.call_count == 3

    def test_empty_response_returns_empty_dataframe(self, fetcher):
        fetcher.exchange.fetch_funding_rate_history.return_value = []

        df = fetcher.fetch_funding_rates("2024-01-01", "2024-01-02")

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_network_error_handled_gracefully(self, fetcher):
        fetcher.exchange.fetch_funding_rate_history.side_effect = ccxt.NetworkError("timeout")

        df = fetcher.fetch_funding_rates("2024-01-01", "2024-01-02")

        assert df.empty

    def test_exchange_error_handled_gracefully(self, fetcher):
        fetcher.exchange.fetch_funding_rate_history.side_effect = ccxt.ExchangeError("bad request")

        df = fetcher.fetch_funding_rates("2024-01-01", "2024-01-02")

        assert df.empty

    def test_deduplicates_by_timestamp(self, fetcher):
        fetcher.exchange.fetch_funding_rate_history.side_effect = [
            [
                {"timestamp": 1704067200000, "fundingRate": 0.0001},
                {"timestamp": 1704067200000, "fundingRate": 0.0003},
                {"timestamp": 1704096000000, "fundingRate": 0.0002},
            ],
            [],
        ]

        df = fetcher.fetch_funding_rates("2024-01-01", "2024-01-02")

        assert len(df) == 2
        assert df["funding_rate"].iloc[0] == 0.0003  # last value kept

    def test_skips_records_with_missing_fields(self, fetcher):
        fetcher.exchange.fetch_funding_rate_history.side_effect = [
            [
                {"timestamp": 1704067200000, "fundingRate": 0.0001},
                {"timestamp": None, "fundingRate": 0.0002},  # missing ts
                {"timestamp": 1704096000000, "fundingRate": None},  # missing rate
                {"timestamp": 1704124800000, "fundingRate": 0.0003},
            ],
            [],
        ]

        df = fetcher.fetch_funding_rates("2024-01-01", "2024-01-02")

        assert len(df) == 2

    def test_prevents_infinite_loop(self, fetcher):
        """Test pagination stops when last_ts <= since."""
        stuck_records = [{"timestamp": 1704067200000, "fundingRate": 0.0001}]
        fetcher.exchange.fetch_funding_rate_history.return_value = stuck_records

        df = fetcher.fetch_funding_rates("2024-01-01", "2024-01-02")

        # Should not loop forever — called once, then breaks
        assert fetcher.exchange.fetch_funding_rate_history.call_count == 1


class TestOffsetPagination:
    """Tests for offset-based pagination (coinbaseinternational)."""

    @pytest.fixture
    def fetcher(self):
        mock_exchange = Mock()
        with patch.object(FundingRateFetcher, "_create_exchange", return_value=mock_exchange):
            f = FundingRateFetcher("coinbaseinternational", "BTC/USDC:USDC", "1h")
            yield f

    def test_uses_result_offset_param(self, fetcher):
        page1 = [{"timestamp": 1704067200000 + i * 3600000, "fundingRate": 0.0001} for i in range(1000)]
        page2 = [{"timestamp": 1704067200000 + (1000 + i) * 3600000, "fundingRate": 0.0002} for i in range(500)]

        fetcher.exchange.fetch_funding_rate_history.side_effect = [page1, page2, []]

        df = fetcher.fetch_funding_rates("2024-01-01", "2025-01-01")

        calls = fetcher.exchange.fetch_funding_rate_history.call_args_list
        # First call: offset=0
        p0 = calls[0].kwargs.get("params") or calls[0][1].get("params", {})
        assert p0.get("result_offset") == 0
        # Second call: offset=1000
        p1 = calls[1].kwargs.get("params") or calls[1][1].get("params", {})
        assert p1.get("result_offset") == 1000

    def test_filters_by_time_range(self, fetcher):
        start_ts = 1704067200000  # 2024-01-01
        end_ts = 1704153600000  # 2024-01-02
        records = [
            {"timestamp": start_ts - 3600000, "fundingRate": 0.0001},  # before range
            {"timestamp": start_ts, "fundingRate": 0.0002},  # in range
            {"timestamp": start_ts + 3600000, "fundingRate": 0.0003},  # in range
            {"timestamp": end_ts + 3600000, "fundingRate": 0.0004},  # after range
        ]
        fetcher.exchange.fetch_funding_rate_history.side_effect = [records]

        df = fetcher.fetch_funding_rates("2024-01-01", "2024-01-02")

        assert len(df) == 2


class TestValidation:
    @pytest.fixture
    def fetcher(self):
        mock_exchange = Mock()
        with patch.object(FundingRateFetcher, "_create_exchange", return_value=mock_exchange):
            f = FundingRateFetcher("binance", "BTC/USDT:USDT", "8h")
            yield f

    def test_gap_warning_logged_for_8h(self, fetcher, caplog):
        """Test gap detection logs warning for 8h granularity."""
        fetcher.exchange.fetch_funding_rate_history.side_effect = [
            [
                {"timestamp": 1704067200000, "fundingRate": 0.0001},  # 2024-01-01 00:00
                {"timestamp": 1704240000000, "fundingRate": 0.0002},  # 2024-01-03 00:00 (48h gap)
            ],
            [],
        ]

        with caplog.at_level("WARNING"):
            fetcher.fetch_funding_rates("2024-01-01", "2024-01-04")

        assert any("Gap" in record.message for record in caplog.records)

    def test_no_gap_warning_when_within_threshold(self, fetcher, caplog):
        """Test no gap warning for normal 8h intervals."""
        fetcher.exchange.fetch_funding_rate_history.side_effect = [
            [
                {"timestamp": 1704067200000, "fundingRate": 0.0001},  # 00:00
                {"timestamp": 1704096000000, "fundingRate": 0.0002},  # 08:00 (8h gap)
            ],
            [],
        ]

        with caplog.at_level("WARNING"):
            fetcher.fetch_funding_rates("2024-01-01", "2024-01-02")

        gap_warnings = [r for r in caplog.records if "Gap" in r.message]
        assert len(gap_warnings) == 0

    def test_hourly_gap_detection(self):
        """Test gap detection for 1h granularity fetcher."""
        mock_exchange = Mock()
        with patch.object(FundingRateFetcher, "_create_exchange", return_value=mock_exchange):
            f = FundingRateFetcher("hyperliquid", "BTC/USDC:USDC", "1h")

        mock_exchange.fetch_funding_rate_history.side_effect = [
            [
                {"timestamp": 1704067200000, "fundingRate": 0.0001},  # 00:00
                {"timestamp": 1704078000000, "fundingRate": 0.0002},  # 03:00 (3h gap)
            ],
            [],
        ]

        import logging

        with patch.object(logging.getLogger("sparky.data.funding_rate"), "warning") as mock_warn:
            f.fetch_funding_rates("2024-01-01", "2024-01-02")
            assert any("Gap" in str(call) for call in mock_warn.call_args_list)


class TestSyncFundingRates:
    @patch("sparky.data.funding_rate.DataStore")
    @patch("sparky.data.funding_rate.FundingRateFetcher")
    def test_sync_calls_append(self, MockFetcherCls, MockStoreCls):
        mock_df = pd.DataFrame(
            {"funding_rate": [0.0001, 0.0002]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"),
        )

        # Make instances return proper mocks
        mock_instance = Mock()
        mock_instance.fetch_funding_rates.return_value = pd.DataFrame(
            {
                "funding_rate": [0.0001, 0.0002],
                "exchange": ["binance", "binance"],
                "granularity": ["8h", "8h"],
            },
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"),
        )
        mock_instance.granularity = "8h"
        MockFetcherCls.binance.return_value = mock_instance
        MockStoreCls.return_value.get_last_timestamp.return_value = None

        results = sync_funding_rates(asset="BTC", exchanges=["binance"])

        assert results["binance"] == 2
        MockStoreCls.return_value.append.assert_called_once()

    @patch("sparky.data.funding_rate.DataStore")
    @patch("sparky.data.funding_rate.FundingRateFetcher")
    def test_sync_handles_empty_response(self, MockFetcherCls, MockStoreCls):
        mock_instance = Mock()
        mock_instance.fetch_funding_rates.return_value = pd.DataFrame()
        MockFetcherCls.binance.return_value = mock_instance
        MockStoreCls.return_value.get_last_timestamp.return_value = None

        results = sync_funding_rates(asset="BTC", exchanges=["binance"])

        assert results["binance"] == 0
        MockStoreCls.return_value.append.assert_not_called()

    @patch("sparky.data.funding_rate.DataStore")
    @patch("sparky.data.funding_rate.FundingRateFetcher")
    def test_sync_handles_fetch_exception(self, MockFetcherCls, MockStoreCls):
        mock_instance = Mock()
        mock_instance.fetch_funding_rates.side_effect = Exception("connection refused")
        MockFetcherCls.coinbase_intl.return_value = mock_instance
        MockStoreCls.return_value.get_last_timestamp.return_value = None

        results = sync_funding_rates(asset="BTC", exchanges=["coinbase_intl"])

        assert results["coinbase_intl"] == 0

    @patch("sparky.data.funding_rate.DataStore")
    @patch("sparky.data.funding_rate.FundingRateFetcher")
    def test_sync_unknown_exchange_skipped(self, MockFetcherCls, MockStoreCls):
        results = sync_funding_rates(asset="BTC", exchanges=["kraken"])

        assert "kraken" not in results

    @patch("sparky.data.funding_rate.DataStore")
    @patch("sparky.data.funding_rate.FundingRateFetcher")
    def test_sync_incremental_from_last_timestamp(self, MockFetcherCls, MockStoreCls):
        from datetime import datetime as dt

        mock_instance = Mock()
        mock_instance.fetch_funding_rates.return_value = pd.DataFrame()
        MockFetcherCls.binance.return_value = mock_instance
        MockStoreCls.return_value.get_last_timestamp.return_value = dt(2024, 6, 15, tzinfo=timezone.utc)

        sync_funding_rates(asset="BTC", exchanges=["binance"])

        start_arg = mock_instance.fetch_funding_rates.call_args[0][0]
        assert start_arg == "2024-06-15T00:00:00+00:00"
