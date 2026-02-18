"""Tests for CCXT price fetcher."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import ccxt
import pandas as pd
import pytest

from sparky.data.price import MAX_CANDLES_PER_REQUEST, MS_PER_DAY, CCXTPriceFetcher


@pytest.fixture
def mock_exchange():
    """Mock CCXT exchange instance."""
    exchange = Mock(spec=ccxt.Exchange)
    exchange.fetch_ohlcv = Mock()
    return exchange


@pytest.fixture
def fetcher(mock_exchange):
    """CCXTPriceFetcher with mocked exchange."""
    with patch("sparky.data.price.ccxt.binance") as mock_binance_class:
        mock_binance_class.return_value = mock_exchange
        fetcher = CCXTPriceFetcher()
        return fetcher


def create_candles(start_ts: int, count: int) -> list:
    """Helper to generate mock OHLCV candles.

    Returns list of [timestamp, open, high, low, close, volume].
    """
    candles = []
    for i in range(count):
        ts = start_ts + (i * MS_PER_DAY)
        # Generate realistic OHLCV data
        base_price = 100.0 + i
        candles.append(
            [
                ts,
                base_price,  # open
                base_price + 5.0,  # high
                base_price - 3.0,  # low
                base_price + 2.0,  # close
                1000.0 + (i * 10),  # volume
            ]
        )
    return candles


class TestBasicFetch:
    """Test basic fetch functionality."""

    def test_fetch_returns_correct_structure(self, fetcher, mock_exchange):
        """Fetch should return DataFrame with correct columns and DatetimeIndex."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candles = create_candles(start_ts, 10)
        mock_exchange.fetch_ohlcv.return_value = candles

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-11")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz == timezone.utc

    def test_fetch_calls_exchange_with_correct_params(self, fetcher, mock_exchange):
        """Fetch should call exchange with correct symbol and timeframe."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        mock_exchange.fetch_ohlcv.return_value = create_candles(start_ts, 5)

        fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-06")

        mock_exchange.fetch_ohlcv.assert_called_once()
        args = mock_exchange.fetch_ohlcv.call_args
        assert args[0][0] == "BTC/USDT"
        assert args[0][1] == "1d"
        assert args[1]["since"] == start_ts
        assert args[1]["limit"] == MAX_CANDLES_PER_REQUEST

    def test_fetch_without_end_date_uses_now(self, fetcher, mock_exchange):
        """Fetch without end_date should use current time."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        mock_exchange.fetch_ohlcv.return_value = create_candles(start_ts, 5)

        with patch("sparky.data.price.datetime") as mock_datetime:
            mock_now = datetime(2024, 2, 1, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_now
            mock_datetime.strptime = datetime.strptime

            fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01")

            # Should still call exchange (end_ts is validated in _paginated_fetch)
            assert mock_exchange.fetch_ohlcv.called

    def test_empty_response_returns_empty_dataframe(self, fetcher, mock_exchange):
        """Empty exchange response should return empty DataFrame."""
        mock_exchange.fetch_ohlcv.return_value = []

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-10")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestPagination:
    """Test pagination logic."""

    def test_pagination_multiple_requests(self, fetcher, mock_exchange):
        """Should make multiple requests when data exceeds one page."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        # Need end_ts far enough in future to accommodate 1500 days
        end_ts = int(datetime(2028, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)

        # First page: 1000 candles
        page1 = create_candles(start_ts, 1000)
        # Second page: 500 more candles (starting one day after last candle from page1)
        page2_start = start_ts + (999 * MS_PER_DAY)  # Last candle in page1 is at index 999
        page2 = create_candles(page2_start, 501)  # Include overlap at index 0, then 500 new

        # Third call returns empty (no more data)
        mock_exchange.fetch_ohlcv.side_effect = [page1, page2, []]

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2028-03-01")

        # Should have 1500 unique timestamps (1000 from page1 + 500 new from page2, dedup removes 1)
        assert len(df) == 1500
        assert mock_exchange.fetch_ohlcv.call_count == 3

        # Verify second call used correct since timestamp
        second_call_args = mock_exchange.fetch_ohlcv.call_args_list[1]
        expected_since = page1[-1][0] + MS_PER_DAY
        assert second_call_args[1]["since"] == expected_since

    def test_pagination_stops_on_empty_response(self, fetcher, mock_exchange):
        """Pagination should stop when exchange returns empty list."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

        page1 = create_candles(start_ts, 100)
        mock_exchange.fetch_ohlcv.side_effect = [page1, []]

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-12-31")

        assert len(df) == 100
        assert mock_exchange.fetch_ohlcv.call_count == 2

    def test_pagination_prevents_infinite_loop(self, fetcher, mock_exchange):
        """Should break if exchange returns same timestamp repeatedly."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

        # Return same candle repeatedly (edge case)
        same_candle = create_candles(start_ts, 1)
        mock_exchange.fetch_ohlcv.return_value = same_candle

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-12-31")

        # Should break after first call due to non-advancing timestamp
        assert mock_exchange.fetch_ohlcv.call_count == 1


class TestFailover:
    """Test exchange failover logic."""

    def test_failover_on_network_error(self, fetcher):
        """Should try failover exchanges on NetworkError."""
        with patch("sparky.data.price.ccxt") as mock_ccxt:
            # Create proper exception instances
            network_error = ccxt.NetworkError("Connection failed")

            # Setup: binance fails, bybit succeeds
            mock_binance = Mock()
            mock_binance.fetch_ohlcv.side_effect = network_error

            mock_bybit = Mock()
            start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
            mock_bybit.fetch_ohlcv.return_value = create_candles(start_ts, 10)

            mock_ccxt.binance.return_value = mock_binance
            mock_ccxt.bybit.return_value = mock_bybit
            # Mock the exception classes themselves
            mock_ccxt.NetworkError = ccxt.NetworkError
            mock_ccxt.ExchangeError = ccxt.ExchangeError

            fetcher = CCXTPriceFetcher()
            df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-11")

            assert len(df) == 10
            assert mock_binance.fetch_ohlcv.called
            assert mock_bybit.fetch_ohlcv.called

    def test_failover_on_exchange_error(self, fetcher):
        """Should try failover exchanges on ExchangeError."""
        with patch("sparky.data.price.ccxt") as mock_ccxt:
            # Setup: binance and bybit fail, okx succeeds
            mock_binance = Mock()
            mock_binance.fetch_ohlcv.side_effect = ccxt.ExchangeError("Rate limit")

            mock_bybit = Mock()
            mock_bybit.fetch_ohlcv.side_effect = ccxt.NetworkError("Timeout")

            mock_okx = Mock()
            start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
            mock_okx.fetch_ohlcv.return_value = create_candles(start_ts, 10)

            mock_ccxt.binance.return_value = mock_binance
            mock_ccxt.bybit.return_value = mock_bybit
            mock_ccxt.okx.return_value = mock_okx
            # Mock the exception classes themselves
            mock_ccxt.NetworkError = ccxt.NetworkError
            mock_ccxt.ExchangeError = ccxt.ExchangeError

            fetcher = CCXTPriceFetcher()
            df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-11")

            assert len(df) == 10
            assert mock_okx.fetch_ohlcv.called

    def test_all_exchanges_fail_returns_empty(self, fetcher):
        """Should return empty DataFrame if all exchanges fail."""
        with patch("sparky.data.price.ccxt") as mock_ccxt:
            # All exchanges fail
            for exchange_id in ["binance", "bybit", "okx", "coinbase"]:
                mock_exchange = Mock()
                mock_exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("Failed")
                setattr(mock_ccxt, exchange_id, Mock(return_value=mock_exchange))

            # Mock the exception classes themselves
            mock_ccxt.NetworkError = ccxt.NetworkError
            mock_ccxt.ExchangeError = ccxt.ExchangeError

            fetcher = CCXTPriceFetcher()
            df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-11")

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0

    def test_primary_exchange_works_no_failover(self, fetcher, mock_exchange):
        """Should not try failover if primary exchange succeeds."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        mock_exchange.fetch_ohlcv.return_value = create_candles(start_ts, 10)

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-11")

        assert len(df) == 10
        # Only primary exchange should be called
        assert mock_exchange.fetch_ohlcv.called


class TestValidation:
    """Test data validation logic."""

    def test_removes_negative_prices(self, fetcher, mock_exchange):
        """Should filter out candles with negative prices."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candles = create_candles(start_ts, 10)

        # Inject negative prices
        candles[2][1] = -10.0  # negative open
        candles[5][4] = -5.0  # negative close

        mock_exchange.fetch_ohlcv.return_value = candles

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-11")

        # Should have 8 rows (10 - 2 invalid)
        assert len(df) == 8

    def test_removes_zero_prices(self, fetcher, mock_exchange):
        """Should filter out candles with zero prices."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candles = create_candles(start_ts, 10)

        # Inject zero prices
        candles[1][2] = 0.0  # zero high
        candles[7][3] = 0.0  # zero low

        mock_exchange.fetch_ohlcv.return_value = candles

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-11")

        # Should have 8 rows (10 - 2 invalid)
        assert len(df) == 8

    def test_removes_negative_volume(self, fetcher, mock_exchange):
        """Should filter out candles with negative volume."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candles = create_candles(start_ts, 10)

        # Inject negative volume
        candles[3][5] = -100.0
        candles[6][5] = -50.0

        mock_exchange.fetch_ohlcv.return_value = candles

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-11")

        # Should have 8 rows (10 - 2 invalid)
        assert len(df) == 8
        # All remaining volume should be non-negative
        assert (df["volume"] >= 0).all()

    def test_allows_zero_volume(self, fetcher, mock_exchange):
        """Zero volume is valid and should be kept."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candles = create_candles(start_ts, 5)

        # Set some volumes to zero
        candles[1][5] = 0.0
        candles[3][5] = 0.0

        mock_exchange.fetch_ohlcv.return_value = candles

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-06")

        # All 5 rows should be present
        assert len(df) == 5
        assert (df["volume"] >= 0).all()


class TestDeduplication:
    """Test duplicate timestamp handling."""

    def test_deduplicates_timestamps_keeps_last(self, fetcher, mock_exchange):
        """Should deduplicate timestamps, keeping last occurrence."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candles = create_candles(start_ts, 5)

        # Add duplicate timestamp with different values
        duplicate_ts = start_ts + (2 * MS_PER_DAY)
        candles.append(
            [
                duplicate_ts,
                999.0,  # Different open
                1000.0,
                998.0,
                999.5,
                5000.0,  # Different volume
            ]
        )

        mock_exchange.fetch_ohlcv.return_value = candles

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-06")

        # Should have 5 unique timestamps
        assert len(df) == 5
        assert not df.index.duplicated().any()

        # Should keep last occurrence
        dup_date = pd.Timestamp(datetime(2024, 1, 3, tzinfo=timezone.utc))
        assert df.loc[dup_date, "open"] == 999.0
        assert df.loc[dup_date, "volume"] == 5000.0

    def test_handles_multiple_duplicates(self, fetcher, mock_exchange):
        """Should handle multiple duplicate timestamps."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candles = create_candles(start_ts, 10)

        # Add several duplicates
        candles.append(candles[0].copy())  # Duplicate first
        candles.append(candles[5].copy())  # Duplicate middle
        candles.append(candles[9].copy())  # Duplicate last

        mock_exchange.fetch_ohlcv.return_value = candles

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-11")

        # Should still have 10 unique timestamps
        assert len(df) == 10
        assert not df.index.duplicated().any()


class TestDateRange:
    """Test date range handling."""

    def test_date_range_boundaries(self, fetcher, mock_exchange):
        """Should fetch data within specified date range."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 11, tzinfo=timezone.utc).timestamp() * 1000)

        candles = create_candles(start_ts, 10)
        mock_exchange.fetch_ohlcv.return_value = candles

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-11")

        # Check date range
        assert df.index.min() == pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert df.index.max() == pd.Timestamp(datetime(2024, 1, 10, tzinfo=timezone.utc))

    def test_sorted_by_date(self, fetcher, mock_exchange):
        """DataFrame should be sorted by timestamp."""
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        candles = create_candles(start_ts, 10)

        # Shuffle candles
        import random

        random.shuffle(candles)

        mock_exchange.fetch_ohlcv.return_value = candles

        df = fetcher.fetch_daily_ohlcv("BTC/USDT", "2024-01-01", "2024-01-11")

        # Should be sorted
        assert df.index.is_monotonic_increasing


class TestExchangeCreation:
    """Test exchange instance creation."""

    def test_creates_exchange_with_rate_limiting(self):
        """Should create exchange with rate limiting enabled."""
        with patch("sparky.data.price.ccxt.binance") as mock_binance_class:
            mock_exchange = Mock()
            mock_binance_class.return_value = mock_exchange

            fetcher = CCXTPriceFetcher()

            mock_binance_class.assert_called_once_with({"enableRateLimit": True})

    def test_custom_exchange_id(self):
        """Should support custom exchange ID."""
        with patch("sparky.data.price.ccxt.okx") as mock_okx_class:
            mock_exchange = Mock()
            mock_okx_class.return_value = mock_exchange

            fetcher = CCXTPriceFetcher(exchange_id="okx")

            assert fetcher.exchange_id == "okx"
            mock_okx_class.assert_called_once()
