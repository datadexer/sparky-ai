"""Tests for CoinGecko market context data fetcher."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from sparky.data.market_context import BASE_URL, CoinGeckoFetcher


@pytest.fixture
def fetcher():
    """CoinGeckoFetcher instance without API key."""
    return CoinGeckoFetcher()


@pytest.fixture
def fetcher_with_key():
    """CoinGeckoFetcher instance with API key."""
    return CoinGeckoFetcher(api_key="test-api-key-123")


@pytest.fixture
def sample_market_data():
    """Sample market data response from CoinGecko API."""
    return [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "current_price": 50000.0,
            "market_cap": 1000000000000,
            "total_volume": 50000000000,
            "circulating_supply": 19000000,
            "fully_diluted_valuation": 1050000000000,
            "price_change_percentage_24h": 2.5,
            "price_change_percentage_7d_in_currency": 5.0,
            "price_change_percentage_30d_in_currency": -3.2,
            "ath": 69000.0,
        },
        {
            "id": "ethereum",
            "symbol": "eth",
            "current_price": 3000.0,
            "market_cap": 360000000000,
            "total_volume": 20000000000,
            "circulating_supply": 120000000,
            "fully_diluted_valuation": None,
            "price_change_percentage_24h": -1.2,
            "price_change_percentage_7d_in_currency": 8.5,
            "price_change_percentage_30d_in_currency": 15.3,
            "ath": 4800.0,
        },
    ]


@pytest.fixture
def sample_historical_data():
    """Sample historical chart data from CoinGecko API."""
    return {
        "prices": [
            [1704067200000, 42000.0],  # 2024-01-01
            [1704153600000, 43000.0],  # 2024-01-02
            [1704240000000, 41500.0],  # 2024-01-03
        ],
        "market_caps": [
            [1704067200000, 820000000000],
            [1704153600000, 840000000000],
            [1704240000000, 810000000000],
        ],
        "total_volumes": [
            [1704067200000, 25000000000],
            [1704153600000, 30000000000],
            [1704240000000, 22000000000],
        ],
    }


class TestCoinGeckoFetcherInit:
    def test_init_without_api_key(self, fetcher):
        assert isinstance(fetcher.session, requests.Session)
        assert "x-cg-demo-key" not in fetcher.session.headers
        assert fetcher._last_request_time == 0.0
        assert fetcher._request_count == 0

    def test_init_with_api_key(self, fetcher_with_key):
        assert fetcher_with_key.session.headers["x-cg-demo-key"] == "test-api-key-123"


class TestFetchMarketData:
    @patch("requests.Session.get")
    def test_fetch_market_data_returns_correct_dataframe(self, mock_get, fetcher, sample_market_data):
        mock_response = Mock()
        mock_response.json.return_value = sample_market_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = fetcher.fetch_market_data(top_n=2)

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.index.name == "coin_id"
        assert list(df.index) == ["bitcoin", "ethereum"]

        # Verify columns exist
        expected_cols = [
            "symbol",
            "current_price",
            "market_cap",
            "total_volume",
            "circulating_supply",
            "fdv",
            "price_change_24h_pct",
            "price_change_7d_pct",
            "price_change_30d_pct",
            "ath_distance_pct",
        ]
        for col in expected_cols:
            assert col in df.columns

        # Verify data values
        assert df.loc["bitcoin", "symbol"] == "BTC"
        assert df.loc["bitcoin", "current_price"] == 50000.0
        assert df.loc["ethereum", "symbol"] == "ETH"
        assert df.loc["ethereum", "current_price"] == 3000.0

    @patch("requests.Session.get")
    def test_ath_distance_calculation(self, mock_get, fetcher, sample_market_data):
        mock_response = Mock()
        mock_response.json.return_value = sample_market_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = fetcher.fetch_market_data()

        # Bitcoin: (50000 - 69000) / 69000 * 100 = -27.536...
        bitcoin_ath_distance = df.loc["bitcoin", "ath_distance_pct"]
        assert bitcoin_ath_distance is not None
        assert abs(bitcoin_ath_distance - (-27.536231884057972)) < 0.001

        # Ethereum: (3000 - 4800) / 4800 * 100 = -37.5
        ethereum_ath_distance = df.loc["ethereum", "ath_distance_pct"]
        assert ethereum_ath_distance is not None
        assert abs(ethereum_ath_distance - (-37.5)) < 0.001

    @patch("requests.Session.get")
    def test_empty_response_returns_empty_dataframe(self, mock_get, fetcher):
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = fetcher.fetch_market_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("requests.Session.get")
    def test_api_request_params(self, mock_get, fetcher, sample_market_data):
        mock_response = Mock()
        mock_response.json.return_value = sample_market_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher.fetch_market_data(top_n=50, vs_currency="eur")

        # Verify API was called with correct parameters
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == f"{BASE_URL}/coins/markets"
        assert call_args[1]["params"]["vs_currency"] == "eur"
        assert call_args[1]["params"]["per_page"] == 50
        assert call_args[1]["params"]["order"] == "market_cap_desc"
        assert call_args[1]["timeout"] == 30

    @patch("requests.Session.get")
    def test_api_key_added_to_headers(self, mock_get, fetcher_with_key, sample_market_data):
        mock_response = Mock()
        mock_response.json.return_value = sample_market_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher_with_key.fetch_market_data()

        assert fetcher_with_key.session.headers["x-cg-demo-key"] == "test-api-key-123"

    @patch("requests.Session.get")
    def test_request_exception_raises(self, mock_get, fetcher):
        mock_get.side_effect = requests.RequestException("Network error")

        with pytest.raises(requests.RequestException):
            fetcher.fetch_market_data()

    @patch("requests.Session.get")
    def test_malformed_coin_data_skipped(self, mock_get, fetcher):
        # Missing required 'id' field
        malformed_data = [
            {
                "symbol": "btc",
                "current_price": 50000.0,
            },
            {
                "id": "ethereum",
                "symbol": "eth",
                "current_price": 3000.0,
                "market_cap": 360000000000,
                "total_volume": 20000000000,
                "circulating_supply": 120000000,
                "ath": 4800.0,
            },
        ]
        mock_response = Mock()
        mock_response.json.return_value = malformed_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = fetcher.fetch_market_data()

        # Only ethereum should be in result
        assert len(df) == 1
        assert "ethereum" in df.index

    @patch("requests.Session.get")
    def test_ath_distance_none_when_ath_zero(self, mock_get, fetcher):
        data_with_zero_ath = [
            {
                "id": "newcoin",
                "symbol": "new",
                "current_price": 1.0,
                "ath": 0,
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = data_with_zero_ath
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = fetcher.fetch_market_data()

        assert pd.isna(df.loc["newcoin", "ath_distance_pct"])


class TestFetchHistoricalMarketChart:
    @patch("requests.Session.get")
    def test_fetch_historical_returns_correct_dataframe(self, mock_get, fetcher, sample_historical_data):
        mock_response = Mock()
        mock_response.json.return_value = sample_historical_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = fetcher.fetch_historical_market_chart("bitcoin", days=3)

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert isinstance(df.index, pd.DatetimeIndex)

        # Verify columns
        assert "price" in df.columns
        assert "market_cap" in df.columns
        assert "volume" in df.columns

        # Verify data values
        assert df["price"].iloc[0] == 42000.0
        assert df["price"].iloc[1] == 43000.0
        assert df["price"].iloc[2] == 41500.0
        assert df["market_cap"].iloc[0] == 820000000000
        assert df["volume"].iloc[0] == 25000000000

    @patch("requests.Session.get")
    def test_historical_timestamps_are_utc(self, mock_get, fetcher, sample_historical_data):
        mock_response = Mock()
        mock_response.json.return_value = sample_historical_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = fetcher.fetch_historical_market_chart("bitcoin")

        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    @patch("requests.Session.get")
    def test_historical_api_request_params(self, mock_get, fetcher, sample_historical_data):
        mock_response = Mock()
        mock_response.json.return_value = sample_historical_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher.fetch_historical_market_chart("ethereum", days=30, vs_currency="eur")

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == f"{BASE_URL}/coins/ethereum/market_chart"
        assert call_args[1]["params"]["vs_currency"] == "eur"
        assert call_args[1]["params"]["days"] == 30
        assert call_args[1]["params"]["interval"] == "daily"
        assert call_args[1]["timeout"] == 30

    @patch("requests.Session.get")
    def test_empty_historical_response_returns_empty_dataframe(self, mock_get, fetcher):
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = fetcher.fetch_historical_market_chart("bitcoin")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("requests.Session.get")
    def test_historical_request_exception_raises(self, mock_get, fetcher):
        mock_get.side_effect = requests.RequestException("API error")

        with pytest.raises(requests.RequestException):
            fetcher.fetch_historical_market_chart("bitcoin")

    @patch("requests.Session.get")
    def test_historical_deduplicates_timestamps(self, mock_get, fetcher):
        # Duplicate timestamps in response
        data_with_duplicates = {
            "prices": [
                [1704067200000, 42000.0],
                [1704067200000, 42500.0],  # Duplicate timestamp
                [1704153600000, 43000.0],
            ],
            "market_caps": [
                [1704067200000, 820000000000],
                [1704067200000, 825000000000],
                [1704153600000, 840000000000],
            ],
            "total_volumes": [
                [1704067200000, 25000000000],
                [1704067200000, 26000000000],
                [1704153600000, 30000000000],
            ],
        }
        mock_response = Mock()
        mock_response.json.return_value = data_with_duplicates
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        df = fetcher.fetch_historical_market_chart("bitcoin")

        # Should keep last duplicate
        assert len(df) == 2
        assert df["price"].iloc[0] == 42500.0


class TestRateLimiting:
    @patch("requests.Session.get")
    @patch("time.sleep")
    @patch("time.time")
    def test_rate_limiting_enforced(self, mock_time, mock_sleep, mock_get, fetcher, sample_market_data):
        mock_response = Mock()
        mock_response.json.return_value = sample_market_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Simulate time progression
        # Need: check elapsed (first), set last_request_time, check elapsed (second), set last_request_time
        time_values = [0.0, 0.5, 0.5, 3.0]
        mock_time.side_effect = time_values

        # First request - _last_request_time is 0.0, so elapsed = 0.0 - 0.0 = 0.0
        # Since elapsed (0.0) < REQUEST_INTERVAL (2.5), it should sleep
        fetcher.fetch_market_data()
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(pytest.approx(2.5, abs=0.01))

        # Reset mock to track second call
        mock_sleep.reset_mock()

        # Second request - _last_request_time is now 0.5, elapsed = 0.5 - 0.5 = 0.0
        # Should sleep for 2.5s
        fetcher.fetch_market_data()
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(pytest.approx(2.5, abs=0.01))

    @patch("requests.Session.get")
    def test_request_count_incremented(self, mock_get, fetcher, sample_market_data):
        mock_response = Mock()
        mock_response.json.return_value = sample_market_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        assert fetcher._request_count == 0

        fetcher.fetch_market_data()
        assert fetcher._request_count == 1

        fetcher.fetch_market_data()
        assert fetcher._request_count == 2

    @patch("requests.Session.get")
    @patch("time.sleep")
    @patch("time.time")
    def test_no_sleep_when_interval_elapsed(self, mock_time, mock_sleep, mock_get, fetcher, sample_market_data):
        mock_response = Mock()
        mock_response.json.return_value = sample_market_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # First request at t=0, second at t=3.0 (> REQUEST_INTERVAL)
        # Need: check elapsed (first), set last_request_time, check elapsed (second), set last_request_time
        mock_time.side_effect = [0.0, 0.5, 3.5, 4.0]

        fetcher.fetch_market_data()
        # First call should sleep since _last_request_time starts at 0
        assert mock_sleep.call_count == 1

        mock_sleep.reset_mock()
        fetcher.fetch_market_data()

        # Second call: elapsed = 3.5 - 0.5 = 3.0, which is >= REQUEST_INTERVAL (2.5)
        # Should not sleep
        assert mock_sleep.call_count == 0
