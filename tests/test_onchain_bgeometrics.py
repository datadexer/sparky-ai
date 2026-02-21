"""Tests for BGeometrics on-chain data fetcher."""

from datetime import timezone
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from sparky.data.onchain_bgeometrics import (
    BASE_URL,
    METRIC_ENDPOINTS,
    BGeometricsFetcher,
    sync_bgeometrics,
)


@pytest.fixture
def fetcher():
    """BGeometricsFetcher without token."""
    return BGeometricsFetcher()


@pytest.fixture
def fetcher_with_token():
    """BGeometricsFetcher with token."""
    return BGeometricsFetcher(token="test_token_123")


@pytest.fixture
def mock_sopr_response():
    """Mock API response for SOPR metric (single page)."""
    return {
        "content": [
            {"d": "2024-01-01", "sopr": "1.05"},
            {"d": "2024-01-02", "sopr": "0.98"},
            {"d": "2024-01-03", "sopr": "1.12"},
        ],
        "last": True,
    }


@pytest.fixture
def mock_paginated_response_page1():
    """Mock API response - first page."""
    return {
        "content": [
            {"d": "2024-01-01", "sopr": "1.05"},
            {"d": "2024-01-02", "sopr": "0.98"},
        ],
        "last": False,
    }


@pytest.fixture
def mock_paginated_response_page2():
    """Mock API response - second page (last)."""
    return {
        "content": [
            {"d": "2024-01-03", "sopr": "1.12"},
            {"d": "2024-01-04", "sopr": "1.01"},
        ],
        "last": True,
    }


@pytest.fixture
def mock_mvrv_response():
    """Mock API response for MVRV metric."""
    return {
        "content": [
            {"d": "2024-01-01", "mvrv": "1.5"},
            {"d": "2024-01-02", "mvrv": "1.6"},
        ],
        "last": True,
    }


@pytest.fixture
def mock_nupl_response():
    """Mock API response for NUPL metric."""
    return {
        "content": [
            {"d": "2024-01-01", "nupl": "0.45"},
            {"d": "2024-01-02", "nupl": "0.48"},
        ],
        "last": True,
    }


class TestFetchMetric:
    def test_fetch_metric_returns_correct_dataframe(self, fetcher, mock_sopr_response):
        """Test fetch_metric returns DataFrame with correct structure."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = mock_sopr_response
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("sopr", "2024-01-01", "2024-01-03")

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert "sopr" in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)
            assert df.index.tz == timezone.utc
            assert df["sopr"].iloc[0] == 1.05
            assert df["sopr"].iloc[1] == 0.98
            assert df["sopr"].iloc[2] == 1.12

    def test_fetch_metric_makes_correct_api_call(self, fetcher, mock_sopr_response):
        """Test fetch_metric makes API call with correct parameters."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = mock_sopr_response
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            fetcher.fetch_metric("sopr", "2024-01-01", "2024-12-31")

            expected_url = f"{BASE_URL}/v1/sopr"
            expected_params = {
                "startday": "2024-01-01",
                "endday": "2024-12-31",
                "page": 0,
                "size": 5000,
            }
            mock_get.assert_called_once_with(expected_url, params=expected_params, timeout=30)

    def test_fetch_metric_defaults_to_today_when_no_end_date(self, fetcher, mock_sopr_response):
        """Test fetch_metric uses current date as default end_date."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = mock_sopr_response
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            with patch("sparky.data.onchain_bgeometrics.datetime") as mock_dt:
                mock_now = Mock()
                mock_now.strftime.return_value = "2026-02-15"
                mock_dt.now.return_value = mock_now

                fetcher.fetch_metric("sopr", "2024-01-01")

                call_args = mock_get.call_args
                assert call_args[1]["params"]["endday"] == "2026-02-15"

    def test_fetch_metric_unknown_metric_raises_error(self, fetcher):
        """Test fetch_metric raises ValueError for unknown metric."""
        with pytest.raises(ValueError) as exc_info:
            fetcher.fetch_metric("unknown_metric", "2024-01-01")

        assert "Unknown metric 'unknown_metric'" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_fetch_metric_empty_response_returns_empty_dataframe(self, fetcher):
        """Test fetch_metric returns empty DataFrame when API returns no data."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {"content": [], "last": True}
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("sopr", "2024-01-01", "2024-01-03")

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
            assert df.empty

    def test_fetch_metric_handles_pagination(
        self,
        fetcher,
        mock_paginated_response_page1,
        mock_paginated_response_page2,
    ):
        """Test fetch_metric correctly handles paginated responses."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp1 = Mock()
            mock_resp1.json.return_value = mock_paginated_response_page1
            mock_resp1.raise_for_status = Mock()

            mock_resp2 = Mock()
            mock_resp2.json.return_value = mock_paginated_response_page2
            mock_resp2.raise_for_status = Mock()

            mock_get.side_effect = [mock_resp1, mock_resp2]

            df = fetcher.fetch_metric("sopr", "2024-01-01", "2024-01-04")

            assert len(df) == 4
            assert df["sopr"].iloc[0] == 1.05
            assert df["sopr"].iloc[1] == 0.98
            assert df["sopr"].iloc[2] == 1.12
            assert df["sopr"].iloc[3] == 1.01
            assert mock_get.call_count == 2

            # Verify pagination parameters
            first_call = mock_get.call_args_list[0]
            second_call = mock_get.call_args_list[1]
            assert first_call[1]["params"]["page"] == 0
            assert second_call[1]["params"]["page"] == 1

    def test_fetch_metric_skips_invalid_values(self, fetcher):
        """Test fetch_metric skips records with invalid/missing values."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "content": [
                    {"d": "2024-01-01", "sopr": "1.05"},
                    {"d": "2024-01-02", "sopr": "invalid"},  # Invalid float
                    {"d": "2024-01-03", "sopr": None},  # None value
                    {"d": None, "sopr": "1.12"},  # Missing date
                    {"d": "2024-01-05", "sopr": "0.98"},
                ],
                "last": True,
            }
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("sopr", "2024-01-01", "2024-01-05")

            # Only valid records should be included
            assert len(df) == 2
            assert df.loc[pd.Timestamp("2024-01-01", tz="UTC"), "sopr"] == pytest.approx(1.05)
            assert df.loc[pd.Timestamp("2024-01-05", tz="UTC"), "sopr"] == pytest.approx(0.98)

    def test_fetch_metric_removes_duplicate_dates(self, fetcher):
        """Test fetch_metric removes duplicate dates, keeping last value."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "content": [
                    {"d": "2024-01-01", "sopr": "1.05"},
                    {"d": "2024-01-02", "sopr": "0.98"},
                    {"d": "2024-01-01", "sopr": "1.99"},  # Duplicate, should be kept
                ],
                "last": True,
            }
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("sopr", "2024-01-01", "2024-01-02")

            assert len(df) == 2
            # Should keep the last value for 2024-01-01
            assert df.loc["2024-01-01", "sopr"] == 1.99

    def test_fetch_metric_sorts_by_date(self, fetcher):
        """Test fetch_metric returns sorted DataFrame."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "content": [
                    {"d": "2024-01-03", "sopr": "1.12"},
                    {"d": "2024-01-01", "sopr": "1.05"},
                    {"d": "2024-01-02", "sopr": "0.98"},
                ],
                "last": True,
            }
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("sopr", "2024-01-01", "2024-01-03")

            assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
            assert df.index[1] == pd.Timestamp("2024-01-02", tz="UTC")
            assert df.index[2] == pd.Timestamp("2024-01-03", tz="UTC")


class TestFetchAllMetrics:
    def test_fetch_all_metrics_merges_multiple_metrics(self, fetcher, mock_sopr_response, mock_mvrv_response):
        """Test fetch_all_metrics merges multiple metrics correctly."""
        with patch.object(fetcher.session, "get") as mock_get:
            # Set up responses for two metrics
            mock_resp1 = Mock()
            mock_resp1.json.return_value = mock_sopr_response
            mock_resp1.raise_for_status = Mock()

            mock_resp2 = Mock()
            mock_resp2.json.return_value = mock_mvrv_response
            mock_resp2.raise_for_status = Mock()

            mock_get.side_effect = [mock_resp1, mock_resp2]

            df = fetcher.fetch_all_metrics("2024-01-01", "2024-01-03", metrics=["sopr", "mvrv_zscore"])

            assert len(df.columns) == 2
            assert "sopr" in df.columns
            assert "mvrv_zscore" in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)

    def test_fetch_all_metrics_defaults_to_all_metrics(self, fetcher):
        """Test fetch_all_metrics uses all available metrics by default."""
        with patch.object(fetcher, "fetch_metric") as mock_fetch:
            # Return DataFrame with unique column name based on metric name
            def mock_return(metric_name, start_date, end_date):
                return pd.DataFrame(
                    {metric_name: [1.0]},
                    index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
                )

            mock_fetch.side_effect = mock_return

            fetcher.fetch_all_metrics("2024-01-01", "2024-01-03")

            # Should be called once for each metric in METRIC_ENDPOINTS
            assert mock_fetch.call_count == len(METRIC_ENDPOINTS)

    def test_fetch_all_metrics_handles_outer_join(self, fetcher, mock_sopr_response, mock_nupl_response):
        """Test fetch_all_metrics performs outer join for metrics with different dates."""
        with patch.object(fetcher.session, "get") as mock_get:
            # SOPR has 3 days, NUPL has 2 days (different dates)
            sopr_response = {
                "content": [
                    {"d": "2024-01-01", "sopr": "1.05"},
                    {"d": "2024-01-02", "sopr": "0.98"},
                    {"d": "2024-01-03", "sopr": "1.12"},
                ],
                "last": True,
            }
            nupl_response = {
                "content": [
                    {"d": "2024-01-02", "nupl": "0.45"},
                    {"d": "2024-01-04", "nupl": "0.48"},
                ],
                "last": True,
            }

            mock_resp1 = Mock()
            mock_resp1.json.return_value = sopr_response
            mock_resp1.raise_for_status = Mock()

            mock_resp2 = Mock()
            mock_resp2.json.return_value = nupl_response
            mock_resp2.raise_for_status = Mock()

            mock_get.side_effect = [mock_resp1, mock_resp2]

            df = fetcher.fetch_all_metrics("2024-01-01", "2024-01-04", metrics=["sopr", "nupl"])

            # Should have 4 dates (outer join)
            assert len(df) == 4
            assert "sopr" in df.columns
            assert "nupl" in df.columns

            # Check that NaN appears where metrics don't overlap
            assert pd.notna(df.loc["2024-01-01", "sopr"])
            assert pd.isna(df.loc["2024-01-01", "nupl"])
            assert pd.notna(df.loc["2024-01-04", "nupl"])
            assert pd.isna(df.loc["2024-01-04", "sopr"])

    def test_fetch_all_metrics_skips_failed_metrics(self, fetcher):
        """Test fetch_all_metrics continues when individual metrics fail."""
        with patch.object(fetcher, "fetch_metric") as mock_fetch:
            # First metric succeeds, second fails, third succeeds
            mock_fetch.side_effect = [
                pd.DataFrame(
                    {"sopr": [1.05]},
                    index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
                ),
                Exception("API error"),
                pd.DataFrame(
                    {"nupl": [0.45]},
                    index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
                ),
            ]

            df = fetcher.fetch_all_metrics("2024-01-01", "2024-01-03", metrics=["sopr", "mvrv_zscore", "nupl"])

            # Should have 2 columns (failed metric excluded)
            assert len(df.columns) == 2
            assert "sopr" in df.columns
            assert "nupl" in df.columns
            assert "mvrv_zscore" not in df.columns

    def test_fetch_all_metrics_skips_empty_dataframes(self, fetcher):
        """Test fetch_all_metrics skips empty DataFrames."""
        with patch.object(fetcher, "fetch_metric") as mock_fetch:
            # First metric returns data, second returns empty DataFrame
            mock_fetch.side_effect = [
                pd.DataFrame(
                    {"sopr": [1.05]},
                    index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
                ),
                pd.DataFrame(),  # Empty
            ]

            df = fetcher.fetch_all_metrics("2024-01-01", "2024-01-03", metrics=["sopr", "mvrv_zscore"])

            # Should only have sopr column
            assert len(df.columns) == 1
            assert "sopr" in df.columns

    def test_fetch_all_metrics_returns_empty_when_all_fail(self, fetcher):
        """Test fetch_all_metrics returns empty DataFrame when all metrics fail."""
        with patch.object(fetcher, "fetch_metric") as mock_fetch:
            mock_fetch.side_effect = Exception("API error")

            df = fetcher.fetch_all_metrics("2024-01-01", "2024-01-03", metrics=["sopr"])

            assert isinstance(df, pd.DataFrame)
            assert df.empty


class TestTokenAuthentication:
    def test_token_passed_in_url_params(self, fetcher_with_token, mock_sopr_response):
        """Test token is included in URL parameters when provided."""
        with patch.object(fetcher_with_token.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = mock_sopr_response
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            fetcher_with_token.fetch_metric("sopr", "2024-01-01", "2024-01-03")

            call_args = mock_get.call_args
            assert call_args[1]["params"]["token"] == "test_token_123"

    def test_no_token_param_when_not_provided(self, fetcher, mock_sopr_response):
        """Test token is not included in parameters when not provided."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = mock_sopr_response
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            fetcher.fetch_metric("sopr", "2024-01-01", "2024-01-03")

            call_args = mock_get.call_args
            assert "token" not in call_args[1]["params"]

    def test_token_read_from_env_var(self, monkeypatch, mock_sopr_response):
        """Test token is read from BGEOMETRICS_TOKEN env var when not passed explicitly."""
        monkeypatch.setenv("BGEOMETRICS_TOKEN", "env_token_abc")
        env_fetcher = BGeometricsFetcher()
        with patch.object(env_fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = mock_sopr_response
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            env_fetcher.fetch_metric("sopr", "2024-01-01", "2024-01-03")

            call_args = mock_get.call_args
            assert call_args[1]["params"]["token"] == "env_token_abc"

    def test_explicit_token_overrides_env_var(self, monkeypatch, mock_sopr_response):
        """Test explicit constructor token takes precedence over env var."""
        monkeypatch.setenv("BGEOMETRICS_TOKEN", "env_token_abc")
        explicit_fetcher = BGeometricsFetcher(token="explicit_token_xyz")
        with patch.object(explicit_fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = mock_sopr_response
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            explicit_fetcher.fetch_metric("sopr", "2024-01-01", "2024-01-03")

            call_args = mock_get.call_args
            assert call_args[1]["params"]["token"] == "explicit_token_xyz"


class TestAvailableMetrics:
    def test_available_metrics_returns_all_metrics(self, fetcher):
        """Test available_metrics property returns list of all metric names."""
        metrics = fetcher.available_metrics

        assert isinstance(metrics, list)
        assert len(metrics) == len(METRIC_ENDPOINTS)
        assert "sopr" in metrics
        assert "mvrv_zscore" in metrics
        assert "nupl" in metrics
        assert "realized_price" in metrics
        assert "cdd" in metrics
        assert "puell_multiple" in metrics
        assert "active_addresses" in metrics
        assert "hash_rate" in metrics
        assert "supply_in_profit" in metrics


class TestMetricEndpoints:
    def test_all_metrics_have_correct_structure(self):
        """Test METRIC_ENDPOINTS has correct structure for all metrics."""
        for metric_name, config in METRIC_ENDPOINTS.items():
            assert "endpoint" in config, f"{metric_name} missing endpoint"
            assert "field" in config, f"{metric_name} missing field"
            assert isinstance(config["endpoint"], str)
            if config.get("multi_field"):
                assert config["field"] is None, f"{metric_name}: multi_field should have field=None"
            else:
                assert isinstance(config["field"], str), f"{metric_name}: field must be str"
            assert config["endpoint"].startswith("/v1/")

    def test_new_metric_endpoints_present(self):
        """Test that P002 metric endpoints are registered."""
        expected = [
            "sth_sopr",
            "lth_sopr",
            "sth_mvrv",
            "lth_mvrv",
            "nupl_sth",
            "nupl_lth",
            "exchange_inflow_btc",
            "exchange_outflow_btc",
            "exchange_netflow_btc",
            "exchange_reserve_btc",
            "lth_position_change_30d",
            "open_interest_futures",
            "funding_rate_aggregate",
            "stablecoin_supply",
            "etf_aggregate",
            "vdd_multiple",
            "realized_pl_ratio",
        ]
        for name in expected:
            assert name in METRIC_ENDPOINTS, f"{name} not in METRIC_ENDPOINTS"

    def test_fetch_uses_correct_endpoint_and_field(self, fetcher):
        """Test fetch_metric uses correct endpoint and field for standard metrics."""
        # Only test standard (non-multi-field, non-best-effort) metrics
        standard = {k: v for k, v in METRIC_ENDPOINTS.items() if not v.get("multi_field") and not v.get("best_effort")}
        for metric_name, config in standard.items():
            with patch.object(fetcher.session, "get") as mock_get:
                mock_resp = Mock()
                mock_resp.json.return_value = {
                    "content": [
                        {"d": "2024-01-01", config["field"]: "1.0"},
                    ],
                    "last": True,
                }
                mock_resp.raise_for_status = Mock()
                mock_get.return_value = mock_resp

                df = fetcher.fetch_metric(metric_name, "2024-01-01", "2024-01-01")

                expected_url = f"{BASE_URL}{config['endpoint']}"
                assert mock_get.call_args[0][0] == expected_url
                assert metric_name in df.columns


class TestSyncBgeometrics:
    @patch("sparky.data.onchain_bgeometrics.DataStore")
    @patch("sparky.data.onchain_bgeometrics.BGeometricsFetcher")
    def test_sync_calls_append_for_new_data(self, MockFetcherCls, MockStoreCls):
        mock_df = pd.DataFrame(
            {"sopr": [1.05, 0.98]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"),
        )
        MockFetcherCls.return_value.fetch_metric.return_value = mock_df
        MockStoreCls.return_value.get_last_timestamp.return_value = None
        MockStoreCls.return_value.load.return_value = (mock_df, {})

        results = sync_bgeometrics(metrics=["sopr"])

        assert results["sopr"] == 2
        MockStoreCls.return_value.append.assert_called_once()
        call_path = str(MockStoreCls.return_value.append.call_args[0][1])
        assert "sopr" in call_path

    @patch("sparky.data.onchain_bgeometrics.DataStore")
    @patch("sparky.data.onchain_bgeometrics.BGeometricsFetcher")
    def test_sync_incremental_from_last_timestamp(self, MockFetcherCls, MockStoreCls):
        from datetime import datetime as dt

        MockFetcherCls.return_value.fetch_metric.return_value = pd.DataFrame()
        MockStoreCls.return_value.get_last_timestamp.return_value = dt(2024, 1, 15, tzinfo=timezone.utc)
        MockStoreCls.return_value.load.return_value = (pd.DataFrame(), {})

        sync_bgeometrics(metrics=["sopr"])

        start_arg = MockFetcherCls.return_value.fetch_metric.call_args[0][1]
        assert start_arg == "2024-01-16"

    @patch("sparky.data.onchain_bgeometrics.DataStore")
    @patch("sparky.data.onchain_bgeometrics.BGeometricsFetcher")
    def test_sync_stops_on_rate_limit(self, MockFetcherCls, MockStoreCls):
        MockFetcherCls.return_value.fetch_metric.side_effect = RuntimeError("Rate limited")
        MockStoreCls.return_value.get_last_timestamp.return_value = None

        results = sync_bgeometrics(metrics=["sopr", "nupl"])

        assert MockFetcherCls.return_value.fetch_metric.call_count == 1


class TestDiscoverField:
    def test_discover_field_returns_non_d_keys(self, fetcher):
        """Test _discover_field returns field names excluding 'd'."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "content": [{"d": "2024-01-01", "usdtSupply": "100", "usdcSupply": "50"}],
                "last": True,
            }
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            fields = fetcher._discover_field("/v1/stablecoin-supply")
            assert "usdtSupply" in fields
            assert "usdcSupply" in fields
            assert "d" not in fields

    def test_discover_field_handles_list_response(self, fetcher):
        """Test _discover_field handles raw list responses."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = [{"d": "2024-01-01", "val": "1.0"}]
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            fields = fetcher._discover_field("/v1/test")
            assert fields == ["val"]

    def test_discover_field_returns_empty_on_error(self, fetcher):
        """Test _discover_field returns empty list on error."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_get.side_effect = Exception("fail")
            fields = fetcher._discover_field("/v1/test")
            assert fields == []

    def test_discover_field_returns_empty_for_empty_response(self, fetcher):
        """Test _discover_field returns empty list for empty content."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {"content": [], "last": True}
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            fields = fetcher._discover_field("/v1/test")
            assert fields == []


class TestBestEffortMetrics:
    def test_best_effort_returns_empty_on_403(self, fetcher):
        """Test best-effort metrics return empty DataFrame on 403."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.status_code = 403
            mock_resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden", response=mock_resp)
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("open_interest_futures", "2024-01-01", "2024-01-03")
            assert isinstance(df, pd.DataFrame)
            assert df.empty

    def test_best_effort_returns_empty_on_401(self, fetcher):
        """Test best-effort metrics return empty DataFrame on 401."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.status_code = 401
            mock_resp.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized", response=mock_resp)
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("funding_rate_aggregate", "2024-01-01", "2024-01-03")
            assert isinstance(df, pd.DataFrame)
            assert df.empty

    def test_non_best_effort_still_raises_on_403(self, fetcher):
        """Test regular metrics still raise on 403."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.status_code = 403
            mock_resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden", response=mock_resp)
            mock_get.return_value = mock_resp

            with pytest.raises(requests.HTTPError):
                fetcher.fetch_metric("sopr", "2024-01-01", "2024-01-03")

    def test_best_effort_works_when_api_succeeds(self, fetcher):
        """Test best-effort metrics work normally when API returns data."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "content": [
                    {"d": "2024-01-01", "openInterestFutures": "25000000000"},
                ],
                "last": True,
            }
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("open_interest_futures", "2024-01-01", "2024-01-01")
            assert len(df) == 1
            assert "open_interest_futures" in df.columns


class TestMultiFieldMetrics:
    def test_multi_field_extracts_all_non_d_keys(self, fetcher):
        """Test multi-field endpoints extract all non-'d' keys."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "content": [
                    {"d": "2024-01-01", "usdt": "80000000000", "usdc": "30000000000"},
                    {"d": "2024-01-02", "usdt": "81000000000", "usdc": "31000000000"},
                ],
                "last": True,
            }
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("stablecoin_supply", "2024-01-01", "2024-01-02")

            assert len(df) == 2
            assert "stablecoin_supply_usdt" in df.columns
            assert "stablecoin_supply_usdc" in df.columns
            assert df["stablecoin_supply_usdt"].iloc[0] == 80000000000.0

    def test_multi_field_etf_aggregate(self, fetcher):
        """Test etf_aggregate multi-field extraction."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "content": [
                    {"d": "2024-01-01", "totalNetFlow": "500", "totalHoldings": "900000"},
                ],
                "last": True,
            }
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("etf_aggregate", "2024-01-01", "2024-01-01")

            assert "etf_aggregate_totalNetFlow" in df.columns
            assert "etf_aggregate_totalHoldings" in df.columns

    def test_multi_field_skips_non_numeric(self, fetcher):
        """Test multi-field skips non-numeric values gracefully."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "content": [
                    {"d": "2024-01-01", "val1": "100", "val2": "not_a_number"},
                ],
                "last": True,
            }
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("stablecoin_supply", "2024-01-01", "2024-01-01")

            assert len(df) == 1
            assert "stablecoin_supply_val1" in df.columns

    def test_multi_field_empty_response(self, fetcher):
        """Test multi-field returns empty DataFrame for empty API response."""
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {"content": [], "last": True}
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            df = fetcher.fetch_metric("stablecoin_supply", "2024-01-01", "2024-01-01")
            assert df.empty
