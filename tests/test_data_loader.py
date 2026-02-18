"""Tests for sparky.data.loader — enforced data access layer."""

from unittest.mock import patch

import pandas as pd
import pytest

from sparky.data.loader import _detect_asset, _find_parquet, list_datasets, load


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary data directories with sample parquet files."""
    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    # Create sample BTC parquet
    dates = pd.date_range("2020-01-01", "2025-06-01", freq="D", tz="UTC")
    df = pd.DataFrame({"close": range(len(dates)), "volume": range(len(dates))}, index=dates)
    df.to_parquet(features_dir / "btc_1h_features.parquet")

    # Create sample ETH parquet
    df.to_parquet(features_dir / "eth_daily.parquet")

    return tmp_path


class TestDetectAsset:
    def test_btc_detection(self):
        assert _detect_asset("btc_1h_features") == "btc"

    def test_eth_detection(self):
        assert _detect_asset("eth_daily") == "eth"

    def test_cross_asset_detection(self):
        assert _detect_asset("sol_hourly") == "cross_asset"

    def test_unknown_defaults_to_cross_asset(self):
        assert _detect_asset("mystery_data") == "cross_asset"


class TestFindParquet:
    def test_finds_exact_match(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            path = _find_parquet("btc_1h_features")
            assert path is not None
            assert path.exists()

    def test_returns_none_for_missing(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            assert _find_parquet("nonexistent_dataset") is None


class TestListDatasets:
    def test_lists_available_datasets(self, tmp_data_dir):
        with patch(
            "sparky.data.loader.DATA_DIRS",
            [
                tmp_data_dir / "data" / "features",
                tmp_data_dir / "data" / "processed",
            ],
        ):
            datasets = list_datasets()
            names = [d["name"] for d in datasets]
            assert "btc_1h_features" in names
            assert "eth_daily" in names

    def test_dataset_has_required_keys(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            datasets = list_datasets()
            for d in datasets:
                assert "name" in d
                assert "path" in d
                assert "asset" in d


class TestLoad:
    def test_training_truncates_at_holdout(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            df = load("btc_1h_features", purpose="training", asset="btc")
            # Holdout is 2024-07-01, embargo 30 days -> max 2024-06-01
            assert df.index.max() <= pd.Timestamp("2024-06-01", tz="UTC")

    def test_analysis_returns_full_data(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            df = load("btc_1h_features", purpose="analysis", asset="btc")
            # Should include data past holdout boundary
            assert df.index.max() > pd.Timestamp("2024-07-01", tz="UTC")

    def test_training_shorter_than_analysis(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            train_df = load("btc_1h_features", purpose="training", asset="btc")
            full_df = load("btc_1h_features", purpose="analysis", asset="btc")
            assert len(train_df) < len(full_df)

    def test_invalid_purpose_raises(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            with pytest.raises(ValueError, match="Invalid purpose"):
                load("btc_1h_features", purpose="oos_evaluation")

    def test_missing_dataset_raises(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            with pytest.raises(FileNotFoundError):
                load("nonexistent_dataset")

    def test_validation_also_truncates(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            df = load("btc_1h_features", purpose="validation", asset="btc")
            assert df.index.max() <= pd.Timestamp("2024-06-01", tz="UTC")

    def test_utc_enforcement(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            df = load("btc_1h_features", purpose="training", asset="btc")
            assert df.index.tz is not None  # must be tz-aware

    def test_load_by_full_path(self, tmp_data_dir):
        full_path = tmp_data_dir / "data" / "features" / "btc_1h_features.parquet"
        df = load(str(full_path), purpose="training", asset="btc")
        assert len(df) > 0
