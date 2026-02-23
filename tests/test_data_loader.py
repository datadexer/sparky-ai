"""Tests for sparky.data.loader — enforced data access layer."""

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from sparky.data.loader import _detect_asset, _find_parquet, _in_vault, list_datasets, load
from sparky.oversight.holdout_guard import HoldoutGuard, HoldoutViolation


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


class TestVaultExclusion:
    """Vault paths must never leak into non-OOS lookups."""

    def test_in_vault_detects_vault_paths(self, tmp_path):
        vault = tmp_path / "data" / ".oos_vault"
        vault.mkdir(parents=True)
        with patch("sparky.data.loader.VAULT_DIR", vault):
            assert _in_vault(vault / "data" / "raw" / "btc" / "ohlcv.parquet")
            assert not _in_vault(tmp_path / "data" / "raw" / "btc" / "ohlcv.parquet")

    def test_find_parquet_excludes_vault(self, tmp_data_dir):
        vault_dir = tmp_data_dir / "data" / ".oos_vault" / "data" / "raw"
        vault_dir.mkdir(parents=True)
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D", tz="UTC")
        df = pd.DataFrame({"close": range(len(dates))}, index=dates)
        df.to_parquet(vault_dir / "secret_data.parquet")

        with (
            patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data"]),
            patch("sparky.data.loader.VAULT_DIR", tmp_data_dir / "data" / ".oos_vault"),
        ):
            assert _find_parquet("secret_data") is None

    def test_list_datasets_excludes_vault(self, tmp_data_dir):
        vault_dir = tmp_data_dir / "data" / ".oos_vault" / "data" / "raw"
        vault_dir.mkdir(parents=True)
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D", tz="UTC")
        df = pd.DataFrame({"close": range(len(dates))}, index=dates)
        df.to_parquet(vault_dir / "vault_only.parquet")

        with (
            patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data"]),
            patch("sparky.data.loader.VAULT_DIR", tmp_data_dir / "data" / ".oos_vault"),
            patch("sparky.data.loader._DATASET_ALIASES", {}),
        ):
            names = [d["name"] for d in list_datasets()]
            assert "vault_only" not in names


class TestDatasetAliases:
    def test_alias_resolves(self, tmp_data_dir):
        alias_path = tmp_data_dir / "data" / "raw" / "eth" / "ohlcv_hourly.parquet"
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="h", tz="UTC")
        df = pd.DataFrame({"close": range(len(dates))}, index=dates)
        df.to_parquet(alias_path)

        with (
            patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data"]),
            patch("sparky.data.loader._DATASET_ALIASES", {"eth_ohlcv_hourly": alias_path}),
        ):
            result = _find_parquet("eth_ohlcv_hourly")
            assert result is not None
            assert result == alias_path

    def test_aliases_in_list_datasets(self, tmp_data_dir):
        alias_path = tmp_data_dir / "data" / "raw" / "eth" / "ohlcv_hourly.parquet"
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="h", tz="UTC")
        df = pd.DataFrame({"close": range(len(dates))}, index=dates)
        df.to_parquet(alias_path)

        with (
            patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data"]),
            patch("sparky.data.loader.VAULT_DIR", tmp_data_dir / "data" / ".oos_vault"),
            patch("sparky.data.loader._DATASET_ALIASES", {"eth_ohlcv_hourly": alias_path}),
        ):
            names = [d["name"] for d in list_datasets()]
            assert "eth_ohlcv_hourly" in names


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
    def test_training_truncates_at_holdout(self, tmp_data_dir, holdout_dates):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            df = load("btc_1h_features", purpose="training", asset="btc")
            assert df.index.max() <= holdout_dates["max_training_ts"]

    def test_analysis_returns_full_data(self, tmp_data_dir, holdout_dates):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            df = load("btc_1h_features", purpose="analysis", asset="btc")
            # Should include data past holdout boundary
            assert df.index.max() > holdout_dates["oos_start_ts"]

    def test_training_shorter_than_analysis(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            train_df = load("btc_1h_features", purpose="training", asset="btc")
            full_df = load("btc_1h_features", purpose="analysis", asset="btc")
            assert len(train_df) < len(full_df)

    def test_invalid_purpose_raises(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            with pytest.raises(ValueError, match="Invalid purpose"):
                load("btc_1h_features", purpose="something_invalid")

    def test_missing_dataset_raises(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            with pytest.raises(FileNotFoundError):
                load("nonexistent_dataset")

    def test_validation_also_truncates(self, tmp_data_dir, holdout_dates):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            df = load("btc_1h_features", purpose="validation", asset="btc")
            assert df.index.max() <= holdout_dates["max_training_ts"]

    def test_utc_enforcement(self, tmp_data_dir):
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            df = load("btc_1h_features", purpose="training", asset="btc")
            assert df.index.tz is not None  # must be tz-aware

    def test_load_by_full_path(self, tmp_data_dir):
        full_path = tmp_data_dir / "data" / "features" / "btc_1h_features.parquet"
        df = load(str(full_path), purpose="training", asset="btc")
        assert len(df) > 0


class TestOosEvaluation:
    """Tests for purpose='oos_evaluation' (deprecated) — the holdout enforcement critical path."""

    def test_oos_without_guard_raises(self, tmp_data_dir):
        """No oos_guard at all → HoldoutViolation."""
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            with pytest.warns(DeprecationWarning):
                with pytest.raises(HoldoutViolation, match="requires explicit authorization"):
                    load("btc_1h_features", purpose="oos_evaluation")

    def test_oos_with_unauthorized_guard_raises(self, tmp_data_dir):
        """Guard present but authorize_oos_evaluation() never called → HoldoutViolation."""
        guard = HoldoutGuard()
        with patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]):
            with pytest.warns(DeprecationWarning):
                with pytest.raises(HoldoutViolation, match="requires explicit authorization"):
                    load("btc_1h_features", purpose="oos_evaluation", oos_guard=guard)

    def test_oos_with_authorized_guard_reads_holdout(self, tmp_data_dir, holdout_dates):
        """Authorized guard with holdout data → returns full dataset including OOS rows."""
        holdout_dir = tmp_data_dir / "data" / "holdout" / "btc"
        holdout_dir.mkdir(parents=True)
        dates = pd.date_range("2020-01-01", "2025-06-01", freq="D", tz="UTC")
        df_full = pd.DataFrame({"close": range(len(dates))}, index=dates)
        df_full.to_parquet(holdout_dir / "1h_features.parquet")

        guard = HoldoutGuard()
        guard.authorize_oos_evaluation(
            model_name="test_model",
            approach_family="test",
            approved_by="human-ak",
            in_sample_sharpe=1.5,
        )

        with (
            patch("sparky.data.loader.DATA_DIRS", [tmp_data_dir / "data" / "features"]),
            patch("sparky.data.loader.HOLDOUT_DIR", tmp_data_dir / "data" / "holdout"),
        ):
            with pytest.warns(DeprecationWarning):
                df = load("btc_1h_features", purpose="oos_evaluation", oos_guard=guard)

        assert df.index.max() > holdout_dates["oos_start_ts"]
        assert df.index.tz is not None


class TestEvaluationPurpose:
    """Tests for purpose='evaluation' — env-var-gated holdout access."""

    def test_evaluation_blocked_without_env(self, tmp_data_dir):
        """Without SPARKY_OOS_ENABLED=1 → PermissionError."""
        env = os.environ.copy()
        env.pop("SPARKY_OOS_ENABLED", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(PermissionError, match="OOS data access denied"):
                load("btc_ohlcv_8h", purpose="evaluation")

    def test_evaluation_allowed_with_env(self, tmp_data_dir):
        """With SPARKY_OOS_ENABLED=1 + holdout data → works."""
        holdout_dir = tmp_data_dir / "data" / "holdout" / "btc"
        holdout_dir.mkdir(parents=True)
        dates = pd.date_range("2020-01-01", "2025-06-01", freq="8h", tz="UTC")
        df = pd.DataFrame({"close": range(len(dates))}, index=dates)
        df.to_parquet(holdout_dir / "ohlcv_8h.parquet")

        with (
            patch.dict(os.environ, {"SPARKY_OOS_ENABLED": "1"}),
            patch("sparky.data.loader.HOLDOUT_DIR", tmp_data_dir / "data" / "holdout"),
        ):
            result = load("btc_ohlcv_8h", purpose="evaluation")
            assert len(result) > 0
            assert result.index.tz is not None

    def test_evaluation_missing_data_raises(self):
        """With env var set but no data → FileNotFoundError."""
        with (
            patch.dict(os.environ, {"SPARKY_OOS_ENABLED": "1"}),
            patch("sparky.data.loader.HOLDOUT_DIR", Path("/nonexistent/holdout")),
        ):
            with pytest.raises(FileNotFoundError, match="No holdout data"):
                load("btc_ohlcv_8h", purpose="evaluation")


class TestP003DataPaths:
    """Test that P003 data directories are properly integrated."""

    def test_p003_asset_pattern(self):
        from sparky.data.loader import _detect_asset

        assert _detect_asset("p003_INJUSDT") == "cross_asset"

    def test_p003_binance_perps_in_data_dirs(self):
        from sparky.data.loader import DATA_DIRS

        dir_strs = [str(d) for d in DATA_DIRS]
        assert "data/p003/binance_perps" in dir_strs
        assert "data/p003/funding_rates" in dir_strs

    def test_find_parquet_resolves_p003(self, tmp_path, monkeypatch):
        import pandas as pd

        from sparky.data.loader import DATA_DIRS, _find_parquet

        p003_dir = tmp_path / "binance_perps"
        p003_dir.mkdir()
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", "2023-12-31", freq="D", tz="UTC"),
                "close": range(365),
            }
        )
        df.to_parquet(p003_dir / "TESTUSDT.parquet")

        monkeypatch.setattr("sparky.data.loader.DATA_DIRS", [p003_dir] + list(DATA_DIRS))
        result = _find_parquet("TESTUSDT")
        assert result is not None
        assert "TESTUSDT.parquet" in str(result)
