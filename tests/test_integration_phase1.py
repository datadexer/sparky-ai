"""Phase 1 integration tests — DataStore -> DataQualityChecker pipeline.

Verifies that data saved through DataStore round-trips correctly and
passes through DataQualityChecker with expected results. Uses synthetic
data only — no API calls.
"""

import json

import numpy as np
import pandas as pd
import pytest

from sparky.data.quality import DataQualityChecker
from sparky.data.storage import DataStore


@pytest.fixture
def synthetic_ohlcv():
    """Generate synthetic OHLCV data with UTC DatetimeIndex."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365, freq="D", tz="UTC")
    close = 30000 + np.cumsum(np.random.normal(0, 500, 365))
    high = close + np.abs(np.random.normal(200, 100, 365))
    low = close - np.abs(np.random.normal(200, 100, 365))
    open_ = close + np.random.normal(0, 100, 365)
    volume = np.abs(np.random.normal(1e9, 2e8, 365))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def store(tmp_path):
    """Create a DataStore that writes manifest to tmp."""
    return DataStore(manifest_path=tmp_path / "test_manifest.json")


class TestDataStoreRoundTrip:
    """DataStore save -> load preserves data and metadata."""

    def test_save_load_preserves_data(self, store, tmp_path, synthetic_ohlcv):
        """DataFrame round-trips through Parquet without data loss."""
        path = tmp_path / "btc" / "ohlcv.parquet"
        metadata = {"source": "binance", "asset": "btc"}

        store.save(synthetic_ohlcv, path, metadata=metadata)
        loaded_df, loaded_meta = store.load(path)

        pd.testing.assert_frame_equal(
            loaded_df, synthetic_ohlcv, check_freq=False
        )

    def test_save_load_preserves_metadata(self, store, tmp_path, synthetic_ohlcv):
        """Metadata round-trips through Parquet schema."""
        path = tmp_path / "btc" / "ohlcv.parquet"
        metadata = {"source": "binance", "asset": "btc"}

        store.save(synthetic_ohlcv, path, metadata=metadata)
        _, loaded_meta = store.load(path)

        assert loaded_meta["source"] == "binance"
        assert loaded_meta["asset"] == "btc"
        assert loaded_meta["row_count"] == 365
        assert "saved_at" in loaded_meta

    def test_manifest_updated_with_sha256(self, store, tmp_path, synthetic_ohlcv):
        """Saving updates the manifest file with SHA-256 hash."""
        manifest_path = store.manifest_path
        path = tmp_path / "btc" / "ohlcv.parquet"

        store.save(synthetic_ohlcv, path, metadata={"source": "test"})

        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert str(path) in manifest
        entry = manifest[str(path)]
        assert "sha256" in entry
        assert len(entry["sha256"]) == 64  # SHA-256 hex digest length


class TestDataStoreToQualityChecker:
    """DataStore -> DataQualityChecker end-to-end pipeline."""

    def test_stored_data_passes_quality_checks(
        self, store, tmp_path, synthetic_ohlcv
    ):
        """Data saved through DataStore passes quality checks."""
        path = tmp_path / "btc" / "ohlcv.parquet"
        store.save(synthetic_ohlcv, path, metadata={"source": "binance"})

        # Load and run quality checks
        loaded_df, _ = store.load(path)
        checker = DataQualityChecker()
        report = checker.run_all_checks(
            loaded_df, asset="btc", source="binance", price_range=(1.0, 1_000_000)
        )

        # Completeness should pass (no gaps in synthetic data)
        assert report["checks"]["completeness"]["pass"] is True
        assert report["checks"]["completeness"]["total_rows"] == 365

        # Price range should pass
        assert report["checks"]["price_range"]["pass"] is True

    def test_quality_checker_detects_gaps_in_stored_data(
        self, store, tmp_path
    ):
        """Quality checker catches gaps in data that went through DataStore."""
        # Create data with a 5-day gap
        dates_before = pd.date_range("2023-01-01", periods=30, freq="D", tz="UTC")
        dates_after = pd.date_range("2023-02-05", periods=30, freq="D", tz="UTC")
        dates = dates_before.union(dates_after)

        np.random.seed(42)
        df = pd.DataFrame(
            {"close": np.random.normal(30000, 500, len(dates))}, index=dates
        )

        path = tmp_path / "gapped.parquet"
        store.save(df, path, metadata={"source": "test"})

        loaded_df, _ = store.load(path)
        checker = DataQualityChecker()
        completeness = checker.check_completeness(loaded_df, max_gap_days=3)

        assert completeness["pass"] is False
        assert len(completeness["gaps"]) > 0

    def test_get_last_timestamp_after_save(self, store, tmp_path, synthetic_ohlcv):
        """get_last_timestamp returns the correct last date after save."""
        path = tmp_path / "btc" / "ohlcv.parquet"
        store.save(synthetic_ohlcv, path, metadata={"source": "test"})

        last_ts = store.get_last_timestamp(path)
        assert last_ts is not None
        expected = synthetic_ohlcv.index.max().to_pydatetime()
        assert last_ts == expected
