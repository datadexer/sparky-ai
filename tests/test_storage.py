"""Tests for Parquet storage layer."""

import json
from pathlib import Path

import pandas as pd
import pytest

from sparky.data.storage import DataStore, MANIFEST_PATH


@pytest.fixture
def store():
    return DataStore()


@pytest.fixture
def sample_df():
    """Sample DataFrame with DatetimeIndex."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {"close": [100.0, 110.0, 105.0, 115.0, 108.0], "volume": [1000, 1200, 900, 1500, 1100]},
        index=dates,
    )


class TestDataStoreSaveLoad:
    def test_save_and_load(self, store, sample_df, tmp_path):
        path = tmp_path / "test.parquet"
        store.save(sample_df, path, metadata={"source": "test"})

        loaded_df, meta = store.load(path)
        pd.testing.assert_frame_equal(loaded_df, sample_df, check_freq=False)
        assert meta["source"] == "test"
        assert meta["row_count"] == 5

    def test_metadata_includes_date_range(self, store, sample_df, tmp_path):
        path = tmp_path / "test.parquet"
        store.save(sample_df, path)
        _, meta = store.load(path)
        assert "date_range_start" in meta
        assert "date_range_end" in meta
        assert "saved_at" in meta

    def test_creates_parent_dirs(self, store, sample_df, tmp_path):
        path = tmp_path / "deep" / "nested" / "dir" / "test.parquet"
        store.save(sample_df, path)
        assert path.exists()

    def test_empty_dataframe(self, store, tmp_path):
        path = tmp_path / "empty.parquet"
        df = pd.DataFrame({"a": []})
        store.save(df, path)
        loaded, meta = store.load(path)
        assert len(loaded) == 0


class TestDataStoreIncremental:
    def test_get_last_timestamp(self, store, sample_df, tmp_path):
        path = tmp_path / "test.parquet"
        store.save(sample_df, path)
        last_ts = store.get_last_timestamp(path)
        assert last_ts == sample_df.index.max().to_pydatetime()

    def test_get_last_timestamp_missing_file(self, store, tmp_path):
        assert store.get_last_timestamp(tmp_path / "missing.parquet") is None

    def test_append_new_data(self, store, sample_df, tmp_path):
        path = tmp_path / "test.parquet"
        store.save(sample_df, path)

        # Append 3 more days
        new_dates = pd.date_range("2024-01-06", periods=3, freq="D")
        new_df = pd.DataFrame(
            {"close": [112.0, 118.0, 109.0], "volume": [1300, 1400, 1000]},
            index=new_dates,
        )
        store.append(new_df, path)

        loaded, _ = store.load(path)
        assert len(loaded) == 8

    def test_append_deduplicates(self, store, sample_df, tmp_path):
        path = tmp_path / "test.parquet"
        store.save(sample_df, path)

        # Append overlapping data (last 2 days + 2 new)
        overlap_dates = pd.date_range("2024-01-04", periods=4, freq="D")
        overlap_df = pd.DataFrame(
            {"close": [999.0, 998.0, 112.0, 118.0], "volume": [9, 8, 1300, 1400]},
            index=overlap_dates,
        )
        store.append(overlap_df, path)

        loaded, _ = store.load(path)
        assert len(loaded) == 7  # 5 original + 2 new, 2 overlapping replaced
        # Overlapping values should be from the new data (keep="last")
        assert loaded.loc["2024-01-04", "close"] == 999.0


class TestDataManifest:
    def test_manifest_updated_on_save(self, store, sample_df, tmp_path, monkeypatch):
        manifest_path = tmp_path / "manifest.json"
        monkeypatch.setattr("sparky.data.storage.MANIFEST_PATH", manifest_path)

        path = tmp_path / "test.parquet"
        store.save(sample_df, path)

        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert str(path) in manifest
        entry = manifest[str(path)]
        assert "sha256" in entry
        assert len(entry["sha256"]) == 64  # SHA-256 hex digest length
        assert "updated_at" in entry
        assert "size_bytes" in entry
