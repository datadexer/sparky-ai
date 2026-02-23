"""Tests for sparky.data.holdout_split module."""

import polars as pl
import pytest

from sparky.data.holdout_split import (
    scan_all,
    split_directory,
    split_parquet_at_holdout,
    validate_directory,
)
from sparky.oversight.holdout_guard import HoldoutGuard


@pytest.fixture
def oos_boundary():
    guard = HoldoutGuard()
    oos_start, _ = guard.get_oos_boundary("cross_asset")
    return oos_start


@pytest.fixture
def sample_parquet(tmp_path, oos_boundary):
    """Create a parquet file straddling the OOS boundary."""
    import pandas as pd

    oos_ts = pd.Timestamp(oos_boundary, tz="UTC")
    timestamps = pd.date_range(oos_ts - pd.Timedelta(days=30), oos_ts + pd.Timedelta(days=29), freq="D", tz="UTC")
    df = pl.DataFrame(
        {
            "timestamp": timestamps.tolist(),
            "close": list(range(len(timestamps))),
        }
    )
    path = tmp_path / "test_data.parquet"
    df.write_parquet(path)
    return path, len([t for t in timestamps if t < oos_ts]), len([t for t in timestamps if t >= oos_ts])


class TestSplitParquetAtHoldout:
    def test_splits_correctly(self, sample_parquet, tmp_path):
        src, expected_is, expected_holdout = sample_parquet
        holdout_dir = tmp_path / "holdout"

        result = split_parquet_at_holdout(src, holdout_dir, timestamp_col="timestamp", asset="cross_asset")

        assert not result["skipped"]
        assert result["is_rows"] == expected_is
        assert result["holdout_rows"] == expected_holdout

        # Source file should contain only IS rows
        df_is = pl.read_parquet(src)
        assert len(df_is) == expected_is

        # Holdout file should exist with OOS rows
        holdout_file = holdout_dir / src.name
        assert holdout_file.exists()
        df_oos = pl.read_parquet(holdout_file)
        assert len(df_oos) == expected_holdout

    def test_skips_non_parquet(self, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("not a parquet")
        result = split_parquet_at_holdout(txt, tmp_path / "holdout")
        assert result["skipped"]
        assert "not a parquet" in result["reason"]

    def test_skips_missing_timestamp_col(self, tmp_path):
        df = pl.DataFrame({"price": [1.0, 2.0]})
        path = tmp_path / "no_ts.parquet"
        df.write_parquet(path)
        result = split_parquet_at_holdout(path, tmp_path / "holdout")
        assert result["skipped"]
        assert "timestamp" in result["reason"]

    def test_all_is_data(self, tmp_path):
        """File with only IS data — no holdout file created."""
        import pandas as pd

        guard = HoldoutGuard()
        oos_start, _ = guard.get_oos_boundary("cross_asset")
        oos_ts = pd.Timestamp(oos_start, tz="UTC")
        timestamps = pd.date_range(oos_ts - pd.Timedelta(days=10), oos_ts - pd.Timedelta(days=1), freq="D", tz="UTC")
        df = pl.DataFrame({"timestamp": timestamps.tolist(), "v": list(range(len(timestamps)))})
        path = tmp_path / "is_only.parquet"
        df.write_parquet(path)

        holdout_dir = tmp_path / "holdout"
        result = split_parquet_at_holdout(path, holdout_dir)
        assert result["holdout_rows"] == 0
        assert result["is_rows"] == len(timestamps)
        assert not (holdout_dir / "is_only.parquet").exists()


class TestSplitDirectory:
    def test_processes_multiple_files(self, tmp_path):
        import pandas as pd

        guard = HoldoutGuard()
        oos_start, _ = guard.get_oos_boundary("cross_asset")
        oos_ts = pd.Timestamp(oos_start, tz="UTC")

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        holdout_dir = tmp_path / "holdout"

        for name in ["A.parquet", "B.parquet"]:
            timestamps = pd.date_range(oos_ts - pd.Timedelta(days=5), oos_ts + pd.Timedelta(days=4), freq="D", tz="UTC")
            df = pl.DataFrame({"timestamp": timestamps.tolist(), "v": list(range(len(timestamps)))})
            df.write_parquet(src_dir / name)

        results = split_directory(src_dir, holdout_dir)
        assert len(results) == 2
        for r in results:
            assert not r["skipped"]
            assert r["is_rows"] > 0
            assert r["holdout_rows"] > 0
            assert "file" in r


class TestValidateDirectory:
    def test_returns_violations(self, sample_parquet):
        src, _, _ = sample_parquet
        violations = validate_directory(src.parent, timestamp_col="timestamp", asset="cross_asset")
        assert len(violations) > 0
        assert "test_data.parquet" in violations[0]["file"]

    def test_passes_clean_dir(self, tmp_path):
        import pandas as pd

        guard = HoldoutGuard()
        oos_start, _ = guard.get_oos_boundary("cross_asset")
        oos_ts = pd.Timestamp(oos_start, tz="UTC")
        timestamps = pd.date_range(oos_ts - pd.Timedelta(days=10), oos_ts - pd.Timedelta(days=1), freq="D", tz="UTC")
        df = pl.DataFrame({"timestamp": timestamps.tolist(), "v": list(range(len(timestamps)))})
        df.write_parquet(tmp_path / "clean.parquet")

        result = validate_directory(tmp_path, timestamp_col="timestamp", asset="cross_asset")
        assert result == []


class TestScanAll:
    def test_excludes_holdout_dir(self, tmp_path):
        import pandas as pd

        guard = HoldoutGuard()
        oos_start, _ = guard.get_oos_boundary("cross_asset")
        oos_ts = pd.Timestamp(oos_start, tz="UTC")
        future_ts = pd.date_range(oos_ts, oos_ts + pd.Timedelta(days=5), freq="D", tz="UTC")
        df = pl.DataFrame({"timestamp": future_ts.tolist(), "v": list(range(len(future_ts)))})

        # Put OOS data inside holdout/ subdir — should be excluded
        holdout_dir = tmp_path / "holdout"
        holdout_dir.mkdir()
        df.write_parquet(holdout_dir / "oos_data.parquet")

        # Put OOS data inside .oos_vault/ subdir — should be excluded
        vault_dir = tmp_path / ".oos_vault"
        vault_dir.mkdir()
        df.write_parquet(vault_dir / "vault_data.parquet")

        violations = scan_all(tmp_path)
        assert len(violations) == 0

    def test_detects_violation_outside_excluded(self, tmp_path):
        import pandas as pd

        guard = HoldoutGuard()
        oos_start, _ = guard.get_oos_boundary("cross_asset")
        oos_ts = pd.Timestamp(oos_start, tz="UTC")
        future_ts = pd.date_range(oos_ts, oos_ts + pd.Timedelta(days=5), freq="D", tz="UTC")
        df = pl.DataFrame({"timestamp": future_ts.tolist(), "v": list(range(len(future_ts)))})

        subdir = tmp_path / "p003"
        subdir.mkdir()
        df.write_parquet(subdir / "bad.parquet")

        violations = scan_all(tmp_path)
        assert len(violations) == 1
        assert "bad.parquet" in violations[0]["file"]
