"""Holdout split utilities — reusable library for splitting parquet files at the OOS boundary.

Used by download scripts, standalone scanner, and project-specific split drivers.
All functions read the OOS boundary dynamically from HoldoutGuard.
"""

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from sparky.oversight.holdout_guard import HoldoutGuard, HoldoutViolation

logger = logging.getLogger(__name__)

_guard = HoldoutGuard()

# Directories excluded from scan_all()
_EXCLUDED_DIRS = {"holdout", ".oos_vault"}


def _get_oos_start(asset: str = "cross_asset") -> pd.Timestamp:
    oos_start_str, _ = _guard.get_oos_boundary(asset)
    return pd.Timestamp(oos_start_str, tz="UTC")


def _to_utc_timestamp(ts) -> pd.Timestamp:
    """Convert a polars scalar (datetime/Timestamp/str) to a UTC pd.Timestamp."""
    result = pd.Timestamp(ts)
    if result.tz is None:
        return result.tz_localize("UTC")
    return result.tz_convert("UTC")


def split_parquet_at_holdout(
    src_path: Path,
    holdout_dest_dir: Path,
    timestamp_col: str = "timestamp",
    asset: str = "cross_asset",
) -> dict:
    """Split one parquet file. Overwrites src with IS-only, writes holdout portion to dest dir.

    Returns dict with keys: is_rows, holdout_rows, skipped (bool), reason (str if skipped).
    """
    src_path = Path(src_path)
    holdout_dest_dir = Path(holdout_dest_dir)

    if src_path.suffix != ".parquet":
        return {"is_rows": 0, "holdout_rows": 0, "skipped": True, "reason": "not a parquet file"}

    try:
        df = pl.read_parquet(src_path)
    except Exception as e:
        return {"is_rows": 0, "holdout_rows": 0, "skipped": True, "reason": f"read error: {e}"}

    if timestamp_col not in df.columns:
        return {"is_rows": 0, "holdout_rows": 0, "skipped": True, "reason": f"no '{timestamp_col}' column"}

    oos_start = _get_oos_start(asset)

    # Ensure timestamp column is comparable
    ts = df[timestamp_col]
    if ts.dtype == pl.Utf8:
        df = df.with_columns(pl.col(timestamp_col).str.to_datetime().alias(timestamp_col))
    elif hasattr(ts.dtype, "time_zone") and ts.dtype.time_zone is None:
        df = df.with_columns(pl.col(timestamp_col).dt.replace_time_zone("UTC").alias(timestamp_col))

    # Split
    is_mask = pl.col(timestamp_col) < oos_start
    df_is = df.filter(is_mask)
    df_holdout = df.filter(~is_mask)

    # Write IS back to source
    if len(df_is) > 0:
        df_is.write_parquet(src_path)
    else:
        # Schema-preserving empty parquet
        df.head(0).write_parquet(src_path)

    # Write holdout
    if len(df_holdout) > 0:
        holdout_dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = holdout_dest_dir / src_path.name
        df_holdout.write_parquet(dest_path)

    return {
        "is_rows": len(df_is),
        "holdout_rows": len(df_holdout),
        "skipped": False,
        "reason": None,
    }


def split_directory(
    is_dir: Path,
    holdout_dir: Path,
    timestamp_col: str = "timestamp",
    asset: str = "cross_asset",
) -> list[dict]:
    """Split all parquets in a directory. Returns list of per-file results."""
    is_dir = Path(is_dir)
    holdout_dir = Path(holdout_dir)
    results = []

    for pq in sorted(is_dir.glob("*.parquet")):
        result = split_parquet_at_holdout(pq, holdout_dir, timestamp_col, asset)
        result["file"] = str(pq)
        results.append(result)
        if not result["skipped"]:
            logger.info(f"  {pq.name}: {result['is_rows']} IS, {result['holdout_rows']} holdout")
        else:
            logger.info(f"  {pq.name}: SKIPPED ({result['reason']})")

    return results


def validate_directory(
    data_dir: Path,
    timestamp_col: str = "timestamp",
    asset: str = "cross_asset",
) -> list[dict]:
    """Scan a directory for holdout violations. Returns list of violations.

    Raises HoldoutViolation if any file has post-holdout timestamps.
    """
    data_dir = Path(data_dir)
    oos_start = _get_oos_start(asset)
    violations = []

    for pq in sorted(data_dir.rglob("*.parquet")):
        try:
            df = pl.read_parquet(pq, columns=[timestamp_col] if timestamp_col else None)
        except Exception:  # noqa: S112
            continue

        if timestamp_col not in df.columns:
            continue

        ts = df[timestamp_col]
        if ts.dtype == pl.Utf8:
            ts = ts.str.to_datetime()

        max_ts = ts.max()
        if max_ts is None:
            continue

        max_ts_pd = _to_utc_timestamp(max_ts)

        if max_ts_pd >= oos_start:
            holdout_count = ts.filter(ts >= oos_start).len()
            violations.append(
                {
                    "file": str(pq),
                    "max_timestamp": str(max_ts),
                    "holdout_rows": holdout_count,
                }
            )

    if violations:
        raise HoldoutViolation(
            f"{len(violations)} file(s) in {data_dir} contain post-holdout data. "
            f"Run split_directory() or setup script to fix."
        )

    return violations


def scan_all(data_root: Path = Path("data")) -> list[dict]:
    """Scan ALL data/ subdirectories (excluding holdout/vault) for violations.

    Returns list of violations (empty = clean). Does NOT raise.
    """
    data_root = Path(data_root)
    oos_start = _get_oos_start("cross_asset")
    violations = []

    for pq in sorted(data_root.rglob("*.parquet")):
        # Skip excluded directories
        rel = pq.relative_to(data_root)
        if any(part in _EXCLUDED_DIRS for part in rel.parts):
            continue

        try:
            # Try reading just the timestamp column for efficiency
            schema = pl.read_parquet_schema(pq)
        except Exception:  # noqa: S112
            continue

        # Find a timestamp-like column
        ts_col = None
        for col_name in ["timestamp", "date", "datetime", "time"]:
            if col_name in schema:
                ts_col = col_name
                break

        if ts_col is None:
            continue

        try:
            df = pl.read_parquet(pq, columns=[ts_col])
        except Exception:  # noqa: S112
            continue

        ts = df[ts_col]
        if ts.dtype == pl.Utf8:
            try:
                ts = ts.str.to_datetime()
            except Exception:  # noqa: S112
                continue

        max_ts = ts.max()
        if max_ts is None:
            continue

        try:
            max_ts_pd = _to_utc_timestamp(max_ts)
        except Exception:  # noqa: S112
            continue

        if max_ts_pd >= oos_start:
            holdout_count = ts.filter(ts >= oos_start).len()
            violations.append(
                {
                    "file": str(pq),
                    "max_timestamp": str(max_ts),
                    "holdout_rows": holdout_count,
                }
            )

    return violations
