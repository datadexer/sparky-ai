"""Crypto Fear & Greed Index fetcher (Alternative.me API)."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

__all__ = ["fetch_fgi", "load_fgi"]


def fetch_fgi(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """Fetch the full Crypto Fear & Greed Index history.

    Parameters
    ----------
    cache_path : Path, optional
        If provided and exists and is <24h old, load from parquet instead
        of hitting the API. After a successful fetch, saves to this path.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC, date-normalized), columns: fgi (int), fgi_class (str).
        Sorted ascending by date, deduplicated.
    """
    # Check cache
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            age_hours = (
                datetime.now(tz=timezone.utc)
                - datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
            ).total_seconds() / 3600
            if age_hours < 24:
                logger.info("Loading FGI from cache: %s (%.1fh old)", cache_path, age_hours)
                return load_fgi(cache_path)
            logger.info("Cache expired (%.1fh old), re-fetching", age_hours)

    # Fetch from API
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    logger.info("Fetching Fear & Greed Index from %s", url)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"FGI API request failed: {exc}") from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise ValueError(
            f"FGI API returned non-JSON response: {resp.text[:200]!r}"
        ) from exc
    if "data" not in data:
        raise ValueError("FGI API response missing 'data' key")

    records = []
    for entry in data["data"]:
        ts = pd.Timestamp(int(entry["timestamp"]), unit="s", tz="UTC").normalize()
        records.append(
            {
                "date": ts,
                "fgi": int(entry["value"]),
                "fgi_class": entry["value_classification"],
            }
        )

    if not records:
        raise ValueError("FGI API returned zero usable records after parsing")

    df = pd.DataFrame(records)
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    logger.info(
        "Fetched %d FGI records: %s to %s",
        len(df),
        df.index.min().date(),
        df.index.max().date(),
    )

    # Save cache
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.rename(cache_path)
        logger.info("Saved FGI cache to %s", cache_path)

    return df


def load_fgi(path: Path) -> pd.DataFrame:
    """Load FGI data from a parquet file.

    Parameters
    ----------
    path : Path
        Path to the parquet file.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC), columns: fgi (int), fgi_class (str).
    """
    df = pd.read_parquet(path)
    expected_cols = {"fgi", "fgi_class"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"FGI cache at {path} has unexpected schema: {list(df.columns)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"FGI cache at {path} does not have a DatetimeIndex")
    return df
