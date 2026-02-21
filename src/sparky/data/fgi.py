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
    resp = requests.get(url, timeout=30)  # noqa: S113
    if resp.status_code != 200:
        raise RuntimeError(f"FGI API returned status {resp.status_code}")

    data = resp.json()
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
        df.to_parquet(cache_path)
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
    return pd.read_parquet(path)
