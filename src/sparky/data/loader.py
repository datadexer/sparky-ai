"""Enforced data access layer with holdout protection.

All data loading for model work MUST go through this module.
Training/validation loads are automatically truncated at the holdout
embargo boundary. Analysis loads return full data with a warning.

Usage:
    from sparky.data.loader import load, list_datasets

    # Training (auto-truncated at embargo boundary)
    df = load("btc_1h_features", purpose="training")

    # Analysis (full data, logged warning)
    df = load("btc_1h_features", purpose="analysis")
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sparky.oversight.holdout_guard import HoldoutGuard

logger = logging.getLogger(__name__)

# Directories to search for parquet files (in priority order)
DATA_DIRS = [
    Path("data/features"),
    Path("data/processed"),
    Path("data"),
]

# Map dataset name patterns to assets for holdout enforcement
_ASSET_PATTERNS = {
    "btc": "btc",
    "eth": "eth",
    "sol": "cross_asset",
    "dot": "cross_asset",
    "link": "cross_asset",
    "ada": "cross_asset",
    "avax": "cross_asset",
    "matic": "cross_asset",
}

_guard = HoldoutGuard()


def _detect_asset(dataset: str) -> str:
    """Detect asset from dataset name for holdout enforcement."""
    name_lower = dataset.lower()
    for pattern, asset in _ASSET_PATTERNS.items():
        if pattern in name_lower:
            return asset
    return "cross_asset"  # conservative default


def _find_parquet(dataset: str) -> Optional[Path]:
    """Find a parquet file matching the dataset name.

    Searches DATA_DIRS for files matching:
    - Exact name: {dataset}.parquet
    - With subdirectory: */{dataset}.parquet
    """
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue
        # Exact match
        exact = data_dir / f"{dataset}.parquet"
        if exact.exists():
            return exact
        # Search subdirectories
        for path in sorted(data_dir.rglob(f"{dataset}.parquet")):
            return path
        # Partial match: dataset name is a substring
        for path in sorted(data_dir.rglob("*.parquet")):
            if dataset in path.stem:
                return path
    return None


def list_datasets() -> list[dict[str, str]]:
    """List all available parquet datasets.

    Returns:
        List of dicts with 'name', 'path', and 'asset' keys.
    """
    datasets = []
    seen = set()
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.rglob("*.parquet")):
            if path.name in seen:
                continue
            seen.add(path.name)
            name = path.stem
            datasets.append({
                "name": name,
                "path": str(path),
                "asset": _detect_asset(name),
            })
    return datasets


def load(
    dataset: str,
    purpose: str = "training",
    asset: Optional[str] = None,
) -> pd.DataFrame:
    """Load a dataset with holdout enforcement.

    For purpose="training" or "validation", data is truncated at the
    embargo boundary (holdout start - embargo days). For purpose="analysis",
    full data is returned with a warning logged.

    Args:
        dataset: Dataset name (without .parquet) or full path.
        purpose: "training", "validation", or "analysis".
        asset: Override asset detection (btc, eth, cross_asset).

    Returns:
        DataFrame with DatetimeIndex.

    Raises:
        FileNotFoundError: If dataset parquet file not found.
        ValueError: If purpose is invalid.
    """
    if purpose not in ("training", "validation", "analysis"):
        raise ValueError(f"Invalid purpose '{purpose}'. Use 'training', 'validation', or 'analysis'.")

    # Resolve path
    path = Path(dataset)
    if not path.exists() or not path.suffix:
        path = _find_parquet(dataset)
        if path is None:
            raise FileNotFoundError(
                f"Dataset '{dataset}' not found. Searched: {[str(d) for d in DATA_DIRS]}. "
                f"Available: {[d['name'] for d in list_datasets()]}"
            )

    df = pd.read_parquet(path)
    logger.info(f"[LOADER] Loaded {len(df)} rows from {path}")

    # Ensure DatetimeIndex is UTC
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    elif "timestamp" in df.columns:
        df = df.set_index("timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    # Apply holdout enforcement
    resolved_asset = asset or _detect_asset(dataset)

    if purpose in ("training", "validation"):
        max_date = _guard.get_max_training_date(resolved_asset)
        original_len = len(df)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df[df.index <= max_date]
        if len(df) < original_len:
            removed = original_len - len(df)
            logger.info(
                f"[LOADER] Holdout enforcement: removed {removed} rows after "
                f"{max_date.date()} for {purpose} (asset={resolved_asset})"
            )
    elif purpose == "analysis":
        logger.warning(
            f"[LOADER] Loading FULL dataset for analysis (includes holdout period). "
            f"Do NOT use for training or model evaluation."
        )

    logger.info(f"[LOADER] Returning {len(df)} rows for {purpose} ({resolved_asset})")
    return df
