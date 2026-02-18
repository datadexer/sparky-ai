"""Enforced data access layer with holdout protection.

All data loading for model work MUST go through this module.
Training/validation loads are automatically truncated at the holdout
embargo boundary. OOS evaluation requires explicit authorization.

Data files in data/ are split at the holdout boundary — they only contain
in-sample data. Full data (including OOS) is stored in the vault and
accessible ONLY via purpose="oos_evaluation" with HoldoutGuard authorization.

Usage:
    from sparky.data.loader import load, list_datasets

    # Training (auto-truncated at embargo boundary)
    df = load("btc_1h_features", purpose="training")

    # Analysis (in-sample only, for plotting/exploration)
    df = load("btc_1h_features", purpose="analysis")

    # OOS evaluation (requires authorization from AK or RBM)
    from sparky.oversight.holdout_guard import HoldoutGuard
    guard = HoldoutGuard()
    guard.authorize_oos_evaluation(
        model_name="my_model", approach_family="tree",
        approved_by="human-ak", in_sample_sharpe=1.2,
    )
    df = load("btc_1h_features", purpose="oos_evaluation", oos_guard=guard)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sparky.oversight.holdout_guard import HoldoutGuard, HoldoutViolation

logger = logging.getLogger(__name__)

# Directories to search for parquet files (in priority order)
DATA_DIRS = [
    Path("data/features"),
    Path("data/processed"),
    Path("data"),
]

# OOS vault — contains full data (including holdout period).
# Only accessible via purpose="oos_evaluation" with HoldoutGuard authorization.
VAULT_DIR = Path("data/.oos_vault")

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


def _find_parquet(dataset: str, search_dirs: Optional[list[Path]] = None) -> Optional[Path]:
    """Find a parquet file matching the dataset name.

    Searches the given directories (default DATA_DIRS) for files matching:
    - Exact name: {dataset}.parquet
    - With subdirectory: */{dataset}.parquet
    """
    dirs = search_dirs or DATA_DIRS
    for data_dir in dirs:
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


def _find_vault_parquet(dataset: str) -> Optional[Path]:
    """Find a parquet file in the OOS vault.

    The vault mirrors the original data/ structure under VAULT_DIR.
    """
    if not VAULT_DIR.exists():
        return None
    vault_dirs = [VAULT_DIR / d for d in DATA_DIRS]
    return _find_parquet(dataset, search_dirs=vault_dirs)


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
            datasets.append(
                {
                    "name": name,
                    "path": str(path),
                    "asset": _detect_asset(name),
                }
            )
    return datasets


def load(
    dataset: str,
    purpose: str = "training",
    asset: Optional[str] = None,
    oos_guard: Optional[HoldoutGuard] = None,
) -> pd.DataFrame:
    """Load a dataset with holdout enforcement.

    Data files in data/ contain ONLY in-sample data (split at holdout boundary).
    This means even direct pd.read_parquet() calls on original files are safe.

    Purposes:
        "training"/"validation": In-sample data with holdout truncation enforced.
        "analysis": In-sample data for exploration/plotting.
        "oos_evaluation": Full data from vault. Requires oos_guard with
            authorize_oos_evaluation() called (needs AK or RBM approval).

    Args:
        dataset: Dataset name (without .parquet) or full path.
        purpose: "training", "validation", "analysis", or "oos_evaluation".
        asset: Override asset detection (btc, eth, cross_asset).
        oos_guard: HoldoutGuard with OOS authorization. Required for oos_evaluation.

    Returns:
        DataFrame with DatetimeIndex.

    Raises:
        FileNotFoundError: If dataset parquet file not found.
        ValueError: If purpose is invalid.
        HoldoutViolation: If oos_evaluation requested without authorization.
    """
    valid_purposes = ("training", "validation", "analysis", "oos_evaluation")
    if purpose not in valid_purposes:
        raise ValueError(f"Invalid purpose '{purpose}'. Use one of {valid_purposes}.")

    # OOS evaluation: read from vault with authorization check
    if purpose == "oos_evaluation":
        if oos_guard is None or oos_guard._oos_authorization is None:
            raise HoldoutViolation(
                "OOS evaluation requires explicit authorization. "
                "Call guard.authorize_oos_evaluation(approved_by='human-ak', ...) first, "
                "then pass oos_guard=guard to load()."
            )
        vault_path = _find_vault_parquet(dataset)
        if vault_path is None:
            raise FileNotFoundError(f"No vault data for '{dataset}'. Run scripts/split_holdout_data.py first.")
        df = pd.read_parquet(vault_path)
        logger.info(f"[LOADER] OOS EVALUATION: Loaded {len(df)} rows from vault ({vault_path})")

        # Ensure DatetimeIndex is UTC
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
        elif "timestamp" in df.columns:
            df = df.set_index("timestamp")
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

        logger.info(f"[LOADER] Returning {len(df)} rows for oos_evaluation")
        return df

    # Standard path: read from IS-only data files
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
        logger.info(
            "[LOADER] Loading in-sample data for analysis. "
            "OOS data requires purpose='oos_evaluation' with authorization."
        )

    logger.info(f"[LOADER] Returning {len(df)} rows for {purpose} ({resolved_asset})")
    return df
