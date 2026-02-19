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

_DATASET_ALIASES = {
    "eth_ohlcv_hourly": Path("data/raw/eth/ohlcv_hourly.parquet"),
    "eth_ohlcv_daily": Path("data/raw/eth/ohlcv.parquet"),
    "btc_ohlcv_hourly": Path("data/raw/btc/ohlcv_hourly_max_coverage.parquet"),
    "btc_ohlcv_daily": Path("data/raw/btc/ohlcv.parquet"),
    # Pre-computed resampled OHLCV (built by scripts/infra/build_resampled_ohlcv.py)
    "btc_ohlcv_2h": Path("data/processed/btc_ohlcv_2h.parquet"),
    "btc_ohlcv_4h": Path("data/processed/btc_ohlcv_4h.parquet"),
    "btc_ohlcv_8h": Path("data/processed/btc_ohlcv_8h.parquet"),
    "eth_ohlcv_2h": Path("data/processed/eth_ohlcv_2h.parquet"),
    "eth_ohlcv_4h": Path("data/processed/eth_ohlcv_4h.parquet"),
    "eth_ohlcv_8h": Path("data/processed/eth_ohlcv_8h.parquet"),
}

_guard = HoldoutGuard()


def _detect_asset(dataset: str) -> str:
    """Detect asset from dataset name for holdout enforcement."""
    name_lower = dataset.lower()
    for pattern, asset in _ASSET_PATTERNS.items():
        if pattern in name_lower:
            return asset
    return "cross_asset"  # conservative default


def _in_vault(path: Path) -> bool:
    """Check if a path is inside the OOS vault."""
    try:
        path.resolve().relative_to(VAULT_DIR.resolve())
        return True
    except ValueError:
        return False


def _find_parquet(
    dataset: str, search_dirs: Optional[list[Path]] = None, *, allow_vault: bool = False
) -> Optional[Path]:
    """Find a parquet file matching the dataset name.

    Searches the given directories (default DATA_DIRS) for files matching:
    - Exact name: {dataset}.parquet
    - With subdirectory: */{dataset}.parquet

    Vault paths are excluded unless allow_vault=True (used by _find_vault_parquet).
    """
    # Check aliases first (only for non-vault lookups with default dirs)
    if not allow_vault and search_dirs is None and dataset in _DATASET_ALIASES:
        alias_path = _DATASET_ALIASES[dataset]
        if alias_path.exists():
            return alias_path

    dirs = search_dirs or DATA_DIRS
    for data_dir in dirs:
        if not data_dir.exists():
            continue
        # Exact match
        exact = data_dir / f"{dataset}.parquet"
        if exact.exists() and (allow_vault or not _in_vault(exact)):
            return exact
        # Search subdirectories
        for path in sorted(data_dir.rglob(f"{dataset}.parquet")):
            if allow_vault or not _in_vault(path):
                return path
        # Partial match: dataset name is a substring
        for path in sorted(data_dir.rglob("*.parquet")):
            if dataset in path.stem and (allow_vault or not _in_vault(path)):
                return path
    return None


def _find_vault_parquet(dataset: str) -> Optional[Path]:
    """Find a parquet file in the OOS vault.

    The vault mirrors the original data/ structure under VAULT_DIR.
    """
    if not VAULT_DIR.exists():
        return None
    vault_dirs = [VAULT_DIR / d for d in DATA_DIRS]
    return _find_parquet(dataset, search_dirs=vault_dirs, allow_vault=True)


def list_datasets() -> list[dict[str, str]]:
    """List all available parquet datasets.

    Returns:
        List of dicts with 'name', 'path', and 'asset' keys.
    """
    datasets = []
    seen = set()
    # Registered aliases first
    for alias_name, alias_path in _DATASET_ALIASES.items():
        if alias_path.exists() and alias_name not in seen:
            seen.add(alias_name)
            datasets.append({"name": alias_name, "path": str(alias_path), "asset": _detect_asset(alias_name)})
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.rglob("*.parquet")):
            if _in_vault(path) or path.name in seen:
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
