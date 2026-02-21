"""Enforced data access layer with holdout protection.

All data loading for model work MUST go through this module.
Training/validation loads are automatically truncated at the holdout
embargo boundary. OOS evaluation requires explicit authorization.

Data files in data/ are split at the holdout boundary — they only contain
in-sample data. Full data (including OOS) is stored in data/holdout/
(OS-level permission restricted) and accessible via purpose="evaluation"
with the SPARKY_OOS_ENABLED=1 env var set.

Usage:
    from sparky.data.loader import load, list_datasets

    # Training (auto-truncated at embargo boundary)
    df = load("btc_1h_features", purpose="training")

    # Analysis (in-sample only, for plotting/exploration)
    df = load("btc_1h_features", purpose="analysis")

    # OOS evaluation (env var + filesystem permissions gate access)
    # SPARKY_OOS_ENABLED=1 must be set, and data/holdout/ must be readable
    df = load("btc_ohlcv_8h", purpose="evaluation")
"""

import logging
import os
import warnings
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

# OOS vault (DEPRECATED — scheduled for removal after data migration to data/holdout/).
# Kept only so _in_vault() can exclude stale vault files from IS lookups.
VAULT_DIR = Path("data/.oos_vault")

# Holdout directory — OS-level permission restricted (owned by sparky-oos user).
# Accessible via purpose="evaluation" when SPARKY_OOS_ENABLED=1 is set.
HOLDOUT_DIR = Path("data/holdout")
_OOS_ENV_VAR = "SPARKY_OOS_ENABLED"

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
    # On-chain metrics (BGeometrics)
    # NOTE: May have <9 columns if rate limit was hit during initial sync.
    # Expected columns: sopr, nupl, mvrv_zscore, realized_price, cdd,
    #   puell_multiple, active_addresses, hash_rate, supply_in_profit
    "btc_onchain_bgeometrics": Path("data/raw/onchain/bgeometrics_combined.parquet"),
    "btc_mvrv_zscore": Path("data/raw/onchain/bgeometrics/mvrv_zscore.parquet"),
    "btc_sopr": Path("data/raw/onchain/bgeometrics/sopr.parquet"),
    "btc_nupl": Path("data/raw/onchain/bgeometrics/nupl.parquet"),
    "btc_realized_price": Path("data/raw/onchain/bgeometrics/realized_price.parquet"),
    "btc_cdd": Path("data/raw/onchain/bgeometrics/cdd.parquet"),
    "btc_puell_multiple": Path("data/raw/onchain/bgeometrics/puell_multiple.parquet"),
    "btc_active_addresses": Path("data/raw/onchain/bgeometrics/active_addresses.parquet"),
    "btc_hash_rate": Path("data/raw/onchain/bgeometrics/hash_rate.parquet"),
    "btc_supply_in_profit": Path("data/raw/onchain/bgeometrics/supply_in_profit.parquet"),
    # On-chain metrics (CoinMetrics Community)
    "btc_onchain_coinmetrics": Path("data/raw/onchain/coinmetrics_btc_daily.parquet"),
    # On-chain metrics (Blockchain.com)
    "btc_onchain_blockchain_com": Path("data/raw/onchain/blockchain_com_btc_daily.parquet"),
    # Funding rates
    "funding_rate_btc_binance": Path("data/raw/funding_rates/btc_binance.parquet"),
    "funding_rate_eth_binance": Path("data/raw/funding_rates/eth_binance.parquet"),
    "funding_rate_btc_hyperliquid": Path("data/raw/funding_rates/btc_hyperliquid.parquet"),
    "funding_rate_eth_hyperliquid": Path("data/raw/funding_rates/eth_hyperliquid.parquet"),
    "funding_rate_btc_coinbase_intl": Path("data/raw/funding_rates/btc_coinbase_intl.parquet"),
    "funding_rate_eth_coinbase_intl": Path("data/raw/funding_rates/eth_coinbase_intl.parquet"),
    # BGeometrics advanced on-chain metrics
    "btc_sth_sopr": Path("data/raw/onchain/bgeometrics/sth_sopr.parquet"),
    "btc_lth_sopr": Path("data/raw/onchain/bgeometrics/lth_sopr.parquet"),
    "btc_sth_mvrv": Path("data/raw/onchain/bgeometrics/sth_mvrv.parquet"),
    "btc_lth_mvrv": Path("data/raw/onchain/bgeometrics/lth_mvrv.parquet"),
    "btc_nupl_sth": Path("data/raw/onchain/bgeometrics/nupl_sth.parquet"),
    "btc_nupl_lth": Path("data/raw/onchain/bgeometrics/nupl_lth.parquet"),
    "btc_exchange_inflow": Path("data/raw/onchain/bgeometrics/exchange_inflow_btc.parquet"),
    "btc_exchange_outflow": Path("data/raw/onchain/bgeometrics/exchange_outflow_btc.parquet"),
    "btc_exchange_netflow": Path("data/raw/onchain/bgeometrics/exchange_netflow_btc.parquet"),
    "btc_exchange_reserve": Path("data/raw/onchain/bgeometrics/exchange_reserve_btc.parquet"),
    "btc_lth_position_change_30d": Path("data/raw/onchain/bgeometrics/lth_position_change_30d.parquet"),
    "btc_open_interest_futures": Path("data/raw/onchain/bgeometrics/open_interest_futures.parquet"),
    "btc_funding_rate_aggregate": Path("data/raw/onchain/bgeometrics/funding_rate_aggregate.parquet"),
    "btc_stablecoin_supply": Path("data/raw/onchain/bgeometrics/stablecoin_supply.parquet"),
    "btc_etf_btc_total": Path("data/raw/onchain/bgeometrics/etf_btc_total.parquet"),
    "btc_vdd_multiple": Path("data/raw/onchain/bgeometrics/vdd_multiple.parquet"),
    "btc_realized_pl_ratio": Path("data/raw/onchain/bgeometrics/realized_pl_ratio.parquet"),
}

# Earliest usable date for datasets with known early-period quality issues.
# Coinbase INTX: first 6 months (Mar-Sep 2023) have extreme negative funding
# rates from low liquidity at exchange launch. See reports/p002/funding_rate_investigation.md.
_START_DATE_OVERRIDES = {
    "funding_rate_btc_coinbase_intl": pd.Timestamp("2023-10-01", tz="UTC"),
    "funding_rate_eth_coinbase_intl": pd.Timestamp("2023-10-01", tz="UTC"),
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
                logger.warning(
                    f"[LOADER] Partial dataset name match for '{dataset}': using '{path}'. "
                    "Use the full alias or exact filename to suppress this warning."
                )
                return path
    return None


def _find_holdout_parquet(dataset: str) -> Optional[Path]:
    """Find a parquet file in the holdout directory.

    Searches HOLDOUT_DIR/{asset}/ for matching parquets, then falls back
    to flat search under HOLDOUT_DIR.
    """
    if not HOLDOUT_DIR.exists():
        return None
    # Direct match: data/holdout/{asset}/{dataset}.parquet
    asset = _detect_asset(dataset)
    asset_dir = HOLDOUT_DIR / asset
    if asset_dir.exists():
        exact = asset_dir / f"{dataset}.parquet"
        if exact.exists():
            return exact
        # Strip asset prefix for matching (e.g. "btc_ohlcv_8h" -> "ohlcv_8h")
        short_name = dataset
        for prefix in _ASSET_PATTERNS:
            if short_name.startswith(prefix + "_"):
                short_name = short_name[len(prefix) + 1 :]
                break
        exact = asset_dir / f"{short_name}.parquet"
        if exact.exists():
            return exact
    # Recursive fallback
    for path in sorted(HOLDOUT_DIR.rglob(f"*{dataset}*.parquet")):
        return path
    return None


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

    Purposes:
        "training"/"validation": In-sample data with holdout truncation enforced.
        "analysis": In-sample data for exploration/plotting.
        "evaluation": OOS data from data/holdout/. Requires SPARKY_OOS_ENABLED=1.
        "oos_evaluation": Deprecated — routes to holdout dir with guard. Will be removed.
    """
    valid_purposes = ("training", "validation", "analysis", "evaluation", "oos_evaluation")
    if purpose not in valid_purposes:
        raise ValueError(f"Invalid purpose '{purpose}'. Use one of {valid_purposes}.")

    # New env-var-gated holdout path
    if purpose == "evaluation":
        if os.environ.get(_OOS_ENV_VAR) != "1":
            raise PermissionError(f"OOS data access denied. Set {_OOS_ENV_VAR}=1 in the environment.")
        holdout_path = _find_holdout_parquet(dataset)
        if holdout_path is None:
            raise FileNotFoundError(
                f"No holdout data for '{dataset}'. "
                f"Searched: {HOLDOUT_DIR}. Run scripts/build_holdout_resampled.py first."
            )
        df = pd.read_parquet(holdout_path)
        logger.info(f"[LOADER] OOS EVALUATION: Loaded {len(df)} rows from holdout ({holdout_path})")
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
        elif "timestamp" in df.columns:
            df = df.set_index("timestamp")
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
        return df

    # Deprecated: purpose="oos_evaluation" — routes to holdout dir with guard check.
    # The legacy data/.oos_vault/ is no longer used. Migrate data to data/holdout/.
    if purpose == "oos_evaluation":
        warnings.warn(
            'purpose="oos_evaluation" is deprecated and will be removed. '
            f'Use purpose="evaluation" with {_OOS_ENV_VAR}=1 instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        if oos_guard is None or oos_guard._oos_authorization is None:
            raise HoldoutViolation(
                "OOS evaluation requires explicit authorization. "
                "Call guard.authorize_oos_evaluation(approved_by='human-ak', ...) first, "
                "then pass oos_guard=guard to load()."
            )
        holdout_path = _find_holdout_parquet(dataset)
        if holdout_path is None:
            raise FileNotFoundError(
                f"No holdout data for '{dataset}'. Searched: {HOLDOUT_DIR}. "
                "The legacy data/.oos_vault/ is no longer used — migrate data to data/holdout/."
            )
        df = pd.read_parquet(holdout_path)
        logger.info(f"[LOADER] OOS EVALUATION: Loaded {len(df)} rows from {holdout_path}")

        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
        elif "timestamp" in df.columns:
            df = df.set_index("timestamp")
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

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

    # Apply data quality start-date filters
    if dataset in _START_DATE_OVERRIDES and isinstance(df.index, pd.DatetimeIndex):
        min_date = _START_DATE_OVERRIDES[dataset]
        before = len(df)
        df = df[df.index >= min_date]
        if len(df) < before:
            logger.info(
                f"[LOADER] Data quality filter: excluded {before - len(df)} early rows "
                f"before {min_date.date()} for {dataset}"
            )

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
            "OOS data requires purpose='evaluation' with SPARKY_OOS_ENABLED=1."
        )

    logger.info(f"[LOADER] Returning {len(df)} rows for {purpose} ({resolved_asset})")
    return df
