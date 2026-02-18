#!/usr/bin/env python3
"""Split parquet data files at the holdout boundary.

Reads the OOS boundary from configs/holdout_policy.yaml and splits every
parquet file in data/ into:
  - In-sample only (truncated at embargo boundary) → stays in original location
  - Full data (including OOS) → moved to data/.oos_vault/

This ensures that even direct file access (pd.read_parquet, polars, pyarrow)
only returns in-sample data. The loader's purpose="analysis" reads from the vault.

Usage:
    python scripts/split_holdout_data.py          # split all
    python scripts/split_holdout_data.py --dry-run # show what would happen
    python scripts/split_holdout_data.py --restore # restore full data from vault
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sparky.oversight.holdout_guard import HoldoutGuard

VAULT_DIR = Path("data/.oos_vault")
DATA_DIRS = [Path("data"), Path("data/processed"), Path("data/raw")]
# Skip these directories when scanning
SKIP_DIRS = {".oos_vault"}


def get_embargo_date() -> pd.Timestamp:
    guard = HoldoutGuard()
    # Use the earliest embargo boundary across all assets
    dates = []
    for asset in ["btc", "eth", "cross_asset"]:
        dates.append(guard.get_max_training_date(asset))
    return min(dates)


def find_all_parquets() -> list[Path]:
    """Find all parquet files under data/, excluding the vault."""
    data_root = Path("data")
    results = []
    for p in sorted(data_root.rglob("*.parquet")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        results.append(p)
    return results


def has_time_index(df: pd.DataFrame) -> bool:
    """Check if DataFrame has a DatetimeIndex or timestamp column."""
    if isinstance(df.index, pd.DatetimeIndex):
        return True
    if "timestamp" in df.columns:
        return True
    return False


def get_max_date(df: pd.DataFrame) -> pd.Timestamp:
    """Get the max date from a DataFrame."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.max()
    if "timestamp" in df.columns:
        return pd.Timestamp(df["timestamp"].max())
    raise ValueError("No time column found")


def truncate_at(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Truncate a DataFrame at a cutoff date."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            cutoff = cutoff.tz_localize(None)
        return df[idx <= cutoff]
    if "timestamp" in df.columns:
        col = pd.to_datetime(df["timestamp"])
        if col.dt.tz is None:
            cutoff = cutoff.tz_localize(None)
        return df[col <= cutoff]
    return df


def split_file(parquet_path: Path, embargo_date: pd.Timestamp, dry_run: bool = False) -> dict:
    """Split a single parquet file. Returns info dict."""
    df = pd.read_parquet(parquet_path)

    if not has_time_index(df):
        return {"path": str(parquet_path), "action": "skip", "reason": "no time index"}

    max_date = get_max_date(df)
    # Normalize timezone for comparison
    if hasattr(max_date, "tz") and max_date.tz is None:
        compare_embargo = embargo_date.tz_localize(None)
    else:
        compare_embargo = embargo_date

    if max_date <= compare_embargo:
        return {
            "path": str(parquet_path),
            "action": "skip",
            "reason": f"already IS-only (max={max_date.date()})",
        }

    # This file extends past the embargo — needs splitting
    original_len = len(df)
    truncated = truncate_at(df, embargo_date)
    removed = original_len - len(truncated)

    # Vault path mirrors the original structure
    vault_path = VAULT_DIR / parquet_path
    info = {
        "path": str(parquet_path),
        "action": "split",
        "original_rows": original_len,
        "truncated_rows": len(truncated),
        "removed_rows": removed,
        "max_date_before": str(max_date.date()),
        "max_date_after": str(get_max_date(truncated).date()) if len(truncated) > 0 else "empty",
        "vault_path": str(vault_path),
    }

    if not dry_run:
        # Save full data to vault
        vault_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parquet_path, vault_path)

        # Overwrite original with truncated data
        truncated.to_parquet(parquet_path)

    return info


def restore_from_vault():
    """Restore full data files from the vault."""
    if not VAULT_DIR.exists():
        print("No vault directory found. Nothing to restore.")
        return

    count = 0
    for vault_file in sorted(VAULT_DIR.rglob("*.parquet")):
        # Reconstruct original path
        relative = vault_file.relative_to(VAULT_DIR)
        original = Path(relative)
        print(f"  Restoring {original}")
        shutil.copy2(vault_file, original)
        count += 1

    print(f"Restored {count} files from vault.")


def main():
    parser = argparse.ArgumentParser(description="Split parquet data at holdout boundary")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--restore", action="store_true", help="Restore full data from vault")
    args = parser.parse_args()

    if args.restore:
        restore_from_vault()
        return

    embargo_date = get_embargo_date()
    print(f"Embargo boundary: {embargo_date.date()}")
    print(f"Vault directory: {VAULT_DIR}")
    print()

    parquets = find_all_parquets()
    print(f"Found {len(parquets)} parquet files\n")

    split_count = 0
    skip_count = 0
    for pq in parquets:
        info = split_file(pq, embargo_date, dry_run=args.dry_run)
        if info["action"] == "split":
            split_count += 1
            prefix = "[DRY RUN] " if args.dry_run else ""
            print(
                f"  {prefix}SPLIT {info['path']}: "
                f"{info['original_rows']}→{info['truncated_rows']} rows "
                f"(removed {info['removed_rows']} after {embargo_date.date()})"
            )
        else:
            skip_count += 1

    print(f"\nSplit: {split_count}, Skipped: {skip_count}")
    if args.dry_run and split_count > 0:
        print("Run without --dry-run to apply.")


if __name__ == "__main__":
    main()
