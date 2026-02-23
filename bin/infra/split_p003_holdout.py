#!/usr/bin/env python3
"""Split P003 data at the holdout boundary.

Processes binance_perps/, funding_rates/, dvol/ directories.
Directories not listed in SPLIT_DIRS (e.g. unlocks/) are not processed.

Usage:
    .venv/bin/python bin/infra/split_p003_holdout.py
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sparky.data.holdout_split import scan_all, split_directory  # noqa: E402
from sparky.oversight.holdout_guard import HoldoutGuard  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")

P003_DIR = PROJECT_ROOT / "data" / "p003"
HOLDOUT_P003 = PROJECT_ROOT / "data" / "holdout" / "p003"

# Directories to split and their timestamp columns
SPLIT_DIRS = {
    "binance_perps": "timestamp",
    "funding_rates": "timestamp",
    "dvol": "timestamp",
}


def main():
    guard = HoldoutGuard()
    oos_start, _ = guard.get_oos_boundary("cross_asset")
    print(f"Splitting P003 data at holdout boundary (OOS: {oos_start})")
    print(f"Source: {P003_DIR}")
    print(f"Holdout dest: {HOLDOUT_P003}")
    print()

    total_is = 0
    total_holdout = 0
    total_files = 0

    for subdir, ts_col in SPLIT_DIRS.items():
        src = P003_DIR / subdir
        if not src.exists():
            print(f"  SKIP: {subdir}/ does not exist")
            continue

        dest = HOLDOUT_P003 / subdir
        print(f"--- {subdir}/ ---")
        results = split_directory(src, dest, timestamp_col=ts_col, asset="cross_asset")

        for r in results:
            if not r["skipped"]:
                total_is += r["is_rows"]
                total_holdout += r["holdout_rows"]
                total_files += 1

        print()

    print("=" * 50)
    print(f"SUMMARY: {total_files} files split")
    print(f"  IS rows:      {total_is:,}")
    print(f"  Holdout rows:  {total_holdout:,}")
    print()

    # Verify no violations remain
    violations = [v for v in scan_all(PROJECT_ROOT / "data") if "p003" in v["file"]]
    if violations:
        print(f"ERROR: {len(violations)} P003 violations remain after split!")
        for v in violations:
            print(f"  {v['file']}: max={v['max_timestamp']}")
        return 1

    print("Verification: 0 P003 holdout violations remain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
