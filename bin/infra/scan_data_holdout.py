#!/usr/bin/env python3
"""Scan all data/ parquet files for holdout violations.

Checks that no in-sample data directory contains rows past the OOS boundary.
Exit code 0 = clean, 1 = violations found.

Usage:
    .venv/bin/python bin/infra/scan_data_holdout.py
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sparky.data.holdout_split import scan_all  # noqa: E402
from sparky.oversight.holdout_guard import HoldoutGuard  # noqa: E402


def main():
    guard = HoldoutGuard()
    oos_start, _ = guard.get_oos_boundary("cross_asset")
    print(f"Scanning data/ for holdout violations (OOS boundary: {oos_start})...")
    print()

    violations = scan_all(PROJECT_ROOT / "data")

    if not violations:
        print("PASS: No holdout violations found.")
        return 0

    # Group by directory
    from collections import defaultdict

    by_dir = defaultdict(list)
    for v in violations:
        parent = str(Path(v["file"]).parent)
        by_dir[parent].append(v)

    for dir_path, dir_violations in sorted(by_dir.items()):
        print(f"  {dir_path}/")
        for v in dir_violations:
            fname = Path(v["file"]).name
            print(f"    VIOLATION: {fname} — max timestamp {v['max_timestamp']} ({v['holdout_rows']} holdout rows)")
        print()

    print(f"FAIL: {len(violations)} violation(s) in {len(by_dir)} directory(ies)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
