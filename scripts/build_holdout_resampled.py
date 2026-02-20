#!/usr/bin/env python3
"""Build resampled OHLCV (8h) from hourly holdout data.

Run as sparky-oos user after migrating data to data/holdout/:
    sudo -u sparky-oos .venv/bin/python scripts/build_holdout_resampled.py

Reads hourly parquets from data/holdout/{asset}/ and writes resampled
versions (ohlcv_8h.parquet) alongside them.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

HOLDOUT_DIR = ROOT / "data" / "holdout"
ASSETS = ["btc", "eth"]
HOURLY_NAMES = ["ohlcv_hourly.parquet", "ohlcv_hourly_max_coverage.parquet"]
RESAMPLE_FREQ = "8h"


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(freq).agg(agg).dropna(subset=["close"])


def preflight():
    """Verify the venv and holdout dir are accessible to current user."""
    import importlib

    for mod in ["pandas", "pyarrow"]:
        try:
            importlib.import_module(mod)
        except ImportError:
            print(
                f"ERROR: Cannot import '{mod}'. If running as sparky-oos, verify "
                f".venv/lib/ is world-readable: chmod -R o+rX .venv/lib/"
            )
            sys.exit(1)

    if not HOLDOUT_DIR.exists():
        print(f"ERROR: {HOLDOUT_DIR} does not exist. Run the data migration first.")
        sys.exit(1)

    # Check we can read at least one asset dir
    readable = False
    for asset in ASSETS:
        if (HOLDOUT_DIR / asset).exists():
            readable = True
            break
    if not readable:
        print(
            f"ERROR: No asset directories found in {HOLDOUT_DIR}. "
            f"Expected: {', '.join(str(HOLDOUT_DIR / a) for a in ASSETS)}"
        )
        sys.exit(1)


def main():
    preflight()
    for asset in ASSETS:
        asset_dir = HOLDOUT_DIR / asset
        if not asset_dir.exists():
            print(f"  SKIP {asset}: {asset_dir} not found")
            continue

        # Find hourly source
        src = None
        for name in HOURLY_NAMES:
            candidate = asset_dir / name
            if candidate.exists():
                src = candidate
                break
        if src is None:
            print(f"  SKIP {asset}: no hourly parquet in {asset_dir}")
            continue

        df = pd.read_parquet(src)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[cols]

        out = resample_ohlcv(df, RESAMPLE_FREQ)
        dest = asset_dir / "ohlcv_8h.parquet"
        out.to_parquet(dest)
        print(f"  {asset}/ohlcv_8h.parquet: {len(out)} rows ({out.index.min()} -> {out.index.max()})")

    print("Done.")


if __name__ == "__main__":
    main()
