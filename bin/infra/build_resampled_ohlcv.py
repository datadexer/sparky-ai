"""Build resampled OHLCV files (2h/4h/8h) from hourly data.

Run once from main checkout (needs data access):
    .venv/bin/python scripts/infra/build_resampled_ohlcv.py

Output files go to data/processed/ (gitignored).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from sparky.data.loader import load

RESAMPLE_MAP = {"2h": "2h", "4h": "4h", "8h": "8h"}
ASSETS = {
    "btc": "btc_ohlcv_hourly",
    "eth": "eth_ohlcv_hourly",
}
OUT_DIR = ROOT / "data" / "processed"


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(freq).agg(agg).dropna(subset=["close"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for asset, dataset in ASSETS.items():
        print(f"Loading {dataset}...")
        df = load(dataset, purpose="training")
        # Ensure OHLCV columns exist
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[cols]
        for label, freq in RESAMPLE_MAP.items():
            out = resample_ohlcv(df, freq)
            fname = f"{asset}_ohlcv_{label}.parquet"
            path = OUT_DIR / fname
            out.to_parquet(path)
            print(f"  {fname}: {len(out)} rows ({out.index.min()} → {out.index.max()})")
    print("Done.")


if __name__ == "__main__":
    main()
