"""Import BGeometrics CSVs from bgeometrics_out/ into DataStore."""

import pandas as pd
from pathlib import Path
from sparky.data.storage import DataStore

store = DataStore()
indir = Path("bgeometrics_out")
if not indir.exists():
    print(f"{indir} not found")
    exit(1)

for f in sorted(indir.glob("*.csv")):
    df = pd.read_csv(f, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
    df = df.set_index("date")
    drop = [c for c in df.columns if "unixts" in c.lower()]
    if drop:
        df = df.drop(columns=drop)
    metric = f.stem
    store.append(
        df,
        f"data/raw/onchain/bgeometrics/{metric}.parquet",
        metadata={"source": "bgeometrics", "metric": metric, "asset": "btc"},
    )
    print(f"{metric}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
