#!/usr/bin/env python3
"""Download daily OHLCV for top 50 USDT-M perpetual futures from data.binance.vision."""

import argparse
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"

TOP_50_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "LTCUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "FILUSDT",
    "APTUSDT",
    "NEARUSDT",
    "ARBUSDT",
    "OPUSDT",
    "MKRUSDT",
    "AAVEUSDT",
    "SUSHIUSDT",
    "COMPUSDT",
    "SNXUSDT",
    "CRVUSDT",
    "LDOUSDT",
    "INJUSDT",
    "SUIUSDT",
    "SEIUSDT",
    "TIAUSDT",
    "JUPUSDT",
    "WLDUSDT",
    "PYTHUSDT",
    "STXUSDT",
    "IMXUSDT",
    "MANAUSDT",
    "SANDUSDT",
    "AXSUSDT",
    "GALAUSDT",
    "ENAUSDT",
    "FTMUSDT",
    "ALGOUSDT",
    "TRXUSDT",
    "XLMUSDT",
    "VETUSDT",
    "ICPUSDT",
    "HBARUSDT",
    "RNDRUSDT",
    "FETUSDT",
]

CSV_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

KEEP_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "sparky-ai/binance-perp-download"})


def generate_months(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def download_with_retry(url: str, max_retries: int = 3) -> bytes | None:
    for attempt in range(max_retries):
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"    Failed after {max_retries} retries: {e}")
                return None


def parse_csv_bytes(csv_bytes: bytes) -> pl.DataFrame:
    # Detect header: older months have no header (first char is a digit)
    first_char = csv_bytes.lstrip()[:1]
    has_header = not first_char.isdigit()

    if has_header:
        df = pl.read_csv(
            csv_bytes,
            has_header=True,
            schema_overrides={c: pl.Float64 for c in CSV_COLUMNS if c != "open_time"} | {"open_time": pl.Int64},
        )
    else:
        df = pl.read_csv(
            csv_bytes,
            has_header=False,
            new_columns=CSV_COLUMNS,
            schema_overrides={c: pl.Float64 for c in CSV_COLUMNS if c != "open_time"} | {"open_time": pl.Int64},
        )

    # Normalize column names (some files might have slight variations)
    if "open_time" not in df.columns:
        df = df.rename({df.columns[0]: "open_time"})

    return df


def download_symbol(symbol: str, start: tuple, end: tuple) -> pl.DataFrame | None:
    all_rows = []
    months_ok = 0
    months_404 = 0

    for year, month in generate_months(*start, *end):
        url = f"{BASE_URL}/{symbol}/1d/{symbol}-1d-{year}-{month:02d}.zip"
        data = download_with_retry(url)
        if data is None:
            months_404 += 1
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    csv_bytes = f.read()
        except (zipfile.BadZipFile, IndexError) as e:
            print(f"    Bad zip for {symbol} {year}-{month:02d}: {e}")
            continue

        try:
            df = parse_csv_bytes(csv_bytes)
        except Exception as e:
            print(f"    Parse error for {symbol} {year}-{month:02d}: {e}")
            continue

        all_rows.append(df.select(["open_time", "open", "high", "low", "close", "volume", "quote_volume"]))
        months_ok += 1

    if not all_rows:
        return None

    combined = pl.concat(all_rows)
    result = (
        combined.with_columns(
            pl.from_epoch(pl.col("open_time"), time_unit="ms").dt.replace_time_zone("UTC").alias("timestamp")
        )
        .select(KEEP_COLUMNS)
        .sort("timestamp")
        .unique(subset=["timestamp"], keep="last")
    )

    print(f"  {symbol}: {len(result)} rows, {months_ok} months ok, {months_404} months 404")
    return result


def main():
    parser = argparse.ArgumentParser(description="Download Binance USDT-M perp daily klines")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbol list (default: top 50)")
    parser.add_argument(
        "--output-dir", type=str, default="data/p003/binance_perps", help="Output directory for parquet files"
    )
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else TOP_50_SYMBOLS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = (2020, 1)
    end = (2026, 2)

    print(f"Downloading {len(symbols)} symbols, {start[0]}-{start[1]:02d} to {end[0]}-{end[1]:02d}")
    print(f"Output: {output_dir.resolve()}\n")

    summary = {}
    skipped = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}")
        df = download_symbol(symbol, start, end)

        if df is None or len(df) == 0:
            print(f"  WARNING: {symbol} has no data, skipping")
            skipped.append(symbol)
            continue

        out_path = output_dir / f"{symbol}.parquet"
        df.write_parquet(out_path)

        ts_min = df["timestamp"].min().strftime("%Y-%m-%d")
        ts_max = df["timestamp"].max().strftime("%Y-%m-%d")
        summary[symbol] = {"rows": len(df), "start": ts_min, "end": ts_max}

    # Print and save summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(summary)} symbols downloaded, {len(skipped)} skipped")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    print(f"{'=' * 60}")

    for sym, info in sorted(summary.items()):
        print(f"  {sym:12s}  {info['rows']:5d} rows  {info['start']} -> {info['end']}")

    summary_path = output_dir / "download_summary.json"
    summary_data = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "n_symbols": len(summary),
        "skipped": skipped,
        "symbols": summary,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2))
    print(f"\nSummary written to {summary_path}")

    # Warn if holdout data present (operator must run split script)
    try:
        from sparky.data.holdout_split import validate_directory

        validate_directory(output_dir, timestamp_col="timestamp", asset="cross_asset")
        print("\nHoldout check: PASS (no post-OOS data)")
    except Exception as e:
        print(f"\nWARNING: {e}")
        print("Run bin/infra/setup_p003_holdout.sh to split holdout data before research use.")


if __name__ == "__main__":
    main()
