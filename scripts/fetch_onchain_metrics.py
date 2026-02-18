#!/usr/bin/env python3
"""
Fetch BTC on-chain metrics from free APIs and save as parquet files.

Data sources:
1. CoinMetrics Community API (free, no auth)
2. Blockchain.com API (free, no auth)

Output:
- data/raw/onchain/coinmetrics_btc_daily.parquet
- data/raw/onchain/blockchain_com_btc_daily.parquet
"""

import time
from pathlib import Path

import pandas as pd
import requests


def fetch_coinmetrics_data() -> pd.DataFrame:
    """
    Fetch BTC metrics from CoinMetrics Community API.

    Metrics (community/free tier):
    - CapMrktCurUSD: Market capitalization (current)
    - CapMVRVCur: MVRV ratio (market cap / realized cap)
    - SplyCur: Current supply
    - HashRate: Hash rate
    - AdrActCnt: Active addresses count
    - AdrBalCnt: Address balance count
    - TxCnt: Transaction count
    - TxTfrCnt: Transfer transaction count
    - BlkCnt: Block count
    - FeeTotNtv: Total fees (native units)
    - FlowInExNtv: Flow into exchanges (native)
    - FlowOutExNtv: Flow out of exchanges (native)
    - PriceUSD: Price in USD

    Returns:
        DataFrame with UTC DatetimeIndex and metric columns
    """
    print("\n[CoinMetrics] Fetching BTC on-chain metrics...")

    base_url = "https://community-api.coinmetrics.io/v4"
    endpoint = "/timeseries/asset-metrics"

    # Using only community (free) tier metrics
    params = {
        "assets": "btc",
        "metrics": "CapMrktCurUSD,CapMVRVCur,SplyCur,HashRate,AdrActCnt,AdrBalCnt,TxCnt,TxTfrCnt,BlkCnt,FeeTotNtv,FlowInExNtv,FlowOutExNtv,PriceUSD",
        "start_time": "2017-01-01",
        "end_time": "2026-02-16",
        "frequency": "1d",
        "page_size": 10000,
    }

    url = f"{base_url}{endpoint}"
    print(f"[CoinMetrics] Request URL: {url}")
    print(f"[CoinMetrics] Parameters: {params}")

    all_records = []
    next_page_token = None
    page_num = 1

    while True:
        if next_page_token:
            params["next_page_token"] = next_page_token

        print(f"[CoinMetrics] Fetching page {page_num}...")
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()

        if "data" not in data:
            raise ValueError(f"Unexpected response format: {data.keys()}")

        records = data["data"]
        all_records.extend(records)
        print(f"[CoinMetrics] Page {page_num}: received {len(records)} records (total: {len(all_records)})")

        # Check for next page
        next_page_token = data.get("next_page_token")
        if not next_page_token:
            break

        page_num += 1
        time.sleep(1)  # Rate limiting

    records = all_records
    print(f"[CoinMetrics] Total records fetched: {len(records)}")

    # Parse into DataFrame
    rows = []
    for record in records:
        row = {"time": record["time"]}
        # Metrics are in a nested dict
        for metric, value in record.items():
            if metric != "time":
                row[metric] = value
        rows.append(row)

    df = pd.DataFrame(rows)

    # Convert time to datetime and set as index
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    df = df.sort_index()

    # Drop unnecessary columns (status columns and asset)
    cols_to_drop = [
        col for col in df.columns if col.endswith("-status") or col.endswith("-status-time") or col == "asset"
    ]
    df = df.drop(columns=cols_to_drop)

    # Convert all columns to numeric (they come as strings)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[CoinMetrics] DataFrame shape: {df.shape}")
    print(f"[CoinMetrics] Date range: {df.index.min()} to {df.index.max()}")
    print(f"[CoinMetrics] Columns: {list(df.columns)}")

    return df


def fetch_blockchain_com_data() -> pd.DataFrame:
    """
    Fetch BTC metrics from Blockchain.com API.

    Metrics:
    - hash_rate: Network hash rate
    - unique_addresses: Number of unique addresses
    - n_transactions: Number of transactions

    Returns:
        DataFrame with UTC DatetimeIndex and metric columns
    """
    print("\n[Blockchain.com] Fetching BTC on-chain metrics...")

    base_url = "https://api.blockchain.info/charts"

    metrics = {"hash_rate": "hash-rate", "unique_addresses": "n-unique-addresses", "n_transactions": "n-transactions"}

    dfs = []

    for metric_name, chart_name in metrics.items():
        print(f"[Blockchain.com] Fetching {metric_name}...")

        url = f"{base_url}/{chart_name}"
        params = {"timespan": "9years", "format": "json", "sampled": "false"}

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()

        if "values" not in data:
            raise ValueError(f"Unexpected response format for {metric_name}: {data.keys()}")

        values = data["values"]
        print(f"[Blockchain.com] Received {len(values)} records for {metric_name}")

        # Parse into DataFrame
        df_metric = pd.DataFrame(values)
        df_metric["time"] = pd.to_datetime(df_metric["x"], unit="s", utc=True)
        df_metric = df_metric.set_index("time")
        df_metric = df_metric.rename(columns={"y": metric_name})
        df_metric = df_metric[[metric_name]]  # Drop 'x' column

        dfs.append(df_metric)

        # Rate limit: 1 request per second
        time.sleep(1)

    # Merge all metrics
    df = pd.concat(dfs, axis=1)
    df = df.sort_index()

    print(f"[Blockchain.com] DataFrame shape: {df.shape}")
    print(f"[Blockchain.com] Date range: {df.index.min()} to {df.index.max()}")
    print(f"[Blockchain.com] Columns: {list(df.columns)}")

    return df


def save_and_summarize(df: pd.DataFrame, output_path: Path, source: str) -> None:
    """
    Save DataFrame to parquet and print summary statistics.

    Args:
        df: DataFrame to save
        output_path: Path to save parquet file
        source: Data source name for logging
    """
    # Save to parquet
    df.to_parquet(output_path)
    print(f"\n[{source}] Saved to: {output_path}")

    # Summary statistics
    print(f"\n[{source}] Summary:")
    print(f"  - Number of metrics: {len(df.columns)}")
    print(f"  - Number of rows: {len(df)}")
    print(f"  - Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  - Index frequency: {df.index.inferred_freq or 'irregular'}")

    print(f"\n[{source}] NaN counts by column:")
    nan_counts = df.isna().sum()
    for col, count in nan_counts.items():
        pct = 100 * count / len(df)
        print(f"  - {col}: {count} ({pct:.2f}%)")

    print(f"\n[{source}] First 5 rows:")
    print(df.head())

    print(f"\n[{source}] Last 5 rows:")
    print(df.tail())


def main():
    """Main execution function."""
    print("=" * 80)
    print("BTC On-Chain Metrics Fetcher")
    print("=" * 80)

    output_dir = Path("/home/akamath/sparky-ai/data/raw/onchain")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Fetch CoinMetrics data
        df_coinmetrics = fetch_coinmetrics_data()
        coinmetrics_path = output_dir / "coinmetrics_btc_daily.parquet"
        save_and_summarize(df_coinmetrics, coinmetrics_path, "CoinMetrics")

        # Wait before next API
        time.sleep(1)

        # Fetch Blockchain.com data
        df_blockchain = fetch_blockchain_com_data()
        blockchain_path = output_dir / "blockchain_com_btc_daily.parquet"
        save_and_summarize(df_blockchain, blockchain_path, "Blockchain.com")

        print("\n" + "=" * 80)
        print("SUCCESS: All on-chain metrics fetched and saved")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] Failed to fetch on-chain metrics: {e}")
        raise


if __name__ == "__main__":
    main()
