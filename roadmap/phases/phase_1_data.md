# Phase 1: Clean Data Layer

## Purpose
Fetch, validate, and store BTC + ETH price and on-chain data from free APIs.
Produce a reliable, deduplicated dataset with quality metadata that all downstream
phases can trust without re-checking raw sources.

## Tasks

| Task | Description |
|------|-------------|
| `ccxt_price_fetcher` | OHLCV price data via CCXT (Binance, Coinbase, Kraken) |
| `bgeometrics_fetcher` | On-chain metrics from bgeometrics.com free API |
| `coinmetrics_fetcher` | Community (free) on-chain data from Coin Metrics |
| `blockchain_com_fetcher` | Hash rate, difficulty, mempool stats from blockchain.com |
| `coingecko_fetcher` | Market cap, volume, supply data from CoinGecko free tier |
| `source_selector` | Priority/fallback logic — pick best source per metric, handle outages |
| `data_quality_checker` | Detect gaps, outliers, stale data; produce quality scores per series |
| `storage_layer` | Parquet-based local storage with partitioning and schema versioning |
| `fetch_historical_data` | Backfill script to pull full history within API rate limits |
| `data_validation_report` | Summary report: coverage, quality scores, cross-source consistency |

## Completion Criteria
- At least 2 years of daily BTC + ETH price data stored and validated
- On-chain data from at least 2 independent sources with cross-checks
- Data quality report shows <1% missing values after imputation
- All fetchers handle rate limits, retries, and partial failures gracefully
- Storage layer supports incremental updates without full re-fetch

## Human Gate
**Type: Approve**
Human reviews data validation report and approves dataset quality before any
modeling work begins.
