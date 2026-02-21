# Funding Rate Investigation: Coinbase INTX vs Hyperliquid

**Date:** 2026-02-20
**Status:** RESOLVED — early-period artifact + base rate formula difference

## Headline

The -30.8% annualized Coinbase INTX carry is **not real**. It's driven by extreme negative rates in the first 6 months after exchange launch (March-September 2023). After maturity, Coinbase INTX carry is +9.1% annualized — within ~8.6% of Hyperliquid (+17.7%), roughly matching the base rate formula difference.

## Root Cause

Two factors fully explain the divergence:

### 1. Low-Liquidity Launch Artifact (primary)

| Period | Coinbase INTX | Hyperliquid | Gap |
|--------|--------------|-------------|-----|
| Early (Mar-Sep 2023) | **-227.6%** ann | N/A (starts May) | - |
| Mature (Sep 2023+) | **+9.1%** ann | **+17.7%** ann | 8.6% |
| Worst month (Apr 2023) | **-1395.7%** ann | N/A | - |

- 302 five-sigma outliers on Coinbase, ALL negative, ALL in April 2023
- 931 zero-value funding rates (early period low activity)
- Coinbase INTX launched March 2023 with minimal liquidity

### 2. Base Rate Formula Difference (secondary)

Coinbase INTX does **not** include the standard ~0.01% per 8h base interest rate (~10.95% annualized) that most exchanges use. Adjusting for this:

| Metric | Raw | Base-Rate Adjusted |
|--------|-----|-------------------|
| CB INTX (overlap period) | +9.06% | +20.01% |
| Hyperliquid (overlap period) | +16.29% | +16.29% |
| **Residual gap** | **7.2%** | **-3.7%** |

After base rate adjustment, the overlap-period gap is only **-3.7%** — within noise.

## Data Quality Summary

| Metric | Hyperliquid | Coinbase INTX |
|--------|------------|---------------|
| Observations | 23,817 | 25,608 |
| Range | May 2023 – Feb 2026 | Mar 2023 – Feb 2026 |
| Granularity | Hourly (mode 01:00:00.005) | Hourly (clean 01:00:00) |
| NaN | 0 | 0 |
| Zeros | 0 | 931 |
| Hourly correlation | 0.451 | — |
| Sign agreement | 78.7% | — |

## Recommendations for P002

1. **Exclude early Coinbase INTX data**: Use `start_date >= 2023-10-01` for carry signal features. The first 6 months are low-liquidity noise.
2. **Primary carry source: Hyperliquid** — longer maturity window, cleaner data, includes base rate.
3. **Coinbase INTX as secondary**: Institutional-grade data, useful as hedger-sentiment indicator. The structural difference (no base rate, no clamping) makes it a complementary rather than redundant signal.
4. **Base rate normalization**: When comparing cross-exchange funding rates, add `0.0000125/hour` to Coinbase INTX values to normalize against base-rate exchanges.
5. **For carry trading signals**: Use Hyperliquid as the primary input. Coinbase INTX spread vs Hyperliquid could be an independent feature (institutional vs retail sentiment divergence).

## Raw Data

Script: `scripts/funding_rate_investigation.py`
