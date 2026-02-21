"""Coinbase INTX vs Hyperliquid funding rate divergence investigation."""

import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sparky.data.loader import load

# ============================================================
# 1. Raw Data Sanity Check
# ============================================================
print("=" * 60)
print("1. RAW DATA SANITY CHECK")
print("=" * 60)

hl = load("funding_rate_btc_hyperliquid", purpose="analysis")
cb = load("funding_rate_btc_coinbase_intl", purpose="analysis")

for name, df in [("Hyperliquid", hl), ("Coinbase INTX", cb)]:
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index range: {df.index.min()} to {df.index.max()}")
    print(f"funding_rate dtype: {df['funding_rate'].dtype}")
    print(f"NaN count: {df['funding_rate'].isna().sum()}")
    print(f"Zero count: {(df['funding_rate'] == 0).sum()}")
    print(f"Inf count: {np.isinf(df['funding_rate']).sum()}")
    print("\nSample values (first 5):")
    print(df["funding_rate"].head())
    print("\nDescriptive stats:")
    print(df["funding_rate"].describe())

# ============================================================
# 2. Distribution Analysis
# ============================================================
print("\n" + "=" * 60)
print("2. DISTRIBUTION ANALYSIS")
print("=" * 60)

for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    hl_q = hl["funding_rate"].quantile(pct / 100)
    cb_q = cb["funding_rate"].quantile(pct / 100)
    print(f"P{pct:2d}: HL={hl_q:+.6f}  CB={cb_q:+.6f}  spread={hl_q - cb_q:+.6f}")

for name, df in [("Hyperliquid", hl), ("Coinbase INTX", cb)]:
    outliers_5s = df[df["funding_rate"].abs() > df["funding_rate"].std() * 5]
    print(f"\n{name} 5-sigma outliers: {len(outliers_5s)} ({len(outliers_5s) / len(df) * 100:.2f}%)")
    if len(outliers_5s) > 0:
        print(f"  Range: [{outliers_5s['funding_rate'].min():.6f}, {outliers_5s['funding_rate'].max():.6f}]")
        print("  Top 5 by abs value:")
        top5 = outliers_5s.nlargest(5, "funding_rate", keep="first")
        for idx, row in top5.iterrows():
            print(f"    {idx}: {row['funding_rate']:+.6f}")

# ============================================================
# 3. Unit / Scaling Verification
# ============================================================
print("\n" + "=" * 60)
print("3. UNIT / SCALING VERIFICATION")
print("=" * 60)

for name, df in [("Hyperliquid", hl), ("Coinbase INTX", cb)]:
    mean_raw = df["funding_rate"].mean()
    ann_decimal = mean_raw * 24 * 365
    ann_percent = mean_raw / 100 * 24 * 365
    print(f"\n{name}:")
    print(f"  Mean raw value: {mean_raw:.8f}")
    print(f"  If decimal (0.0001 = 0.01%): annualized = {ann_decimal:.2%}")
    print(f"  If percent (0.01 = 0.01%):   annualized = {ann_percent:.4%}")

# Detect time spacing
for name, df in [("Hyperliquid", hl), ("Coinbase INTX", cb)]:
    if len(df) > 1:
        diffs = df.index.to_series().diff().dropna()
        print(f"\n{name} time spacing:")
        print(f"  Mode: {diffs.mode().iloc[0]}")
        print(f"  Min:  {diffs.min()}")
        print(f"  Max:  {diffs.max()}")
        print(f"  Median: {diffs.median()}")

# ============================================================
# 4. Temporal Decomposition
# ============================================================
print("\n" + "=" * 60)
print("4. TEMPORAL DECOMPOSITION")
print("=" * 60)

hl_monthly = hl["funding_rate"].resample("ME").mean() * 24 * 365
cb_monthly = cb["funding_rate"].resample("ME").mean() * 24 * 365

comparison = pd.DataFrame(
    {
        "Hyperliquid": hl_monthly,
        "Coinbase_INTX": cb_monthly,
    }
).dropna()
comparison["Spread"] = comparison["Hyperliquid"] - comparison["Coinbase_INTX"]

print("\nMonthly Annualized Funding Rates:")
for idx, row in comparison.iterrows():
    print(
        f"  {idx.strftime('%Y-%m')}: HL={row['Hyperliquid']:+7.1%}  CB={row['Coinbase_INTX']:+7.1%}  spread={row['Spread']:+7.1%}"
    )

print(f"\nCoinbase months negative: {(cb_monthly < 0).sum()} / {len(cb_monthly)}")
print(f"Coinbase months < -50% ann: {(cb_monthly < -0.50).sum()}")
if len(cb_monthly) > 0:
    print(f"Worst month: {cb_monthly.idxmin().strftime('%Y-%m')} = {cb_monthly.min():.1%}")
    print(f"Best month: {cb_monthly.idxmax().strftime('%Y-%m')} = {cb_monthly.max():.1%}")

# By quarter
hl_quarterly = hl["funding_rate"].resample("QE").mean() * 24 * 365
cb_quarterly = cb["funding_rate"].resample("QE").mean() * 24 * 365
q_comparison = pd.DataFrame(
    {
        "Hyperliquid": hl_quarterly,
        "Coinbase_INTX": cb_quarterly,
    }
).dropna()
q_comparison["Spread"] = q_comparison["Hyperliquid"] - q_comparison["Coinbase_INTX"]

print("\nQuarterly Annualized Funding Rates:")
for idx, row in q_comparison.iterrows():
    print(
        f"  {idx.strftime('%Y-Q')}{'1234'[idx.quarter - 1]}: HL={row['Hyperliquid']:+7.1%}  CB={row['Coinbase_INTX']:+7.1%}  spread={row['Spread']:+7.1%}"
    )

# ============================================================
# 5. Overlapping Period Comparison
# ============================================================
print("\n" + "=" * 60)
print("5. OVERLAPPING PERIOD COMPARISON")
print("=" * 60)

# Round to hour for alignment
hl_hourly = hl["funding_rate"].copy()
hl_hourly.index = hl_hourly.index.round("h")
hl_hourly = hl_hourly[~hl_hourly.index.duplicated(keep="last")]

cb_hourly = cb["funding_rate"].copy()
cb_hourly.index = cb_hourly.index.round("h")
cb_hourly = cb_hourly[~cb_hourly.index.duplicated(keep="last")]

common = pd.DataFrame({"hyperliquid": hl_hourly, "coinbase": cb_hourly}).dropna()

print(f"Overlap period: {common.index.min()} to {common.index.max()}")
print(f"Common timestamps: {len(common)}")
print(f"HL mean (overlap): {common['hyperliquid'].mean():.8f} -> ann: {common['hyperliquid'].mean() * 24 * 365:.2%}")
print(f"CB mean (overlap): {common['coinbase'].mean():.8f} -> ann: {common['coinbase'].mean() * 24 * 365:.2%}")
print(f"Correlation: {common['hyperliquid'].corr(common['coinbase']):.3f}")

sign_agree = (np.sign(common["hyperliquid"]) == np.sign(common["coinbase"])).mean()
print(f"Sign agreement: {sign_agree:.1%}")

# ============================================================
# 6. Base Rate Adjustment
# ============================================================
print("\n" + "=" * 60)
print("6. BASE RATE ADJUSTMENT")
print("=" * 60)

# Standard base rate: 0.01% per 8h = 0.00125% per hour = 0.0000125 decimal/hour
BASE_RATE_PER_HOUR = 0.01 / 100 / 8

cb_adjusted = cb["funding_rate"] + BASE_RATE_PER_HOUR
print(f"Base rate added per hour: {BASE_RATE_PER_HOUR:.8f} ({BASE_RATE_PER_HOUR * 24 * 365:.2%} annualized)")
print(f"Coinbase raw annualized:      {cb['funding_rate'].mean() * 24 * 365:+.2%}")
print(f"Coinbase adjusted annualized:  {cb_adjusted.mean() * 24 * 365:+.2%}")
print(f"Hyperliquid annualized:        {hl['funding_rate'].mean() * 24 * 365:+.2%}")
print(f"Gap after adjustment:          {(hl['funding_rate'].mean() - cb_adjusted.mean()) * 24 * 365:+.2%}")

# Overlap period with adjustment
common_adj = pd.DataFrame({"hyperliquid": hl_hourly, "coinbase_adj": cb_hourly + BASE_RATE_PER_HOUR}).dropna()

print("\nOverlap period adjusted:")
print(f"HL mean:     {common_adj['hyperliquid'].mean() * 24 * 365:+.2%}")
print(f"CB adj mean: {common_adj['coinbase_adj'].mean() * 24 * 365:+.2%}")
print(f"Gap:         {(common_adj['hyperliquid'].mean() - common_adj['coinbase_adj'].mean()) * 24 * 365:+.2%}")

# ============================================================
# 7. Early Period Analysis (low liquidity artifact?)
# ============================================================
print("\n" + "=" * 60)
print("7. EARLY vs MATURE PERIOD")
print("=" * 60)

# Split Coinbase data at 6-month mark from start
cb_start = cb.index.min()
maturity_cutoff = cb_start + pd.Timedelta(days=180)
print(f"Coinbase start: {cb_start}")
print(f"Maturity cutoff (6 months in): {maturity_cutoff}")

cb_early = cb.loc[:maturity_cutoff, "funding_rate"]
cb_mature = cb.loc[maturity_cutoff:, "funding_rate"]

print(f"\nEarly period ({cb_early.index.min().date()} to {cb_early.index.max().date()}):")
print(f"  N={len(cb_early)}, mean={cb_early.mean():.8f}, ann={cb_early.mean() * 24 * 365:+.2%}")
print(f"  std={cb_early.std():.8f}")

print(f"\nMature period ({cb_mature.index.min().date()} to {cb_mature.index.max().date()}):")
print(f"  N={len(cb_mature)}, mean={cb_mature.mean():.8f}, ann={cb_mature.mean() * 24 * 365:+.2%}")
print(f"  std={cb_mature.std():.8f}")

# Same for Hyperliquid
hl_overlap_start = maturity_cutoff  # compare same period
hl_same_period = hl.loc[maturity_cutoff:, "funding_rate"]
print(f"\nHyperliquid same mature period ({hl_same_period.index.min().date()} to {hl_same_period.index.max().date()}):")
print(f"  N={len(hl_same_period)}, mean={hl_same_period.mean():.8f}, ann={hl_same_period.mean() * 24 * 365:+.2%}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
Full-period stats:
  Hyperliquid:  {hl["funding_rate"].mean() * 24 * 365:+.1%} annualized ({len(hl)} hourly obs)
  Coinbase INTX: {cb["funding_rate"].mean() * 24 * 365:+.1%} annualized ({len(cb)} hourly obs)

Base rate adjustment (+{BASE_RATE_PER_HOUR * 24 * 365:.1%} annualized):
  Coinbase adj: {cb_adjusted.mean() * 24 * 365:+.1%} annualized
  Residual gap: {(hl["funding_rate"].mean() - cb_adjusted.mean()) * 24 * 365:+.1%}

Cross-exchange:
  Correlation (hourly): {common["hyperliquid"].corr(common["coinbase"]):.3f}
  Sign agreement: {sign_agree:.1%}

Period analysis:
  Early Coinbase (first 6mo): {cb_early.mean() * 24 * 365:+.1%} ann
  Mature Coinbase (after 6mo): {cb_mature.mean() * 24 * 365:+.1%} ann
""")
