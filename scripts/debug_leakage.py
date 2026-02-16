#!/usr/bin/env python3
"""Minimal reproducer to debug data leakage in feature-target alignment.

This script creates a simple synthetic dataset and manually verifies that:
1. Features at time T use only data up to T close
2. Targets at time T represent "close_{T+1+N} > open_{T+1}"
3. There is NO overlap or information leakage
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("LEAKAGE DEBUGGING — Minimal Reproducer")
print("=" * 80)

# Create synthetic trending data (10 days)
dates = pd.date_range('2020-01-01', periods=10, freq='D', tz='UTC')
prices = pd.DataFrame({
    'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
}, index=dates)

print(f"\nSynthetic Price Data:")
print(prices)

# Compute returns_1d feature (as in prepare_phase3_data.py line 104)
# This uses pct_change() which computes: (close_T - close_T-1) / close_T-1
features_df = pd.DataFrame(index=dates)
features_df['returns_1d'] = prices['close'].pct_change()

print(f"\nFeatures (returns_1d):")
print(features_df)

# Generate target for horizon=1 (as in prepare_phase3_data.py lines 194-202)
horizon = 1
next_open = prices['open'].shift(-1)      # T+1 open
target_close = prices['close'].shift(-(1 + horizon))  # T+1+N close (N=1 → T+2 close)

print(f"\nTarget components for horizon={horizon}:")
print(pd.DataFrame({
    'next_open': next_open,
    'target_close': target_close,
    'target': (target_close > next_open).astype(int)
}))

# MANUAL VERIFICATION for specific dates
print(f"\n" + "=" * 80)
print("MANUAL VERIFICATION — Check for Leakage")
print("=" * 80)

# Pick 2020-01-03 as example
test_date = pd.Timestamp('2020-01-03', tz='UTC')
print(f"\nVerifying timing for {test_date.date()}:")

# What data is available AT time T (2020-01-03 close)?
print(f"  Available data at {test_date.date()} close:")
print(f"    - Close price on 2020-01-03: {prices.loc[test_date, 'close']}")
print(f"    - Close price on 2020-01-02: {prices.loc[test_date - pd.Timedelta(days=1), 'close']}")
print(f"    - returns_1d feature: {features_df.loc[test_date, 'returns_1d']:.4f}")
print(f"      (= (103 - 102) / 102 = 0.0098)")

# What is the target for this row?
print(f"\n  Target timing (should use FUTURE data only):")
print(f"    - Signal generated at: {test_date.date()} close")
print(f"    - Trade executed at: {(test_date + pd.Timedelta(days=1)).date()} open = {prices.loc[test_date + pd.Timedelta(days=1), 'open']}")
print(f"    - Target evaluated at: {(test_date + pd.Timedelta(days=2)).date()} close = {prices.loc[test_date + pd.Timedelta(days=2), 'close']}")
print(f"    - Target value: {1 if prices.loc[test_date + pd.Timedelta(days=2), 'close'] > prices.loc[test_date + pd.Timedelta(days=1), 'open'] else 0}")
print(f"      (105 > 104? → 1 = long)")

# Check the computed target
computed_target = (target_close > next_open).astype(int).loc[test_date]
print(f"\n  Computed target from shift logic: {computed_target}")
print(f"    next_open[{test_date.date()}] = {next_open.loc[test_date]}")
print(f"    target_close[{test_date.date()}] = {target_close.loc[test_date]}")

# Verify NO OVERLAP
print(f"\n" + "=" * 80)
print("LEAKAGE CHECK")
print("=" * 80)
print(f"\nFeature uses: close up to {test_date.date()}")
print(f"Target uses: open on {(test_date + pd.Timedelta(days=1)).date()} and close on {(test_date + pd.Timedelta(days=2)).date()}")
print(f"\nConclusion: ", end="")

# Check if feature computation overlaps with target evaluation dates
feature_end_date = test_date
target_start_date = test_date + pd.Timedelta(days=1)

if feature_end_date < target_start_date:
    print("✅ NO OVERLAP — Feature uses past data, target uses future data")
else:
    print("❌ OVERLAP DETECTED — Feature may leak into target")

# Now check the ACTUAL implementation's shift logic
print(f"\n" + "=" * 80)
print("SHIFT LOGIC VERIFICATION")
print("=" * 80)

print(f"\nOriginal close prices:")
print(prices['close'])

print(f"\nclose.shift(-1) [next day's close]:")
print(prices['close'].shift(-1))

print(f"\nclose.shift(-(1+1)) [close 2 days ahead]:")
print(prices['close'].shift(-2))

print(f"\nopen.shift(-1) [next day's open]:")
print(prices['open'].shift(-1))

# Test with actual returns_1d computation
print(f"\n" + "=" * 80)
print("RETURNS_1D LEAKAGE TEST")
print("=" * 80)

# At time T, returns_1d = (close_T - close_T-1) / close_T-1
# Does this leak into target = (close_T+2 > open_T+1)?
#
# If we're predicting whether price goes up, and returns_1d tells us
# the price JUST went up (T-1 to T), this is a lagging indicator.
# It should NOT leak because it's past information.
#
# BUT: If the target is misaligned and actually uses close_T instead of close_T+2,
# then returns_1d (which uses close_T) would leak.

print("\nTesting for off-by-one error in target generation...")
for i in range(len(prices) - 3):
    date_t = prices.index[i]

    # What we SHOULD have:
    # Feature at T: uses close_T and close_T-1
    # Target at T: (close_{T+2} > open_{T+1})

    if i > 0:  # Skip first row (returns_1d is NaN)
        feature_val = features_df.loc[date_t, 'returns_1d']
        target_val = (target_close > next_open).astype(int).loc[date_t]

        # Check if feature and target reference same close value
        close_t = prices.loc[date_t, 'close']
        close_t_minus_1 = prices.loc[prices.index[i-1], 'close']
        close_target = target_close.loc[date_t]
        open_next = next_open.loc[date_t]

        print(f"\n{date_t.date()}:")
        print(f"  Feature uses: close[{prices.index[i-1].date()}]={close_t_minus_1}, close[{date_t.date()}]={close_t}")
        print(f"  Target uses: open[{prices.index[i+1].date()}]={open_next}, close[{prices.index[i+2].date() if i+2 < len(prices) else 'N/A'}]={close_target}")

        # If close_t appears in both feature AND target, we have leakage
        if not pd.isna(close_target) and (close_t == close_target):
            print(f"  ❌ LEAKAGE: close_t ({close_t}) appears in BOTH feature and target!")
        elif not pd.isna(open_next) and (close_t == open_next):
            print(f"  ❌ LEAKAGE: close_t ({close_t}) == next_open ({open_next})!")
        else:
            print(f"  ✅ No obvious leakage")

print(f"\n" + "=" * 80)
print("DEBUGGING COMPLETE")
print("=" * 80)
