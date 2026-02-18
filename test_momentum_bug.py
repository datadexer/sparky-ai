#!/usr/bin/env python3
"""Test to demonstrate look-ahead bias in momentum strategy.

The bug: momentum at time T uses close[T], and the strategy captures
returns[T] which is (close[T] - close[T-1]) / close[T-1].

This means we're using close[T] to predict a return that INCLUDES close[T]!
"""

import pandas as pd

# Create simple price series
dates = pd.date_range("2025-01-01", periods=10, freq="D")
prices = pd.Series([100, 110, 105, 115, 120, 118, 125, 130, 128, 135], index=dates)

print("=" * 80)
print("DEMONSTRATION OF LOOK-AHEAD BIAS")
print("=" * 80)

# Compute momentum (30-day, but use 3-day for demo)
momentum_period = 3
momentum = (prices - prices.shift(momentum_period)) / prices.shift(momentum_period)

# Compute returns (same as option3_strategic_pivot.py line 157)
returns = prices.pct_change().fillna(0)

print("\nPrices:")
print(prices)
print("\nMomentum (3-day):")
print(momentum)
print("\nReturns (close[t-1] to close[t]):")
print(returns)

# The buggy approach (what option3_strategic_pivot.py does)
print("\n" + "=" * 80)
print("BUGGY APPROACH (what the script does)")
print("=" * 80)
signals_buggy = (momentum > 0.05).astype(int)
strategy_returns_buggy = signals_buggy * returns

print("\nSignals (momentum > 0.05):")
print(signals_buggy)
print("\nStrategy returns (BUGGY):")
print(strategy_returns_buggy)

# Let's look at a specific example
print("\n" + "=" * 80)
print("SPECIFIC EXAMPLE: 2025-01-06")
print("=" * 80)
t = pd.Timestamp("2025-01-06")
print(f"Close on {t - pd.Timedelta(days=3)} (T-3): {prices[t - pd.Timedelta(days=3)]:.2f}")
print(f"Close on {t - pd.Timedelta(days=1)} (T-1): {prices[t - pd.Timedelta(days=1)]:.2f}")
print(f"Close on {t} (T): {prices[t]:.2f}")
print("\nMomentum at T = (close[T] - close[T-3]) / close[T-3]")
print(
    f"             = ({prices[t]:.2f} - {prices[t - pd.Timedelta(days=3)]:.2f}) / {prices[t - pd.Timedelta(days=3)]:.2f}"
)
print(f"             = {momentum[t]:.4f}")
print("\nReturns at T = (close[T] - close[T-1]) / close[T-1]")
print(
    f"            = ({prices[t]:.2f} - {prices[t - pd.Timedelta(days=1)]:.2f}) / {prices[t - pd.Timedelta(days=1)]:.2f}"
)
print(f"            = {returns[t]:.4f}")
print(f"\nSignal at T (momentum > 0.05): {signals_buggy[t]}")
print(f"Captured return: {strategy_returns_buggy[t]:.4f}")
print("\n⚠️  THE BUG: Momentum at T uses close[T]=118, and returns at T is from 120→118.")
print("   We're using the price at T to 'predict' a return that ends at T!")
print("   This is look-ahead bias!")

# The correct approach
print("\n" + "=" * 80)
print("CORRECT APPROACH (shift returns forward)")
print("=" * 80)
# Use momentum at T to predict returns at T+1
signals_correct = (momentum > 0.05).astype(int)
returns_forward = returns.shift(-1)  # Shift returns forward by 1
strategy_returns_correct = signals_correct * returns_forward

print("\nForward returns (close[t] to close[t+1]):")
print(returns_forward)
print("\nStrategy returns (CORRECT):")
print(strategy_returns_correct)

print("\n" + "=" * 80)
print("COMPARISON AT 2025-01-06")
print("=" * 80)
t = pd.Timestamp("2025-01-06")
print(f"Momentum at T: {momentum[t]:.4f} (uses close[T]={prices[t]:.2f})")
print(f"Signal at T: {signals_correct[t]}")
print("\nBuggy approach:")
print(
    f"  Captures returns[T] = {returns[t]:.4f} (from close[T-1]={prices[t - pd.Timedelta(days=1)]:.2f} to close[T]={prices[t]:.2f})"
)
print(f"  Strategy return: {strategy_returns_buggy[t]:.4f}")
print("\nCorrect approach:")
print(
    f"  Captures returns[T+1] = {returns_forward[t]:.4f} (from close[T]={prices[t]:.2f} to close[T+1]={prices[t + pd.Timedelta(days=1)]:.2f})"
)
print(f"  Strategy return: {strategy_returns_correct[t]:.4f}")
print("\n✓ The correct approach uses close[T] to predict the NEXT period's return!")

print("\n" + "=" * 80)
print("IMPACT ON CUMULATIVE RETURNS")
print("=" * 80)

# Remove NaN values for fair comparison
valid_buggy = strategy_returns_buggy.dropna()
valid_correct = strategy_returns_correct.dropna()

# Align them to the same index
common_idx = valid_buggy.index.intersection(valid_correct.index)
valid_buggy = valid_buggy.loc[common_idx]
valid_correct = valid_correct.loc[common_idx]

cum_buggy = (1 + valid_buggy).cumprod().iloc[-1] - 1
cum_correct = (1 + valid_correct).cumprod().iloc[-1] - 1

print(f"Buggy approach total return: {cum_buggy:.2%}")
print(f"Correct approach total return: {cum_correct:.2%}")
print(f"\nDifference: {(cum_buggy - cum_correct):.2%}")
print("\nThe buggy approach will ALWAYS look better because it has look-ahead bias!")
