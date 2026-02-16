# BUG REPORT: Look-Ahead Bias in Simple Strategy Backtests

**Date Reported**: 2026-02-15
**Severity**: CRITICAL
**Status**: CONFIRMED - Awaiting Fix
**Reporter**: Human (AK)
**Assignee**: CEO Agent

---

## Executive Summary

**CRITICAL BUG**: The claimed Sharpe ratio of 2.556 for the "momentum > 0.05" strategy is **completely false** due to look-ahead bias in the backtest implementation.

**True Performance**: Sharpe **-0.27** (loses money), NOT +2.56

**Root Cause**: Using `close[T]` in the signal calculation to predict a return that **ends at** `close[T]`.

**Impact**: All simple strategy backtests in Phase 3 validation are invalid and must be re-run.

---

## Bug Description

### The Problem

In `scripts/option3_strategic_pivot.py` (and similar scripts), the backtest computes:

1. **Momentum at time T**: `momentum[T] = (close[T] - close[T-30]) / close[T-30]`
2. **Returns at time T**: `returns[T] = (close[T] - close[T-1]) / close[T-1]`
3. **Signal at time T**: `position[T] = 1 if momentum[T] > 0.05`
4. **Strategy return at T**: `strategy_return[T] = position[T] * returns[T]`

**The bug**: The momentum calculation uses `close[T]` to generate a signal, then captures `returns[T]`, which is the return **ending at** `close[T]`. This means the signal uses information (the close price at T) that is contained within the return period being predicted.

### Why This is Wrong

In real trading:
- At **close of day T**: You see `close[T]`, compute momentum, generate signal
- At **open of day T+1**: You execute the trade (cannot trade at yesterday's close!)
- You capture the return from **close[T] to close[T+1]** (or later)

The current code captures the return from **close[T-1] to close[T]**, which is **already known** when you compute momentum at time T.

### Correct Timing

```python
# WRONG (current implementation)
position[T] = signal_based_on(close[T])
return[T] = (close[T] - close[T-1]) / close[T-1]  # ← Ends at close[T]!
strategy_return[T] = position[T] * return[T]  # ← Look-ahead bias!

# CORRECT (what it should be)
position[T] = signal_based_on(close[T])
forward_return[T] = (close[T+1] - close[T]) / close[T]  # ← Starts at close[T]
strategy_return[T] = position[T] * forward_return[T]  # ← No bias
```

---

## Evidence

### Reproduction

Run `prove_bug.py` (created during investigation):

```bash
source .venv/bin/activate
python prove_bug.py
```

**Results**:

| Metric | Buggy (claimed) | Correct (actual) | Degradation |
|--------|----------------|------------------|-------------|
| Sharpe | **+2.5564** | **-0.2725** | **-2.83** |
| Return | +17.46% | -2.70% | -20.15% |
| Max DD | 3.63% | 9.67% | +6.04% |

The "breakthrough" strategy is actually a **losing strategy**.

### Test Case

See `test_momentum_bug.py` for a minimal demonstration with synthetic data showing:
- Buggy approach total return: 21.90%
- Correct approach total return: 17.39%
- Difference: 4.51% inflation due to look-ahead bias

---

## Affected Files

### Primary Bug Location

1. **`scripts/option3_strategic_pivot.py`**
   - Lines 157, 174-183 (momentum strategies)
   - Lines 214-223 (RSI strategies)
   - Line 44 in `run_experiment()` function

### Potentially Affected Files

The following scripts use similar patterns and should be audited:

2. **`scripts/option2_debug_simple.py`** - If it exists and uses simple strategies
3. **`scripts/option2_debug_overfitting.py`** - If it tests simple rules
4. **Any other scripts** in `scripts/` that:
   - Compute `returns = prices.pct_change()`
   - Generate signals based on features at time T
   - Apply signals to returns at the same time T

### Files That Are CORRECT

The following files use proper timing and do NOT have this bug:

- ✅ **`src/sparky/backtest/engine.py`** - WalkForwardBacktester is correct
- ✅ **`scripts/prepare_phase3_data.py`** - Target generation uses proper forward returns
- ✅ **`scripts/run_baseline.py`** - Uses `returns.shift(-1)` for targets (correct)
- ✅ **`scripts/validate_holdout.py`** - Uses trained model with proper backtester

**Why these are correct**: They use the `WalkForwardBacktester` which expects:
- `y[T]` = target label at T (which represents a **forward** return)
- `returns[T]` = **realized** return at T (used for performance calculation, not prediction)

The backtester internally handles the timing correctly by training on T, predicting at T, and measuring the **forward** return.

---

## Root Cause Analysis

### Why The Bug Exists

The `option3_strategic_pivot.py` script was written as a "quick and dirty" exploration to bypass the ML pipeline. It directly generates signals from features without using the proper backtesting infrastructure.

**Design flaw**: The `run_experiment()` function (line 32-67) treats signals as if they were generated out-of-sample, but the timing alignment is wrong.

### Why It Wasn't Caught

1. **No integration test**: The simple strategy backtests weren't part of the main backtesting pipeline
2. **Isolated script**: Written in a hurry during Option 3 pivot, not reviewed
3. **Confirmation bias**: The high Sharpe looked like a "breakthrough" so it wasn't questioned
4. **Data snooping**: The focus was on the p-hacking issue, not the implementation

---

## Fix Instructions

### Step 1: Fix `option3_strategic_pivot.py`

**Location**: Line 44 in `run_experiment()` function

**Current code**:
```python
def run_experiment(name, model_or_signals, X_train, y_train, X_holdout, y_holdout, returns_holdout, is_signals=False):
    """Run single experiment."""
    if is_signals:
        # Direct signals provided
        predictions = model_or_signals
    else:
        # Train model
        model_or_signals.fit(X_train, y_train)
        predictions = model_or_signals.predict(X_holdout)

    # Evaluate
    positions = predictions
    strategy_returns = positions * returns_holdout  # ← BUG HERE
```

**Fixed code**:
```python
def run_experiment(name, model_or_signals, X_train, y_train, X_holdout, y_holdout, returns_holdout, is_signals=False):
    """Run single experiment."""
    if is_signals:
        # Direct signals provided
        predictions = model_or_signals
    else:
        # Train model
        model_or_signals.fit(X_train, y_train)
        predictions = model_or_signals.predict(X_holdout)

    # Evaluate
    positions = predictions
    # FIX: Use forward returns (shift -1 to align with proper trading timing)
    # Position at T (based on close[T]) captures return from close[T] to close[T+1]
    forward_returns = returns_holdout.shift(-1).fillna(0)
    strategy_returns = positions * forward_returns  # ← FIXED
```

**Alternative fix** (more explicit):
```python
    # Compute forward returns explicitly
    # This makes it clear we're predicting the NEXT period's return
    forward_returns = pd.Series(0.0, index=returns_holdout.index)
    forward_returns[:-1] = returns_holdout.iloc[1:].values
    strategy_returns = positions * forward_returns
```

### Step 2: Re-run All Affected Experiments

After fixing the code, re-run:

```bash
source .venv/bin/activate
python scripts/option3_strategic_pivot.py
```

**Expected outcome**: All Sharpe ratios will drop significantly (likely to negative or near-zero values).

### Step 3: Update Results

1. **Delete or mark as INVALID**:
   - `results/experiments/option3_pivot_results.json`
   - Any RESEARCH_LOG.md entries claiming Sharpe 2.556

2. **Add new entries** with corrected results

3. **Update DECISIONS.md** with finding:
   ```
   [2026-02-15] BUG FOUND: Option 3 results were invalid due to look-ahead bias
   - Claimed Sharpe 2.556 was false (actual: -0.27)
   - All simple strategies in Option 3 need re-validation
   - Root cause: signals used close[T] to predict returns ending at close[T]
   ```

### Step 4: Verify Fix

Run the proof script to confirm the fix:

```bash
python prove_bug.py
```

**Success criteria**:
- Script should report that buggy and correct approaches now produce the **same** results
- Or remove the buggy approach test since the code is now fixed

### Step 5: Add Regression Test

Create `tests/test_simple_strategy_timing.py`:

```python
"""Test that simple strategy backtests use correct timing (no look-ahead bias)."""

import numpy as np
import pandas as pd
from sparky.backtest.costs import TransactionCostModel
from scripts.option3_strategic_pivot import run_experiment

def test_simple_strategy_no_lookahead():
    """Verify simple strategies don't have look-ahead bias."""
    # Create synthetic data where look-ahead bias is detectable
    dates = pd.date_range('2025-01-01', periods=100, freq='D')

    # Price pattern: alternating up/down days
    prices = pd.Series([100 * (1.01 if i % 2 == 0 else 0.99) ** (i // 2)
                       for i in range(100)], index=dates)

    # Returns: alternating positive/negative
    returns = prices.pct_change().fillna(0)

    # Signal: always long (should capture forward returns, not same-period returns)
    signals = pd.Series(1, index=dates)

    # Run experiment (assuming it's been fixed)
    result = run_experiment("test", signals, None, None, None, None, returns, is_signals=True)

    # With look-ahead bias: would capture same-period returns (mix of +/-)
    # Without look-ahead bias: captures forward returns (should be shifted)

    # Verify the strategy returns are shifted (not same as input returns)
    # This is a basic sanity check that the fix was applied
    assert result is not None  # Placeholder - implement actual verification

if __name__ == "__main__":
    test_simple_strategy_no_lookahead()
    print("✓ Test passed")
```

---

## Verification Checklist

Before closing this bug:

- [ ] Fix applied to `scripts/option3_strategic_pivot.py`
- [ ] Fix applied to any other affected scripts (TBD after audit)
- [ ] All experiments re-run with corrected code
- [ ] Results files updated/marked invalid
- [ ] RESEARCH_LOG.md updated with corrected findings
- [ ] DECISIONS.md updated with bug report
- [ ] Regression test added to prevent recurrence
- [ ] `prove_bug.py` confirms fix works
- [ ] Git commit with message: `fix: correct look-ahead bias in simple strategy backtests`

---

## Additional Context

### Related Issues

1. **Data Snooping** ([DATA_SNOOPING_ISSUE.md](roadmap/DATA_SNOOPING_ISSUE.md)):
   - Even WITHOUT this bug, the results are invalid due to p-hacking
   - This bug COMPOUNDS the data snooping issue
   - Both issues must be addressed

2. **Phase 3 Validation Status**:
   - All Option 3 results are invalid
   - Option 1 and 2 results need audit to check for same bug
   - Recommend full restart of validation with proper methodology

### Lessons Learned

1. **Don't bypass the backtester**: The `WalkForwardBacktester` exists for a reason
2. **Always verify timing**: When in doubt, draw a timeline of what's known when
3. **Test with synthetic data**: Alternating patterns make look-ahead bias obvious
4. **Peer review critical findings**: High Sharpe ratios should trigger extra scrutiny

### References

- **Bug proof script**: `prove_bug.py`
- **Minimal test case**: `test_momentum_bug.py`
- **Original buggy script**: `scripts/option3_strategic_pivot.py`
- **Correct implementation**: `src/sparky/backtest/engine.py` (use this as reference)

---

## Questions for CEO Agent

When you pick up this bug to fix:

1. **Scope**: Should we fix and re-run, or abandon Option 3 entirely given the data snooping issue?
2. **Validation**: After fixing, should we continue with simple strategies or return to ML models?
3. **Timeline**: Is this blocking Phase 3 completion, or can we proceed with other tasks?
4. **Audit**: Should we audit ALL scripts for similar timing bugs before proceeding?

**Recommendation**: Fix the bug for code hygiene, but **DO NOT** re-run Option 3 experiments. The data snooping issue makes them invalid regardless. Instead, proceed to Phase 3 data expansion (10K+ samples) and test strategies on truly held-out 2026 data.

---

**End of Bug Report**
