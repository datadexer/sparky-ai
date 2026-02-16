# Look-Ahead Bias Bug - Quick Summary

**Date**: 2026-02-15
**Status**: ❌ CONFIRMED - Sharpe 2.556 claim is FALSE

---

## The Claim

> "Simple momentum > 0.05 rule achieved Sharpe 2.556 on 2025 holdout"

## The Truth

**Actual Sharpe: -0.27** (loses money!)

---

## The Bug

**Location**: `scripts/option3_strategic_pivot.py` lines 157, 178, 44

```python
# Line 157: Compute returns
returns = prices.pct_change()  # returns[T] = close[T-1] to close[T]

# Line 178: Generate signal
signals = (momentum > 0.05)    # momentum[T] uses close[T]

# Line 44: Apply signal to SAME-period return
strategy_returns = signals * returns  # ← BUG: Using close[T] to predict returns ending at close[T]
```

**The Problem**: Signal uses information (`close[T]`) that is contained in the return being predicted.

**The Fix**: Use forward returns:
```python
forward_returns = returns.shift(-1)  # Return from close[T] to close[T+1]
strategy_returns = signals * forward_returns
```

---

## The Proof

Run: `python prove_bug.py`

| Metric | Buggy (claimed) | Correct (actual) | Difference |
|--------|----------------|------------------|------------|
| **Sharpe** | **+2.56** | **-0.27** | **-2.83** |
| **Return** | +17.5% | -2.7% | -20.2% |

---

## Impact

1. ❌ All Option 3 results are INVALID
2. ❌ "Momentum > 0.05" is NOT a breakthrough - it's a losing strategy
3. ❌ Combined with data snooping, Phase 3 validation has failed completely

---

## What To Do

**For CEO Agent**:
1. Read full report: [roadmap/BUG_REPORT_LOOKAHEAD_BIAS.md](roadmap/BUG_REPORT_LOOKAHEAD_BIAS.md)
2. Fix the bug in `scripts/option3_strategic_pivot.py`
3. DO NOT re-run Option 3 (data snooping issue remains)
4. Proceed to Phase 3 data expansion (get 10K+ samples)
5. Test strategies on truly unseen 2026 data

**For Human**:
- Decision needed on path forward (see DECISIONS.md)
- Options: Wait for 2026 data / Frame as hypothesis / Terminate project

---

## Bottom Line

The "breakthrough" was a bug. The strategy loses money. Back to the drawing board.
