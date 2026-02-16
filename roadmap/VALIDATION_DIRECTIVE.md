# VALIDATION DIRECTIVE — Phase 4 Gate

**Date**: 2026-02-16 01:40 UTC
**Priority**: BLOCKING
**Status**: STOP and validate before proceeding

---

## 🛑 STOP Phase 4 Multi-Seed Validation

Before continuing to multi-seed validation, complete these BLOCKING validations.

## Concern

Sharpe 0.999 on 30d horizon is highly suspicious:
1. Same horizon that previously FAILED leakage detection (0.86 Sharpe but failed shuffled-label)
2. After fixing leakage, performance IMPROVED (0.86 → 0.999) - backwards from expected
3. Sharpe 0.999 is unrealistically high for crypto with transaction costs
4. Contradicts earlier clean-data results (Phase 1: Sharpe 0.0037 on 7d)
5. On-chain "adds no value" contradicts foundational hypothesis

**This pattern suggests either:**
- New leakage source introduced
- Overfitting to train/test split
- Implementation bug
- Statistical anomaly

## Required Validations (in order)

### VALIDATION 1: Holdout Test (HIGHEST PRIORITY)

**Objective**: Test on truly unseen data to catch overfitting

**Action**:
```python
# Test technical-only, 30d on holdout set
# Period: 2025-10-01 to 2025-12-31 (92 days)
# This data was NEVER used in training or Phase 2-3
```

**Pass criteria**:
- Holdout Sharpe within 0.3 of train/test Sharpe (i.e., > 0.7)
- If holdout Sharpe < 0.5 → Result is OVERFITTING, not real alpha

**Why this is critical**:
- Multi-seed validation uses same train/test split across seeds
- Only holdout test uses completely fresh data
- This is THE smoking gun for overfitting

---

### VALIDATION 2: Leakage Re-Audit

**Objective**: Verify 30d horizon isn't still leaking data

**Action**:
```python
# Re-run leakage detector SPECIFICALLY on technical-only, 30d
detector.run_all_checks(
    model=xgboost_technical_30d,
    X_train=X_train_technical,
    y_train=y_train_30d,
    X_test=X_test_technical,
    y_test=y_test_30d,
    n_shuffle_trials=20  # Increase from 10 for confidence
)
```

**Pass criteria**:
- Shuffled-label accuracy: 48-52% (random performance)
- Temporal boundary: Pass
- Index overlap: Pass

**Also check**:
- How many trades executed? (0 trades = costs not applied)
- Is model just "buy and hold"? (Check position changes)
- What are entry/exit signals? (Inspect predictions)

---

### VALIDATION 3: Sanity Checks

**Objective**: Verify implementation correctness

**Actions**:

1. **Baseline verification**:
   ```python
   # Re-run BuyAndHold on same data
   # Should still show Sharpe ~0.79
   # If baseline changed, data/cost model is broken
   ```

2. **Trade inspection**:
   ```python
   # Print first 10 trades:
   # [Date, Signal, Entry Price, Exit Price, P&L, Cost]
   # Verify:
   # - Trades make logical sense
   # - Transaction costs applied (0.13% per trade)
   # - Entry/exit timing correct
   ```

3. **Target variable audit**:
   ```python
   # For 30d horizon, verify:
   # T close → T+1 open (entry) → T+31 close (target)
   # Check for off-by-one errors
   # Confirm no future data in features
   ```

4. **Feature importance**:
   ```python
   # Show which features drive predictions
   # RSI, Momentum, EMA ratios should make sense
   # Check for any suspicious patterns
   ```

---

## Blocking Questions to Answer

Before proceeding, provide clear answers:

1. **Why did fixing leakage IMPROVE performance?**
   - Before: 0.86 Sharpe (with leakage) → FAILED test
   - After: 0.999 Sharpe (leakage "fixed") → claims PASS
   - This is backwards - fixing leakage should REDUCE performance

2. **Why is 30d the winner?**
   - 30d was the exact horizon that failed leakage before
   - Now it's suddenly the best performer?
   - Shorter horizons (1d, 3d, 7d) all worse - suspicious pattern

3. **How did drawdown improve?**
   - Phase 1 clean results: 87.8% max drawdown
   - Phase 2-3: 60.1% max drawdown
   - Removing returns_1d made drawdown BETTER? How?

4. **Why does technical-only win?**
   - Entire project hypothesis: on-chain metrics provide edge
   - Now technical-only beats all features by 4% Sharpe
   - This invalidates Strategic Goal #1

5. **What is the model predicting?**
   - Show predictions vs actuals
   - Show feature importance
   - Explain the trading logic

---

## Revised Workflow

```
1. ✅ Phase 2-3: Feature ablation + horizon optimization (UNVERIFIED)
2. 🔴 Validation 1: Holdout test ← YOU ARE HERE (BLOCKING)
3. 🔴 Validation 2: Leakage re-audit (BLOCKING)
4. 🔴 Validation 3: Sanity checks (BLOCKING)
5. ⏸️  Phase 4: Multi-seed validation (ONLY if validations 1-3 PASS)
6. ⏸️  Phase 5: Final report and decision
```

**Do NOT proceed to Phase 4 until validations complete and pass.**

---

## Why This Order Matters

**Holdout test must come BEFORE multi-seed** because:

| Validation Type | Catches Overfitting? | Catches Leakage? | Catches Implementation Bugs? |
|-----------------|---------------------|------------------|------------------------------|
| **Holdout test** | ✅ YES | ❌ No | ❌ No |
| **Leakage detector** | ❌ No | ✅ YES | ❌ No |
| **Sanity checks** | ❌ No | ⚠️ Partial | ✅ YES |
| **Multi-seed** | ❌ NO | ❌ NO | ❌ NO |

Multi-seed validation is useful for checking stability AFTER we know the result is valid.

Running multi-seed first wastes compute on a potentially bogus result.

---

## Expected Outcome

**If validations PASS**:
- Holdout Sharpe > 0.7 ✅
- Leakage detector passes ✅
- Sanity checks pass ✅
- **Then**: Proceed to multi-seed validation
- **Result**: Genuine breakthrough (would be extraordinary)

**If validations FAIL**:
- Holdout Sharpe < 0.5 ❌
- **Then**: Result is overfitting or leakage, not real alpha
- **Action**: Debug root cause, don't proceed to multi-seed

---

## Deliverable

After running validations, report:

```
VALIDATION RESULTS:

1. Holdout Test:
   - Holdout Sharpe: [X.XX]
   - Status: PASS/FAIL
   - Interpretation: [...]

2. Leakage Re-Audit:
   - Shuffled accuracy: [XX.X%]
   - Status: PASS/FAIL
   - Interpretation: [...]

3. Sanity Checks:
   - Baseline Sharpe: [X.XX] (should be ~0.79)
   - Trade count: [N]
   - Sample trades: [show 5]
   - Status: PASS/FAIL

CONCLUSION:
- All validations passed → Proceed to Phase 4 multi-seed
- Any validation failed → DEBUG before proceeding
```

---

**Priority**: Run holdout test first. It's fast (5 min) and definitive.

If holdout Sharpe ~= 0.999 → Maybe real (shocking but possible)
If holdout Sharpe << 0.999 → Overfitting confirmed, stop and debug
