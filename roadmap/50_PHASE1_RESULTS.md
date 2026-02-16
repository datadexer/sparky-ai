# Phase 1 Results — Baseline Models with Leakage-Free Data

**Date**: 2026-02-15 20:24 UTC
**Status**: ✅ XGBoost completed | ⚠️ LSTM training error
**Key Finding**: **NO ALPHA DETECTED** with clean data (honest result)

---

## Summary

After resolving the `returns_1d` data leakage issue (Phase 0), we re-ran baseline models on clean 6-feature data. **XGBoost showed near-zero Sharpe (0.0037)**, confirming that:

1. ✅ The leakage fix worked (previous high Sharpe was due to leaked information)
2. ✅ The model produces honest results (no false positives)
3. ❌ There is NO exploitable alpha with current features + 7d horizon

---

## XGBoost Results

### Configuration
- **Horizon**: 7 days
- **Features**: 6 (rsi_14, momentum_30d, ema_ratio_20d, hash_ribbon_btc, address_momentum_btc, volume_momentum_btc)
- **Train period**: 2019-01-15 to 2022-01-01 (1082 samples)
- **Test period**: 2022-01-01 to 2025-09-30 (1461 samples)
- **Walk-forward folds**: 76
- **Transaction costs**: 0.13% per trade (TransactionCostModel.for_btc())
- **Random seed**: 42

### Leakage Detection
✅ **ALL CHECKS PASSED**
- `shuffled_label`: ✅ PASS (0.513 accuracy — random performance)
- `temporal_boundary`: ✅ PASS (1-day gap between train/test)
- `index_overlap_audit`: ✅ PASS (no timestamp overlap)

### Performance Metrics
| Metric | Value | Baseline (BuyAndHold) | Delta |
|--------|-------|----------------------|-------|
| **Sharpe** | 0.0037 | 0.7892 | **-0.7855** |
| **Sharpe 95% CI** | (-0.66, 0.69) | (0.14, 1.48) | — |
| **Max DD** | 87.82% | 76.63% | +11.19% (worse) |
| **Total Return** | -53.7% | 1027.76% | — |

### Interpretation

1. **Near-zero Sharpe (0.0037)**:
   - Model has no predictive power
   - Wide confidence interval (-0.66 to 0.69) includes zero
   - Not statistically significant

2. **Massive drawdown (87.82%)**:
   - Worse than baseline (76.63%)
   - Model loses half the capital
   - Unacceptable for deployment

3. **Negative total return (-53.7%)**:
   - Strategy loses money after transaction costs
   - Baseline gained 1027%
   - Clear underperformance

4. **Delta vs Baseline (-0.79)**:
   - XGBoost significantly underperforms simple buy-and-hold
   - Not even close to beating baseline (would need +0.2 minimum)

### Conclusion: NO ALPHA

**The model does not generate exploitable alpha on BTC 7d predictions with current features.**

This is an **honest null result**, which is valuable:
- Confirms leakage fix worked (no false positives)
- Establishes true performance floor
- Guides next steps (need different approach)

---

## LSTM Results

⚠️ **Training Error**: LSTM encountered `RuntimeError: all elements of input should be between 0 and 1` during BCELoss computation.

**Likely causes**:
1. NaN values in on-chain features (36.6% NaN rate) propagating through LSTM
2. Numerical instability in sequence normalization
3. Issue with sigmoid output handling

**Status**: Needs debugging (deferred to focus on XGBoost findings)

---

## Strategic Decision Point

### Current Situation
- ✅ Data is leakage-free (validated)
- ✅ Models pass all integrity checks
- ❌ XGBoost shows NO ALPHA (Sharpe ≈ 0)
- ❌ Performance worse than baseline

### Options

**Option A: Pivot Strategy (RECOMMENDED)**
- **Issue**: Current features (technical + on-chain BTC) don't predict 7d BTC returns
- **Action**: Test alternative approaches:
  1. Different horizon (1d, 14d, 30d instead of 7d) — Phase 2
  2. Feature ablation to identify which features are useful — Phase 2
  3. Different asset (ETH with gas/staking features)
  4. Different target (volatility, regime change instead of direction)

**Option B: Terminate Project**
- If feature ablation + horizon optimization still show Sharpe < 0.50
- Acknowledge that BTC is too efficient for simple ML models
- Focus resources elsewhere

**Option C: Continue with Current Approach (NOT RECOMMENDED)**
- Hyperparameter tuning unlikely to add 0.8 Sharpe
- Risk wasting time on fundamentally flawed hypothesis

### Recommended Next Steps

1. **Phase 2: Feature Ablation** (2-3 hours)
   - Test if on-chain features add ANY value vs technical-only
   - If delta Sharpe < 0.05 → on-chain is useless, confirm termination
   - If delta Sharpe > 0.1 → maybe salvageable

2. **Phase 3: Horizon Optimization** (2-3 hours)
   - Test 1d, 3d, 14d, 30d horizons
   - Identify if any horizon shows Sharpe > 0.50
   - If all negative → strong evidence of NO ALPHA

3. **Decision Gate** (after Phases 2-3)
   - If best Sharpe still < 0.50 → **TERMINATE** or pivot to ETH
   - If best Sharpe > 0.70 → Continue to multi-seed validation

---

## Comparison to Original Findings

### Before Leakage Fix
- 30d horizon: Sharpe = 0.86 (but FAILED leakage test)
- 7d horizon: Sharpe = -0.45

### After Leakage Fix
- 7d horizon: Sharpe = 0.0037 (leakage-free, honest)

The 30d result was **fiction** (data leakage). The honest result is near-zero Sharpe.

---

## Files Generated
- `scripts/run_phase1_baselines.py` — Phase 1 experiment script
- `roadmap/PHASE1_RESULTS.md` — This file

## Next Session TODO
1. Fix LSTM training error (NaN handling in sequence creation)
2. Run Phase 2: Feature ablation to test on-chain value
3. Run Phase 3: Horizon optimization (test 1d, 3d, 14d, 30d)
4. Make GO/NO-GO decision based on Phase 2-3 results

---

**Phase 1 Status**: ✅ COMPLETE (XGBoost validated, LSTM needs debug)
**Key Finding**: ❌ NO ALPHA with current approach
**Recommendation**: Proceed to Phase 2-3 to test alternatives before terminating
