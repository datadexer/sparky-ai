# CONTRACT #003: FINAL CORRECTED VERDICT

**Date**: 2026-02-16
**Validated by**: Oversight Opus (with CEO + Validation Agent + RBM inputs)
**Status**: DEBUNKED — regime approaches provide no improvement over baseline

---

## Executive Summary

The claimed Sharpe 2.656 "breakthrough" from the Regime-Weighted Ensemble was caused by
**look-ahead bias in the backtest framework**. After fixing the bug (shifting signals by 1 day),
no regime approach beats the simple Multi-Timeframe Donchian baseline.

---

## Look-Ahead Bias (Root Cause)

**Bug**: `strategy_returns = signals * price_returns` where signal[T] uses close[T].
Since return[T] = close[T]/close[T-1] - 1, the strategy captures the very price move
used to generate the signal.

**Fix**: `actual_positions = signals.shift(1); strategy_returns = actual_positions * price_returns`

**Impact**: All Sharpe ratios were inflated by 43-256%.

**Fix merged**: PR #12 (2026-02-16)

---

## Corrected Results (Post Look-Ahead Fix)

| Rank | Approach | Corrected Sharpe | Old (Biased) | Inflation |
|------|----------|-----------------|-------------|-----------|
| 1 | **BASELINE: Multi-TF Donchian (20/40/60)** | **1.062** | 1.878 | +77% |
| 2 | Markov-Switching Ensemble | 1.029 | — | — |
| 3 | Regime-Weighted Ensemble (aggressive) | 1.017 | 2.656 | +161% |
| 4 | Regime-Weighted Ensemble (balanced) | 1.017 | 2.656 | +161% |
| 5 | Volatility Term Structure Ensemble | 0.943 | — | — |
| 6 | Adaptive Lookback Ensemble | 0.868 | — | — |
| 7 | Regime Multi-TF Ensemble (aggressive) | 0.851 | — | — |
| 8 | HMM 2-State | 0.742 | 2.641 | +256% |
| 9 | HMM 3-State | 0.612 | — | — |

**Key finding**: The simple baseline BEATS all regime approaches after correction.

---

## Yearly Breakdown (Regime-Weighted Ensemble)

| Year | Biased Sharpe | Corrected Sharpe | Note |
|------|--------------|-----------------|------|
| 2019 | 3.024 | 1.256 | Bull market, bias extreme |
| 2020 | 3.884 | 2.064 | Strong uptrend, less bias |
| 2021 | 2.646 | 0.592 | Significant inflation |
| 2022 | 0.172 | **-1.349** | Bear market — LOST MONEY |
| 2023 | 3.552 | 1.692 | Recovery, high inflation |

**2022 is particularly telling**: The biased backtest showed a small positive (0.172),
but the corrected version shows the strategy LOST money (-1.349). The look-ahead bias
allowed the strategy to "avoid" losses it would have actually taken.

---

## Secondary Findings

### Multi-Seed Stability (HMM Approaches)
- HMM 2-State: CV = 0.093 — STABLE (but Sharpe only 0.742 after correction)
- HMM 3-State: CV = 0.093 — STABLE (but Sharpe only 0.612 after correction)
- Label switching detected: Expected behavior, fixable with state sorting
- **Conclusion**: Multi-seed stability is MOOT — the approaches don't beat baseline

### Parameter Sensitivity (Regime Ensemble)
- CEO ran 140 parameter combinations on BIASED framework
- Reported 79.3% produced "good" results (Sharpe > 1.5)
- **These numbers are invalid** — need re-run with corrected framework
- Relative robustness (plateau vs peak) likely holds, but absolute performance is much lower

### WalkForwardBacktester Engine
- Same look-ahead pattern exists in `_compute_equity_curve` when used with
  signals that depend on same-day close prices
- Added documentation warning to engine.py
- ML model predictions (which use features computed at T-1) are NOT affected

---

## What IS Validated

**Multi-Timeframe Donchian (20/40/60)**: Corrected Sharpe **1.062**
- Survives look-ahead bias correction
- Still beats Buy & Hold
- Simple, deterministic, no regime detection needed
- **Ready for paper trading** (pending Human Gate approval)

---

## Recommendations

### Immediate
1. **Accept corrected baseline (Sharpe 1.062)** as the best validated strategy
2. **Abandon regime ensemble research** — no improvement over baseline after correction
3. **Request Human Gate** for paper trading with Multi-TF Donchian

### Strategic
1. Look-ahead bias fix must be applied to ANY future backtest
2. Parameter sensitivity should be re-run with corrected framework (but low priority)
3. Consider whether ML approaches (which don't have this bias) can beat 1.062
4. The corrected HMM approaches (0.742, 0.612) are WORSE than baseline — don't pursue

---

## Lessons Learned

1. **Always verify backtest timing**: Signal at T must use only data available before T
2. **Breakout strategies are most affected**: The entry signal fires when the breakout happens
3. **Bear markets reveal bias**: The biased 2022 Sharpe was 0.172 (slightly positive),
   corrected is -1.349 (significant loss)
4. **RBM was right**: The 70%+ overfitting probability estimate was essentially correct,
   though the mechanism was look-ahead bias, not statistical overfitting
5. **Validation protocol works**: The 6-step validation caught the bug (Step 4 caveat was prophetic)

---

## Files

- **Look-ahead bias analysis**: `CRITICAL_FINDING_LOOKAHEAD_BIAS.md`
- **Multi-seed audit**: `MULTI_SEED_AUDIT_SUMMARY.md`, `multi_seed_audit.md`
- **Original verdict**: `VALIDATION_VERDICT.md` (includes addendum with correction)
- **Corrected comparison**: `regime_approaches_comparison.json` (re-run with fix)
