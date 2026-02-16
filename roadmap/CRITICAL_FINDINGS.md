# CRITICAL FINDINGS — Phase 3

## 1. Severe Underperformance vs Baseline (BLOCKING ISSUE)

**Baseline (BuyAndHold BTC)**: Sharpe = 0.79, MaxDD = 76.6%

**XGBoost Results**:
- 1d horizon: Sharpe = -0.43, MaxDD = 91.9%
- 3d horizon: Sharpe = -0.56, MaxDD = 93.8%
- 7d horizon: Sharpe = -0.45, MaxDD = 93.9%
- 14d horizon: Sharpe = 0.22, MaxDD = 82.1%
- 30d horizon: Sharpe = 0.86, MaxDD = 57.7% **[LEAKAGE DETECTED]**

**Analysis**:
- Most horizons (1d-7d) produce NEGATIVE Sharpe ratios
- All horizons underperform simple buy-and-hold when accounting for costs
- Only 30d horizon beats baseline, but fails leakage detection
- Leakage check failure: `shuffled_label` test failed (model performs well even with random labels)

## 2. Data Leakage Hypothesis

**Evidence**:
1. 30d horizon model achieves Sharpe=0.86 but fails shuffled_label test
2. Removing "returns_1d" feature IMPROVES performance (+0.48 Sharpe delta)
3. Pattern: Longer horizons → better performance (suspicious trend)

**Potential Causes**:
1. **Target variable leakage**: Despite correct target timing logic, the alignment or shifting may be off
2. **Feature leakage**: The returns_1d feature might be computed incorrectly or aligned incorrectly
3. **Backtester implementation**: Execution timing may not match documented specification
4. **Data quality**: NaN handling (46% NaN in features) may be causing spurious patterns

## 3. Feature Ablation Paradox

Removing the "returns" feature group IMPROVED Sharpe from -0.45 to +0.03. This suggests:
- The returns_1d feature is actively harmful
- Possible forward-looking bias in how returns are computed
- Target variable may be correlated with returns in unexpected ways

## 4. Recommendation

**IMMEDIATE ACTION REQUIRED**:
1. Audit target variable generation in `prepare_phase3_data.py`
2. Verify backtester execution timing matches specification (T close → T+1 open → T+1+N close)
3. Check returns_1d feature computation for off-by-one errors
4. Re-run leakage detector on ALL experiments, not just 30d

**PHASE 3 STATUS**: NO-GO for paper trading until leakage issue resolved.

All subsequent experiments (Tasks 5-11) completed for documentation purposes only.
Results cannot be trusted until root cause identified and fixed.

---
*Created: 2026-02-16 01:02:00 UTC*
*Author: CEO agent*
