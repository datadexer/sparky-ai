# ML Underperformance Analysis

**Date**: 2026-02-16
**Context**: 58-feature hyperparameter sweep showing best Sharpe 0.967 vs baseline 1.062

---

## Observed Problem

After extensive feature engineering (23 → 58 features) and hyperparameter sweeps:

| Approach | Best Sharpe | vs Baseline | Accuracy |
|----------|-------------|-------------|----------|
| Multi-TF Donchian | 1.062 | baseline | N/A |
| Single Donchian(40/20) | 1.243 | +17% | N/A |
| XGBoost (23 features) | 0.050 | -95% | 51.5% |
| CatBoost (58 features, interim) | 0.967 | -9% | 51.7-53.3% |
| Cross-asset XGBoost | N/A | N/A | 53.4% |

**Key finding**: ML models struggle to beat simple rule-based breakout strategies, even with 58 features.

---

## Hypothesis: Why ML Underperforms

### 1. Crypto Regime Simplicity
**Theory**: BTC price dynamics from 2019-2023 may be simpler than expected.

- **2020-2021**: Strong bull market (easy: just be long)
- **2022**: Bear market (Donchian exits quickly via breakdowns)
- **2023**: Recovery (Donchian catches new uptrend)

Donchian works because crypto trends are STRONG and PERSISTENT. ML tries to predict day-ahead moves, but crypto doesn't mean-revert on daily timescales — it trends for weeks/months.

**Evidence**:
- Donchian Sharpe 1.062-1.243 (simple trend following)
- ML accuracy 51-53% (barely better than random)
- ML models overfit to noise, miss big trends

### 2. Look-Ahead Bias Still Present?
**Theory**: Subtle timing issues remain in feature calculation.

**Evidence against**:
- Fixed signal.shift(1) in backtest
- Features computed from hourly data with proper lagging
- Validation spans multiple regimes (2020-2023)

**Verdict**: Unlikely. Bias would inflate Sharpe, not deflate it.

### 3. Sample Size Insufficient
**Theory**: 4,795 daily samples not enough for 58 features.

**Math**:
- 4,795 samples / 4 years validation = 1,199 samples/year
- 58 features → ~20 samples per feature per year
- Rule of thumb: need 10-50 samples per feature for tree models

**Verdict**: Borderline. More data would help, but cross-asset (11,931 samples) also failed (53.4% accuracy).

### 4. Feature Quality vs Feature Quantity
**Theory**: Adding 35 features added more noise than signal.

**Evidence**:
- 23 features: Sharpe 0.050 (terrible)
- 58 features: Sharpe 0.967 (better, but still below baseline)
- Accuracy stayed flat (51-53%) despite +35 features

**Interpretation**: New features improved Sharpe by reducing overfitting (more regularization via noise), but didn't add genuine signal.

### 5. Wrong Target Definition
**Theory**: Predicting next-day direction is wrong objective for crypto.

**Current target**: `target = (close[t+1] > close[t])`

**Problems**:
1. Crypto gaps overnight → next-day close may not reflect trend
2. Trends last weeks/months, not days
3. Transaction costs (0.26%) require >52% accuracy to break even

**Alternative targets**:
- Next 7-day return (trend prediction)
- Breakout persistence (does breakout hold for N days?)
- Volatility regime (classify high/low vol, adjust sizing)

### 6. Model Mismatch to Regime Structure
**Theory**: Tree models (XGBoost, CatBoost, LightGBM) poor fit for time-series trends.

**Evidence**:
- Tree models excellent for cross-sectional data (fraud, credit scoring)
- Tree models struggle with temporal dependencies (trends, momentum)
- LSTMs/transformers better for sequences, but need >10K samples

**Verdict**: Likely. Trees don't naturally capture "if price is rising, keep rising" logic. They split on static thresholds, not dynamic regimes.

---

## What Works (Confirmed)

1. **Donchian breakouts** (Sharpe 1.062-1.243)
   - Captures strong crypto trends
   - Low turnover (cost-efficient)
   - Robust across regimes (2020-2023)

2. **Multi-timeframe ensembles** (not yet tested with ML)
   - Combine 20/10, 40/20, 60/30 Donchian channels
   - Reduce false breakouts via consensus

3. **Simple = Better**
   - 2-parameter Donchian beats 58-feature XGBoost
   - Overfitting is the enemy, not complexity ceiling

---

## Next Steps (Graduated Exploration)

### Tier 1: Finish Current Sweep (MANDATORY)
- Wait for 54/54 configs to complete
- Extract top 5 configs by Sharpe
- Test ensemble methods (weighted, stacking, voting)

**Success criteria**: Ensemble Sharpe >1.062
**Estimated time**: 12 hours (sweep) + 2 hours (ensemble testing)

### Tier 2: Hybrid Approaches
- **ML Filter on Donchian**: Only take trades when ML confidence >0.6
- **ML Position Sizing**: Scale position by ML probability
- **Regime-Aware ML**: Train separate models for high/low volatility

**Success criteria**: Hybrid Sharpe >1.062
**Estimated time**: 4-6 hours

### Tier 3: Alternative Targets
- Train on 7-day forward returns (trend prediction)
- Train on breakout persistence (will breakout hold?)
- Train on volatility regime classification

**Success criteria**: Sharpe >0.7 (TIER 2 threshold)
**Estimated time**: 8-10 hours

### Tier 4: Alternative Model Families
- LSTMs (if sample size expands to 10K+ via hourly targets)
- Transformers (if sample size expands to 20K+)
- Gaussian Processes (non-parametric, good for small data)

**Success criteria**: Sharpe >0.7
**Estimated time**: 10-15 hours

### Tier 5: Accept Simple Baselines
If Tiers 1-4 all fail (Sharpe <1.062 after 30+ configs), conclusion:

**Simple breakout strategies are superior to ML for crypto (2019-2023 regime).**

Deploy Multi-TF Donchian (Sharpe 1.062, validated, robust) and move to:
- Paper trading validation
- Live trading preparation
- Continuous monitoring for regime shifts

---

## Open Questions

1. **Why does Donchian work so well?**
   - Crypto trends are stronger than traditional assets?
   - BTC 2019-2023 was a unique regime (will it persist)?
   - Breakouts are self-fulfilling (more traders use them)?

2. **Can ML add ANY value?**
   - Even 1% improvement (1.062 → 1.073) would justify hybrid approach
   - ML filter to reduce false breakouts?
   - ML for exit timing (let Donchian enter, ML optimize exit)?

3. **Is more data the answer?**
   - 4,795 daily samples → 115K hourly samples
   - Could train hourly models, aggregate to daily signals
   - Risk: overfitting to microstructure noise

---

## Conclusion (Interim)

After 14/54 configs, evidence suggests:

1. **ML struggles with crypto trend prediction** (51-53% accuracy, Sharpe 0.629-0.967)
2. **Simple baselines excel** (Donchian Sharpe 1.062-1.243)
3. **Feature engineering helps but not enough** (23 → 58 features, 0.050 → 0.967 Sharpe, still below baseline)

**Next actions**:
1. Complete sweep (40 more configs)
2. Test ensembles (may push above 1.062 via variance reduction)
3. Test hybrid approaches (ML filter on Donchian)
4. If all fail: accept simple baselines, deploy, iterate in live trading

**Critical insight**: Overfitting to in-sample data is EASY. Building a model that generalizes across regimes (2020 bull, 2022 bear, 2023 recovery) is HARD. Donchian passes this test. ML has not (yet).
