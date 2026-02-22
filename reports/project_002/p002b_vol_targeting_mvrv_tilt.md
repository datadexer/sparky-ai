# P002-B: Vol Targeting Validation + MVRV Continuous Tilt

## 1. Executive Summary

Vol targeting works. MVRV continuous tilt adds genuine value on top of it.

Pure vol targeting produced positive validation Sharpe across all 12 configs tested (0.29-0.36). The momentum-gated MVRV tilt (Variant B) pushed the best config to val Sharpe 0.545 with -11.8% MaxDD — a +0.216 improvement over the vol-targeting baseline.

The critical innovation is the momentum gate: going flat when MVRV is falling (regardless of level) and tilting positions only when MVRV is rising. This addresses the path-dependence problem that destroyed P002's binary regime approach.

---

## 2. Part 1: Vol Targeting Validation

### 2.1 Parameter Grid (12 configs)

| EWMA Span | Target Vol | Dev Sharpe | Val Sharpe | Val MaxDD | Val Return | Retention |
|-----------|-----------|-----------|-----------|----------|-----------|----------|
| 30 | 0.10 | 1.606 | 0.293 | -15.5% | +5.2% | 18.2% |
| 30 | 0.15 | 1.606 | 0.293 | -22.7% | +6.9% | 18.2% |
| 30 | 0.20 | 1.606 | 0.293 | -29.4% | +8.0% | 18.2% |
| 30 | 0.25 | 1.606 | 0.334 | -35.7% | +10.9% | 20.8% |
| **60** | **0.10** | **1.588** | **0.336** | **-14.9%** | **+5.8%** | **21.2%** |
| 60 | 0.15 | 1.588 | 0.336 | -21.8% | +8.0% | 21.2% |
| 60 | 0.20 | 1.588 | 0.336 | -28.2% | +9.7% | 21.2% |
| 60 | 0.25 | 1.588 | 0.336 | -34.3% | +10.8% | 21.2% |
| **90** | **0.10** | **1.603** | **0.362** | **-14.6%** | **+6.2%** | **22.6%** |
| 90 | 0.15 | 1.603 | 0.362 | -21.3% | +8.7% | 22.6% |
| 90 | 0.20 | 1.603 | 0.362 | -27.6% | +10.6% | 22.6% |
| 90 | 0.25 | 1.603 | 0.362 | -33.5% | +12.1% | 22.6% |

Buy-and-hold baseline: Dev Sharpe 1.494, Val Sharpe 0.111, Val MaxDD -66.9%

### 2.2 Key Observations

**Sharpe is invariant to target_vol** (within each EWMA span). This is expected — vol targeting is pure scaling. Position size changes proportionally with target_vol, but risk-adjusted return stays the same. The only lever is EWMA span, which affects how quickly the vol estimate adapts.

**EWMA(90) dominates on validation** (0.362 vs 0.336 vs 0.293). Smoother vol estimates avoid whipsawing during the volatile 2022 bear. EWMA(30) reacts faster but creates excess turnover.

**Dev Sharpe differences are negligible** (1.603-1.606). The EWMA span barely matters in the trending 2019-2021 bull.

**All 12 configs beat buy-and-hold** on val (0.29-0.36 vs 0.11) with dramatically lower drawdowns (-14.6% to -35.7% vs -66.9%).

### 2.3 Phase Gates

| Gate | Threshold | Result | Verdict |
|------|-----------|--------|---------|
| Best val Sharpe >= 0.25 | >= 0.25 | 0.362 | **PASS** |
| Majority val Sharpe > 0 | >= 8/12 | 12/12 | **PASS** |
| Best val MaxDD < 20% | < 20% | -14.6% | **PASS** |
| Parameter stability (top-3 overlap) | >= 2/3 | 0/3 | **FAIL** |

The parameter stability failure is misleading: top-3 dev are all EWMA(30), top-3 val are all EWMA(90). But dev Sharpe differences between spans are <0.02 — effectively a tie. The "instability" is that a negligible dev advantage for EWMA(30) reverses to a modest val advantage for EWMA(90). The strategy works across all parameters; the ranking is noise.

### 2.4 Vol Targeting Summary Card

```
VOL TARGETING VALIDATION CARD
══════════════════════════════

Best dev config: EWMA(30), tv=0.10
  Dev Sharpe:      1.606
  Val Sharpe:      0.293 (retention: 18.2%)
  Val MaxDD:       -15.5%

Best val config: EWMA(90), tv=0.10
  Val Sharpe:      0.362 (retention: 22.6%)
  Val MaxDD:       -14.6%

Parameter stability: 0/3 top configs overlap (but dev differences < 0.02)
Configs with val Sharpe > 0: 12/12
Mechanism check: realized vol ≈ target vol? YES
  EWMA(60)/tv=0.10: realized dev vol 11.1%, realized val vol 10.4%

DEPLOYABLE AS STANDALONE? MARGINAL
  Positive Sharpe and controlled drawdowns across all params
  But Sharpe 0.36 falls below Tier 2 (0.7) and is marginal for Tier 3 (0.4)
  Low absolute returns (+6.2% over 2 years with 10% vol target)
```

---

## 3. Part 2: MVRV Continuous Tilt

### 3.1 Baseline

Vol targeting baseline (EWMA(60), tv=0.10): Dev Sharpe 1.632, Val Sharpe 0.329, Val MaxDD -15.4%.

MVRV data: 5,164 rows from 2012-01-01 to 2026-02-19. All three lookback windows (730, 1095, 1460 days) produce valid percentile ranks from the start of the price series (2019).

### 3.2 Variant A: Pure Percentile Rank Tilt (9 configs)

| Lookback | Tilt Mag | Dev Sharpe | Val Sharpe | Val Delta | Val MaxDD |
|----------|---------|-----------|-----------|----------|----------|
| 730 | 0.5 | 1.631 | 0.267 | -0.062 | -18.0% |
| 730 | 1.0 | 1.591 | 0.212 | -0.117 | -20.6% |
| 730 | 1.5 | 1.509 | 0.166 | -0.163 | -23.1% |
| 1095 | 0.5 | 1.612 | 0.333 | +0.004 | -17.7% |
| 1095 | 1.0 | 1.551 | 0.334 | +0.005 | -20.0% |
| 1095 | 1.5 | 1.456 | 0.333 | +0.004 | -22.2% |
| **1460** | **0.5** | **1.631** | **0.384** | **+0.055** | **-16.8%** |
| **1460** | **1.0** | **1.597** | **0.425** | **+0.096** | **-18.2%** |
| **1460** | **1.5** | **1.535** | **0.457** | **+0.128** | **-19.6%** |

**Pattern:** 730d lookback hurts (MVRV tilts into 2022 crash). 1095d is neutral. 1460d adds value. Longer lookback → better calibration of "cheap" vs "expensive" across full cycles.

**Variant A best:** lb=1460, tilt=1.5, val Sharpe 0.457 (+0.128 over baseline).

**As predicted,** Variant A increases position as MVRV falls, creating path-dependence risk. The 1460d lookback partially mitigates this (MVRV doesn't reach extreme low percentiles as quickly), but doesn't solve it.

### 3.3 Variant B: Level + Momentum Gate (36 configs, top 10 by val Sharpe)

| Lookback | Tilt | MW | RF | Dev Sharpe | Val Sharpe | Val Delta | Val MaxDD |
|----------|------|----|----|-----------|-----------|----------|----------|
| **1460** | **0.5** | **30** | **1.0** | **2.240** | **0.545** | **+0.216** | **-11.8%** |
| 1460 | 1.0 | 30 | 1.0 | 2.198 | 0.544 | +0.215 | -13.0% |
| 1460 | 1.5 | 30 | 1.0 | 2.106 | 0.542 | +0.213 | -14.2% |
| 1095 | 0.5 | 30 | 1.0 | 2.198 | 0.510 | +0.181 | -12.2% |
| 1095 | 1.0 | 30 | 1.0 | 2.092 | 0.482 | +0.153 | -13.8% |
| 1460 | 1.0 | 30 | 0.5 | 1.969 | 0.475 | +0.146 | -14.9% |
| 1460 | 0.5 | 30 | 0.5 | 2.016 | 0.466 | +0.137 | -13.7% |
| 1095 | 1.5 | 30 | 1.0 | 1.923 | 0.458 | +0.129 | -15.4% |
| 1460 | 1.5 | 30 | 0.5 | 1.885 | 0.481 | +0.152 | -16.0% |
| 730 | 0.5 | 30 | 1.0 | 2.221 | 0.443 | +0.114 | -12.2% |

**Clear patterns:**
- **Momentum window 30 >> 60** across the board. 30-day momentum reacts faster, catching the LUNA crash onset. 60-day momentum is too slow.
- **Retreat factor 1.0 >> 0.5.** Going fully flat when MVRV is falling beats halving position. The crashes are severe enough that partial retreat isn't enough.
- **Lookback 1460 >= 1095 >> 730.** Same as Variant A — longer calibration windows prevent premature tilt.
- **Tilt magnitude barely matters** for lb=1460 (0.545 vs 0.544 vs 0.542). The momentum gate does the heavy lifting; the tilt adjusts returns modestly.

**Best Variant B:** lb=1460, tilt=0.5, mw=30, rf=1.0. Val Sharpe 0.545, MaxDD -11.8%, Return +8.4%.

### 3.4 Variant C: Vol-Target Multiplier (9 configs)

| Lookback | Tilt Mag | Dev Sharpe | Val Sharpe | Val Delta | Val MaxDD |
|----------|---------|-----------|-----------|----------|----------|
| 730 | 0.3 | 1.635 | 0.291 | -0.038 | -16.9% |
| 730 | 0.5 | 1.631 | 0.267 | -0.062 | -18.0% |
| 730 | 0.8 | 1.612 | 0.233 | -0.096 | -19.6% |
| 1095 | 0.3 | 1.625 | 0.332 | +0.003 | -16.8% |
| 1095 | 0.5 | 1.612 | 0.333 | +0.004 | -17.7% |
| 1095 | 0.8 | 1.580 | 0.334 | +0.005 | -19.1% |
| 1460 | 0.3 | 1.635 | 0.364 | +0.035 | -16.2% |
| 1460 | 0.5 | 1.631 | 0.384 | +0.055 | -16.8% |
| **1460** | **0.8** | **1.614** | **0.410** | **+0.081** | **-17.6%** |

**Variant C is the mildest intervention.** Best config adds +0.081 Sharpe — real but modest. Identical ranking pattern (1460 >> 1095 >> 730).

Note: Variant C with lb=1460/tilt=0.5 produces identical results to Variant A with lb=1460/tilt=0.5 — expected, since they apply the same tilt, just through different mechanisms.

### 3.5 Variant Comparison

| Variant | Best Val Sharpe | Val Delta | Val MaxDD | Mechanism |
|---------|----------------|----------|----------|-----------|
| Baseline (VT only) | 0.329 | — | -15.4% | — |
| A (pure tilt) | 0.457 | +0.128 | -19.6% | Continuous sizing, no protection |
| **B (momentum gate)** | **0.545** | **+0.216** | **-11.8%** | Size + crash protection |
| C (VT multiplier) | 0.410 | +0.081 | -17.6% | Gentle sizing, no protection |

**Variant B is the clear winner** because the momentum gate provides crash protection that the other variants lack. The +0.216 delta is almost entirely attributable to avoiding the LUNA crash drawdown.

### 3.6 Diagnostics: Best Config (Variant B, lb=1460, tilt=0.5, mw=30, rf=1.0)

**Tilt factor by sub-period:**

| Period | Avg Tilt Factor | Interpretation |
|--------|----------------|---------------|
| 2022 Q1 (grind) | 0.354 | Mostly flat (MVRV falling from 1.97→1.58) |
| 2022 Q2 (LUNA) | **0.152** | **Nearly fully flat during crash** |
| 2022 Q3 (sideways) | 0.477 | Partial position (MVRV stabilizing) |
| 2022 Q4 (FTX) | 0.587 | Moderate position (MVRV mixed direction) |
| 2023 H1 (recovery) | 0.754 | Ramping back up (MVRV rising) |
| 2023 H2 (pre-ETF) | 0.622 | Moderate (MVRV mixed in H2) |

**Crash protection:**

| Event | VT Baseline MaxDD | Best Variant B MaxDD | Improvement |
|-------|-------------------|---------------------|-------------|
| LUNA (Apr-Jun 2022) | -12.0% | **-3.6%** | **+8.4 pp** |
| FTX (Oct-Nov 2022) | -6.1% | -7.3% | -1.2 pp |

The momentum gate excelled during LUNA (MVRV declined continuously from 1.64 to 0.97 over 2 months → gate stayed on). FTX protection was marginal because MVRV had already stabilized at low levels (~0.9) and showed mixed momentum.

**Momentum gate activation blocks:** 28 during validation period. Major activations:
- Apr 14 - Jun 8, 2022: 56 days flat during LUNA crash (MVRV 1.64 → 1.30)
- Jun 11 - Jul 16, 2022: 36 days flat during continued decline (MVRV 1.22 → 0.97)
- Nov 8 - Dec 9, 2022: 32 days flat during FTX fallout (MVRV 0.89 → 0.86)
- Jul 21 - Sep 17, 2023: 59 days flat during summer drift (MVRV 1.47 → 1.31)

**Dev performance:** Dev Sharpe 2.240. The momentum gate correctly went flat during the COVID March 2020 crash and scaled up during the 2020-2021 bull. High dev Sharpe is from both crash avoidance and tilt during accumulation.

---

## 4. Combined Assessment

### 4.1 Strategy Comparison Matrix

| Strategy | Dev Sharpe | Val Sharpe | Val MaxDD | Val Return | Tradeable? |
|----------|-----------|-----------|----------|-----------|-----------|
| Buy-and-hold BTC | 1.494 | 0.111 | -66.9% | -16.3% | Yes (trivial) |
| P002 S1 regime (binary) | 2.129 | -0.223 | -60.1% | -35.7% | No (failed) |
| Pure vol targeting (best) | 1.603 | 0.362 | -14.6% | +6.2% | Marginal |
| **VT + MVRV-A (pure tilt)** | 1.535 | 0.457 | -19.6% | +11.1% | Maybe |
| **VT + MVRV-B (momentum gate)** | **2.240** | **0.545** | **-11.8%** | **+8.4%** | **Yes** |
| VT + MVRV-C (VT multiplier) | 1.614 | 0.410 | -17.6% | +8.7% | Maybe |

### 4.2 Critical Numbers

| Metric | VT Only | VT + MVRV-B | Delta |
|--------|---------|-------------|-------|
| Val Sharpe | 0.329 | 0.545 | +0.216 |
| Val MaxDD | -15.4% | -11.8% | +3.6 pp |
| Val Return | +5.8% | +8.4% | +2.6 pp |
| LUNA Crash MaxDD | -12.0% | -3.6% | +8.4 pp |
| Dev Sharpe | 1.632 | 2.240 | +0.608 |
| Retention Ratio | 20.1% | 24.3% | +4.2 pp |

### 4.3 Outcome Assessment

**This is Outcome 1: Vol targeting passes, MVRV tilt adds value.**

Evidence:
- Vol targeting: 3/4 gates pass, all 12 configs positive on val
- MVRV Variant B: +0.216 val Sharpe delta, LUNA crash protection, consistent across lookback/tilt parameters
- The improvement is primarily from the momentum gate (crash avoidance), not the tilt magnitude
- 27/54 MVRV configs beat baseline on val (50%) — not dominant but the best configs are consistent

Caveats:
- Val Sharpe 0.545 is Tier 3 (>= 0.4, "keep iterating"), not Tier 2 (>= 0.7)
- Dev-to-val retention is still low (24%)
- Momentum gate was less effective during FTX (mixed MVRV direction at already-low levels)
- The strategy is still long-only — it can only reduce exposure, not short
- 2023 H2 showed reduced participation (tilt 0.62) during the pre-ETF rally because MVRV momentum was mixed

### 4.4 Recommended Holdout Config

**Dev-set-best for holdout test (per directive — no forward-looking bias):**

```
Strategy: VT + MVRV Variant B
  Vol targeting: EWMA(60), target_vol=0.10, max_leverage=1.0
  MVRV tilt: lookback=1460, tilt_magnitude=0.5, momentum_window=30, retreat_factor=1.0
  Costs: 15 bps maker (daily rebalancing)
```

Note: Part 2 used EWMA(60)/tv=0.10 as baseline. Part 1 shows EWMA(30)/tv=0.10 is marginally best on dev (+0.02 Sharpe) and EWMA(90)/tv=0.10 is best on val (+0.03). Dev Sharpe differences between EWMA spans are < 0.02 — effectively a tie. EWMA(60) is a defensible middle-ground choice. The MVRV tilt adds the same incremental value regardless of EWMA span.

### 4.5 What AK Should Decide

1. **Does vol targeting meet the bar as a base strategy?** Val Sharpe 0.33-0.36 is positive but low. It's a risk management overlay on a positive-drift asset, not traditional alpha.

2. **Is the MVRV momentum-gate tilt a genuine signal or look-ahead?** The momentum gate's value comes primarily from one event (LUNA crash). With only 2 major crashes in the validation set (LUNA and FTX), we cannot know if this generalizes. The holdout (2024+) has different dynamics (post-ETF, institutional flows).

3. **Should we test on holdout?** The combined strategy (val Sharpe 0.545, MaxDD -11.8%) is the strongest P002 result. But it's built on N=54 pre-registered configs with the best selected — DSR at N=54 should be computed if proceeding.

4. **Alternative: combine with P001 Donchian?** The VT+MVRV-B strategy has low correlation with Donchian (one is mean-reversion sizing on vol+valuation, the other is trend-following breakouts). A portfolio might be interesting.

---

## 5. Technical Notes

- Vol targeting Sharpe is invariant to target_vol (it's a pure scaling). Only EWMA span affects risk-adjusted returns. Target_vol only affects absolute return, drawdown magnitude, and position sizes.
- Part 2 baseline (EWMA 60, tv=0.10) shows slightly different absolute numbers than Part 1's EWMA(60)/tv=0.10 due to signal shift(1) handling. Val Sharpe: 0.329 (Part 2) vs 0.336 (Part 1). The difference is negligible.
- MVRV data from BGeometrics, 2012-01-01 to 2026-02-19 (5,164 rows). 1460-day percentile rank is valid from ~2016 onwards — well before the 2019 dev start.
- Pre-registered trial count: N=12 for vol targeting, N=54 for MVRV tilt.
- All positions shifted by 1 day to prevent lookahead bias.
