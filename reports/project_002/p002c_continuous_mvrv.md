# P002-C: Continuous MVRV Tilt — Research Report

**Date**: 2026-02-21
**Pre-registered configs**: 99 (9 variants: A, C, D, E, F, G, H, I, J)
**Cost model**: 15 bps maker, daily rebalancing
**Data**: BTC daily, dev=2019-2021, val=2022-2023

## Executive Summary

**NEGATIVE RESULT.** No continuous MVRV tilt variant beats VT-only on BOTH val Sharpe AND val MaxDD.

- 63/99 configs beat VT on val Sharpe alone
- Best val Sharpe delta: +0.143 (Variant D, sigmoid, lb=1460, steepness=8)
- **0/99 configs beat VT on BOTH val Sharpe AND val MaxDD**
- Root cause: MVRV reads "cheap" during crashes (pctrank below median) → tilts INCREASE position during crashes → systematically worsens drawdowns
- The val Sharpe improvement is entirely from overweighting during the 2023 recovery, not from crash protection

**Conclusion**: Close MVRV tilt research. VT-only remains the P002 strategy.

## VT-Only Baseline

| Metric | Dev (2019-2021) | Val (2022-2023) |
|--------|----------------|-----------------|
| Sharpe | 1.632 | 0.329 |
| MaxDD | -11.3% | -15.4% |
| Annual Return | 19.2% | 3.0% |

## Phase 1: Signal Characterization

### Key Finding: Tilts Are ABOVE 1.0 During Crashes

During 2022 crashes, MVRV had fallen from 2021 highs. With a 1460-day window including the bubble, MVRV percentile rank was below 0.5 → all variants read "cheap" → tilts go UP:

| Event | A | C | D | G | I |
|-------|------|------|------|------|------|
| COVID crash (2020-03) | 1.43 | 1.21 | — | — | — |
| BTC ATH (2021-11) | 0.64 | 0.82 | — | — | — |
| LUNA (2022-05) | 1.46 | 1.23 | — | — | — |
| Capitulation (2022-06) | 1.50 | 1.25 | — | — | — |
| FTX (2022-11) | 1.49 | 1.25 | — | — | — |
| Recovery (2023-01) | 1.47 | 1.24 | — | — | — |

This is the "MVRV cheap trap" — the signal correctly identifies expensive markets (reduces at ATH) but during crashes, the falling MVRV triggers "cheap" → overweights into further losses.

### Correlation Matrix

Variants A, C, D, G, J are highly correlated (r>0.99) — all monotonic transforms of the same MVRV percentile rank. Only Variant F (momentum modifier) differs meaningfully.

## Phase 2: Dev Results (2019-2021)

| Variant | Best Dev Sharpe | Delta | MaxDD | Config |
|---------|----------------|-------|-------|--------|
| I (Dual SOPR) | 1.680 | +0.048 | -10.7% | lb=1460, tilt=0.5, sw=3, spw=0.5 |
| F (Momentum) | 1.637 | +0.005 | -11.3% | lb=1460, mw=30, s=1 |
| C (VolTarget) | 1.635 | +0.003 | -11.3% | lb=730, tilt=0.3 |
| E (MultiHorizon) | 1.632 | +0.000 | -11.3% | sw=730, lw=1460, w=0.5 |
| A (Linear) | 1.631 | -0.001 | -11.3% | lb=1460, tilt=0.5 |
| D (Sigmoid) | 1.623 | -0.009 | -11.4% | lb=730, steep=2 |
| G (Asymmetric) | 1.598 | -0.034 | -11.5% | lb=1460, cm=0.3, em=0.5 |
| J (Floor/Ceil) | 1.597 | -0.035 | -11.5% | fl=0.0, cl=1.5 |
| H (Z-Score) | 1.373 | -0.259 | -15.8% | lb=1095, tilt=0.5, s=1.0 |

Dev results are mixed — most variants barely improve or slightly hurt on dev. H (z-score) is notably bad on dev, suggesting the z-score transform is too aggressive.

## Phase 3: Validation Results (2022-2023)

### Best per Variant (Val)

| Variant | Val Sharpe | Delta | Val MaxDD | Config |
|---------|-----------|-------|-----------|--------|
| D (Sigmoid) | 0.473 | +0.143 | -21.7% | lb=1460, steep=8 |
| A (Linear) | 0.457 | +0.128 | -19.6% | lb=1460, tilt=1.5 |
| G (Asymmetric) | 0.457 | +0.128 | -17.9% | lb=1460, cm=0.5, em=1.0 |
| J (Floor/Ceil) | 0.425 | +0.096 | -18.2% | fl=0.0, cl=1.5 |
| C (VolTarget) | 0.410 | +0.081 | -17.6% | lb=1460, tilt=0.8 |
| H (Z-Score) | 0.401 | +0.072 | -22.1% | lb=1460, tilt=0.8, s=1.0 |
| I (Dual SOPR) | 0.388 | +0.059 | -16.7% | lb=1460, tilt=0.5, sw=3, spw=0.5 |
| F (Momentum) | 0.351 | +0.022 | -16.3% | lb=1460, mw=30, s=1 |
| E (MultiHorizon) | 0.349 | +0.020 | -17.1% | lb=730, lw=1460, w=0.7 |

### Lookback = 1460 Dominates

All top configs use lb=1460 (4-year window). lb=730 uniformly hurts on val. This makes sense: a 2-year window during 2022-2023 only contains bear market data → distorted percentile rank.

### Beat-VT Rates

| Variant | N | Beat Val Sharpe | Beat Both (Sharpe + MaxDD) |
|---------|---|----------------|---------------------------|
| A | 9 | 6 | 0 |
| C | 9 | 7 | 0 |
| D | 9 | 6 | 0 |
| E | 12 | 4 | 0 |
| F | 12 | 5 | 0 |
| G | 12 | 12 | 0 |
| H | 12 | 6 | 0 |
| I | 16 | 13 | 0 |
| J | 8 | 4 | 0 |
| **Total** | **99** | **63** | **0** |

### Dev-Best → Val Check

| Variant | Dev Best | Val of Dev Best | Val Delta |
|---------|---------|-----------------|-----------|
| A | 1.631 | 0.384 | +0.055 |
| C | 1.635 | 0.291 | -0.038 |
| D | 1.623 | 0.238 | -0.091 |
| E | 1.632 | 0.326 | -0.004 |
| F | 1.637 | 0.351 | +0.022 |
| G | 1.598 | 0.407 | +0.078 |
| H | 1.373 | 0.357 | +0.028 |
| I | 1.680 | 0.388 | +0.059 |
| J | 1.597 | 0.425 | +0.096 |

Mixed — some dev-best configs do beat VT on val (A, F, G, H, I, J). But none beat on MaxDD.

### Parameter Sensitivity — Variant G (best risk-adjusted)

| Config | Val Sharpe | Delta | Val MaxDD |
|--------|-----------|-------|-----------|
| lb=1460, cm=0.5, em=1.0 | 0.457 | +0.128 | -17.9% |
| lb=1460, cm=0.5, em=0.8 | 0.444 | +0.115 | -18.0% |
| lb=1460, cm=0.3, em=1.0 | 0.442 | +0.113 | -16.7% |
| lb=1460, cm=0.3, em=0.8 | 0.428 | +0.099 | -16.8% |
| lb=1460, cm=0.5, em=0.5 | 0.425 | +0.096 | -18.2% |
| lb=1460, cm=0.3, em=0.5 | 0.407 | +0.078 | -16.9% |
| lb=1095, cm=0.3, em=1.0 | 0.344 | +0.015 | -18.0% |
| lb=1095, cm=0.3, em=0.8 | 0.341 | +0.012 | -18.0% |
| lb=1095, cm=0.5, em=1.0 | 0.341 | +0.012 | -19.9% |
| lb=1095, cm=0.5, em=0.8 | 0.338 | +0.009 | -19.9% |
| lb=1095, cm=0.3, em=0.5 | 0.337 | +0.008 | -18.1% |
| lb=1095, cm=0.5, em=0.5 | 0.334 | +0.005 | -20.0% |

Smooth plateau within lb=1460 (all 6 configs, delta +0.078 to +0.128). Sharp drop at lb=1095.

## Crash-Period Diagnostics

### LUNA Crash (Apr-Jun 2022)

| Strategy | Return | MaxDD | Avg Position | Avg Tilt |
|----------|--------|-------|-------------|----------|
| VT-only | -11.1% | -12.0% | 0.149 | 1.00 |
| D (steep=8) | -15.6% | -16.7% | 0.209 | **1.449** |
| A (tilt=1.5) | -14.3% | -15.4% | 0.192 | **1.328** |
| G (cm=0.5, em=1.0) | -13.1% | -14.0% | 0.175 | **1.206** |
| I (tilt=0.5, sopr) | -12.0% | -13.0% | 0.161 | **1.093** |

**Every tilt variant has a tilt ABOVE 1.0 during the LUNA crash.** MVRV was "cheap" → signal says "buy more" → larger positions into the crash → worse drawdown.

### FTX Crash (Oct-Nov 2022)

| Strategy | Return | MaxDD | Avg Position | Avg Tilt |
|----------|--------|-------|-------------|----------|
| VT-only | -2.4% | -6.1% | 0.181 | 1.00 |
| D (steep=8) | -4.0% | -10.3% | 0.313 | **1.737** |
| A (tilt=1.5) | -3.2% | -9.1% | 0.289 | **1.613** |
| G (cm=0.5, em=1.0) | -3.0% | -8.1% | 0.253 | **1.409** |
| I (tilt=0.5, sopr) | -2.6% | -7.1% | 0.210 | **1.158** |

By FTX, MVRV was even lower → tilts even higher → even worse relative performance. D's tilt of 1.737 means 74% higher position than VT-only during FTX collapse.

### 2023 Recovery (Jan-Oct 2023)

| Strategy | Return | MaxDD | Avg Position | Avg Tilt |
|----------|--------|-------|-------------|----------|
| VT-only | +12.0% | -6.6% | 0.235 | 1.00 |
| D (steep=8) | +22.3% | -9.2% | 0.362 | 1.548 |
| A (tilt=1.5) | +20.1% | -7.9% | 0.313 | 1.335 |
| G (cm=0.5, em=1.0) | +17.3% | -7.5% | 0.287 | 1.223 |
| I (tilt=0.5, sopr) | +14.2% | -7.1% | 0.261 | 1.117 |

Recovery overweighting drives the val Sharpe improvement. D captures +22.3% vs VT's +12.0% — but this is not "alpha," it's just being systematically overleveraged during both crashes and recoveries.

## Answers to Research Questions

**Q1: Does ANY variant beat VT on val with better MaxDD?** No. 0/99.

**Q2: Sigmoid (D) vs linear (A)?** D wins on val Sharpe (0.473 vs 0.457) but has worse MaxDD (-21.7% vs -19.6%). Sigmoid saturation doesn't help because the problem isn't extreme tilts — it's that tilts are in the WRONG direction during crashes.

**Q3: Multi-horizon (E) vs single-horizon (A)?** E is worse (0.349 vs 0.457). Combining horizons dilutes the signal.

**Q4: Continuous momentum (F)?** Nearly inert (delta +0.022). Momentum filter dampens the tilt signal to near-zero.

**Q5: Asymmetric (G) vs symmetric (A)?** G has similar Sharpe (0.457) but better MaxDD (-17.9% vs -19.6%). The aggressive expensive-side reduction doesn't fire during 2022 crashes because MVRV was below median — the cheap side dominates.

**Q6: Z-score (H) vs percentile rank (A)?** H is worse on dev (0.499 Sharpe!) because z-scores are too aggressive in bull markets. Better on val (+0.072) but with terrible MaxDD (-22.1%).

**Q7: MVRV+SOPR dual (I)?** Modest improvement (0.388 vs 0.329 VT). SOPR slightly moderates the MVRV "cheap trap" but not enough to prevent MaxDD degradation. Best MaxDD penalty is only +1.3%.

**Q8: Floor/ceiling (J)?** floor=0.0, ceiling=1.5 is best (0.425). Lower ceiling helps by limiting the upside tilt during "cheap" (crash) periods.

**Q9: Smoothest plateau?** G has excellent plateau within lb=1460 (6 configs, delta +0.078 to +0.128). But the sharp drop at lb=1095 shows sensitivity to lookback.

## Phase 4: Candidate Selection

### Selection Criteria

| Criterion | Result |
|-----------|--------|
| Val Sharpe delta > 0 vs VT | 63/99 pass |
| Val MaxDD no worse than VT | **0/99 pass** |
| Both criteria | **0/99 pass** |

### Verdict: NO CANDIDATES

The hard MaxDD gate eliminates all configs. The best near-miss is Variant I (lb=1460, tilt=0.5, sopr_window=3, sopr_weight=0.5) with MaxDD only 1.3% worse than VT (-16.7% vs -15.4%) and val Sharpe delta +0.059.

But the mechanism analysis shows this improvement is fragile:
1. **Val Sharpe gain comes from overweighting during 2023 recovery** — not from crash protection or valuation timing
2. **Every config systematically increases position during crashes** because MVRV reads "cheap" when it should read "danger"
3. **On holdout (2024+), MVRV is elevated** → tilt would reduce exposure during the bull market → same failure mode as P002-B (lesson #28)

## Root Cause Analysis

MVRV tilt fails because of a **structural mechanism mismatch**:

1. MVRV is a **valuation indicator**, not a timing indicator. It identifies expensive/cheap relative to realized value, but the lag from "expensive" to "crash" is months.

2. During crashes, MVRV falls rapidly → percentile rank drops below median → all continuous tilts read "cheap" → **increase position into the crash**. This is the "cheap trap" — buying the dip works in brief V-shaped recoveries (COVID 2020) but fails in prolonged bear markets (2022).

3. Vol targeting already provides mechanically faster crash protection. Adding a slower-moving MVRV signal either conflicts with VT (during crashes) or is redundant (during recoveries).

4. The dev-val asymmetry (massive bull → prolonged bear) means MVRV behavior is fundamentally different between periods. No continuous mapping overcomes this.

This confirms memory lesson #23 and #28. MVRV information is real (IC exists) but cannot be profitably converted into position sizing.

## Artifacts

- Raw results: `scratch/p002c_results.json` (99 configs)
- Script: `scratch/p002c_continuous_mvrv.py`
- wandb: tag `p002c_continuous_mvrv`
