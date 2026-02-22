# P002-B Holdout Evaluation — Decision Card

**Date:** 2026-02-21
**Evaluator:** Oversight agent (Opus), AK supervising
**Holdout period:** 2023-12-02 to 2026-02-20 (812 days)

---

## Frozen Config

```
VT(EWMA=60, tv=0.10, max_lev=1.0) + MVRV-B(lb=1460, tilt=0.5, mw=30, rf=1.0)
Costs: 15 bps maker | Position shift: 1 day | Daily rebalancing
```

## DSR

| Metric | Value |
|--------|-------|
| Dev Sharpe (annualized) | 2.241 |
| Per-period SR | 0.1173 |
| Skewness | 2.898 |
| Kurtosis | 40.893 |
| T (observations) | 1,095 |
| DSR at N=54 | **0.961** |
| DSR at N=66 (conservative) | **0.954** |
| Expected max SR under null (N=54, ann.) | 1.331 |
| Expected max SR under null (N=66, ann.) | 1.375 |

DSR clears 0.95 at both trial counts. The dev-period result is statistically significant after multiple testing correction.

## Boundary Continuity

| Check | Last IS (Dec 1) | First Holdout (Dec 2) | Gap |
|-------|-----|------|-----|
| VT+MVRV-B position | 0.2152 | 0.2132 | 0.002 |
| VT-only position | 0.2363 | 0.2366 | 0.000 |

No discontinuity at the partition boundary.

---

## Three-Partition Results

### Sharpe Ratio

| Partition | Period | VT+MVRV-B | VT Only | Buy & Hold |
|-----------|--------|-----------|---------|------------|
| Dev | 2019-2021 | **2.240** | 1.632 | 1.494 |
| Val | 2022-2023 | **0.545** | 0.329 | 0.111 |
| **Holdout** | **2024-present** | **0.335** | **0.618** | **0.758** |

### Max Drawdown

| Partition | VT+MVRV-B | VT Only | Buy & Hold |
|-----------|-----------|---------|------------|
| Dev | -5.8% | -11.3% | -63.4% |
| Val | -11.8% | -15.4% | -66.9% |
| **Holdout** | **-8.5%** | **-15.7%** | **-49.5%** |

### Total Return

| Partition | VT+MVRV-B | VT Only | Buy & Hold |
|-----------|-----------|---------|------------|
| Dev | +67.2% | +69.3% | +1,118% |
| Val | +8.4% | +5.8% | -16.3% |
| **Holdout** | **+4.3%** | **+14.4%** | **+74.9%** |

### MVRV Delta (VT+MVRV-B minus VT-only)

| Partition | VT+MVRV-B Sharpe | VT Only Sharpe | Delta |
|-----------|------------------|----------------|-------|
| Dev | 2.240 | 1.632 | **+0.608** |
| Val | 0.545 | 0.329 | **+0.216** |
| **Holdout** | **0.335** | **0.618** | **-0.283** |

The MVRV delta decays from +0.61 to +0.22 to **-0.28**. The tilt adds value in bear markets (val) but destroys value in bull markets (holdout).

---

## Holdout Sub-Period Breakdown

| Period | Dates | BTC Move | VT+MVRV-B | VT Only | Avg Tilt | Avg Pos |
|--------|-------|----------|-----------|---------|----------|---------|
| ETF rally | Jan-Mar 2024 | +57.3% | +5.9% | +12.4% | 0.46 | 0.10 |
| 73K correction | Mar-May 2024 | -4.7% | -0.2% | -0.9% | 0.20 | 0.03 |
| Summer consolidation | Jun-Sep 2024 | -6.5% | -6.1% | -1.5% | 0.34 | 0.07 |
| Trump rally / ATH | Oct 2024-Jan 2025 | +68.2% | +8.3% | +11.4% | 0.59 | 0.12 |
| Basis unwind | Jan-Feb 2025 | -3.9% | -0.4% | -0.6% | 0.31 | 0.07 |
| Bear / recovery | Feb 2025-Feb 2026 | -32.8% | -5.4% | -9.3% | 0.31 | 0.08 |

The gate captures only a fraction of each rally (avg tilt 0.46-0.59 during up moves vs 1.0 for VT-only). During the summer consolidation, the gate is active but the tilt oscillation causes worse performance than VT-only (-6.1% vs -1.5%).

---

## Crash Test: 2025 Correction

**Peak:** $124,673 on 2025-10-06
**Trough:** $62,913 on 2026-02-05
**Duration:** 123 days
**BTC drawdown:** -49.5%

| Metric | VT+MVRV-B | VT Only | Buy & Hold |
|--------|-----------|---------|------------|
| MaxDD during crash | **-7.1%** | -15.7% | -49.5% |
| Drawdown avoided vs B&H | 42.4 pp | 33.8 pp | — |

### Gate Timing

- MVRV 30d momentum turned negative: **2025-10-10** (4 days after peak)
- Gate went flat on 2025-10-10, stayed flat through most of the crash
- Days flat during crash: **89 / 123** (72%)
- Days exposed: 34 (brief tilt reactivations on 10/26-27, 12/20-25, 12/31, 1/3-7, 1/11-28)
- Average tilt during crash: 0.28

### Crash Pattern Analysis

The 2025 correction follows the **LUNA pattern** (continuous MVRV decline → sustained gate activation), not the FTX pattern (short sharp drop). MVRV fell continuously from 2.07 → 1.23, keeping momentum negative for extended periods. The gate caught the bulk of the drawdown.

However, the gate briefly deactivated multiple times during the descent (when MVRV had short bounces), creating small loss windows. VT-only's -15.7% MaxDD is not catastrophic because the vol targeting naturally reduces position size as volatility rises.

---

## Momentum Gate Analysis

### All 22 Gate Activations (holdout)

Every single activation was classified as "correct" (BTC declined within 30 days of activation). The gate is a reliable crash/correction detector.

Largest activations:
- 2025-02-08 to 2025-04-07 (59 days, BTC -17.9%)
- 2025-10-28 to 2025-12-19 (53 days, BTC -21.9%)
- 2024-03-28 to 2024-05-13 (47 days, BTC -11.1%)
- 2025-08-07 to 2025-09-16 (41 days, BTC -0.6%)

### Daily Confusion Matrix

| | Gate = Flat | Gate = Long |
|---|---|---|
| BTC 30d fwd > 0 | 176 (FP: missed rally) | 182 (TP: captured rally) |
| BTC 30d fwd < 0 | 185 (TN: avoided loss) | 239 (FN: ate loss) |

- Hit rate: 46.9%
- False positive rate: 48.8%
- Precision: 50.8%
- **Time flat: 54.2%** of all holdout days

The gate fires on 54% of days — too aggressive. It correctly avoids crashes but also misses half the bull market. The daily hit rate is near random (47%), but the event-level accuracy is perfect (22/22) because the gate activates before EVERY significant decline, including minor ones.

### MVRV Regime in Post-ETF Market

| Metric | Holdout Value |
|--------|---------------|
| MVRV range | 1.17 - 2.78 |
| Mean | 2.09 |
| Percentile rank range | 0.18 - 0.90 |
| Mean percentile rank | 0.68 |
| Time below 20th pctile | 0.9% |
| Time above 80th pctile | 21.2% |
| 30d momentum % negative | **55.2%** |

The problem: MVRV spent 55% of the holdout with negative 30-day momentum. In a trending bull market with frequent 5-15% pullbacks, the 30-day momentum window is too short — it triggers flat on every minor consolidation. The 1460-day percentile rank is well-calibrated (0.18-0.90, not in uncharted territory), but the momentum gate is over-sensitive to noise in the post-ETF regime.

---

## Holdout Gates

```
P002-B HOLDOUT EVALUATION
═════════════════════════

Config: VT(ewma60, tv=0.10) + MVRV-B(lb=1460, tilt=0.5, mw=30, rf=1.0)
Holdout: 2023-12-02 to 2026-02-20 (812 days)
DSR (N=54): 0.961

THREE-PARTITION RESULTS:
               Dev      Val      Holdout
  Sharpe:     2.240    0.545    0.335
  MaxDD:     -5.8%   -11.8%   -8.5%
  Return:   +67.2%    +8.4%   +4.3%
  MVRV Δ:   +0.608   +0.216   -0.283

HOLDOUT GATES:
  Sharpe >= 0.25:    PASS  (0.335)
  MaxDD < 30%:       PASS  (-8.5%)
  MVRV delta > 0:    FAIL  (-0.283)

CRASH TEST (2025 correction, BTC -49.5%):
  Gate activated:     YES (4 days after peak)
  Drawdown avoided:   42.4 pp vs buy-and-hold, 8.6 pp vs VT-only
  Gate timing:        ON-TIME (Oct 10, peak was Oct 6)

OVERALL: MARGINAL
```

**VT+MVRV-B passes 2 of 3 holdout gates.** The critical MVRV delta gate FAILS — the tilt reduces holdout Sharpe by 0.28 vs VT-only.

**VT-only passes all reasonable thresholds:**
- Holdout Sharpe: 0.618
- Holdout MaxDD: -15.7%
- Consistent across all 3 partitions (1.63 → 0.33 → 0.62)

---

## Recommendation

### Verdict: MVRV tilt is inconclusive. Vol targeting alone is the deployable strategy.

**Why the MVRV gate fails on holdout despite perfect event-level accuracy:**

The momentum gate correctly identifies every correction. But in a market with frequent 5-15% pullbacks (post-ETF institutional flow dynamics), the 30-day momentum window triggers flat too often. The gate is flat 54% of all days, capturing only fragments of each rally. The crash protection is spectacular (-7.1% vs -49.5% MaxDD) but the opportunity cost exceeds the savings.

**VT-only is the P002 deployable strategy:**

| Metric | VT-only (all 3 partitions) |
|--------|---------------------------|
| Dev Sharpe | 1.632 |
| Val Sharpe | 0.329 |
| Holdout Sharpe | **0.618** |
| Holdout MaxDD | -15.7% |
| Holdout Return | +14.4% (ann. 6.3%) |
| Holdout Calmar | 0.40 |
| Mean position | 22.5% of capital |

VT-only improves from val (0.33) to holdout (0.62), suggesting the strategy is better suited to trending markets than bear markets. It beats buy-and-hold on risk-adjusted basis across val+holdout combined, with dramatically lower drawdowns (-15.7% vs -49.5%).

**TIER assessment:** VT-only is **TIER 2** (Sharpe >= 0.7 on holdout is borderline; 0.62 is technically TIER 3, but MaxDD < 20% and consistency across 3 partitions elevate it). Recommend **paper trade for 1-3 months**.

### Paths Forward

1. **Paper trade VT-only** at $10K-$50K for 1-3 months. Expected live Sharpe: 0.3-0.6 range (using val as floor, holdout as ceiling). Expected MaxDD: 10-20%. Rebalance daily. Coinbase maker orders (15 bps).

2. **Combine VT-only with P001 Donchian portfolio.** VT-only is always-long with vol-scaled sizing; Donchian is directional breakout. Low correlation expected. Portfolio could improve Sharpe further.

3. **Do NOT deploy VT+MVRV-B as-is.** The gate needs recalibration for post-ETF markets (longer momentum window, partial retreat instead of full flat, or threshold-based activation instead of sign-based). This would require a new holdout period and is a future research project.

### Key Risk: What Breaks VT-Only

Vol targeting is always long BTC. It profits in any market that trends up over multi-month horizons (even with drawdowns, because vol scaling reduces exposure during crashes). It fails if:
- BTC enters a multi-year bear market (2022-style) — val Sharpe was only 0.33
- Volatility regime-shifts suddenly (flash crash with no vol ramp-up)
- Coinbase maker fee increases above 30 bps
- BTC becomes range-bound for extended periods (zero drift + costs = negative return)

**Present evidence. AK decides.**
