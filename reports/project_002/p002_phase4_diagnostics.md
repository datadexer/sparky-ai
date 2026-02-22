# P002 Phase 4 — Deep Diagnostic Analysis

## 1. Executive Summary

The P002 on-chain regime strategy has no recoverable edge. Every signal component (MVRV, SOPR, Netflow) degrades or fails on the 2022-2023 validation set, no voting scheme rescues the composite, and pure vol targeting (always long, EWMA-scaled) outperforms every variant of the S1 signal. The on-chain regime approach should be killed.

---

## 2. MVRV Lookback Sensitivity

| Window | % Bullish (Val) | Val Sharpe | Val MaxDD | Bearish Before LUNA | Bearish Before FTX |
|--------|----------------|------------|-----------|--------------------|--------------------|
| 365 | 54.0% | -0.680 | -66.7% | No | No |
| 500 | 61.1% | -0.305 | -66.7% | No | No |
| 730 | 82.9% | -0.120 | -66.7% | No | No |
| 900 | 91.1% | +0.215 | -62.0% | Yes | Yes |
| 1095 | 90.4% | +0.238 | -62.0% | Yes | Yes |

**Paradox:** Shorter windows (365, 500) correctly reduce bullish persistence to ~54-61%, but produce *worse* Sharpe because they whipsaw in and out during the volatile period. Longer windows (900, 1095) are so persistently bullish they catch recovery moves, producing slightly positive Sharpe — but at the cost of riding through both crashes fully long. No window navigates the 2022 bear.

**Adaptive lookback** (min(900, days_since_ATH)): Increased bearish fraction from 8.9% to 40.1%, but Sharpe = 0.014 with -62.7% MaxDD. The adaptive rule helps detection but not timing — it flips too late and too often.

**Conclusion:** The MVRV rolling median approach is structurally broken for bear market detection regardless of window choice. The problem is not the 900-day parameter — it's the median-comparison mechanism itself.

---

## 3. Voting Mechanism Counterfactuals

| Scheme | Dev Sharpe | Val Sharpe | Val MaxDD | Flat LUNA | Flat FTX |
|--------|-----------|------------|-----------|-----------|----------|
| Majority 2/3 (actual S1) | 2.074 | -0.140 | -53.4% | 43% | 18% |
| Unanimous 3/3 | 0.631 | -1.656 | -38.0% | 100% | 100% |
| SOPR veto | 1.561 | -0.218 | -37.7% | 89% | 100% |
| MVRV excluded | 0.983 | -1.186 | -47.1% | 89% | 100% |

**Unanimous** avoids both crashes (100% flat) but destroys dev performance (0.63 → requires different strategy entirely) and still loses -1.66 on validation because being flat during 2023 recovery is fatal.

**SOPR veto** is the best crash-avoidance scheme (89% flat LUNA, 100% flat FTX, MaxDD -37.7% vs -53.4%) but still produces negative val Sharpe (-0.218). The protection comes too late and exits too slowly.

**MVRV excluded** (SOPR + Netflow only): Dev Sharpe drops to 0.98 and val Sharpe is -1.19. Without MVRV, the other two signals are even worse.

**No voting scheme produces positive validation Sharpe.** The problem is not MVRV domination — it's that ALL three signals degrade on validation.

---

## 4. SOPR Standalone

| Period | Sharpe | MaxDD | Return | Trades | Win Rate |
|--------|--------|-------|--------|--------|----------|
| Dev (2019-2021) | 1.514 | -42.3% | — | — | — |
| Val (2022-2023) | -0.259 | -44.0% | -19.8% | 20 | 19.3% |

SOPR correctly identified the 2022 bear (bearish 58% of time in Phase 4 analysis), but as a standalone trading signal it loses money on validation. Dev Sharpe 1.51 → Val -0.26 is a classic IS/OOS degradation. The 19.3% win rate on validation with 20 trades shows the signal's timing is off even when directional calls are roughly correct.

---

## 5. FGI Overlay Impact

| Variant | Val Sharpe | Val MaxDD | Val Return |
|---------|------------|-----------|------------|
| With FGI (actual S1) | -0.223 | -60.1% | -35.8% |
| Without FGI | -0.140 | -53.4% | -26.9% |
| Regime-conditioned FGI | -0.222 | -56.7% | -32.4% |

**FGI delta: -0.083 Sharpe, -6.7% worse DD, -8.9% worse return.** The FGI overlay amplified losses by boosting positions during extreme fear periods when the base regime was already (incorrectly) long. Regime-conditioned FGI (only boost during fear when regime = bearish) also failed.

---

## 6. Vol Targeting Decomposition

| Strategy | Dev Sharpe | Val Sharpe | Dev MaxDD | Val MaxDD | Val Return |
|----------|-----------|------------|-----------|-----------|------------|
| Buy & Hold | 1.494 | 0.111 | -63.4% | -66.9% | -16.3% |
| **Pure Vol Target** | **1.588** | **0.336** | **-11.6%** | **-14.9%** | **+5.8%** |
| S1 Raw | 2.129 | -0.223 | -26.2% | -60.1% | -35.7% |
| S1 + Vol Target | 2.101 | -0.198 | -5.2% | -13.9% | -4.3% |

**Pure vol targeting (always long, EWMA-scaled) achieves Val Sharpe 0.336 — the S1 signal DESTROYS value relative to this baseline.** Adding S1's positions to vol targeting takes Sharpe from +0.336 down to -0.198. The on-chain regime signal has negative marginal contribution.

Vol targeting sub-period breakdown:

| Period | VT Sharpe | VT Return | Avg Position |
|--------|-----------|-----------|-------------|
| 2022 Q1 (grind) | +0.067 | +0.04% | 0.155 |
| 2022 Q2 (LUNA/3AC) | -4.292 | -11.1% | 0.148 |
| 2022 Q3 (sideways) | -0.068 | -0.3% | 0.150 |
| 2022 Q4 (FTX) | -1.115 | -2.2% | 0.180 |
| 2022-12 to 2023-03 | +3.321 | +11.8% | 0.200 |
| 2023 Q2 | +0.868 | +1.9% | 0.213 |
| 2023 H2 | +1.444 | +6.5% | 0.278 |

Vol targeting mechanically reduces position to ~0.15 during high-vol 2022 (limiting crash exposure to ~11% loss during LUNA despite -57% BTC) and scales up to ~0.28 during low-vol 2023 recovery. This is a pure risk-management overlay with no signal intelligence — and it works better than the signal.

---

## 7. Signal Information Content

### Forward Return IC (Spearman)

| Signal | Dev IC@7d | Dev IC@30d | Val IC@7d | Val IC@30d | Dev p(30d) | Val p(30d) |
|--------|-----------|------------|-----------|------------|------------|------------|
| MVRV (raw) | -0.063 | **-0.178** | -0.128 | **-0.236** | <0.001 | <0.001 |
| SOPR SMA3 | +0.015 | -0.044 | -0.061 | -0.108 | 0.144 | 0.005 |
| Netflow SMA14 | -0.074 | -0.104 | **+0.107** | -0.076 | <0.001 | 0.048 |
| FGI | +0.085 | +0.066 | -0.008 | +0.095 | 0.028 | 0.013 |

MVRV has consistently significant **negative** IC — higher MVRV predicts lower future returns. The inverted S1 signal exploits this, but the relationship strengthened from -0.178 to -0.236 on validation, yet the strategy still failed. This suggests the binary threshold (median comparison) loses the continuous information.

SOPR flips from insignificant (+0.015) in dev to significantly negative (-0.061) in val at 7d. Netflow **flips sign** at 7d between dev (-0.074) and val (+0.107). FGI loses significance entirely on val at 7d/14d.

### Rolling IC (180d window, mean IC@30d by year)

| Signal | 2019 | 2020 | 2021 | 2022 | 2023 |
|--------|------|------|------|------|------|
| MVRV | -0.38 | -0.57 | -0.26 | -0.42 | -0.53 |
| SOPR | -0.24 | -0.37 | -0.09 | -0.22 | -0.34 |
| Netflow | -0.18 | -0.04 | -0.04 | -0.20 | -0.08 |
| FGI | -0.02 | -0.43 | +0.04 | -0.18 | -0.31 |

**All signals show negative mean rolling IC in nearly every year.** This means the continuous-to-binary mapping in the S1 strategy loses or inverts the information content. The signal processing (median comparison, majority vote, persistence) is not capturing what the raw data contains.

---

## 8. Structural Break Evidence

### Autocorrelation (lag-60)

| Signal | Pre-2021 | Post-2021 | Change |
|--------|----------|-----------|--------|
| MVRV | 0.430 | **0.794** | +0.364 |
| SOPR SMA3 | -0.041 | 0.223 | +0.264 |
| Netflow SMA14 | -0.003 | 0.022 | +0.025 |

MVRV autocorrelation at 60-day lag nearly doubled post-2021 (0.43 → 0.79). The metric became far more persistent/sticky, making rolling-window signals less responsive to regime changes. This is consistent with institutional custody (less frequent on-chain movement) degrading MVRV signal dynamics.

### Pairwise Correlations (yearly means)

| Pair | 2019 | 2020 | 2021 | 2022 | 2023 |
|------|------|------|------|------|------|
| MVRV-SOPR | 0.58 | 0.68 | 0.69 | 0.59 | 0.56 |
| MVRV-Netflow | 0.18 | 0.12 | 0.05 | 0.08 | **0.33** |
| SOPR-Netflow | -0.07 | 0.10 | -0.02 | 0.07 | **0.31** |

MVRV-Netflow and SOPR-Netflow correlations jump sharply in 2023 (from near-zero to ~0.31-0.33). The diversity assumption behind majority voting is breaking down — signals are converging, reducing the value of ensemble voting.

### Distribution Shift (Dev → Val)

| Signal | Dev Mean | Val Mean | Dev Std | Val Std | Shift |
|--------|----------|----------|---------|---------|-------|
| MVRV | 1.94 | 1.30 | 0.70 | 0.31 | Mean -33%, Std -56% |
| SOPR SMA3 | 1.005 | 0.999 | 0.012 | 0.009 | Kurtosis 4.7 → 10.4 |
| Netflow SMA14 | +512 | **-363** | 2710 | 3186 | Mean flipped sign |

**Substantial regime changes across all three signals.** MVRV compressed into a narrow range (std halved), SOPR developed extreme tails (kurtosis doubled), and Netflow mean flipped from net inflow (+512) to net outflow (-363). These are not parameter drift — they're fundamental distributional changes.

---

## 9. Recoverable Edge Assessment

### Confirmed dead ends

- **MVRV rolling median comparison**: Broken at all lookback windows. The median mechanism cannot distinguish "cheap because undervalued" from "cheap because falling." Structural autocorrelation increase (0.43 → 0.79 at lag 60) makes the metric too sticky for regime detection.

- **Majority vote composite (any weighting)**: All voting schemes produce negative validation Sharpe. The problem is not weighting — all three component signals degrade individually.

- **SOPR as trading signal**: Dev 1.51 → Val -0.26 with 19.3% win rate. Classic overfitting profile.

- **FGI as position overlay**: Hurts in bear markets (fear is informative, not contrarian). Both fear-boost and regime-conditioned variants failed.

- **Netflow as timing signal**: IC flips sign between dev and val. Distribution mean changed sign. Not stable.

### Potentially recoverable with new research cycle

- **MVRV as continuous predictor**: Raw MVRV IC actually *strengthened* on validation (-0.178 → -0.236 at 30d). The continuous information is there, but the binary threshold (median comparison) destroys it. A regression-based or quantile-rank approach to MVRV positioning might work — but this is a fundamentally different strategy requiring new pre-registration.

- **Vol targeting as primary strategy**: Pure vol-targeted long-only BTC produced Val Sharpe 0.336 with -14.9% MaxDD — meaningful positive performance through a -67% BTC drawdown. This is not alpha in the traditional sense (it's a risk management overlay on a positive-drift asset), but it's a legitimate strategy component worth investigating further.

### Surprisingly robust

- **EWMA(60) vol targeting**: Sharpe 1.59 dev, 0.34 val — consistent positive performance in both periods. MaxDD reduced from -67% to -15%. The mechanism is transparent (scale down during high vol, scale up during low vol) and does not require signal accuracy. Retention ratio = 21% (0.34/1.59), which is low but still positive, unlike every signal-based strategy tested.

---

## 10. Recommendation for AK

### Option A: Kill P002 entirely

**Evidence:** Every signal component fails on validation. Every voting scheme fails. SOPR standalone fails. FGI overlay hurts. The on-chain regime hypothesis has been thoroughly falsified across 5 lookback windows, 5 voting schemes, standalone tests, and continuous IC analysis.

**Cost:** Writes off P002 investment (~$200 in compute, ~20 sessions). No further research on on-chain regime signals.

### Option B: Pivot P002 to "MVRV continuous + vol targeting"

**Evidence:** Raw MVRV IC strengthened on validation (-0.24 at 30d, p<0.001). Pure vol targeting produced Val Sharpe 0.34. A strategy that uses MVRV *continuously* (regression/quantile-rank positioning, not binary threshold) combined with vol targeting might extract the information that the binary approach destroyed.

**Cost:** New pre-registration. New screening cycle (~$50-80). Different strategy architecture. Risk: the continuous IC might not translate to tradeable positions after transaction costs.

### Option C: Launch P003 around vol-targeted BTC exposure

**Evidence:** Pure vol targeting is the only approach that produced positive validation Sharpe (0.34). It's simple, transparent, and mechanically sound. Combined with P001's Donchian portfolio (which also uses vol dynamics implicitly), this could form a multi-strategy allocation.

**Cost:** Minimal — vol targeting infrastructure already exists. Could be validated quickly. But Sharpe 0.34 alone doesn't meet Tier 1/2 thresholds.

### Option D: Focus exclusively on P001 Donchian portfolio

**Evidence:** P001 champion passed all validation gates including DSR (0.972). P002 failed comprehensively. Concentration on the proven strategy rather than diversifying into a new research direction.

**Cost:** Single-strategy risk. No diversification. But the proven strategy is ready for OOS evaluation.

---

**Do not choose.** Present evidence and let AK decide.
