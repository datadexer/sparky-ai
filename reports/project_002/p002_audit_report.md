# P002 Screening Audit Report

**Date:** 2026-02-21 | **Auditor:** Oversight Agent (Opus)
**Scope:** IS screening sessions 1-4 only. Validation session results NOT examined.

---

## 1. Session-by-Session Trial Count Breakdown

| Session | New Configs | Cumulative N | Best S@30 | Champion Family | Duration |
|---------|-------------|--------------|-----------|-----------------|----------|
| S1 | 297 | 297 | 2.075 | INV_MVRV + FGI | 25 min |
| S2 | 418 | 715 | 2.259 | VDD + MVRV + SOPR + vol | 26 min |
| S3 | 642 | 1,357 | 2.590 | VDD(0.75) + MVRV + SOPR + FGI + vol | 23 min |
| S4 | 371 | 1,728 | 2.590 (unchanged) | Same as S3 | 17 min |
| **Total** | **1,728** | | | | **91 min** |

**Pre-registered max: 350 configs. Actual: 1,728 (4.9x overshoot).**

Cost: $4.82 of $80 budget. The budget was not the constraint — the orchestrator kept spawning sessions because stall detection overrode `max_sessions: 1`.

---

## 2. Full List of Signals Tested

### Pre-Registered (in screening.yaml strategy_space)

| Signal | Used in Champion? | Notes |
|--------|-------------------|-------|
| MVRV z-score | YES (inverted direction) | Pre-registered as "bullish above median"; champion uses BELOW median |
| SOPR | YES | SMA(3) > 1.0 = bullish, as designed |
| Exchange netflow | No | Tested S1 only, IC borderline (p=0.09), dropped |
| FGI overlay | YES | fear_only=True mode, ft=28, adj=0.4 |
| Vol targeting | YES | ewma_60, tv=0.10 |
| Funding rate carry | No | Only 1yr data (2021), S=0.50, inconclusive |

### Non-Pre-Registered (introduced mid-research)

| Signal | Session Introduced | IC | Used in Champion? |
|--------|-------------------|-----|-------------------|
| **VDD (Value Days Destroyed)** | S2 | -0.149*** (30d) | **YES — primary new signal** |
| LTH MVRV | S2 | -0.094** (7d) | No (inferior to standard MVRV) |
| LTH SOPR | S2 | -0.078** (7d) | No (inferior to standard SOPR) |
| NUPL | S2 | -0.060* (7d) | No (inferior composite leg) |
| Puell Multiple | S2 | -0.209*** (30d) | No (best alternative, S=2.32) |
| STH MVRV | S4 | n.s. | No |
| Exchange Reserve ROC | S2 | n.s. | No |
| Stablecoin Supply ROC | S2 | n.s. | No |
| MVRV percentile rank | S1 | -0.115*** (7d) | No (transform of MVRV, continuous version inferior) |

**8 non-pre-registered signals tested.** The champion relies on VDD, which was never in the pre-registration.

---

## 3. Champion Trade Profile

**Config:** VDD(0.75) + INV_MVRV(pw=900) + SOPR(ss=3), pers=11, FGI(ft=28, adj=0.4), vol(ewma_60, tv=0.10)
**Dev set:** 2019-01-01 to 2021-12-31

### Trade Count & Hold Periods

| Metric | Value |
|--------|-------|
| Completed round trips | **5** |
| Average hold period | 81.6 days |
| Median hold period | 58 days |
| Min / Max hold | 17 / 163 days |
| Win rate | **100%** (5/5 wins) |
| Days in position | 408 / 1,096 (**37.2%**) |
| Days flat | 688 / 1,096 (**62.8%**) |

### Individual Trades

| # | Entry | Exit | Days | Avg Pos | PnL | BTC Move |
|---|-------|------|------|---------|-----|----------|
| 1 | 2019-01-13 | 2019-06-25 | 163 | 0.192 | +25.3% | +240% |
| 2 | 2019-12-17 | 2020-02-13 | 58 | 0.185 | +9.0% | +54% |
| 3 | 2020-04-04 | 2020-08-07 | 125 | 0.135 | +8.7% | +69% |
| 4 | 2021-07-22 | 2021-08-08 | 17 | 0.137 | +5.1% | +36% |
| 5 | 2021-09-07 | 2021-10-22 | 45 | 0.146 | +4.1% | +30% |

### Regime Attribution

| Period | PnL | % of Total | Avg Position | Sharpe |
|--------|-----|------------|--------------|--------|
| 2019 recovery | +27.5% | **52.9%** | 0.098 | 3.10 |
| 2020 pre-COVID (Jan-Mar 15) | +6.7% | 12.9% | 0.103 | 4.44 |
| 2020 COVID crash + recovery | +3.6% | 7.0% | 0.094 | 1.89 |
| 2020 DeFi summer + year-end | +5.1% | 9.7% | 0.038 | 2.18 |
| **2021 Jan-Apr (bull to 1st top)** | **0.0%** | **0.0%** | **0.000** | **flat** |
| **2021 Apr-Jul (China ban crash)** | **0.0%** | **0.0%** | **0.000** | **flat** |
| 2021 Jul-Nov (second top) | +9.1% | 17.5% | 0.081 | 3.63 |
| **2021 Nov-Dec (bear begins)** | **0.0%** | **0.0%** | **0.000** | **flat** |

**Key finding: 52.9% of total PnL comes from a single 163-day trade in 2019.** The top 2 periods (2019 recovery + 2021 second top) account for 70.4% of total PnL.

The strategy is completely flat for 252 days (23% of dev set) during the entire 2021 bull run peak, China crash, and bear onset.

### The -2.7% MaxDD Explanation

The MaxDD is NOT a bug. The mechanism:

1. **Vol targeting keeps positions tiny**: target_vol=0.10 vs BTC realized vol of 0.35-1.88 annualized. Average position size when in-market: **0.168** (max: 0.29). The strategy never holds more than ~29% of capital in BTC.

2. **Strategy is flat during all crashes**:
   - COVID crash (2020-02-20 to 2020-03-23): **100% flat, zero exposure**
   - China ban crash (2021-05-10 to 2021-07-20): **100% flat, zero exposure**

3. The actual max drawdown (-2.69%, Sep 18-28 2021) is a minor dip during trade #5. It recovered in 13 days.

**Assessment:** The low MaxDD is a mechanical consequence of (a) being flat 63% of the time, (b) vol targeting capping position sizes at ~17% of capital, and (c) the on-chain signals going bearish before/during crashes. Whether this transfers to bear markets depends entirely on VDD and MVRV continuing to work out-of-sample.

---

## 4. PBO Path Sharpe Distribution

CPCV with 12 groups, k_test=2 produces 924 paths (C(12,6)).

**IMPORTANT NOTE:** The PBO audit was run with a signal reconstruction that produced Sharpe=1.94 (the s004 truncated-history approach), NOT the claimed 2.59 (s003 full-history approach). See Section 7 for details on this discrepancy. The PBO distribution below corresponds to the 1.94 Sharpe version.

| Percentile | Path Sharpe |
|------------|-------------|
| Min | 0.000 |
| p5 | 1.102 |
| p10 | 1.245 |
| p25 | 1.538 |
| **Median** | **1.831** |
| p75 | 2.114 |
| p90 | 2.281 |
| p95 | 2.441 |
| Max | 2.593 |

- **PBO = 0.000** (0 of 924 paths negative)
- **Min path Sharpe = 0.000** — one path is borderline zero. PBO = 0.000 is marginally fragile.
- Mean: 1.807, Std: 0.409

**Assessment:** The PBO result is genuine — the strategy produces positive Sharpe across essentially all time-period combinations. Even at the lower Sharpe of 1.94, the time-stability is strong. However, the min path being exactly 0.000 means PBO could flip to non-zero with small perturbations.

---

## 5. DSR Analysis

### DSR at Correct Sharpe (2.59, from s003 full-history approach)

From session 3 results:
- **DSR at N=1,357: 0.923** (from s003_r4_final.json)
- DSR fails the 0.95 threshold at the actual trial count

### DSR at Lower Sharpe (1.94, from s004 truncated approach)

| N (trials) | DSR |
|------------|-----|
| 1 | 0.999 |
| 10 | 0.981 |
| **22** | **0.950 (max N for DSR > 0.95)** |
| 50 | 0.896 |
| 100 | 0.832 |
| 200 | 0.754 |
| **350 (pre-registered)** | **0.684** |
| 715 | 0.583 |
| 1,000 | 0.545 |
| **1,728 (actual)** | **0.474** |

**Assessment:** At the lower Sharpe (1.94), DSR fails at any N above 22. At the correct Sharpe (2.59), DSR reaches 0.923 at N=1,357 — better, but still below the 0.95 gate. **Even at the pre-registered N=350, DSR likely does not exceed 0.95** (extrapolating from the N=1,357 datapoint, DSR at N=350 would be higher than 0.923 but the gap to 0.95 is uncertain without direct computation).

### The PBO-DSR Tension

- **PBO says:** Strategy is time-stable. All 924 CPCV paths are profitable.
- **DSR says:** After testing 1,728 configs, the Sharpe is not statistically distinguishable from data-mining.

These test different things: PBO tests time-stability (given this config), DSR tests whether the config was cherry-picked from noise. Both can be simultaneously true — the config could be genuine AND selected from too many trials.

---

## 6. Validation Partition Contamination Assessment

### Was the validation session killed before completing?

**YES.** The validation orchestrator log shows:
- Session started reading files (directive, previous scripts, data availability)
- Was about to write the validation script ("Now I have everything I need. Let me write the comprehensive validation script")
- **Killed before any computation ran**

### Did it write any results?

| Location | Status |
|----------|--------|
| `results/p002_validation/` | **EMPTY** (directory created but no files) |
| wandb (validation/p002_val tags) | **ZERO runs** with validation tags |
| Orchestrator state | `session_count: 0`, `total_cost: 0.0` |
| Telemetry | No validation session telemetry |

**The validation partition is CLEAN.** No 2022-2023 data was accessed, no results were computed, no wandb runs were logged. The validation set remains usable for the champion config.

---

## 7. Implementation Discrepancy: Sharpe 2.59 vs 1.94

A critical finding from the audit: **two different signal reconstruction approaches produce different Sharpes.**

| Approach | Sharpe@30bps | Source | Mechanism |
|----------|-------------|--------|-----------|
| **Full-history (s003)** | **2.5895** | Session 3 scripts | MVRV rolling 900d median computed on full on-chain history (5,164 points from 2012) |
| **Truncated (s004)** | **1.9355** | Session 4 scripts | On-chain data reindexed to price index (1,096 points from 2019) BEFORE computing rolling windows |

The difference: 263 days where the MVRV signal disagrees, 163 days where the composite position differs, pct_long drops from 37.2% to 22.4%.

**Which is correct?** The s003 approach is methodologically correct — a 900-day rolling median should use 900 days of history, not be truncated to the dev set. The s004 approach is a bug (documented in s004 summary itself). However, this means:

1. The champion's signal depends on MVRV data from 2012-2018 (outside the dev set) for its lookback window
2. The Sharpe is sensitive to the lookback implementation — 0.65 Sharpe swing (33%) from this single choice
3. Any validation must use the correct (full-history) approach consistently

---

## 8. Summary of Findings

### Red Flags

1. **5 trades in 3 years.** The strategy's entire edge comes from 5 well-timed entries. Statistical reliability of a 5-trade sample is very low regardless of CPCV results.

2. **52.9% of PnL from one trade.** The 2019 recovery trade (163 days, $3.5k → $11.8k) dominates the Sharpe. Remove it and the strategy looks very different.

3. **100% win rate is suspicious.** 5/5 winners on a 3-year dev set with only bullish regime exposure (the strategy is flat during all bearish periods) means this is fundamentally a "buy-the-dip-in-a-bull-market" strategy tested on a bull market.

4. **VDD not pre-registered.** The champion's primary differentiating signal (VDD) was not in the pre-registration. This is an undocumented researcher degree of freedom.

5. **N=1,728 vs N=350 pre-registration.** 5x overshoot. DSR fails at all reasonable N values.

6. **Sharpe depends on implementation detail.** 0.65 Sharpe swing (2.59 vs 1.94) from how MVRV lookback is computed.

### Green Flags

1. **PBO = 0.000 across 924 paths.** Genuine time-stability.
2. **MC p-value = 0.005.** Signal is better than circular-shift permutations.
3. **Plateau coverage = 1.000.** All parameter neighbors produce similar Sharpes — not a cliff.
4. **Crash avoidance is mechanical, not fitted.** The VDD/MVRV signals go bearish during crashes via the actual on-chain dynamics (coins being spent, MVRV above historical median). This is an economic mechanism, not a fitted pattern.
5. **VDD has clear economic rationale.** VDD < 0.75 means long-term holders are NOT spending their coins (accumulation). This is a well-known on-chain heuristic with academic precedent.

---

## 9. Decision Options for AK

### Option A: Champion is legitimate, validation is recoverable

**Evidence for:** PBO=0.000, MC p=0.005, plateau=1.000, crash avoidance is mechanistic (on-chain signals genuinely go bearish during crashes), VDD has real economic rationale, validation partition is clean.

**Evidence against:** Only 5 trades, 53% PnL from one trade, VDD not pre-registered, N=1,728 exceeds budget 5x, DSR < 0.95 at any N.

**Action:** Re-run validation interactively with AK present, using correct full-history approach. Use actual N=1,728 for DSR. If DSR > 0.95 on combined IS+Val period (more observations = more statistical power), proceed to OOS.

### Option B: Champion is legitimate but needs protocol cleanup

**Evidence for:** Same as A. Validation partition confirmed clean.

**Action:** (i) Amend pre-registration to include VDD retroactively (document the rationale). (ii) Re-run validation interactively. (iii) Use N=1,728 honestly — if the signal is real, more data (2022-2023) may push DSR above 0.95 despite the high N.

### Option C: Champion is suspicious

**Evidence for:** 5 trades, 53% PnL from one trade (2019 recovery), strategy is a "buy VDD dips in a bull market" pattern tested exclusively on a bull market. 100% win rate on 5 trades is not statistically meaningful. The 2022 bear market (locked validation set) is the only real test — and the strategy will likely be flat the entire time (producing Sharpe ≈ 0 during the bear, which is actually good relative to buy-and-hold but reveals the strategy has no bear-market alpha).

**Action:** Kill the VDD champion. Review whether any pre-registered-signal-only configs from S1 (INV_MVRV+SOPR+FGI, S@30=2.075, n_trades=69) show more promise as a validation candidate. These had more trades (69 vs 5) and used only pre-registered signals. DSR would be computed at N=297 (S1 trial count only).

### Option D: Pre-registration violation is disqualifying

**Action:** Restart P002 screening with: (i) amended pre-registration including VDD, (ii) hard kill at N=350, (iii) mandatory Phase 1 IC characterization for all signals before optimization. Most conservative option.

---

## 10. Auditor's Assessment

The PBO and plateau results are genuinely strong — the strategy IS time-stable across arbitrary sub-periods, and the parameter space IS smooth. These are not artifacts. The crash avoidance mechanism (VDD + MVRV going bearish during real on-chain distress) has legitimate economic underpinning.

However, **5 trades is not enough to trust.** The fundamental issue is not data-mining (PBO says it isn't) — it's that 5 observations in a 3-year bull market provide almost no statistical power. The 2019 recovery trade alone is responsible for half the Sharpe.

The strongest counterargument to "this is just a bull-market bet" is the 2022-2023 validation set. If the strategy stays flat during the 2022 bear (as the signals suggest it should) and then re-enters during the 2023 recovery, that would be genuine evidence of crash avoidance alpha. If instead it produces Sharpe ≈ 0 by being flat the entire time, it confirms the strategy has no edge beyond avoiding losses.

**The S1 pre-registered-signal-only champion (INV_MVRV+SOPR+FGI, S@30=2.075, 69 trades, N=297) may be a cleaner validation candidate** despite the lower Sharpe, because it has 14x more trades and stays within the pre-registration protocol.

---

*Report generated: 2026-02-21 ~12:15 PM EST*
*Audit scripts: `scratch/p002_audit_trades.py`, `scratch/p002_audit_pbo_dsr.py`*
