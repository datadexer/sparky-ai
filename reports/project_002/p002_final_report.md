# Project 002 — Final Report

**Status:** COMPLETE — all research directions closed
**Dates:** 2026-02-21 to 2026-02-22
**Total cost:** ~$8.81 across 8 sessions (~138 minutes)
**Final strategy:** Vol Targeting only (EWMA-60, tv=0.10, 15 bps)
**Holdout Sharpe:** 0.618 | **Tier:** 2 (paper trade candidate)

---

## 1. Objective

Develop a BTC trading strategy using on-chain regime signals (MVRV, SOPR, Netflow, VDD), funding rate carry, ML models, and vol targeting overlays. The hypothesis: on-chain metrics can identify market regimes (accumulation vs distribution) and provide crash protection when combined with vol targeting.

## 2. Research Directions

| Sub-project | Approach | Configs | Verdict |
|-------------|----------|---------|---------|
| P002 Screening | On-chain regime voting + FGI | 1,728 | IS Sharpe 2.13, val FAIL |
| P002-B | Vol targeting + MVRV momentum gate | 54 | VT-only PASS, MVRV FAIL |
| P002-C | Continuous MVRV tilt (9 variants) | 99 | All negative vs VT-only |
| P002-D | ML vol forecast + tilt prediction | 945 | Best val 0.349, gate FAIL |
| P002-E | Funding rate carry | 0 (analysis) | NOT_VIABLE, MONITOR |
| P002-F | VDD tilt | 24 | Phase 0 STOP, cheap trap |

---

## 3. Screening (S1-S4)

### Champion config
```
Signal: INV_MVRV(pw=900) + SOPR(ss=3) + Netflow(14d)
Voting: majority (2/3), persistence=5 days
Overlay: FGI(ft=25, adj=0.3) + vol targeting
Cost: 15 bps/side
```

### IS performance (2019-2021)
| Metric | Value |
|--------|-------|
| Sharpe @15bps | 2.129 |
| MaxDD | -26.2% (vol-targeted: -5.1%) |
| CPCV PBO | 0.000 (924 paths) |
| MC p-value | 0.003 |
| DSR (N=297) | 0.805 (fails 0.95 threshold) |

### Validation (2022-2023): ALL 4 GATES FAIL
| Gate | Value | Threshold |
|------|-------|-----------|
| Val Sharpe | **-0.223** | >= 0.3 |
| Val MaxDD | **-60.1%** | < 25% |
| WF Retention | **-10.5%** | >= 50% |
| Worst fold | **-1.45** | > -1.0 |

### Root cause
MVRV 900d rolling median includes 2021 bubble prices. Post-bubble, all prices look "cheap" relative to this window. Signal stays 91% bullish through the entire 2022 bear market: 67% long during LUNA crash (BTC -57%), 82% long during FTX collapse. The 2022-2023 validation decomposes as:

| Component | Val Sharpe |
|-----------|-----------|
| Raw on-chain regime | -0.223 |
| + Vol targeting | -0.198 |
| **Pure vol targeting (no signal)** | **+0.336** |
| Buy & hold | +0.111 |

The on-chain signal destroyed 0.56 Sharpe vs vol targeting alone.

---

## 4. P002-B: Vol Targeting + MVRV Momentum Gate

### Vol targeting alone (12 configs)
All positive on validation. Best: EWMA(90)/tv=0.10 val Sharpe 0.362. Sharpe is invariant to target_vol (pure position scaling) — only the EWMA span matters.

### MVRV momentum gate (54 configs)
Best config (Variant B): VT + MVRV lb=1460, tilt=0.5, momentum_window=30, retreat_factor=1.0.

| Partition | VT-only | VT+MVRV-B | Delta |
|-----------|---------|-----------|-------|
| Dev (2019-2021) | 1.632 | 2.240 | **+0.608** |
| Val (2022-2023) | 0.329 | 0.545 | **+0.216** |
| Holdout (2024-2026) | **0.618** | 0.335 | **-0.283** |

**MVRV delta decays across partitions.** Tilt adds value during bears (compression/expansion cycles), destroys value during bulls (always looks "overvalued"). Post-ETF markets with frequent 5-15% pullbacks cause the 30d momentum gate to fire on 54% of all days, missing the bulk of rally gains.

### Crash test (2025 BTC correction, -49.5%)
The gate correctly detected the crash (flat 72% of correction days, MaxDD -7.1% vs B&H -49.5%) but the same gate that protects during crashes also fires during normal pullbacks, destroying alpha in trending markets.

### Holdout verdict
- **VT+MVRV-B:** FAIL (MVRV delta < 0)
- **VT-only:** PASS — Sharpe 0.618, MaxDD -15.7%, return +14.4%
- DSR: 0.961 (N=54) — dev Sharpe statistically significant

---

## 5. P002-C: Continuous MVRV Tilt

99 configs across 9 transform variants (linear, sigmoid, z-score, asymmetric, floor/ceiling, momentum, multi-horizon, dual-SOPR, vol-multiplier). All with lookback 730/1095/1460.

**Result:** 0/99 configs beat VT-only on BOTH val Sharpe AND val MaxDD.

The "cheap trap" is structural: during LUNA crash, MVRV pctrank < 0.5, so ALL tilt variants increased position size into the crash. Best variant (Sigmoid D) had val Sharpe 0.473 but MaxDD -21.7% (worse than VT baseline -15.4%).

---

## 6. P002-D: ML Vol Forecasting + Tilt Prediction

630 pre-registered configs + 315 exploratory.

**Track A (vol forecasting, 270 configs):** Best val Sharpe 0.165 — EWMA(60) already well-calibrated, ML adds noise.

**Track B (tilt prediction, 360 configs):** Best val Sharpe 0.349 with MaxDD -0.153. Fails DD gate by 0.002. The beta model still predicts tilt=1.37 during LUNA (inherits MVRV cheap trap through feature correlations).

**Exploratory crash classifiers (315 configs):** All degenerate — logistic regression outputs constant P~0.5, producing monotonic position scaling. One sparse gate (vol_spike + neg_mom, 4% trigger rate) showed val Sharpe 0.499 — noted as future P001 candidate but not pursued further.

---

## 7. P002-E: Funding Rate Carry

Quick investigation (5 minutes, $0.81). Blueprint claim that carry is "negative in 2025" was wrong — actual 2025 mean was +4.54% annualized.

| Metric | Pre-ETF | Post-ETF | Feb 2026 |
|--------|---------|----------|----------|
| BTC ann carry | 15.8% | 7.6% | 0.98% |
| ETH ann carry | — | — | -2.31% |

ETF structural break halved carry but didn't kill it. Leading indicator: BTC 30d price momentum leads funding by ~7 days (corr 0.84). Currently NOT_VIABLE. Funding rate monitor deployed to alert when carry regime recovers.

---

## 8. P002-F: VDD Investigation

Phase 0 characterization (6.8 min, $1.08). Two STOP conditions triggered immediately.

**STOP 1 — VDD has the cheap trap:**
| Crash | VDD pctrank (1460d) | Signal |
|-------|-------------------|--------|
| LUNA (May 2022) | 0.433 | "accumulate" (wrong) |
| FTX (Nov 2022) | 0.457 | "accumulate" (wrong) |

**STOP 2 — Catastrophic autocorrelation:**
ac1 = 0.9962, effective N ~ 3 (from 1,797 nominal). DSR validation statistically impossible.

VDD's mechanism differs from MVRV (behavioral: bear markets suppress old coin movement, so VDD stays low) but the result is identical: signal reads "accumulation" during crashes.

Confirmatory session (24 configs): high-threshold variants produce tilt=1.0 always (no signal); momentum variants increase exposure into LUNA.

---

## 9. Structural Lessons

### The on-chain cheap trap is universal
All tested on-chain metrics (MVRV, SOPR, VDD, Netflow) share a bear-market suppression problem. During crashes, on-chain activity drops — fewer transactions, less coin movement, lower network value metrics. These conditions look identical to accumulation bottoms. The metrics were designed to identify distribution tops (high activity = sell), not crash bottoms (low activity != buy).

This was confirmed across five independent approaches:
- P002 screening: raw regime voting fails
- P002-B: MVRV momentum gate works in bears, fails in bulls
- P002-C: 9 continuous tilt transforms, all fail
- P002-D: ML inherits same biases through feature correlations
- P002-F: VDD behavioral trap (different mechanism, same outcome)

### Strong IS metrics don't guarantee OOS success
The S1 champion had PBO=0.000 and MC p=0.003 — mathematically zero probability of backtest overfitting within 2019-2021. Yet it produced Sharpe -0.22 on 2022-2023. The IS tests confirm the signal was real within that regime, but cannot detect regime-dependent mechanism failure.

### Vol targeting is mechanically robust
Pure EWMA-based position scaling (always long, size inversely proportional to realized vol) produced consistent results across all three partitions. It has no signal extraction risk, no regime dependency, and naturally reduces exposure during high-vol crashes. The tradeoff: it cannot avoid losses, only reduce their magnitude through position sizing.

---

## 10. Final Strategy Specification

```
Strategy:      Vol Targeting (VT-only)
Asset:         BTC perpetual / spot
Signal:        None (always long)
Position:      target_vol / EWMA_vol(span=60)
Target vol:    10% annualized
Max leverage:  1.0x (capped)
Rebalancing:   Daily
Cost tier:     Spot maker (15 bps/side)
```

### Three-partition performance

| Partition | Period | Sharpe | MaxDD | Return |
|-----------|--------|--------|-------|--------|
| Dev | 2019-2021 | 1.632 | -11.3% | +69.3% |
| Val | 2022-2023 | 0.329 | -15.4% | +5.8% |
| **Holdout** | **2024-2026** | **0.618** | **-15.7%** | **+14.4%** |

### OOS evaluation (logged to `results/oos_evaluations.jsonl`)
- Holdout Sharpe: 0.618
- DSR: 0.961 (N=54), 0.954 (N=66) — statistically significant
- Tier: **2 (paper trade candidate)**
- wandb run: `p002b_holdout_eval` (d1i45rux)

---

## 11. Disposition

| Option | Recommendation |
|--------|---------------|
| Paper trade VT-only | **Recommended.** $10-50K, 1-3 months. Expected Sharpe 0.3-0.6. |
| Combine with P001 Donchian | Low correlation, diversification benefit. Evaluate jointly. |
| Deploy VT+MVRV-B | **Do NOT deploy.** MVRV delta negative on holdout. |
| Continue on-chain research | **Closed.** 5 independent approaches exhausted. |
| Monitor funding carry | Deployed. Timer alerts when carry recovers above 5.5% ann. |

---

## 12. Session Telemetry

| Sub-project | Session | Start (EST) | End (EST) | Duration | Configs | Cost | wandb |
|-------------|---------|-------------|-----------|----------|---------|------|-------|
| P002 S1 | 20260221_152336 | 10:23 | 10:49 | 25.5 min | 297 | $0.09 | 9 runs |
| P002 S2 | 20260221_154908 | 10:49 | 11:15 | 26.1 min | 418 | $2.49 | 10 runs |
| P002 S3 | 20260221_161521 | 11:15 | 11:38 | 22.9 min | 642 | $2.24 | 11 runs |
| P002 S4 | 20260221_163827 | 11:38 | 11:56 | 17.3 min | 371 | $2.03 | 4 runs |
| P002-D | 20260222_054207 | 00:42 | 01:12 | 30.1 min | 945 | $0.07 | 6 runs |
| P002-E | 20260222_153836 | 10:38 | 10:44 | 4.9 min | 0 | $0.81 | 1 run |
| P002-F S1 | 20260222_155646 | 10:57 | 11:04 | 6.8 min | 0 | $1.08 | 1 run |
| P002-F S2 | (confirmatory) | — | — | ~5 min | 24 | — | 1 run |
| **Total** | **8 sessions** | | | **~138 min** | **~2,697** | **~$8.81** | **43 runs** |

---

## 13. Artifacts

| Artifact | Path |
|----------|------|
| Screening audit | `reports/p002_audit_report.md` |
| S1 validation card | `reports/p002_s1_validation_card.md` |
| Phase 4 validation card | `reports/p002_phase4_validation_card.md` |
| Phase 4 diagnostics | `reports/p002_phase4_diagnostics.md` |
| Vol targeting + MVRV tilt | `reports/p002b_vol_targeting_mvrv_tilt.md` |
| Holdout decision card | `reports/p002b_holdout_decision_card.md` |
| ML models report | `results/p002d/p002d_ml_models_report.md` |
| Funding investigation | `results/p002e/funding_investigation.json` |
| VDD report | `results/p002f/p002f_vdd_report.md` |
| OOS evaluations | `results/oos_evaluations.jsonl` |
| wandb project | `datadex_ai/sparky-ai` (43 runs, tags: p002*) |
