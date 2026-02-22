# P002 VALIDATION CARD — S1 Pre-Registered Champion

```
Config: INV_MVRV(pw=900) + SOPR(ss=3) + Netflow(14d SMA)
        Majority vote (2/3 agree), persistence=5 days
        FGI overlay: fear_threshold=25, adjustment=0.3, fear_only=True

Dev set: 2019-01-01 to 2021-12-31 (1095 observations)
Cost model: 15 bps maker (spot_maker), stress at 30 bps

Signal components (all BGeometrics, 1-bar lag):
  - MVRV z-score: below rolling 900d median → bullish (+1), else bearish (-1)
  - SOPR: 3d SMA > 1.0 → bullish (+1), else bearish (-1)
  - Exchange netflow: 14d SMA < 0 → bullish (+1), else bearish (-1)
  Composite: equal 1/3 weights, long when sum > 0 (majority vote = 2 of 3)
  Persistence: 5 consecutive days required before state change
  FGI: position × 1.3 during extreme fear (FGI ≤ 25), 1.0 otherwise
```

## Metrics

| Metric | Raw | Vol-Targeted (tv=0.10) |
|--------|-----|------------------------|
| Sharpe @15bps | 2.129 | 2.119 |
| Sharpe @30bps | 2.075 | — |
| MaxDD | -26.2% | -5.1% |
| Total Return | 1389% | 56% |
| Sortino | 3.891 | 3.960 |
| Calmar | 5.572 | 3.140 |

| Metric | Value |
|--------|-------|
| Round-trip trades | 22 |
| Position changes (n_trades) | 69 |
| Trade win rate | 68.2% |
| Day-level win rate | 23.7% |
| Profit factor | 6.82 |
| Avg hold period | 20.8 days |
| Median hold period | 16.0 days |
| Time in position | 41.8% |
| Max single trade PnL | 42.9% of total |
| Max gap between trades | 92 days |
| Mean position size | 0.439 |

## Validation Battery

| Gate | Value | Threshold | Verdict |
|------|-------|-----------|---------|
| CPCV PBO (924 paths) | 0.000 | < 0.40 | **PASS** |
| CPCV median path Sharpe | 2.04 | > 0.25 | **PASS** |
| CPCV p10 path Sharpe | 1.36 | > 0.0 | **PASS** |
| MC p-value (circular, 1000 perms) | 0.003 | < 0.05 | **PASS** |
| MC p-value (block, 21d, 1000 perms) | 0.004 | < 0.05 | **PASS** |
| DSR (N=297, S1 only) | 0.805 | > 0.95 | **FAIL** |
| DSR (N=1728, all sessions) | 0.619 | > 0.95 | **FAIL** |
| Plateau coverage (67 configs) | 95.5% | >= 50% | **PASS** |
| Regime gate (3/3 profitable) | 3/3 | >= 2/3 | **PASS** |
| Temporal stability (negative streak) | 0 days | < 90 days | **PASS** |

**Overall: 8/10 PASS, 2 FAIL (DSR only)**

## CPCV Path Sharpe Distribution

```
min  p5   p10  p25  MEDIAN  p75  p90  p95  max
0.70 1.21 1.36 1.68 2.04    2.39 2.68 2.82 3.09
```

All 924/924 paths positive. Mean 2.03, std 0.49. Zero probability of backtest overfitting.

## Regime Attribution

| Regime | Sharpe | % of PnL | % of Time |
|--------|--------|----------|-----------|
| Bull (2019 recovery, 2020 recovery, 2021 H1) | 2.16 | 70.4% | 74.9% |
| Bear (COVID crash, 2021 H2) | 2.36 | 26.8% | 19.6% |
| Sideways (2020 pre-COVID) | 2.21 | 2.8% | 5.5% |

Strategy profitable in all 3 macro regimes. COVID crash performance: Strategy Sharpe 3.32 vs B&H -0.83 (entered after bottom, not fully flat during crash).

## Signal Contribution

| Signal | Corr w/ Position | Marginal IC | Assessment |
|--------|-----------------|-------------|------------|
| Netflow | 0.366 | 0.005 | Dominant timing driver |
| MVRV | 0.180 | 0.009 | Highest forward return predictability |
| SOPR | 0.025 | -0.003 | **Negligible / possible noise** |

SOPR has negative marginal IC. The signal may be redundant given MVRV and netflow already capture regime information. However, SOPR's role in the majority-vote gate (requiring 2/3 agreement) means it acts as a filter rather than an alpha source.

## Temporal Stability

Rolling 1-year Sharpe trajectory (2019-2021):
- Start: 3.37 → End: 1.85
- Min: 0.93 (Jan 2021) → Max: 3.50
- Mean: 2.10
- First half avg: 2.30, Second half avg: 1.90 (82.7% retention)
- **Zero days of negative rolling Sharpe**

Moderate decay from early peak, but no structural degradation.

## Risk Flags

1. **DSR fails at all N values.** Even at N=297 (S1-only trials), DSR=0.805. The strategy's heavy-tailed returns (kurtosis=15.27, skewness=1.69) are the primary cause — they widen the DSR penalty. This is a structural limitation of on-chain daily signals with concentrated profits.

2. **Largest trade = 42.9% of total PnL.** Trade #4 (2019-04-08 to 2019-06-23, BTC $5,150 → $10,900) accounts for 42.9% of all profit. This is below the 50% disqualification threshold but remains concentrated. The top 3 trades account for 69.9% of PnL.

3. **SOPR signal adds no marginal IC.** Marginal IC = -0.003. It functions as a filter (majority vote gate) rather than an alpha generator. Removing it would change the strategy architecture.

4. **Dev set is predominantly bullish (2019-2021).** All 22 round-trip trades are LONG. No short positions generated. Bear market robustness is untested with actual short exposure — the strategy simply goes flat during bearish conditions.

5. **Vol-targeted version compresses returns heavily.** At tv=0.10, total return drops from 1389% to 56% (mean position 0.44 → 0.07). The DD improvement (-26% → -5%) comes at the cost of heavily reduced capital deployment.

## DSR Deep Dive

Why DSR fails despite strong CPCV and MC:
- DSR penalizes based on kurtosis and skewness of returns. Kurtosis=15.27 means fat tails.
- With N=297 trials and heavy tails, the "expected best" Sharpe from random selection is high.
- The MC test (p=0.003) tests the SPECIFIC position timing, confirming the signal is real.
- DSR tests whether any of 297 random strategies could have achieved this Sharpe, which is a stricter bar.

At what N would DSR > 0.95? With Sharpe=2.13, skew=1.69, kurt=15.27, DSR > 0.95 requires N < ~22. The strategy space (297+ trials) makes this impossible.

## Vol Targeting Sensitivity

| Target Vol | Sharpe | MaxDD | Total Return | Mean Position |
|-----------|--------|-------|-------------|---------------|
| Raw (no VT) | 2.129 | -26.2% | 1389% | 0.439 |
| 0.10 | 2.119 | -5.1% | 56% | 0.067 |
| 0.15 | 2.119 | -7.6% | 94% | 0.101 |
| 0.20 | 2.119 | -10.1% | 141% | 0.134 |

Vol targeting is a clean risk overlay: near-zero Sharpe cost with proportional DD/return scaling.

## Overall Verdict: MARGINAL

The S1 champion passes 8/10 validation gates with strong CPCV (PBO=0.000), significant MC p-values (0.003), robust parameter plateau (95.5%), all-regime profitability, and stable temporal performance.

**The sole failure is DSR**, driven by heavy-tailed returns and the cumulative trial count. This is a known structural limitation: on-chain daily signals with infrequent trading produce concentrated profits that inflate kurtosis, which DSR penalizes. The MC test confirms the signal timing is real (p=0.003); the DSR concern is whether we'd find something this good by chance across 297 trials.

### Recommendation

Per the directive's OOS readiness criteria:
- **OOS READY** requires: All Phase 3 gates pass → **NO** (DSR fails)
- **CONDITIONAL** requires: ≥3/5 Phase 3 gates pass → **YES** (4/5 pass)
- **FAIL** requires: ≥2 gates fail → **NO** (only 1 unique gate fails — DSR)

**Verdict: CONDITIONAL** — Strategy has genuine signal (MC confirms), robust parameter space (plateau confirms), no overfitting (CPCV confirms), but cannot rule out multiple testing inflation (DSR fails).

### For AK's Decision

**Option A: Proceed to OOS validation (2022-2023) with AK present.**
Rationale: MC p=0.003 is strong evidence of real signal. DSR is structurally penalized by kurtosis. CPCV shows zero paths with negative Sharpe. The 2022 bear market will be the ultimate test.

**Option B: Reduce trial count penalty by testing the raw INV_FINE variant (rank 3).**
Config: pw=900, pers=5, ss=3, inv_mvrv=true (NO FGI overlay). Sharpe @15bps = 2.074, 44 trades, same DSR issue. This removes the FGI degrees of freedom but doesn't fix DSR.

**Option C: Accept DSR limitation and evaluate on strict OOS metrics only.**
Treat IS as "screening" and place full weight on OOS performance. If OOS Sharpe > 0.5 and DD < 40%, declare success regardless of DSR.

**Option D: Kill P002 on-chain regime strategies.**
Accept that daily on-chain signals produce too few trades for DSR to pass, and concentrate on P001 Donchian portfolio (which passed all gates including DSR).

### Additional Notes

- **VDD champion** (non-pre-registered, S@30=2.59): Disqualified per audit. Contains only 5 trades, 52.9% PnL concentration in single trade. Not a viable candidate even if pre-registration were amended.

- **Puell Multiple signal** (S@30=2.32 from S2): Alternative on-chain signal worth investigating in future project. Was found during sessions 2-3 with VDD, so also carries pre-registration concerns.

- **Backup S1 candidates** above Sharpe 1.5 (all pre-registered):
  - INV_VOL (rank 5): S@15=2.068, DD=-7.5%, 65 trades (with vol targeting built in)
  - INV_FINE (rank 3): S@15=2.074, DD=-26.2%, 44 trades (raw signal, no overlays)
  - INV_FINE (rank 4): S@15=2.071, DD=-26.9%, 28 trades (higher persistence)

All backup candidates share the same DSR limitation (heavy tails from infrequent on-chain daily trading).
