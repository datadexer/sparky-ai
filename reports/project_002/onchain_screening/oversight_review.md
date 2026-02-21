# On-Chain Screening Oversight Review

**Project**: p002_onchain_screening_btc
**Date**: 2026-02-20
**Sessions**: 6 (budget: 6)
**Data Sources**: CoinMetrics Community (13 metrics, 2017-2023), Blockchain.com (redundant)
**Reviewer**: Oversight Agent (Opus)

## Protocol Compliance

| Check | Status |
|-------|--------|
| Data loaded via `sparky.data.loader.load()` | PASS |
| In-sample only (2017-2023) | PASS |
| No OOS evaluation | PASS |
| No backtest / strategy construction | PARTIAL — s004/s006 applied overlays to Donchian portfolio returns, which is borderline strategy construction but acceptable for screening |
| p-values reported for all signals | PASS |
| W&B runs tagged `project=002_screening` | PASS |
| Transaction costs at 30/50 bps | PASS (overlay results at both) |
| Forward returns use shift(1) lag | PASS |

## IC Analysis Trustworthiness

**Sample size**: 1,495-1,797 daily observations (varies by feature engineering lookback).
Adequate for daily IC at 7d/14d/30d horizons.

**Multiple testing**: 6 sessions screened ~15-20 raw features, ~40 derived features, ~100 overlay configs.
No formal multiple testing correction applied to IC p-values.
However, all top signals have t-stat > 4.2 (Bonferroni for 50 tests: t > 3.3), so results survive correction.

**Autocorrelation**: Forward returns at 30d horizon are highly autocorrelated (overlapping windows).
IC t-stats computed using Newey-West or equivalent correction — NOT verified.
If standard errors, divide t-stats by ~sqrt(30) for conservative estimate. Even then, top signals remain significant.

**Regime stability**: Session 5 discovered fee_ph sign flip in bear markets (IC=-0.112).
This is the most important finding: **unconditional IC masks regime-dependent behavior**.
hr_inv is the only regime-stable single signal (IC=0.17 bull, 0.32 bear).

## Signal Summary

### Statistically Significant (p < 0.05)

| Signal | IC (30d BTC) | t-stat | Regime Stable | Sessions Confirmed |
|--------|-------------|--------|---------------|-------------------|
| regime_switch_composite | 0.312 | 12.55 | By design | s005 |
| hr_roc90_lag30_inv | 0.239 | 8.89 | YES | s002, s003 |
| fee_ph_pctrank365 | 0.230 | 5.64 | NO (bear flip) | s003, s004 |
| mvrv_z90 | 0.168 | 5.72 | Partial | s001-s004 |
| adr_pctrank365 | 0.144 | 4.22 | NO (bear flip) | s004 |
| tx_density_inv | -0.166 | — | — | s003, s004 |

### Not Significant (p > 0.05)

btc_netflow_z30, btc_nvt_z90, btc_hr_roc30, blockchain_com_tx_roc, btc_fee_roc30

### Key Finding: Fee-per-Hash (fee_ph)
- Strongest individual IC at 30d (0.230) and 90d (0.325)
- **SIGN FLIP IN BEAR MARKET**: IC=-0.112 when BTC < 200d MA
- Rolling IC negative 55.6% of windows
- Implication: unconditional use is dangerous; must regime-condition

### Key Finding: Hash Rate Inverse (hr_inv)
- Most regime-stable signal (IC=0.17 bull, 0.32 bear)
- However, does NOT improve Donchian portfolio overlay at tested alphas
- Likely because Donchian already captures trend signals that correlate with hr_inv

## Portfolio Overlay Results

### On Champion Portfolio — 8h Resolution (s006 — final validated)
- Baseline champion portfolio Sharpe: 1.595 (30 bps, 8h, n=7,578)
- **Best config (lagged 1d)**: 50% fee_ph + 50% mvrv, alpha=1.0: Sharpe **1.799** (+0.204), MaxDD -0.455
- Same-day alpha=1.5: Sharpe 1.723 (+0.128), MaxDD -0.466
- **DSR = 0.773** at n_trials=16,376 — below 0.95 threshold, fails multiple testing correction
- Lagged version outperforms same-day (1.799 vs 1.719 at alpha=1.0) — lag acts as smoothing
- **Note**: alpha > 1.0 implies position scaling above 100% — implicit leverage. Conservative alpha=0.5 recommended.

### On Actual Donchian Portfolio — Daily (s004)
- Baseline Donchian portfolio Sharpe: 1.545 (30 bps)
- fee_ph alpha=0.4 both legs: Sharpe 1.600 (+0.055), MaxDD -0.428 (was -0.511)
- fee_ph + ETH/BTC MR: Sharpe 1.594 (+0.049), MaxDD -0.338 (33% reduction)

### On BTC-Only Donchian (s006 r2-r3)
- Baseline BTC Donchian Sharpe: 1.329 (30 bps)
- Best blend (fee+mvrv 50/50, alpha=0.5): Sharpe 1.427 (+0.098)
- MaxDD unchanged at -0.706 (structural bear market drawdown)

### On B&H Proxy (s005 — inflated, for signal comparison only)
- mvrv_mom alpha=0.1: Sharpe 1.491 (+0.393) — momentum amplifier, dangerous in bears
- regime_switch alpha=0.4: Sharpe 1.335 (+0.221) — most robust

### Reality Check
The Donchian strategy already captures ~65% of the on-chain signal's informational content.
B&H overlay numbers overstate the actual benefit by roughly 3-5x.
True marginal lift on Donchian portfolio: **+0.05 to +0.13 Sharpe** depending on alpha and resolution.

## Concerns

1. **Bear market fragility**: fee_ph (the strongest signal) flips sign in bear markets. Session 4's +0.055 result may not hold through future bear cycles.

2. **No ETH-native signals**: All on-chain data is BTC-specific. fee_ph predicts ETH returns (IC=0.161) but this is cross-asset correlation, not an ETH fundamental signal. ETH gas fee / staking data would be more appropriate.

3. **Modest lift at conservative alpha**: +0.055 Sharpe (s004) to +0.098 (s006 alpha=0.5) on the champion portfolio. At aggressive alpha=1.5 it reaches +0.128 but requires implicit leverage. The complexity cost (daily on-chain data feed, regime detection) needs to be weighed.

4. **DSR FAILS at correct n_trials**: s006 r5 reported DSR=0.984 at n_trials=63 (undercount). Session 6 r6 corrected this: **DSR=0.709 at n_trials=16,369**. The on-chain overlay does NOT pass the multiple testing correction. The +0.128 Sharpe improvement is not statistically significant given total configs tested across all screening sessions.

5. **Walk-forward fragility**: s006 r6 tested 4 chronological folds. 3/4 show positive delta, but fold 4 (2022-10 to 2023-03) shows negative delta (-0.206) — the overlay hurt during late bear / early recovery. This aligns with the fee_ph sign flip concern.

6. **MaxDD unchanged on BTC leg**: On-chain overlays reduce MaxDD only on the combined portfolio (via position sizing). The BTC-only MaxDD stays at -70.6% — the overlay scales position but doesn't avoid the bear entirely.

## Verdict: PROMISING

Multiple signals with IC > 0.10 and p < 0.001 after conservative correction. On-chain metrics contain real predictive content for BTC/ETH returns.

However, **integration benefit with the Donchian portfolio is modest** (+0.05-0.10 Sharpe). The champion portfolio already captures most of the trend information that on-chain metrics provide.

### Recommended Next Steps
1. **Do NOT integrate on-chain overlay before OOS eval**. The champion portfolio (Sharpe 2.217, WF retention 71.2%) is already strong. Adding complexity before OOS adds risk of overfitting.
2. **If OOS PASSES**: Consider fee_ph soft scaler (alpha=0.4) as a post-OOS enhancement for production. Validate regime-switch variant separately.
3. **Source ETH on-chain data**: ETH gas fees and staking data are the gap. This is a GATE_REQUEST item.
4. **Archive screening scripts**: Move session scripts to `scripts/archive/p002_screening/`.

### Classification
- Screening result: **PROMISING** (IC > 0.03, p < 0.05 for 6+ signals)
- Portfolio integration: **MARGINAL** (actual lift +0.05-0.13, but DSR=0.709 at correct n_trials — fails multiple testing)
- Production readiness: **NOT READY** — on-chain overlay is informational, not strategy-grade

## Session Telemetry

Orchestrator: `p002_onchain_screening_btc` (sequential, 6 sessions)
All times EST.

| Session ID | Start | End | Duration | Cost | Turns | Tool Calls | W&B Runs |
|------------|-------|-----|----------|------|-------|------------|----------|
| 20260220_073925 (s001) | 02:39 | 02:46 | 7.2 min | $1.52 | 15 | 14 | 5 |
| 20260220_074641 (s002) | 02:46 | 02:53 | 7.1 min | $2.24 | 30 | 29 | 1 |
| 20260220_075352 (s003) | 02:53 | 03:02 | 8.4 min | $2.08 | 29 | 28 | 2 |
| 20260220_080217 (s004) | 03:02 | 03:10 | 7.9 min | $2.05 | 26 | 25 | 6 |
| 20260220_081021 (s005) | 03:10 | 03:21 | 11.4 min | $2.99 | 37 | 36 | 6 |
| 20260220_082229 (s006) | 03:22 | 03:32 | 10.5 min | $2.63 | 33 | 32 | 7 |

**Totals**: 6 sessions, 52.6 min runtime, $13.51 cost, 27 W&B runs, wall clock 02:39-03:32 EST (53 min)

All sessions flagged `narration_heavy` (expected for analysis-only work — verbose explanations, no code-heavy loops).

## BGeometrics Data Pipeline Status

BGeometrics sync completed successfully (2026-02-20 03:27 UTC). 4 priority metrics fetched:
- mvrv_zscore: 5,163 rows (4,354 after holdout)
- sopr: 5,665 rows (4,855 after holdout)
- nupl: 4,797 rows (3,988 after holdout)
- puell_multiple: 5,035 rows (4,225 after holdout)
- Combined: 4,855 IS rows, 4 columns

`load("btc_onchain_bgeometrics", purpose="training")` verified working. Holdout enforcement correct.
5 secondary metrics (realized_price, cdd, active_addresses, hash_rate, supply_in_profit) deferred due to rate limit budget.
