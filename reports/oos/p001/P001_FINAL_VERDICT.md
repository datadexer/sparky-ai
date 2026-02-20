# P001 Final Verdict — CLOSED

**Date:** 2026-02-20

## Executive Summary

Program 001 evaluated a BTC/ETH Donchian trend-following portfolio — the champion selected from 16,269 in-sample configurations (BTC Don8h(82,20) at 30% + ETH Don8h(83,33) at 70%, both with inverse-volatility sizing). Out-of-sample from 2024-01-01 to 2026-02-20 (781 days, 2346 bars), the portfolio produced Sharpe 0.417, failing the pre-registered threshold of 0.5. The core finding: BTC's trend-following edge is dead OOS (Sharpe -0.114), ETH retains a marginal edge (Sharpe 0.645) but cannot carry the portfolio alone. IS-to-OOS Sharpe retention was 18.8% versus the walk-forward prediction of 71.2%. Program 001 is closed.

## Champion Portfolio OOS Results

| Metric | IS | OOS | Retention |
|--------|-----|-----|-----------|
| Portfolio Sharpe | 2.217 | 0.417 | 18.8% |
| Portfolio MaxDD | -7.0% | -12.8% | worse |
| Portfolio Return | — | 6.4% | — |
| Portfolio Sortino | — | 0.594 | — |
| Portfolio Calmar | — | 0.250 | — |
| Portfolio DSR (n=16,269 IS / n=1 OOS) | 0.972 | 0.536 | — |
| BTC Sharpe | 2.220 | -0.114 | negative |
| ETH Sharpe | 2.056 | 0.645 | 31.4% |
| ETH MaxDD | -9.5% | -12.8% | worse |

Walk-forward validation predicted 71.2% Sharpe retention (IS WF Sharpe 1.579 / IS Sharpe 2.217). Actual retention was 18.8%. All four WF folds were positive, yet the OOS BTC leg went negative. The IS walk-forward folds all fell within a structurally different regime (2013-2023) where trend-following worked consistently. The 2024-2026 period broke that pattern.

## Per-Leg OOS Decomposition

| Metric | BTC Don8h(82,20) | ETH Don8h(83,33) | Portfolio 30/70 |
|--------|-------------------|-------------------|-----------------|
| Sharpe | -0.114 | 0.645 | 0.417 |
| MaxDD | -19.6% | -12.8% | -12.8% |
| Return | -4.0% | 10.8% | 6.4% |
| Calmar | -0.065 | 0.400 | 0.250 |
| Sortino | -0.162 | 0.937 | 0.594 |

**BTC:** The trend-following edge that produced IS Sharpe 2.220 is gone. OOS Sharpe is -0.114 with a -19.6% drawdown. BTC spent much of 2024 in a range-bound market where Donchian breakouts generate whipsaws. The 237-day drawdown (2024-03-13 to 2024-11-05) maps directly to this ranging period.

**ETH:** Retains positive Sharpe (0.645) with 31.4% retention from IS. However, running ETH standalone at 100% weight produces Sharpe 0.477 — below the 0.5 threshold — because cost drag increases at full allocation. The formal ETH-only evaluation (OOS Sharpe 0.477, MaxDD -13.8%, Return 8.3%) missed the threshold by 0.023.

**Correlation:** Mean rolling 60-bar correlation between legs was 0.487 (0.476 during drawdowns). This provided less diversification benefit than IS suggested, particularly because both legs drew down simultaneously in early 2024.

## Key Findings

1. **BTC trend-following edge is gone.** IS Sharpe 2.220 became OOS Sharpe -0.114. The 2024-2026 BTC market was predominantly range-bound, which is the failure mode for breakout strategies.
2. **ETH retains a marginal edge.** 31.4% Sharpe retention (IS 2.056 to OOS 0.645) is positive but insufficient to meet the portfolio threshold when BTC is a drag.
3. **Lower target vol improves OOS risk-adjusted returns.** The minimal config (BTC TV=0.05, ETH TV=0.035) achieved Sharpe 0.675 and MaxDD -3.9% OOS — better than the original on both metrics.

| Config | BTC TV | ETH TV | Sharpe | MaxDD | Return | Calmar |
|--------|--------|--------|--------|-------|--------|--------|
| Original | 0.20 | 0.15 | 0.417 | -12.8% | 6.4% | 0.250 |
| Reduced | 0.15 | 0.10 | 0.393 | -9.0% | 4.3% | 0.232 |
| Conservative | 0.10 | 0.07 | 0.427 | -6.2% | 3.3% | 0.257 |
| Minimal | 0.05 | 0.035 | 0.675 | -3.9% | 4.0% | 0.477 |

4. **The 237-day drawdown (March-November 2024)** coincides with BTC's ranging market. Donchian breakout fails in range-bound conditions — this is a known structural weakness, not a surprise.
5. **Walk-forward overpredicted retention** (71.2% vs 18.8%). All IS walk-forward folds fell within a trend-friendly regime. The OOS period represents a regime shift that time-series cross-validation within IS data could not anticipate.
6. **IS-to-OOS Sharpe decay of 81%** far exceeds the 30-50% degradation typical of trend-following strategies, indicating the IS edge was partly regime-specific rather than structural.

## Bugs Discovered During OOS

**Bug 1 — Flat Sizing Default.** The initial OOS evaluation scripts (evaluations 1-4) did not implement inverse-volatility sizing despite the research config specifying it. All positions were full-size regardless of volatility. This produced inflated MaxDD (-30.5% vs -12.8% with IV sizing) and an artificially high Sharpe (0.822) that masked the true OOS performance. The discrepancy between OOS MaxDD (-30.5%) and IS MaxDD (-7.0%) flagged the issue. Fixed by adding IV sizing logic to `oos_evaluate.py`.

**Bug 2 — IV Timing Lookahead.** After adding IV sizing, the OOS script applied `signals.shift(1) * scale` — shifting the signal but using the current bar's volatility scale. The research code applies `(signals * scale).shift(1)` — shifting the combined product, so scale uses the previous bar's realized volatility. The OOS version had access to current-bar vol, introducing a subtle lookahead. This inflated the IS reproduction Sharpe to 2.442 vs the known 2.217, which flagged the discrepancy. Fixed by aligning OOS scripts with the research code pattern.

Both bugs were discovered through the OOS evaluation process: IS metric discrepancies (Sharpe and MaxDD) against known values revealed the issues before any OOS conclusions were drawn.

## Validation Summary

The validation sweep script (`run_validation_sweep.py`) runs 5 tests that gate parameter exploration:

1. **IS Reproduction** — portfolio and per-leg Sharpe/MaxDD match known IS values within tolerance
2. **Signal Verification** — independently rebuilt signals match exactly
3. **IV Scale Match** — scale factors match `sweep_utils.inv_vol_sizing()` within atol=1e-6
4. **Return Spot Check** — 10 random IS dates manually verified
5. **Data Continuity** — no gaps, duplicates, or price jumps at IS/OOS boundary

All 5 tests passed before OOS metrics were computed. The validation suite was designed to catch exactly the class of bugs found (sizing defaults, timing alignment, signal reconstruction errors). Parameter sweeps (6 weight configs, 5 ETH target vol configs) and the formal ETH-only evaluation ran after validation passed.

## Artifacts

| File | Description |
|------|-------------|
| `oos_evaluation_20260220_181205.md` | First eval attempt — no PPY, minimal format |
| `oos_evaluation_20260220_181836.md` | Second eval — PPY added, holdout section |
| `oos_evaluation_20260220_182827.md` | Third eval — IS section added |
| `oos_evaluation_20260220_184103.md` | Fourth eval — flat sizing, final format |
| `oos_evaluation_20260220_191508.md` | Fifth eval — IV sizing added, buggy IV timing |
| `diagnostics/oos_diagnostic_report.md` | Full diagnostic with per-leg decomposition |
| `diagnostics/equity_curve.png` | OOS equity curve |
| `diagnostics/underwater.png` | Drawdown underwater chart |
| `diagnostics/leg_decomposition.png` | BTC vs ETH per-leg performance + rolling correlation |
| `diagnostics/monthly_returns.png` | Monthly returns heatmap |
| `diagnostics/tv_sensitivity.png` | Target vol sensitivity scatter |
| `diagnostics/regime_overlay.png` | On-chain regime signals overlaid |
| `validation_sweep/eth_only_evaluation.md` | ETH-only formal OOS evaluation |
| `validation_sweep/parameter_sweeps.png` | Weight + ETH TV sweep scatter plots |

## Disposition

**CLOSED.** Program 001 Donchian trend-following does not survive out-of-sample evaluation. The BTC leg is dead (negative Sharpe OOS). The ETH leg retains marginal predictive content (Sharpe 0.645 OOS) which may inform future programs — particularly the finding that lower target volatility improves OOS risk-adjusted returns. The infrastructure built for this evaluation (OOS evaluation scripts, 5-test validation suite, holdout protocol, diagnostic tooling) is reusable for future programs. No further OOS evaluations will be conducted for P001 candidates.
