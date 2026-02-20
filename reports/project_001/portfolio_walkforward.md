# Portfolio Walk-Forward Validation

**Portfolio**: BTC Don8h(82,20) 30% + ETH Don8h(83,33) 70%
**Date**: 2026-02-20
**Protocol**: 5-fold reverse-chronological, 20% test, 30-bar gap
**Data**: BTC/ETH 8h OHLCV, common range 2017-01-01 to 2023-12-02 (7,579 bars)

## Results Summary

| Metric | 30 bps | 50 bps |
|--------|--------|--------|
| IS Sharpe | 2.217 | 2.099 |
| WF Sharpe | 1.579 | 1.516 |
| Retention | 71.2% | 72.2% |
| WF MaxDD | -5.9% | -6.0% |
| WF Total Return | 90.6% | 85.9% |
| WF Calmar | 2.016 | 1.914 |
| WF PSR | 99.99% | 99.98% |
| WF DSR (n=16,269) | 0.396 | 0.341 |

**VERDICT: PASS** (retention 71.2% > 60% threshold)

## Per-Fold Breakdown

| Fold | Period (approx) | Sharpe (30bps) | BTC Scale | ETH Scale | n_obs |
|------|----------------|----------------|-----------|-----------|-------|
| 0 (most recent) | ~2022-2023 | 0.844 | 0.35 | 0.18 | 1,515 |
| 1 | ~2020-2022 | 1.503 | 0.18 | 0.17 | 1,515 |
| 2 | ~2019-2020 | 2.899 | 0.25 | 0.13 | 1,515 |
| 3 | ~2017-2019 | 0.995 | 0.36 | 0.16 | 1,515 |

- **Mean fold Sharpe**: 1.560
- **Min fold Sharpe**: 0.844
- **Positive folds**: 4/4 (100%)

## Notes

- Fold 4 skipped (insufficient training data after gap deduction)
- IS Sharpe of 2.217 was computed on full aligned range (2017-2023)
- WF Sharpe of 1.579 is computed on concatenated test returns (6,060 bars)
- Donchian params are fixed (not fitted), so WF tests temporal stability of vol sizing and regime persistence
- WF DSR is low (0.396) due to cumulative n_trials=16,269 — this is expected for a post-search validation
- All folds are profitable, with worst fold (0, 2022-2023 bear) still at Sharpe 0.844
- Fold 2 (2019-2020 bull run) has highest Sharpe 2.899 — consistent with momentum strategy
- BTC scale varies 0.18-0.36, ETH scale 0.13-0.18 — inverse vol adapts to regime volatility
- MaxDD of -5.9% in WF vs -7.0% in IS: WF is actually *less* drawdown, consistent with vol targeting

## Comparison with Individual WF

| Component | IS Sharpe | WF Sharpe | Retention |
|-----------|-----------|-----------|-----------|
| BTC Don8h(82,20) individual | 2.220 | 1.708 | 77% |
| Portfolio (this report) | 2.217 | 1.579 | 71% |

Portfolio retention is slightly lower than BTC individual (71% vs 77%), but the portfolio has:
- Much lower MaxDD (-5.9% vs BTC individual ~-19.7%)
- All folds positive (BTC individual had 1 near-zero fold)
- Better risk-adjusted returns via ETH diversification

## OOS Readiness Assessment

This validates criterion #7 (portfolio-level walk-forward). Combined with the 8 criteria
already assessed in the final report, the portfolio is ready for single-shot OOS evaluation.
