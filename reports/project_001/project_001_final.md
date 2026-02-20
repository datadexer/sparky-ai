# Project 001 Final Report — BTC/ETH Donchian Breakout

**Date:** 2026-02-20
**Status:** OOS Pending AK Approval
**Champion Portfolio:** btc82_30_eth83_70

---

## 1. Executive Summary

Project 001 explored 16,269 configurations across BTC and ETH Donchian breakout strategies on hourly data (4h and 8h timeframes). The sweep covered entry/exit window combinations, volatility-adjusted and flat sizing variants, and cross-asset portfolio combinations.

8 individual candidates entered formal validation. 6 survived: 3 BTC (PASS), 3 ETH (CONDITIONAL). 2 ETH flat-sizing candidates failed on bootstrap MaxDD. 55 portfolio configurations were tested across 3 sessions.

The champion portfolio is **btc82_30_eth83_70**: BTC Don8h(82,20) iv(vw30,tv0.2) at 30% and ETH Don8h(83,33) iv(vw30,tv0.15) at 70%. It achieves Sharpe 2.217 at 30bps, DSR 0.972 at n=16,269 trials, MaxDD -7.0%, and 2020+ Sharpe 1.943. It meets 7 of 8 OOS readiness criteria. Portfolio-level walk-forward validation has not yet been run.

This report documents what was found, why it is credible, and where the remaining uncertainty lies.

---

## 2. Individual Candidate Results

| Candidate | Asset | TF | Sharpe@30 | Sharpe@50 | DSR | MaxDD | N_trades | Verdict |
|---|---|---|---|---|---|---|---|---|
| btc_don4h_160_25_iv | BTC | 4h | 2.319 | 2.107 | 0.9999 | -12.5% | 1421 | PASS |
| btc_don4h_60_20_iv | BTC | 4h | 2.042 | 1.808 | 0.9977 | -14.9% | 941 | PASS |
| btc_don8h_82_20_iv | BTC | 8h | 2.220 | 2.088 | 0.9998 | -19.7% | 1074 | PASS |
| eth_don8h_83_33_iv | ETH | 8h | 2.056 | 1.971 | 0.9279 | -9.5% | — | CONDITIONAL |
| eth_don4h_164_47_iv | ETH | 4h | 2.001 | 1.857 | 0.9026 | -8.3% | — | CONDITIONAL |
| eth_don4h_138_47_iv | ETH | 4h | 1.952 | 1.794 | 0.8785 | -9.7% | — | CONDITIONAL |
| eth_don4h_138_47_flat | ETH | 4h | 2.140 | 2.099 | 0.9525 | -38.8% | — | FAIL |
| eth_don4h_164_47_flat | ETH | 4h | 2.133 | 2.095 | 0.9505 | -38.8% | — | FAIL |

**Notes:**

- All 6 passing/conditional candidates share the same 3 soft fails: walk_forward_multi, tail_risk_analysis, correlation_stability. These are structural to trend-following strategies and are not considered disqualifying for this strategy class.
- The 2 ETH flat candidates fail on bootstrap_sharpe hard fail. The -38.8% MaxDD is distributed across 5+ events rather than a single event, which means the bootstrap cannot attribute it to bad luck. The IV variants fix this by reducing position size during high-volatility periods.
- ETH DSR values are all below 0.95 individually. This is a known weakness of the ETH leg and is part of the reason for the CONDITIONAL verdict. The portfolio DSR exceeds 0.95.
- BTC 2020+ sub-period Sharpe degrades to 1.20-1.40 across the BTC candidates. This degradation is real and relevant.
- All 6 survivors share CONDITIONAL status on the full validation battery. The "PASS" label for BTC candidates indicates hard criteria are met; the distinction from ETH "CONDITIONAL" reflects ETH's sub-0.95 individual DSR, not a qualitative difference in validation methodology.

---

## 3. Return Correlation Matrix

```
                   btc_160_4h  btc_60_4h  btc_82_8h  eth_138_4h  eth_164_4h  eth_83_8h
btc_160_4h              1.000      0.852      0.901       0.318       0.307      0.408
btc_60_4h               0.852      1.000      0.809       0.301       0.283      0.389
btc_82_8h               0.901      0.809      1.000       0.352       0.343      0.452
eth_138_4h              0.318      0.301      0.352       1.000       0.980      0.846
eth_164_4h              0.307      0.283      0.343       0.980       1.000      0.860
eth_83_8h               0.408      0.389      0.452       0.846       0.860      1.000
```

Within-asset correlations are high (0.80-0.98), as expected for strategies on the same underlying with similar parameters. Cross-asset correlations are 0.28-0.45, confirming genuine diversification. Combining BTC and ETH legs meaningfully reduces drawdown relative to either leg individually. The portfolio MaxDD of -7.0% versus -19.7% (BTC82) and -9.5% (ETH83) reflects this.

---

## 4. Portfolio Results

55 portfolios were tested across 3 sessions, varying asset weights, timeframe combinations, and whether to include the flat-sizing ETH legs.

### Top 5 Portfolios

| Portfolio | S@30 | S@50 | DSR | MaxDD | S2020+ |
|---|---|---|---|---|---|
| btc82_30_eth83_70 (8h) | 2.217 | 2.099 | 0.972 | -7.0% | 1.943 |
| btc82_35_eth83_65 | 2.213 | — | 0.971 | -7.0% | — |
| btc82_40_eth83_60 | 2.200 | — | 0.969 | -7.3% | 1.914 |
| btc82_50_eth83_50_8h | 2.151 | 2.021 | 0.960 | -8.9% | 1.860 |
| btc160_80_btc60_20_4h | 2.334 | 2.115 | 1.000 | -12.8% | 1.257 |

**Note on btc160_80_btc60_20_4h:** This portfolio has the highest raw Sharpe (2.334) and DSR=1.000, but its 2020+ Sharpe is only 1.257. The edge is concentrated in the pre-2020 low-liquidity era. This does not qualify as deployment quality under the sub-period requirement. It is documented here for completeness but is not the champion.

### Champion: btc82_30_eth83_70

**Configuration:** BTC Don8h(82,20) iv(vw30,tv0.2) 30% + ETH Don8h(83,33) iv(vw30,tv0.15) 70%

| Metric | Value |
|---|---|
| Sharpe at 30bps | 2.217 |
| Sharpe at 50bps | 2.099 |
| Cost retention (30 to 50bps) | 94.7% |
| DSR at n_trials=16,269 | 0.972 |
| MaxDD | -7.0% |
| 2020+ Sharpe | 1.943 |
| BTC-ETH return correlation | 0.466 |
| Validation | CONDITIONAL (0 hard fails, 3 structural soft fails) |

Weight sensitivity is flat: BTC 25-35% / ETH 65-75% are all within 0.005 DSR of the optimum. The portfolio is not at a narrow peak.

### Yearly Breakdown

| Year | Portfolio S | BTC82 S | ETH83 S | Max DD |
|---|---|---|---|---|
| 2019 | 1.088 | 1.649 | 0.412 | -7.0% |
| 2020 | 3.413 | 3.156 | 2.925 | -5.6% |
| 2021 | 2.254 | 0.503 | 2.708 | -4.4% |
| 2022 | -0.261 | -1.688 | +0.443 | -6.7% |
| 2023 | 1.478 | 2.275 | 0.790 | -6.6% |

The 2022 row is the most important. BTC individually has Sharpe -1.688 in the bear market. ETH83 produces Sharpe +0.443 in the same period. The 70% ETH weight limits the portfolio to -0.261 in the worst year. Whether this ETH bear resilience repeats is the central uncertainty for deployment (see Section 6).

### Regime Decomposition

| Leg | Bull S | Bear S | Sideways S |
|---|---|---|---|
| BTC82 8h | 2.826 | -0.979 | 2.355 |
| ETH83 8h | 2.606 | +0.465 | 2.071 |

ETH83 is the only leg with positive bear Sharpe. This is unusual and warrants scrutiny.

### Edge Attribution

| Leg | Signal Edge | % of Sharpe | Sizing Edge |
|---|---|---|---|
| BTC82 8h | 2.68 | 121% | +0.28 |
| ETH83 8h | 1.38 | 67% | -0.02 |

Both legs have genuine signal edge above the 15% threshold. BTC82 derives additional edge from sizing (IV amplification during high-momentum periods). ETH83 sizing is approximately neutral.

---

## 5. Reasons to Believe

**1. Statistical significance after multiple testing correction.**
DSR=0.972 at n_trials=16,269. The Deflated Sharpe Ratio accounts for the number of trials tested. At this DSR, the result would remain significant at substantially higher trial counts. This is not a marginal reading.

**2. Cross-asset diversification is real.**
BTC-ETH return correlation of 0.47 is confirmed in the data. The portfolio MaxDD of -7.0% versus -19.7% and -9.5% for the individual legs quantifies the benefit. The diversification is not assumed; it is measured.

**3. Cost robustness.**
94.7% Sharpe retention when costs increase from 30bps to 50bps. The strategy is not skating on thin margins. Breakeven is at 150+ bps, well above any realistic exchange cost.

**4. Sub-period consistency.**
2020+ Sharpe of 1.943 covers COVID crash (March 2020), 2021 bull run, 2022 bear market, and 2023 recovery. No single regime dominates the result.

**5. MaxDD well-contained.**
-7.0% across the full 2019-2023 period, including the 2022 bear. Individual strategies reach -12.5% to -19.7%. Portfolio construction meaningfully reduces drawdown.

**6. Signal edge confirmed.**
Both legs exceed the 15% signal edge threshold. The performance is not attributable to sizing accidents or random entries.

**7. Parameter robustness.**
BTC 82/20 and ETH 83/33 sit in broad, stable parameter neighborhoods across the 16,269-config sweep. The results are not narrow peaks that disappear with small parameter changes.

**8. Walk-forward retention for BTC82.**
BTC Don8h(82,20) individually produces IS Sharpe 2.220 and walk-forward Sharpe 1.708, a 77% retention rate. This provides partial validation that the signal is not purely in-sample.

---

## 6. Reasons for Skepticism

**1. Long bias in a structurally bullish IS period.**
The in-sample period (2019-2023) includes massive BTC appreciation. Trend-following is structurally long-biased and benefits from sustained uptrends. The post-halving, post-ETF market structure may produce shorter or weaker trends, which this strategy would handle worse.

**2. 2022 bear reveals BTC structural weakness.**
BTC82 Sharpe of -1.688 in 2022 is not a minor blip. The portfolio only survives 2022 because of ETH. If ETH bear resilience disappears in the next bear, the portfolio does not have a fallback.

**3. ETH 2022 resilience may not repeat.**
ETH generating Sharpe +0.443 in the 2022 bear market is notable and may be attributable to the Ethereum Merge narrative, which created sustained directional price action during the broader crypto selloff. That was a one-time event. Future bears may see BTC and ETH fall together with high correlation, eliminating the hedge.

**4. 16,269 configurations is a large search space.**
DSR=0.972 is meaningful but not a guarantee. The risk of data-mining is reduced, not eliminated. The DSR framework assumes the test statistics are approximately independent, which may not hold perfectly given the parameter sweep structure.

**5. All results are in-sample.**
The in-sample period is 2019-07-01 to 2023-12-31. No out-of-sample evaluation has been performed. This is the most important caveat. In-sample performance is necessary but not sufficient for deployment.

**6. Structural soft fails on walk-forward.**
All strategies fail walk_forward_multi at 90-day windows. The strategy has prolonged flat periods where it is in the market but not capturing trend. This reduces capital efficiency and creates extended periods of zero or negative return for investors.

**7. ETH DSR is individually weak.**
eth_don8h_83_33_iv has DSR=0.928, below the 0.95 threshold. The portfolio DSR of 0.972 exceeds the threshold, but the ETH leg is not independently validated to the same standard as the BTC legs.

**8. BTC edge may be weakening in recent years.**
BTC individual Sharpe degrades from approximately 2.2 (full period) to 1.2-1.4 (2020+). The 8h portfolio partially masks this degradation through ETH diversification, but the BTC trend signal appears to have weakened in the more recent and liquid market.

---

## 7. OOS Readiness Checklist

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | DSR > 0.95 at cumulative n_trials | YES | Portfolio DSR=0.972 at n=16,269 |
| 2 | No hard validation fails | YES | 0 hard fails, 3 structural soft fails |
| 3 | MaxDD < -30% | YES | MaxDD=-7.0% |
| 4 | Sharpe > 1.0 at 30bps | YES | S=2.217 |
| 5 | Sharpe > 0.7 at 50bps stress | YES | S=2.099 |
| 6 | 2020+ sub-period Sharpe > 0.7 | YES | S2020+=1.943 |
| 7 | Walk-forward retention > 60% | PARTIAL | BTC82 individual WF=77%; portfolio-level WF not run |
| 8 | Signal edge confirmed (>15% from signal) | YES | BTC82 121%, ETH83 67% |

**Result: 7/8 criteria met, 1 partial.**

Portfolio-level walk-forward validation has not been run. BTC82 was individually walk-forward validated at 77% retention during the exploration phase. The composite portfolio combining BTC82 (30%) and ETH83 (70%) has not been subjected to the same walk-forward test. This gap should be closed before OOS evaluation.

The strategy meets TIER 2 (Paper Trade) criteria (Sharpe >= 0.7, MC > 70%, beats buy-and-hold) and borderline TIER 1 (Sharpe >= 1.0, MC > 80%, MaxDD < 50%). Final tier classification requires OOS confirmation.

---

## 8. Next Steps

**Step 1: Run portfolio-level walk-forward validation.**
Apply the same walk-forward protocol used for BTC82 individual to the btc82_30_eth83_70 composite portfolio. This closes the one partial criterion and is the final in-sample validation step. This does not require AK approval and can proceed immediately.

**Step 2: Request AK approval for OOS evaluation.**
Once portfolio walk-forward is complete (and assuming WF retention > 60%), submit an OOS evaluation request. OOS is a one-shot test — if it fails, the strategy is discarded. Do not request OOS until the portfolio walk-forward is complete.

**Step 3: Do not build paper trading infrastructure yet.**
Standing directive: no paper trading until OOS is validated. This applies even if the portfolio walk-forward passes.

**Step 4: Monitor regime change indicators.**
The ETH bear resilience (positive 2022 Sharpe) is the portfolio's most critical feature and most uncertain assumption. Track BTC-ETH return correlation over time. If they converge toward 0.7+ during the next bear, the -7% MaxDD estimate is stale.

**Step 5: Consider BTC-only portfolio as a contingency.**
btc160_80_btc60_20_4h (S=2.334, DSR=1.000) is a fully documented alternative that does not depend on ETH bear resilience. Its weakness is 2020+ Sharpe of 1.257 and MaxDD of -12.8%. It is not deployment quality under current criteria, but documents the best BTC-only result if ETH diversification is not available.

---

## Appendix: Project Statistics

| Item | Value |
|---|---|
| Total configurations explored | 16,269 |
| Individual candidates validated | 8 |
| -- PASS | 3 (BTC) |
| -- CONDITIONAL | 3 (ETH IV) |
| -- FAIL | 2 (ETH flat) |
| Portfolio configurations tested | 55 |
| Null results documented | 4 (regime Donchian, XGBoost, Kelly sizing, ETH Bollinger) |
| Orchestrator sessions | 9 (3 BTC validation, 3 ETH validation, 3 portfolio) |
| wandb project | datadex_ai/sparky-ai |
| wandb tags | project_001, validation/portfolio |
| In-sample data range | 2019-07-01 to 2023-12-31 |
| Data source | BTC/ETH hourly candles from OKX |
| OOS evaluation | Not performed — pending AK approval |

---

*Generated 2026-02-20. All results are in-sample. OOS evaluation has not been performed. Do not treat in-sample results as validated deployment performance.*
