# P002 PHASE 4 VALIDATION CARD — S1 Champion

```
Config: INV_MVRV(pw=900) + SOPR(ss=3) + Netflow(14d SMA)
        Majority vote (2/3 agree), persistence=5 days
        FGI overlay: fear_threshold=25, adjustment=0.3, fear_only=True

Validation period: 2022-01-01 to 2023-12-01 (700 days)
Cost model: 15 bps maker (spot_maker)
Dev verification: PASS (IS Sharpe = 2.129276, exact match)
Boundary continuity: OK (position = 0.0 at both sides of 2021-12-31/2022-01-01)
```

## Performance

| Metric | Raw | Vol-Targeted (tv=0.10) |
|--------|-----|------------------------|
| Sharpe @15bps | -0.223 | 0.177 |
| Sharpe @30bps | -0.278 | — |
| MaxDD | -60.1% | -10.9% |
| Total Return | -35.7% | +2.5% |
| Sortino | -0.310 | — |
| Calmar | -0.343 | — |
| Win Rate | 34.5% | — |
| Trades (position changes) | 60 | — |
| Round-trip trades | 11 | — |
| Time in position | 74.1% | — |
| Mean position size | 0.795 | 0.175 |

## Walk-Forward (4 folds)

| Fold | Period | Sharpe | Return | MaxDD | Trades | TiP |
|------|--------|--------|--------|-------|--------|-----|
| 0 | 2022-01-02 to 2022-06-24 | **-1.450** | -41.5% | -51.2% | 30 | 66% |
| 1 | 2022-06-25 to 2022-12-15 | -0.559 | -19.3% | -38.0% | 16 | 74% |
| 2 | 2022-12-16 to 2023-06-07 | 0.627 | +8.7% | -15.6% | 8 | 83% |
| 3 | 2023-06-08 to 2023-12-01 | 1.579 | +25.2% | -22.3% | 6 | 73% |

- Aggregate validation Sharpe: -0.223
- WF Retention (val/IS): -10.5%
- All folds positive: **NO** (2 of 4 negative)
- Any fold < -1.0: **YES** (Fold 0: -1.45)

## Bear Market Test

| Period | BTC Move | Strategy Pos | Strategy Return | B&H Return | Correct? |
|--------|----------|-------------|-----------------|------------|----------|
| 2022-01 to 2022-03 (grind down) | -4.6% | Long (89%) | -9.8% | -4.6% | NO |
| 2022-04 to 2022-06 (LUNA/3AC) | **-56.9%** | **Long (67%)** | **-40.3%** | -56.2% | **NO** |
| 2022-07 to 2022-09 (sideways) | +0.8% | Long (62%) | -3.5% | -2.6% | ~OK |
| 2022-10 to 2022-11 (FTX) | -11.1% | **Long (82%)** | **-10.7%** | -11.6% | **NO** |
| 2022-12 to 2023-03 (recovery) | +67.6% | Long (76%) | +19.4% | +65.8% | YES |
| 2023-04 to 2023-06 (consolidation) | +7.1% | Long (100%) | +7.1% | +7.1% | YES |
| 2023-07 to 2023-11 (pre-ETF) | +23.3% | Long (69%) | +5.5% | +23.8% | YES |

**The strategy stayed long through both major 2022 crashes (LUNA/3AC and FTX).** It did not go flat during crashes as designed. The on-chain regime signal failed to detect the bear market.

## Signal Diagnostics

| Signal | % Bullish (val period) | Assessment |
|--------|----------------------|------------|
| MVRV (inverted) | **91.1%** | Almost permanently bullish |
| SOPR (SMA>1.0) | 41.9% | Correctly bearish during bear |
| Netflow (14d SMA<0) | 50.9% | Roughly 50/50 |

**Root cause:** MVRV was below its 900-day rolling median for 91% of the validation period. This is because the 900-day lookback window included 2021 bubble prices, making 2022 prices look "cheap" (below median). The inverted signal interpreted this as bullish.

With MVRV permanently bullish, only ONE of SOPR/Netflow needs to be bullish for the majority vote to trigger — a low bar. The composite was 85% split-vote (2-1) and 0% all-bearish.

MVRV crossovers during validation: 10 total, but 8 occurred in Q1 2022 (choppy near the boundary). After April 6, 2022, MVRV stayed bullish until October 19, 2023 — an uninterrupted 18-month bullish streak through the entire bear market.

FGI: Mean = 38.7, with 210 fear days (FGI <= 25) out of 700. The FGI overlay *boosted* positions during fear (from 1.0 to 1.3), compounding losses during crashes. FGI was designed to increase exposure during fear — but in a bear market, fear is warranted, not contrarian.

## Benchmarks

| Strategy | Sharpe | Return | MaxDD |
|----------|--------|--------|-------|
| Buy-and-hold BTC | 0.080 | -19.0% | -66.7% |
| S1 champion (raw) | **-0.223** | **-35.7%** | **-60.1%** |
| S1 champion (vol-targeted) | 0.177 | +2.5% | -10.9% |
| Risk-free (flat) | 0.0 | 0% | 0% |

**The raw strategy underperforms buy-and-hold BTC by 16.7 percentage points.** It also underperforms staying flat (risk-free). The strategy is worse than doing nothing.

The vol-targeted version performs reasonably (Sharpe 0.18, beats B&H) because the EWMA vol scaler reduces position sizes dramatically during high-vol periods (mean position 0.175 vs 0.795 raw). But this is vol targeting saving a broken signal, not the signal working.

## Trade Profile (Validation)

| Metric | Value |
|--------|-------|
| Round-trip trades | 11 |
| Win rate | 45.5% (5/11) |
| Profit factor | 0.67 |
| Avg duration | 47.1 days |
| Expectancy | -2.9% per trade |
| Max consecutive losses | 2 |
| Max gap | 39 days |

Largest loss: Trade #2 (2022-02-06 to 2022-05-14, 97 days long, -31.4%). The strategy entered at $43.8K and held through the LUNA crash to $30K.

## Phase 4 Gates

| Gate | Value | Threshold | Verdict |
|------|-------|-----------|---------|
| Val Sharpe >= 0.3 | -0.223 | >= 0.3 | **FAIL** |
| WF Retention >= 50% | -10.5% | >= 50% | **FAIL** |
| MaxDD < 25% | 60.1% | < 25% | **FAIL** |
| No fold < -1.0 | Fold 0 = -1.45 | No | **FAIL** |

**OVERALL: 0/4 PASS, 0 MARGINAL, 4 FAIL**

## Outcome: D — Failure

The S1 champion loses money on unseen bear market data. The on-chain regime signal does not work out-of-sample.

### Root Cause Analysis

The failure has a clear mechanism: the 900-day MVRV rolling percentile is structurally unable to detect bear markets that follow multi-year bull markets. The lookback window (900 days ~ 2.5 years) includes the 2020-2021 bull run, so 2022 prices look "cheap" relative to the bubble. The inverted signal interprets "cheap" as "buy" — exactly wrong during a secular decline.

This is not parameter overfitting in the traditional sense (the parameter plateau was robust at 95.5%). It's a **mechanism failure**: the economic hypothesis (MVRV below historical median = undervalued = buy) breaks down when the median is inflated by bubble prices. The signal cannot distinguish between "cheap because undervalued" and "cheap because falling."

SOPR showed the correct regime (bearish 58% of the time), but the majority-vote gate only requires 2/3 agreement, so MVRV + either other signal bullish = composite bullish.

### What Survived

The vol-targeted version (Sharpe 0.18) slightly outperforms B&H. This is entirely attributable to the vol scaler reducing exposure during high-volatility bear periods — it's a vol-targeting effect, not a signal effect. The underlying signal is flat-to-harmful.

The 2023 recovery period shows genuine performance (Fold 2: S=0.63, Fold 3: S=1.58), suggesting the signal works in trending-up environments but fails in bear markets.

### Recommendation

**Kill S1 as a deployment candidate.** The strategy fails all 4 Phase 4 gates, underperforms buy-and-hold, and has a clear mechanism failure. The 2019-2021 IS performance was real within that regime (PBO=0.000 confirms no overfitting within IS), but the signal does not generalize to bear markets.

**P002 disposition options for AK:**

**Option 1: Kill P002 entirely.** The on-chain regime approach has been falsified — the core MVRV signal is regime-dependent and cannot be fixed without introducing new parameters (which would require new screening). Focus on P001 Donchian portfolio, which uses price-based signals that are structurally regime-adaptive.

**Option 2: Test backup candidates.** The 3 backup S1 candidates (INV_VOL, INV_FINE variants) share the same MVRV core signal and will likely fail for the same structural reason. Testing them would confirm but is likely futile.

**Option 3: Investigate adaptive MVRV lookback.** The failure is specifically the 900-day fixed window. An adaptive window that contracts during high-vol regimes or uses percentile ranks instead of medians might work. But this is a new research direction, not a P002 fix — it would require new pre-registration and screening.

**Option 4: Pivot to SOPR-only or Netflow-only signals.** SOPR correctly identified the 2022 bear (bearish 58% of time). But SOPR alone had negative marginal IC in IS and was the weakest of the 3 signals. Netflow was 50/50 — no edge. Neither is likely a standalone solution.
