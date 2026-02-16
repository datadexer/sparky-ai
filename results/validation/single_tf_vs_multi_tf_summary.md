# Single-TF (40/20) vs Multi-TF Baseline: Investigation Summary

**Date**: 2026-02-16
**Backtest**: CORRECTED (signals.shift(1) — no look-ahead bias)
**Transaction costs**: 26 bps per trade
**Data**: BTC daily, in-sample only (2019-2023)

---

## Research Question

Parameter sensitivity analysis showed Single-TF Donchian(40/20) achieves Sharpe 1.243 vs Multi-TF (20/40/60) at 1.062 — a 17% improvement. Is this real and should we switch?

---

## Key Results

### Yearly Walk-Forward Sharpe

| Year | Single-TF (40/20) | Multi-TF (20/40/60) | Multi-TF (20/30/40) |
|------|-------------------|---------------------|---------------------|
| 2019 | **1.801** | 1.487 | 1.652 |
| 2020 | **2.684** | 2.684 | 2.720 |
| 2021 | 0.840 | **1.099** | 0.490 |
| 2022 | **-0.824** | -1.539 | -1.246 |
| 2023 | **1.712** | 1.576 | 1.730 |
| **Mean** | **1.243** | 1.062 | 1.069 |
| **Std** | **1.327** | 1.569 | 1.516 |

### Bootstrap 95% CI

| Strategy | Sharpe | 95% CI |
|----------|--------|--------|
| **Single-TF (40/20)** | **1.337** | **[0.622, 2.043]** |
| Multi-TF (20/40/60) | 1.301 | [0.587, 2.011] |
| Multi-TF (20/30/40) | 1.250 | [0.526, 1.970] |

**CIs heavily overlap** — no statistically significant difference.

### Paired Bootstrap (Head-to-Head)

| Comparison | P(A > B) |
|------------|----------|
| P(Single-TF > Multi-TF 20/40/60) | **0.624** |
| P(Single-TF > Multi-TF 20/30/40) | 0.624 |

**62.4% probability Single-TF wins** — above 50% but well below 75% significance threshold.

### Trade Analysis (2019-2023)

| Metric | Single-TF | Multi-TF (20/40/60) | Multi-TF (20/30/40) |
|--------|-----------|---------------------|---------------------|
| Trades | **15** | 19 | 21 |
| Avg holding | **50 days** | 39 days | 34 days |
| Win rate | **66.7%** | 52.6% | 47.6% |
| Avg return/trade | **28.1%** | 21.9% | 16.3% |

**Single-TF trades less, holds longer, wins more often, and earns more per trade.**

### Signal Overlap

- **96.5% agreement** between Single-TF and Multi-TF
- Only 51 days where Single-TF is LONG and Multi-TF is FLAT (2.8%)
- Only 13 days where Multi-TF is LONG and Single-TF is FLAT (0.7%)

### 2022 Bear Market (Critical Test)

| Strategy | 2022 Sharpe |
|----------|-------------|
| **Single-TF (40/20)** | **-0.824** |
| Multi-TF (20/40/60) | -1.539 |
| Multi-TF (20/30/40) | -1.246 |

**Single-TF loses MUCH LESS in the bear market.** This is the key advantage — the longer lookback (40 days) avoids whipsaws that plague shorter periods.

---

## Why Single-TF is Better

1. **Fewer false breakouts**: 40-day window requires stronger confirmation than 20-day
2. **Fewer trades**: 15 vs 19 (21% fewer), reducing transaction costs
3. **Better bear market protection**: -0.824 vs -1.539 in 2022 (47% less loss)
4. **Higher win rate**: 66.7% vs 52.6% — each trade is more likely to profit
5. **The multi-TF 20-day signal adds noise**: Short-period breakouts trigger premature entries

## Why the Difference is NOT Statistically Significant

1. **Only 5 years of data**: Very small sample for strategy comparison
2. **Overlapping CIs**: Bootstrap 95% CIs overlap substantially
3. **62.4% paired probability**: Better than coin flip but not decisive
4. **96.5% signal overlap**: Strategies are nearly identical in practice

---

## VERDICT

**MODERATE IMPROVEMENT, NOT CONCLUSIVE**

Single-TF (40/20) shows:
- +17% higher mean Sharpe (1.243 vs 1.062)
- +47% better bear market resilience
- 21% fewer trades
- 14pp higher win rate

But statistical significance is weak (p=0.376 against null).

## RECOMMENDATION

**Deploy Single-TF (40/20) as the primary strategy for paper trading.**

Rationale:
1. Higher point estimate (1.243 vs 1.062) — even if not significant, it's the best guess
2. Simpler (1 strategy vs 3 combined) — less complexity, fewer parameters
3. Better bear market protection — the most important practical characteristic
4. Lower transaction costs — fewer trades
5. Conservative choice: if wrong, we lose ~3% of signal overlap (3.5% of days)

The multi-TF ensemble adds complexity but NOT performance. Occam's Razor favors the simpler model.

---

## Full Period Performance (2019-2023)

| Strategy | Sharpe | Return | Trades |
|----------|--------|--------|--------|
| **Single-TF (40/20)** | **1.341** | **2087%** | **31** |
| Multi-TF (20/40/60) | 1.307 | 1892% | 39 |
| Multi-TF (20/30/40) | 1.256 | 1502% | 43 |
