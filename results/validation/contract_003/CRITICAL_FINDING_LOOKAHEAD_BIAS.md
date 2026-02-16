# CRITICAL FINDING: Look-Ahead Bias in Backtest Framework

**Date**: 2026-02-16
**Discovered by**: Oversight Opus during CONTRACT #003 validation
**Severity**: CRITICAL — invalidates ALL prior Sharpe ratio claims

---

## The Bug

In `scripts/validate_regime_approaches.py`, line 82:

```python
strategy_returns = period_signals * price_returns
```

The signal at time T is computed using `close[T]` (the current day's close price),
but the strategy earns `return[T] = close[T]/close[T-1] - 1` — which includes the
very price move that triggered the signal.

In the Donchian strategy (`simple_baselines.py`), line 106:
```python
if current_price >= upper_channel.iloc[i - 1]:
    signals.iloc[i] = 1
```

This means: we know `close[T]` broke above the 20-day high → we "buy" at `close[T-1]`
and earn `close[T]/close[T-1] - 1`. But at `close[T-1]` we didn't know `close[T]`
would break out — we only learn this at end of day T.

## The Fix

```python
strategy_returns = period_signals.shift(1) * price_returns
```

Signal from day T should apply to day T+1's return.

## Impact

All Sharpe ratios reported in Phase 2 and Phase 3 are inflated:

| Approach | Biased Sharpe | Correct Sharpe | Inflation |
|----------|---------------|----------------|-----------|
| Multi-TF Donchian (baseline) | 1.878 | 1.062 | +77% |
| Regime Weighted Ensemble | **2.656** | **1.017** | **+161%** |
| HMM 2-State | 2.641 | 0.742 | +256% |
| Regime Multi-TF (aggressive) | ~1.720 | ~0.8 | est +115% |

## Corrected Results

After fixing the look-ahead bias:
- **Baseline (Multi-TF Donchian)**: Sharpe 1.062 (not 1.878)
- **Regime Ensemble**: Sharpe 1.017 (not 2.656)
- **HMM 2-State**: Sharpe 0.742 (not 2.641)

The regime approaches provide **NO improvement** over the properly computed baseline.
The Regime Ensemble (1.017) is actually **worse** than the baseline (1.062).

## Yearly Breakdown (Regime Ensemble)

| Year | Biased | Correct | Note |
|------|--------|---------|------|
| 2019 | 3.024 | 1.252 | Bull market, bias extreme |
| 2020 | 3.884 | 2.084 | Strong uptrend, less bias |
| 2021 | 2.646 | 1.121 | Significant inflation |
| 2022 | 0.172 | -0.824 | Bear market REVERSES sign |
| 2023 | 3.552 | 1.452 | Recovery, high inflation |

2022 is particularly telling: the biased backtest shows a small positive Sharpe (0.172),
but the correct version shows the strategy LOST money (-0.824). The look-ahead bias
allowed the strategy to "avoid" losses it would have actually taken.

## Root Cause

The backtest framework used a common but incorrect pattern:
`signal[T] * return[T]` instead of `signal[T-1] * return[T]`.

This is especially damaging for breakout strategies because the entry signal
fires precisely when the breakout move happens — capturing the very move
that the strategy couldn't have predicted.

## Action Required

1. Fix `backtest_strategy()` to use `signals.shift(1) * price_returns`
2. Re-run ALL prior validation with the fix
3. Update ALL Sharpe claims in research logs
4. The "breakthrough" Sharpe 2.656 claim is DEBUNKED
5. The validated baseline drops from 0.772 to ~0.5 (needs re-computation with original baseline code)

## Implications for Research

- All Phase 2 regime detection research was built on biased backtests
- The correct question is: does the corrected regime approach (1.017) beat the corrected baseline (1.062)?
- Answer: NO — the regime overlay provides no improvement
- The HMM approach (0.742) is WORSE than baseline (1.062)
- This is an honest negative result

## Note on the WalkForwardBacktester

The main `WalkForwardBacktester` in `sparky/backtest/engine.py` may also have this bug.
It needs to be audited separately. The ad-hoc backtests in `validate_regime_approaches.py`
definitely have this issue.
