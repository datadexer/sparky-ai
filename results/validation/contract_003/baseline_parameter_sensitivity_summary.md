# Parameter Sensitivity: Multi-TF Donchian Baseline (Corrected Backtest)

**Date**: 2026-02-16
**Backtest**: CORRECTED (signals.shift(1) — no look-ahead bias)
**Transaction costs**: 26 bps per trade (TransactionCostModel.for_btc())
**Data**: BTC daily, in-sample only (before 2024-06-01)
**Validation**: Yearly folds 2019-2023

---

## Single-Timeframe Results

| Entry | Exit | Mean Sharpe | 2019 | 2020 | 2021 | 2022 | 2023 |
|-------|------|------------|------|------|------|------|------|
| 10 | 5 | 0.478 | 0.92 | 1.89 | 0.51 | -2.15 | 1.23 |
| 15 | 7 | 0.784 | 1.58 | 1.94 | 0.69 | -1.86 | 1.56 |
| 20 | 10 | 0.903 | 1.37 | 2.16 | 0.50 | -1.24 | 1.72 |
| 25 | 12 | 0.998 | 1.21 | 2.18 | 0.85 | -1.03 | 1.77 |
| 30 | 15 | 1.015 | 1.64 | 2.72 | 0.36 | -1.38 | 1.73 |
| **40** | **20** | **1.243** | **1.80** | **2.68** | **0.84** | **-0.82** | **1.71** |
| 50 | 25 | 0.718 | 1.54 | 1.85 | 0.56 | -2.02 | 1.66 |
| 60 | 30 | 0.772 | 1.47 | 1.50 | 1.09 | -1.69 | 1.49 |
| 80 | 40 | 0.643 | 1.27 | 1.42 | 1.00 | -1.23 | 0.76 |

**Best single-TF**: Entry=40, Sharpe 1.243

---

## Multi-Timeframe (3-Strategy Majority Vote) Results

| Configuration | Mean Sharpe | Status |
|--------------|------------|--------|
| **[20, 30, 40]** | **1.069** | Best multi-TF |
| **[20, 40, 60]** | **1.062** | Current baseline |
| [15, 40, 80] | 0.923 | — |
| [10, 20, 40] | 0.893 | — |
| [10, 30, 60] | 0.833 | — |
| [15, 30, 60] | 0.775 | — |
| [30, 50, 80] | 0.725 | — |
| [25, 50, 80] | 0.645 | — |
| [20, 50, 80] | 0.640 | — |

---

## Robustness Assessment

- **Single-TF**: 7/9 configs (78%) have Sharpe > 0.7
- **Multi-TF**: 7/9 configs (78%) have Sharpe > 0.7
- **Single-TF range**: [0.478, 1.243]
- **Multi-TF range**: [0.640, 1.069]

**VERDICT: ROBUST** — majority of parameter combinations produce acceptable Sharpe.

---

## Key Observations

1. **Baseline [20,40,60] is near top but NOT at peak**: Sharpe 1.062 is 2nd best multi-TF
2. **Entry=40 single-TF is actually better**: Sharpe 1.243 beats all multi-TF configs
3. **2022 bear market destroys all configs**: All have negative Sharpe in 2022
4. **Sweet spot: Entry 25-40**: Best single-TF performance in this range
5. **Short periods (10-15) are worse**: Not enough signal, too many false breakouts
6. **Long periods (60-80) are also worse**: Too slow to react, miss opportunities

## Strategic Implications

1. The baseline is in a PLATEAU (Sharpe 0.9-1.2 for entry periods 20-40)
2. Entry=40/Exit=20 single-TF may be worth investigating as simpler alternative
3. The multi-TF ensemble adds robustness but NOT performance
4. Bear market (2022) is the dominant risk — ALL configs lose money
5. Consider bear market hedging as higher priority than parameter tuning
