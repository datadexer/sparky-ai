# RESEARCH LOG — Sparky AI

Running log of all research findings. Newest entries at the top.

---

## regime_donchian_v3 — NEGATIVE RESULT — 2026-02-18

**DIRECTIVE**: regime_donchian_v3
**STATUS**: NEGATIVE — DSR>0.95 not achieved after 1552 configs
**SESSIONS**: 001 (1507 configs) + 002 (45 configs)
**COSTS**: 30 bps standard, 50 bps stress test (correct two-tier model)
**ARTIFACTS**:
- `results/regime_donchian/session_001_v3_summary.json`
- `results/regime_donchian/session_002_v3_summary.json`
- `results/regime_donchian/session_001_v3_analysis.md`
- `results/regime_donchian/session_001_v3_final_report.md`

### Session 001 — 1507 configs across 5 rounds

| Round | Strategy | N | Best Sharpe | Best DSR@N=1507 |
|-------|----------|---|-------------|-----------------|
| 1 | inverse_vol_sizing | 425 | **1.777** | **0.730** |
| 2 | regime_param_switching | 48 | 1.690 | ~0.65 |
| 3 | vol_momentum_4state | 720 | 1.479 | ~0.40 |
| 4 | adaptive_lookback | 96 | 1.687 | ~0.55 |
| 5 | refined_grid | 218 | 1.518 | ~0.42 |

**Best config**: `inverse_vol_sizing(ep=30, xp=20, vol_window=45, target_vol=0.4)`
- Sharpe=1.777 @ 30 bps, Sharpe=1.730 @ 50 bps — cost-robust
- DSR=0.730 @ N=1507, MaxDD=-35.8% vs baseline -42.9%
- Outperforms baseline in 4/5 years (2019,2020,2021,2023; worse in 2022)
- TIER 3 result

### Session 002 — 45 configs across 4 approaches

| Approach | N | Best Sharpe | Best DSR@N=1552 |
|----------|---|-------------|-----------------|
| Long/Short Donchian | 19 | 1.291 | 0.295 |
| Adaptive Exit Speed | 15 | 1.268 | 0.281 |
| HMM Soft Probability | 11 | 1.209 | 0.239 |
| Deep Validation (session 001 best) | — | 1.777 (confirmed) | 0.727 |

### Key Findings

1. **Statistical bar is extremely high**: At N=1552 cumulative trials, DSR>0.95 requires
   Sharpe ~2.3+ on 1797 daily obs. Best achieved: 1.777 (77% of threshold).

2. **2022 is structural**: BTC lost ~65% in 2022. Any long-only strategy suffers.
   No regime detection approach can predict bear market start ex-ante.

3. **High-vol ≠ bad returns for Donchian**: 2020 (HIGH vol, +3.2 Sharpe) and 2022
   (HIGH vol, -1.0 Sharpe) are both high-vol but opposite outcomes. Vol regime
   cannot distinguish "2020 breakout bull" from "2022 crash bear."

4. **Best approach**: inverse_vol_sizing with slow window (vw=45) gives +12.5% Sharpe
   improvement but does NOT achieve statistical significance.

5. **Long/Short Donchian**: Improves 2022 (-0.51 vs -1.02) but kills bull years
   (total Sharpe 1.291 vs 1.580 baseline). Not a net improvement.

### Baseline

| Period | Baseline Sharpe | Best Config Sharpe | Edge |
|--------|-----------------|--------------------|------|
| 2019 | 2.160 | 2.499 | +0.339 |
| 2020 | 3.202 | 3.287 | +0.085 |
| 2021 | 1.016 | 1.011 | -0.005 |
| 2022 | -1.021 | -1.204 | -0.183 |
| 2023 | 2.006 | 2.331 | +0.325 |
| **Full IS** | **1.580** | **1.777** | **+0.197** |

### Recommendations

- **Option 1 (recommended)**: Hourly data (47,500 candles) → more statistical power,
  lower Sharpe threshold for DSR>0.95
- **Option 2**: Accept documented negative result, pivot to different strategy family
- **DO NOT** continue daily-data regime sweeps — 3922 cumulative configs across v2+v3
  directives, still no DSR>0.95 at standard costs

---

## Prior directives (directive_002) — archived in old log

See `roadmap/02_RESEARCH_LOG__OLD_DONT_USE.md` for sessions 001-003 of directive_002
(10 bps, then 50 bps costs, 2370 configs total — also negative result).

---

## Contract 005 Audit Fixes — 2026-02-18

PRs #35 and #36 merged. Changes:
- Sortino formula corrected: `sqrt(mean(min(r,0)^2))`
- `periods_per_year` default: 252 → 365 everywhere
- `statistics.py`: added `periods_per_year` param to `sharpe_confidence_interval`
- `selection.py`: KFold → TimeSeriesSplit(gap=embargo)
- `sweep_two_stage.py` + `smart_hyperparam_sweep.py`: added DSR, guardrails, load()
- `compute_all_metrics`: "sharpe" returns annualized; "sharpe_per_period" added

---

## XGBoost on hourly features — 2026-02-18

52.9% accuracy on 23 hourly features — does NOT beat baseline.
Feature expansion 58→88 features: best sweep Sharpe 1.19 but NOT statistically significant.

---

## Baseline (validated) — 2026-02-16

- Multi-TF Donchian (walk-forward, 30 bps): **Sharpe 1.062**
- Single-TF Donchian(40/20) (full IS, 30 bps): **Sharpe 1.580**
- Look-ahead bias bug fixed (PR #12): all prior Sharpe claims were inflated 43-256%
