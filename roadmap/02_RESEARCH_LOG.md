# RESEARCH LOG — Sparky AI

Running log of all research findings. Newest entries at the top.

---

## meta_labeling_donchian_20260218 — Session 1 — 2026-02-18

**DIRECTIVE**: meta_labeling_donchian_20260218
**STATUS**: SUCCESS gate hit — Sharpe 1.787, DSR 0.998 at N=123 (independently verified)
**DATA**: ohlcv_hourly_max_coverage (95,689 hourly bars, 2013-2023), resampled to 4h (23,923 bars)
**COSTS**: 30 bps standard, 50 bps stress test
**ARTIFACTS**: results/meta_labeling_donchian_20260218/

### Attribution (ordered by impact)

| Config | Sharpe | Delta vs prev | DSR@123 | Source |
|--------|--------|---------------|---------|--------|
| 4h Donchian(30,20) WITH binary HMM regime filter (no meta) | 0.593 | baseline | 0.402 | primary_4h_baselines.json |
| 4h Donchian(30,20) + meta, WITH regime filter (R1-R4 best) | 0.786 | +0.193 | 0.672 | round1-4 |
| **4h Donchian(30,20) NO regime filter (no meta)** | **1.682** | **+1.089** | **0.997** | primary_4h_noregimedfilter.json |
| 4h meta-labeled, no regime, tight barriers (R5, tp=1.5/sl=1.0/vert=12) | 1.596 | **-0.086** | 0.996 | round5_results.json |
| **4h meta-labeled, no regime, wide barriers (R9, tp=3.0/sl=1.5/vert=30)** | **1.787** | **+0.105** | **0.998** | session1_comprehensive_final.json |

**Largest gain**: removing binary regime filter (+1.089 Sharpe). Meta-labeling itself: barrier-dependent.
Tight barriers hurt (-0.086), wide barriers help (+0.105). Only 13/117 configs (11.1%) beat raw primary.

### Best config

- Donchian(30, 20) on 4h BTC, tp=3.0×ATR, sl=1.5×ATR, vert=30 bars
- LogReg (C=0.1, balanced): trend_r2, regime_proba_3s (3-state HMM), adx_proxy
- Threshold 0.5, Accuracy 53.3% OOF purged CV, 172 trades / 271 signals (36% filtered)
- Stress (50bps): Sharpe 1.641, DSR 0.995

### Independent verification

- Reproduced best config from scratch: Sharpe=1.7865, DSR=0.9984 at n_trials=123
- Raw 4h primary (no meta): Sharpe=1.6824, DSR=0.9970 at n_trials=123
- Buy-and-hold BTC (same 4h data, 2013-2023): Sharpe=1.2882
- DSR remains >0.95 even at n_trials=500

### MaxDD

- Best config: -0.580 (fails <25% deployment criterion)
- Range across 123 configs: -0.486 to -0.637
- Needs Layer 4 (sizing) — 0.25x Kelly → ~-0.29 estimated

### Caveats

- 2013-2023 is overwhelmingly favorable for long BTC — any long-biased strategy looks good
- B&H Sharpe 1.288 on same data confirms the tailwind
- The DSR at n_trials=123 (0.998) is the primary validity measure, not raw Sharpe
- Daily Donchian reproduced at 1.330, not 1.777. The 1.777 was inverse_vol_sizing on
  daily data — a different strategy entirely, with DSR=0.730 (NOT statistically significant)
- Correct baseline for 4h meta-labeling comparison: raw 4h primary = 1.682

### Key findings

1. Binary HMM regime filter destroys 4h performance (0.593 → 1.682 by removing it)
2. Meta-labeling is barrier-param-sensitive: tight barriers hurt, wide barriers help
3. 3-state HMM probability as continuous feature > binary 2-state filter
4. 3 features beat 5-8 features (N=271 signals, overfitting risk)
5. LogReg dominates XGBoost at this sample size
6. Calibration + Kelly sizing does NOT improve at this N

---

## meta_labeling_donchian_20260218 — Session 2 — 2026-02-18

**DIRECTIVE**: meta_labeling_donchian_20260218
**STATUS**: SUCCESS — Sharpe 1.981, DSR 0.999 at N=359 (cumulative)
**DATA**: ohlcv_hourly_max_coverage (95,689 hourly bars, 2013-2023), resampled to 4h (23,923 bars)
**COSTS**: 30 bps standard, 50 bps stress test
**ARTIFACTS**: results/meta_labeling_donchian_20260218/
**BUDGET**: $5.68 total (2 sessions), 32 min wall clock

### Round progression (236 configs, R1-R5A)

| Round | N | Best Sharpe | Finding |
|-------|---|-------------|---------|
| R1 | 30 | 1.787 | Shorter Donchian params produce more signals but lower quality |
| R2A-2E | 100 | 1.800 | 4d features, thr=0.5, C=0.1 optimal; XGBoost/calibration hurt |
| R3A-3C | 67 | 1.891 | **Donchian(30,25) breakthrough** — wider exit retains higher-quality signals |
| R4A-4C | 66 | 1.931 | 4j=[trend_r2, regime, adx, dist_sma_60] further improvement |
| R5A | 36 | **1.981** | 4k=[trend_r2, regime, dist_sma_60, vol_accel] = **session best** |

### Best config

- Donchian(30, 25) on 4h BTC, tp=2.0×ATR, sl=1.5×ATR, vert=20 bars
- LogReg (C=0.1, balanced): trend_r2, regime_proba_3s, dist_sma_60, vol_accel
- Threshold 0.5, Accuracy 56.1% OOF purged CV, 156 trades / 248 signals (37% filtered)
- Sharpe@30bps: **1.981**, DSR: **0.999** at N=359
- Sharpe@50bps: **1.857**, DSR: 0.998 (cost-robust)
- MaxDD: **-0.490** (improved from S1's -0.580, still fails <-0.25 deployment criterion)

### Attribution (S1→S2 gain decomposition)

| Factor | Delta Sharpe | % of total |
|--------|-------------|------------|
| Primary signal (30,20)→(30,25), same barriers | -0.044 | -23% |
| Barrier change tp=3.0→2.0, sl=1.5→1.5, vb=30→20 | **+0.149** | **76%** |
| Feature change 3c→4k (add dist_sma_60, vol_accel) | **+0.090** | **46%** |
| **Total S1→S2** | **+0.195** | **100%** |

Tighter barriers (+0.149) are the largest contributor — shorter holding period reduces drawdown
exposure. New features contribute meaningfully (+0.090): dist_sma_60 captures mean-reversion
risk, vol_accel confirms breakout quality.

### Comparisons

| Config | Sharpe@30 | DSR@359 | MaxDD |
|--------|-----------|---------|-------|
| Buy-and-hold BTC (4h, 2013-2023) | 1.288 | — | -0.852 |
| Raw Donchian(30,20) 4h, no meta | 1.682 | 0.997 | -0.590 |
| Raw Donchian(30,25) 4h, no meta | 1.691 | — | -0.643 |
| S1 best: meta (30,20) 3c tp=3.0 vb=30 | 1.787 | 0.998 | -0.580 |
| **S2 best: meta (30,25) 4k tp=2.0 vb=20** | **1.981** | **0.999** | **-0.490** |

### Sub-period validation

| Period | Sharpe@30 | Sharpe@50 | MaxDD | Ann Return | Trades | Win Rate | B&H Sharpe |
|--------|-----------|-----------|-------|------------|--------|----------|------------|
| Full 2013-2023 | 1.981 | 1.857 | -0.490 | 133.4% | 156 | 0.527 | 1.288 |
| 2017-2023 | 1.609 | 1.479 | -0.436 | 78.1% | 86 | 0.524 | 1.085 |
| 2020-2023 | 1.444 | 1.297 | -0.436 | 57.7% | 49 | 0.520 | 0.978 |

Strategy beats B&H in all sub-periods. Sharpe declines in shorter windows (expected: fewer
bars = noisier estimate, and 2013-2016 was strongly favorable). 2020-2023 includes full
bull+bear cycle and still holds 1.444 @30bps — no flag.

### Caveats (carried forward from S1)

- 2013-2023 is overwhelmingly favorable for long BTC — any long-biased strategy looks good
- B&H Sharpe 1.288 on same data confirms the tailwind
- DSR at N=359 (0.999) is the primary validity measure, not raw Sharpe
- MaxDD -0.490 fails deployment criterion (<-0.25) — needs Layer 4 sizing

### Next step

**Layer 4: position sizing** — fractional Kelly + inverse vol sizing to get MaxDD < -0.25
while preserving Sharpe > 1.5. See `directives/layer4_sizing_donchian_20260218.yaml`.

### Program totals

- 2 sessions, 359 configs tested, $5.68 total cost, 32 min wall clock
- Best: Sharpe 1.981 / DSR 0.999 / MaxDD -0.490 @ 30 bps
- Stress: Sharpe 1.857 / DSR 0.998 @ 50 bps

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
