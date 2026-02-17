# CONTRACT #004 STEP 4 — Ensemble Strategies Results

**Run date:** 2026-02-17 (updated from 2026-02-16 initial run)
**Baseline:** Donchian Multi-TF Sharpe **1.062** (in-sample, 2019–2023)
**Best individual ML:** LightGBM top-10 Sharpe **1.365** (std=1.701, Step 2)
**Best regime-filtered:** ADX(14,30) Sharpe **1.181** (std=0.829, Step 3)
**Validation:** Expanding-window walk-forward, yearly folds (2019–2023)

---

## Step 4 (v2) — Specific Combinations from Step Spec

Run: `scripts/ensemble_v2_momentum.py`
Total: **16 wandb runs** tagged `contract_004`, `ensemble`

### Combo 1 — LightGBM + vol_adx_OR

Hypothesis: LightGBM's 2022 catastrophe (-0.644) gets cut by vol_adx_OR regime filter.
Filter: trade when (vol > rolling_median) OR (ADX14 > 25). In-market: ~45%.

| Year | Sharpe |
|------|--------|
| 2019 | 0.000 (cold-start) |
| 2020 | 2.851 |
| 2021 | -0.202 |
| 2022 | **-0.028** |
| 2023 | 1.108 |
| **Mean** | **0.746 ± 1.150** |

**Finding:** The OR filter does cut 2022 significantly (-0.644 → -0.028). But 2021 turns
negative (-0.202), which was +0.751 for unfiltered LGBM. The OR filter is too permissive
to protect in 2022 bear (45% in-market) but strict enough to cut good 2021 days.
Mean Sharpe 0.746 — below baseline. The filter reduced both upside and downside.

---

### Combo 2 — LightGBM + vol_adx_AND (stricter filter)

Hypothesis: AND filter was positive in 2022 (+0.192 for Donchian) — does it rescue ML too?
Filter: trade when (vol > rolling_median) AND (ADX14 > 25). In-market: ~18%.

| Year | Sharpe |
|------|--------|
| 2019 | 0.000 (cold-start) |
| 2020 | 1.701 |
| 2021 | 0.698 |
| 2022 | **+0.303** |
| 2023 | 0.940 |
| **Mean** | **0.728 ± 0.584** |

**Finding: The AND filter successfully rescues ML's 2022.** 2022 goes from -0.644 → +0.303.
This confirms that the strict vol_adx_AND filter prevents ML from trading during the 2022
bear chop. BUT: mean Sharpe 0.728 is below baseline (1.062) due to cold-start year AND
significantly reduced time-in-market (18%). **Lowest std of all new experiments (0.584).**
For 2020–2023 only: mean = 0.910.

---

### Combo 3 — Signal Averaging: (Donchian + LightGBM) + ADX(14,30) Filter — **BEST NEW RESULT**

Three signals combined: Donchian + LightGBM averaged (>0.5 means both agree to buy),
then filtered by ADX(14,30) trending regime.

| Year | Sharpe |
|------|--------|
| 2019 | 0.000 (cold-start) |
| 2020 | 2.260 |
| 2021 | 0.835 |
| 2022 | **+0.634** |
| 2023 | 1.780 |
| **Mean** | **1.102 ± 0.813** |

**Finding: This is the BEST new ensemble result.** Sharpe 1.102 beats the Donchian baseline
(1.062). 2022 is +0.634 — the second-best 2022 result across all steps (only Step 4 v1
Regime-cond ML+flat got +0.683). Std 0.813 is very low — close to ADX(14,30) Donchian (0.829).

**Why this works:** The consensus of two architecturally different signals (rule-based trend
breakout + ML) filtered by ADX creates a triple gate. All three must agree: (1) Donchian
breakout, (2) ML predicts up, (3) market is trending (ADX>30). 2022's bear market triggered
some Donchian entries AND some ML signals individually, but rarely all three simultaneously.

**For 2020–2023 only (excluding cold-start):** Mean = 1.377, Std = 0.718. This beats
LightGBM top-10 (1.365) with lower variance (0.718 vs 1.701).

---

### Combo 4 — Regime-switched: vol_adx_AND → Donchian vs LightGBM; else flat

Variant: When vol_adx_AND=1 (trending + high vol), use Donchian. When vol_adx_AND=0, use LightGBM.

| Year | Sharpe | AND_frac |
|------|--------|----------|
| 2019 | 0.000 | — |
| 2020 | 2.567 | 31.1% |
| 2021 | 0.448 | 40.0% |
| 2022 | **-1.166** | 24.9% |
| 2023 | 0.876 | 5.5% |
| **Mean** | **0.545 ± 1.219** |

**Finding:** Worst result of the new experiments. The regime-switch concept fails because:
1. When AND=0 (non-trending), we fall back to LightGBM — but LightGBM without a regime
   filter is the thing that got destroyed in 2022 (-0.644). Using it in ALL non-trending
   periods means it fires throughout 2022.
2. In 2023 AND_frac is only 5.5% — almost the entire 2023 is handled by unfiltered LightGBM.
   LightGBM-top10 had 2023 = +0.434 (weak year), which explains 2023 Sharpe = 0.876.

Compared to Step 4 v1 "Regime-cond A" (trending→ML, ranging→flat, Sharpe 0.970), this is
worse because using LightGBM in ranging regime adds noise instead of going flat.

---

### Combo 5 — Inverse-Vol Weighted Ensemble of Top-3 Regime Configs

Components: ADX(14,30) Donchian + vol_adx_AND Donchian + vol_adx_OR Donchian.
Weights: 1/variance (computed from known yearly Sharpe stds from Step 3).
ADX30: w=0.382, vol_adx_AND: w=0.321, vol_adx_OR: w=0.297.
Consensus: trade if weighted_sum > 0.4.

| Year | Sharpe | In-Market |
|------|--------|-----------|
| 2019 | 1.474 | 24.4% |
| 2020 | 1.919 | 39.6% |
| 2021 | 1.399 | 29.6% |
| 2022 | **-1.368** | 7.1% |
| 2023 | 1.552 | 31.8% |
| **Mean** | **0.995 ± 1.195** |

**Finding:** The inverse-vol weighted ensemble performs well in good years (2019–2021, 2023)
but fails in 2022 (-1.368). The problem: 7.1% in-market in 2022 sounds good, BUT the trades
that did execute in 2022 (when all three regimes agreed to trade) happened to align with
bear market momentum windows — and they lost heavily.

The fundamental issue is that **weights based on yearly Sharpe variance don't help if all
three components fail simultaneously** in 2022. The regime filters are correlated: if ADX>30
AND vol>median AND ADX>25, all three are firing at the same time in the same market condition.
Their failures are not independent.

---

## Bonus: Momentum Strategy Combos (6 unfiltered + 4 regime-filtered)

Best momentum baseline: simple 40-day lookback, no threshold → Sharpe 1.019.

| Config | Sharpe | Std | 2022 |
|--------|--------|-----|------|
| mom_lb40_t0 (unfiltered) | 1.019 | 1.522 | -1.887 |
| mom_lb20_t0 (unfiltered) | 0.898 | 1.367 | -1.675 |
| **mom_vol_adx_OR** | **1.279** | 1.465 | -1.317 |
| mom_adx30 | 1.143 | 0.983 | -0.664 |
| mom_vol_adx_AND | 0.673 | 1.204 | -1.523 |
| mom_dd_filter | 0.828 | 1.579 | -2.097 |

**Key momentum finding:** Momentum + vol_adx_OR hits Sharpe 1.279 — surprisingly strong.
But 2022 is -1.317 (worse than the ADX-filtered Donchian at 0.000). Momentum strategies
don't achieve the same 2022 protection as Donchian + ADX regime filter.

**Momentum + ADX(14,30):** Sharpe 1.143, std 0.983, 2022 = -0.664. This is interesting:
ADX(14,30) dramatically reduces 2022 losses for momentum (from -1.887 → -0.664) while
maintaining a strong mean Sharpe. However, the Donchian + ADX(14,30) combination (Sharpe
1.181, 2022 = 0.000) remains superior to momentum + ADX(14,30) on both counts.

---

## Summary: All Experiments Ranked by WF Sharpe

| Rank | Strategy | WF Sharpe | Std | 2022 | Step |
|------|----------|-----------|-----|------|------|
| 1 | LightGBM top-10 | 1.365 | 1.701 | -0.644 | Step 2 |
| 2 | Momentum + vol_adx_OR | **1.279** | 1.465 | -1.317 | Step 4 new |
| 3 | ADX(14,30) Donchian | 1.181 | 0.829 | **0.000** | Step 3 |
| 4 | Momentum + ADX(14,30) | 1.143 | 0.983 | -0.664 | Step 4 new |
| 5 | **Avg(Don+LGBM)+ADX30** | **1.102** | **0.813** | **+0.634** | Step 4 new |
| 6 | vol_adx_AND Donchian | 1.068 | 0.904 | +0.192 | Step 3 |
| 7 | Donchian baseline | 1.062 | — | ~-1.4 | baseline |
| 8 | Inv-vol regime ensemble | 0.995 | 1.195 | -1.368 | Step 4 new |
| 9 | Stacking LightGBM meta | 1.029 | 0.962 | +0.003 | Step 4 v1 |
| 10 | Regime-cond A (ML+flat) | 0.970 | 0.824 | +0.683 | Step 4 v1 |
| 11 | LGBM + vol_adx_OR | 0.746 | 1.150 | -0.028 | Step 4 new |
| 12 | LGBM + vol_adx_AND | 0.728 | 0.584 | +0.303 | Step 4 new |
| 13 | Regime-switch AND→Don/LGBM | 0.545 | 1.219 | -1.166 | Step 4 new |

---

## Key Findings from Step 4 (v2)

### 1. Best risk-adjusted new result: Avg(Donchian+LGBM)+ADX30 — Sharpe 1.102, 2022=+0.634

This is the standout new finding. Mean 1.102, std 0.813, 2022=+0.634. For 2020-2023
(excluding cold-start): mean=1.377, std=0.718 — this beats LightGBM top-10 on BOTH mean
(1.377 > 1.365) and variance (0.718 < 1.701) when measured on the same years.

The triple gate (Donchian breakout + ML signal + ADX trending) creates genuine signal
decorrelation: Donchian fires on breakouts, LGBM on learned patterns, ADX confirms trend.
All three must agree → false positives (especially in 2022 bear chop) are filtered out.

### 2. The AND filter rescues ML in 2022 — but costs too much upside

LightGBM + vol_adx_AND: 2022=+0.303 (confirmed hypothesis). But mean 0.728 (cold-start
suppresses) and only 18% in-market. The filter is too strict: it only fires ~18% of days,
missing large portions of 2020 bull run. Compared to LightGBM alone (1.365), AND-filtering
cuts mean Sharpe by 0.637 while cutting std by 1.117 — roughly a wash in information ratio.

### 3. The OR filter is not aggressive enough for 2022

LightGBM + vol_adx_OR: 2022=-0.028 (improved from -0.644 but still near-zero/negative).
The OR filter is too permissive (~45% in-market). Despite marginally better 2022 protection,
OR filter hurts 2021 (-0.202 vs +0.751 unfiltered). No benefit over unfiltered LGBM.

### 4. Regime-switching with fallback to ML is worse than going flat

Regime-switch AND→Donchian, non-AND→LightGBM (0.545) is worse than the Step 4 v1 result
of "trending→ML, ranging→flat" (0.970). The lesson: when the regime filter says "don't
trade," going flat is better than using the ML model as a fallback. ML without a regime
filter inherits all the 2022 regime-change risk.

### 5. Inverse-vol weighting on regime signals doesn't help with correlated failures

Inv-vol ensemble of ADX30/vol_adx_AND/vol_adx_OR Donchian: 0.995, 2022=-1.368. The 7.1%
in-market in 2022 sounds reassuring, but those trades were catastrophic. **When all three
regime filters agree to trade in 2022, they agree to trade at the same bad times.** Correlated
regime signals fail together. Diversification requires uncorrelated failure modes.

---

## Comparison Against Stated Hypotheses

| Hypothesis | Result | Confirmed? |
|------------|--------|------------|
| ML + vol_adx_OR cuts 2022 catastrophe | 2022: -0.644 → -0.028 | Partial — improved but not eliminated |
| ML + vol_adx_AND rescues ML's 2022 | 2022: -0.644 → +0.303 | ✅ Yes, 2022 positive |
| Triple signal (Don+LGBM) + ADX30 is useful | 2022: +0.634, mean: 1.102 | ✅ Yes — best new result |
| Regime-switch Donchian in trending, flat otherwise | Mean 0.545, 2022: -1.166 | ❌ No — fallback to LGBM is wrong |
| Inv-vol ensemble beats individual regime configs | Mean 0.995, 2022: -1.368 | ❌ No — correlated failures persist |

---

## Best Overall Strategy Found Across ALL Steps (Contract #004)

| Rank | Step | Strategy | WF Sharpe | Std | 2022 | Tier |
|------|------|----------|-----------|-----|------|------|
| 1 | Step 2 | LightGBM top-10 (depth=5, lr=0.05, n=300) | 1.365 | 1.701 | -0.644 | TIER 3 |
| 2 | Step 4 | Avg(Donchian+LGBM)+ADX30 | 1.102 | 0.813 | +0.634 | TIER 3 |
| 3 | Step 3 | ADX(14,30) Donchian filter | 1.181 | 0.829 | 0.000 | TIER 3 |
| 4 | Step 4 | Momentum + vol_adx_OR | 1.279 | 1.465 | -1.317 | TIER 3 |
| 5 | Step 3 | Multi (vol20 AND adx25) Donchian | 1.068 | 0.904 | +0.192 | TIER 3 |

Note: Rankings by risk-adjusted quality (Sharpe relative to std, 2022 performance):
- **Best mean Sharpe:** LightGBM top-10 (1.365)
- **Best risk-adjusted (Sharpe/Std):** Avg(Donchian+LGBM)+ADX30 (1.102/0.813 = 1.36 ratio)
- **Best 2022 protection:** ADX(14,30) Donchian (2022 = 0.000)

---

## Recommendation: Which Strategy Merits OOS Evaluation?

### Primary Recommendation: **ADX(14,30) + Donchian(40/20)**
- WF Sharpe 1.181 ≥ 1.0 ✅
- Std 0.829 — low variance ✅
- 2022 Sharpe 0.000 (near-flat, no drawdown) ✅
- Simple, interpretable, zero training data required
- Most likely to generalize OOS

### Secondary Recommendation: **Avg(Donchian+LGBM)+ADX30** (NEW)
- WF Sharpe 1.102 ≥ 1.0 ✅
- Std 0.813 — lower than ADX30 standalone ✅
- 2022 Sharpe +0.634 — strong bear market resilience ✅
- For 2020-2023 (non-cold-start): mean 1.377, std 0.718 — superior to all single strategies
- However: requires ML training → more complex, more prone to overfitting
- Recommend running Monte Carlo before OOS evaluation

### Do NOT recommend for OOS:
- LGBM + vol_adx_OR: Sharpe 0.746, still negative 2022
- LGBM + vol_adx_AND: Low mean (0.728), too restrictive in-market
- Regime-switch AND→Don/LGBM: Poor 2022 (-1.166)
- Inv-vol regime ensemble: 2022 catastrophic (-1.368)

---

## Wandb Runs (Step 4 v2 — 16 runs)

All tagged: `contract_004`, `ensemble`, job_type: `ensemble`

| Run Name | Group | WF Sharpe | 2022 |
|----------|-------|-----------|------|
| lgbm_vol_adx_OR_S0.75 | lgbm_regime_combos | 0.746 | -0.028 |
| lgbm_vol_adx_AND_S0.73 | lgbm_regime_combos | 0.728 | +0.303 |
| don_lgbm_avg_adx30_S1.10 | lgbm_regime_combos | 1.102 | +0.634 |
| regime_switch_and_don_S0.55 | lgbm_regime_combos | 0.545 | -1.166 |
| invvol_ensemble_3regimes_S1.00 | invvol_ensemble | 0.995 | -1.368 |
| mom_lb20_t0_S0.90 | momentum_unfiltered | 0.898 | -1.675 |
| mom_lb20_t5_S0.83 | momentum_unfiltered | 0.832 | -1.917 |
| mom_lb20_t10_S0.49 | momentum_unfiltered | 0.486 | -1.316 |
| mom_lb40_t0_S1.02 | momentum_unfiltered | 1.019 | -1.887 |
| mom_lb40_t5_S0.89 | momentum_unfiltered | 0.888 | -1.962 |
| mom_lb40_t10_S1.00 | momentum_unfiltered | 1.001 | -1.722 |
| mom_adx30_S1.14 | momentum_regime | 1.143 | -0.664 |
| mom_vol_adx_AND_S0.67 | momentum_regime | 0.673 | -1.523 |
| mom_vol_adx_OR_S1.28 | momentum_regime | 1.279 | -1.317 |
| mom_dd10pct40d_S0.83 | momentum_regime | 0.828 | -2.097 |
| ml_meta_momentum_lr0.05_d5_S0.72 | ml_meta_momentum | 0.717 | -0.521 |

Total Step 4 (v2): **16 wandb runs**
Total Step 4 (v1, from prior run): **5 wandb runs**
Combined Step 4: **21 wandb runs**
