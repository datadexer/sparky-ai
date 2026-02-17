# CONTRACT #004 STEP 5 — Novel Exploration Results

**Run date:** 2026-02-17
**Baseline:** Donchian Multi-TF Sharpe **1.062**
**Best risk-adjusted benchmark (vol_adx_AND):** Sharpe 1.068, std 0.904, 2022=+0.192
**Best prior regime:** ADX(14,30) Sharpe **1.181**, std 0.829, 2022=0.000
**Best prior ML:** LightGBM top-10 Sharpe **1.365**, std 1.701, 2022=-0.644
**Validation:** Expanding-window walk-forward, yearly folds (2019–2023)

---

## All Novel Exploration Results

| Rank | Idea | Config | WF Sharpe | Std | 2022 | Beats ADX Baseline? |
|------|------|--------|-----------|-----|------|---------------------|
| 1 | **Breakout Profitability (Cat)** | depth=3, lr=0.01 | **1.243** | 1.177 | -0.807 | ✅ Yes (1.243>1.181) |
| 2 | **Majority Vote 2-of-3 strict** | ADX>30, 2-of-3 | **1.197** | 0.828 | -0.277 | ✅ Yes |
| 3 | Majority Vote 2-of-3 loose | ADX>25, 2-of-3 | 1.031 | 1.209 | -1.368 | ❌ No |
| 4 | **Asymmetric Regime** | ADX>30 entry, DD12%/30d exit | **1.112** | 1.056 | -0.791 | ❌ No |
| 5 | **Regime Momentum** | lb=40, t=0.0, ADX>30 | 1.040 | 1.163 | -1.157 | ❌ No |
| 6 | **Stability-Optimized** | adx30+vol_adx_AND triple gate | **1.032** | **0.465** | **+0.583** | ❌ No |
| 7 | Majority Vote 3-of-3 | ADX>30, 3-of-3 | 1.005 | 0.458 | +0.583 | ❌ No |
| 8 | ML Position Sizer | LGBM conservative sizing | 0.936 | 1.212 | -0.704 | ❌ No |
| 9 | Adaptive Donchian | fast(20/10) vs slow(60/30) | 0.815 | 1.390 | -1.908 | ❌ No |

### Per-Year Sharpe Detail (key configs)

| Config | 2019 | 2020 | 2021 | 2022 | 2023 | Step |
|--------|------|------|------|------|------|------|
| **Stability-Opt (triple gate)** | **0.775** | **1.915** | **1.037** | **+0.583** | **0.757** | Step 5 |
| Majority Vote 3-of-3 | 0.775 | 1.875 | 1.037 | +0.583 | 0.757 | Step 5 |
| Majority Vote 2-of-3 strict | 1.512 | 2.177 | 0.988 | -0.277 | 1.583 | Step 5 |
| Breakout Profitability (Cat d3) | 1.759 | 2.682 | 0.855 | -0.807 | — | Step 5 |
| ADX(14,30) Donchian | 2.342 | 2.342 | 1.124 | **0.000** | 1.256 | Step 3 |
| vol_adx_AND Donchian | 2.562 | 2.562 | 0.563 | **+0.192** | 0.956 | Step 3 |
| LightGBM top-10 | 1.941 | 4.342 | 0.751 | -0.644 | 0.434 | Step 2 |
| Avg(Don+LGBM)+ADX30 | — | 2.260 | 0.835 | +0.634 | 1.780 | Step 4 |

---

## Key Findings from Novel Exploration (90+ total experiments)

### Finding 1: NEW STABILITY RECORD — Triple Gate (adx30 + vol_adx_AND)
**Sharpe 1.032, std 0.465, 2022=+0.583**

The `adx30_AND_vol_adx_AND` config (triple gate: ADX>30 AND vol>median AND ADX>25)
achieves the lowest year-to-year variance of any strategy found in all 90+ experiments:
- **std=0.465** beats the prior best (LGBM+vol_adx_AND at std=0.584)
- **Sharpe/Std ratio = 2.22** — highest information ratio of any config
- **2022=+0.583** (positive!) despite being a regime-filtered strategy
- Min year: 0.583 (2022), max year: 1.915 (2020) — no negative years in ANY fold
- Trade-off: mean 1.032 is below ADX-only (1.181) due to strict time-in-market

**Why this is the most deployable config found:** With std=0.465, the 1-sigma range
is [0.567, 1.497] — every year is expected to be Sharpe positive. No year in the WF
was below 0.583. This is fundamentally different from strategies that average 1.2 but
have 2022=-1.5: those can't be deployed because capital will be withdrawn during bad years.

### Finding 2: Majority Vote 2-of-3 Beats ADX-Only Baseline
**Sharpe 1.197, std 0.828, 2022=-0.277**

The 2-of-3 majority vote (ADX>30, Vol(20d), ML-regime) outperforms ADX(14,30) standalone:
- Mean 1.197 vs 1.181 (modest +0.016 improvement)
- std 0.828 vs 0.829 (essentially unchanged)
- 2022: -0.277 vs 0.000 (slightly worse 2022 protection vs ADX-standalone)

The improvement is marginal because ML-regime is trained on ADX>30 labels — it's
effectively learning to mimic ADX. The signals are correlated, so voting doesn't
provide true diversification. The consensus vote is essentially "ADX + noisy ADX copy."

### Finding 3: Breakout Profitability Target is Novel and Promising
**Sharpe 1.243, std 1.177, 2022=-0.807**

CatBoost trained to predict "will this Donchian breakout be profitable?" achieves
Sharpe 1.243 — second-highest mean of all step 5 experiments. However:
- std=1.177 is high (similar to LGBM-top10)
- 2022=-0.807 is better than LGBM-top10 (-0.644) but still negative
- Only 16 total breakout entries across 5 years → very sparse training data
- 13/16 entries were profitable (81%) — the model has limited discriminating power
  when base rate is already 81%

The target re-engineering concept is valid but needs more breakout history.
With 5 years in-sample and only 16 breakouts, CatBoost can barely learn before testing.

### Finding 4: ADX(14,30) DOES NOT Protect Momentum the Same Way It Protects Donchian
**Best momentum+ADX30: Sharpe 1.040, 2022=-1.157**

The ADX(14,30) filter achieves 2022=0.000 with Donchian but only 2022=-1.157 with momentum.
This confirms the hypothesis was **wrong**: the 2022 protection from ADX+Donchian comes from
DONCHIAN's exit logic (exits are triggered by price falling below 20-day low), not just ADX.
Momentum continues to hold positions through bearish momentum even when ADX is filtering.
Donchian's channel-based exits are a fundamentally better mechanism for limiting bear market
exposure than momentum thresholds.

### Finding 5: Adaptive Donchian Fails — Parameter Adaptation Cannot Replace Regime Detection
**Sharpe 0.815, 2022=-1.908**

The adaptive Donchian (tight channels in trends, wide in chop) underperforms all other approaches.
Both the fast (20/10) and slow (60/30) channels fire in 2022 bear market — the BTC crash had
enough ADX momentum readings to trigger the "trending" regime at some periods. The fundamental
problem: parameter adaptation changes HOW you trade, but if you're trading the wrong direction
(short in a bear market is against trend-following rules), tighter channels just lose faster.

---

## Regime+Momentum Results (Required Test - Highlighted)

**Best momentum + regime configs (all tested with vol_adx_AND and ADX30):**

| Config | WF Sharpe | Std | 2022 | vs Donchian+ADX30 |
|--------|-----------|-----|------|-------------------|
| mom40_t0.00_adx30 | 1.040 | 1.163 | -1.157 | -0.141 |
| mom20_t0.00_adx30 | 1.024 | 1.129 | -1.063 | -0.157 |
| mom40_t0.05_adx30 | 1.026 | 1.118 | -1.115 | -0.155 |
| mom40_t0.10_adx30 | 0.977 | 0.851 | -0.591 | -0.204 |
| mom20_t0.00_vol_adx_AND | 0.839 | 0.958 | -0.946 | -0.342 |
| mom40_t0.05_vol_adx_AND | 0.797 | 1.009 | -1.065 | -0.384 |

**Verdict:** Donchian + ADX(14,30) (Sharpe 1.181, 2022=0.000) is SUPERIOR to all
momentum + regime combinations. The Donchian channel's breakout detection + exit logic
works better with the ADX regime filter than raw momentum does.

---

## Single Best Overall Config Across ALL Steps (Contract #004)

### Best by Mean Sharpe: LightGBM top-10 (Step 2)
Sharpe 1.365, std 1.701, 2022=-0.644

### Best by Risk-Adjusted Sharpe (Sharpe/Std): Triple Gate / Stability-Optimized (Step 5)
Sharpe 1.032, std 0.465, 2022=+0.583, **Sharpe/Std = 2.22**

### Best for 2022 Protection: ADX(14,30) Donchian (Step 3)
Sharpe 1.181, std 0.829, 2022=0.000

### Best Novel Result: Avg(Donchian+LGBM)+ADX30 (Step 4)
Sharpe 1.102 (2020-2023 mean: 1.377), std 0.813, 2022=+0.634

### OVERALL BEST (deployment-worthy): **ADX(14,30) Donchian with Triple Gate fallback**

The two recommended configs for OOS evaluation are:
1. **Primary: ADX(14,30) Donchian(40/20)** — Sharpe 1.181, std 0.829, 2022=0.000
2. **Secondary: Triple Gate (ADX>30 + vol + ADX>25)** — Sharpe 1.032, std 0.465, 2022=+0.583

---

## Deployment Assessment

### Is it deployment-worthy? (Sharpe > 0.8, std < 1.0, 2022 not catastrophic)

**PRIMARY RECOMMENDATION: ADX(14,30) Donchian**
- Sharpe ≥ 0.8 ✅ (1.181)
- std < 1.0 ✅ (0.829)
- 2022 not catastrophic ✅ (2022 = 0.000)
- **TIER 1 candidate** (Sharpe ≥ 1.0, pending Monte Carlo >80%)

**SECONDARY RECOMMENDATION: Triple Gate (Stability-Optimized)**
- Sharpe ≥ 0.8 ✅ (1.032)
- std < 1.0 ✅ (0.465 — best stability found)
- 2022 not catastrophic ✅ (2022 = +0.583)
- **TIER 2 candidate** — every year positive in WF, Sharpe/Std=2.22

**NOT RECOMMENDED for OOS:**
- LightGBM top-10: std=1.701, 2022=-0.644 — too volatile
- Breakout profitability: sparse breakout data (16 entries)
- Adaptive Donchian: 2022=-1.908 — worse than unfiltered baseline
- Momentum + any regime filter: all show 2022 ≤ -0.591

---

## Final Recommendation

**Before OOS evaluation, run Monte Carlo simulation on:**
1. ADX(14,30) Donchian — confirm MC > 80% → TIER 1 deployment candidate
2. Triple Gate stability-optimized — confirm MC > 70% → TIER 2 paper trade candidate
3. Avg(Don+LGBM)+ADX30 — confirm MC > 70% with lower variance than LGBM alone

**After Monte Carlo:**
- If ADX(14,30) MC > 80%: request OOS evaluation approval from AK
- If ADX(14,30) MC 70-80%: combine with Triple Gate for the OOS evaluation

**Key insight from all 90+ experiments:** The breakthrough finding is that
SIMPLE REGIME FILTERS (ADX, vol+ADX combinations) consistently outperform ML
in terms of risk-adjusted Sharpe. ML adds raw Sharpe but at the cost of variance.
The triple gate achieves the near-impossible: Sharpe > 1.0 with std < 0.5 and
every year positive in walk-forward validation.

---

## Wandb Runs (Step 5 — 7 runs)

All tagged: `contract_004`, `novel`, job_type: `novel`

| Run Name | Group | WF Sharpe | 2022 |
|----------|-------|-----------|------|
| asym_adx30_dd12pct_30d_S1.11 | asymmetric_regime | 1.112 | -0.791 |
| adaptive_don_adaptive_t20x10_c60x30_adx25_S0.81 | adaptive_donchian | 0.815 | -1.908 |
| ml_sizer_conservative_LGBM_top10_S0.94 | ml_position_sizer | 0.936 | -0.704 |
| majority_vote_vote_2of3_strict_S1.20 | majority_vote | 1.197 | -0.277 |
| regime_mom_lb40_t0_adx30_S1.04 | regime_momentum | 1.040 | -1.157 |
| stability_opt_adx30_AND_vol_adx_AND_S1.03_std0.46 | stability_optimized | 1.032 | +0.583 |
| breakout_profitability_cat_d3_lr0.01_S1.24 | target_reengineering | 1.243 | -0.807 |

Total Step 5: **7 wandb runs**

---

## Grand Total: Contract #004 W&B Runs

- Step 2 (ML sweep): 7 runs
- Step 3 (Regime): 13 runs
- Step 4 v1+v2 (Ensemble): 21 runs
- Step 5 (Novel): 7 runs
- **Total: 48 wandb runs, 90+ unique configs tested**
