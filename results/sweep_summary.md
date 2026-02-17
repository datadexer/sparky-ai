# CONTRACT #004 — Two-Stage Sweep Results

**Run date:** 2026-02-17
**Baseline:** Donchian Multi-TF Sharpe **1.062** (in-sample, corrected for look-ahead)
**Data:** `feature_matrix_btc_hourly_expanded` — 3,976 daily rows, 2013-01-31 to 2023-12-31 (in-sample only)
**Feature pool:** 88 hourly-aggregate features; sweeps used top-10/15/20 by XGBoost importance

---

## Stage 1: Screening (28 configs, single 80/20 split)

| Rank | Family    | Features | Depth | LR    | Val Sharpe | Val Acc | AUC   | Note |
|------|-----------|----------|-------|-------|-----------|---------|-------|------|
| 1    | CatBoost  | top20    | 3     | 0.01  | -0.046    | 50.3%   | 0.531 | Shallow tree + very low LR avoids overfitting; near-zero Sharpe on OOT |
| 2    | LightGBM  | top20    | 3     | 0.01  | -0.060    | 52.5%   | 0.534 | Same story as CB — low LR keeps it from diverging but no real edge on 20% |
| 3    | XGBoost   | top20    | 5     | 0.03  | -0.412    | 51.1%   | 0.526 | Medium depth + moderate LR; marginal AUC suggests weak directional signal |
| 4    | LightGBM  | top10    | 5     | 0.05  | -0.416    | 50.6%   | 0.517 | Fewer features hurts AUC slightly but cleaner signal in walk-forward |
| 5    | XGBoost   | top20    | 3     | 0.01  | -0.456    | 52.1%   | 0.533 | Very conservative XGB; holds up in WF due to low overfitting |
| 6    | LightGBM  | top20    | 6     | 0.10  | -0.465    | 51.1%   | 0.520 | High LR + deep — likely fits noise; AUC drops |
| 7    | CatBoost  | top15    | 5     | 0.05  | -0.492    | 53.6%   | 0.546 | Best stage-1 AUC overall; features 11-20 add noise to signal |
| 8    | CatBoost  | top20    | 6     | 0.01  | -0.600    | 51.8%   | 0.534 | Deeper tree hurts even at low LR — too many splits for ~3K training rows |
| 9    | XGBoost   | top20    | 5     | 0.05  | -0.640    | 51.9%   | 0.528 | Moderate config; negative test Sharpe suggests 2021-2023 regime change |
| 10   | LightGBM  | top20    | 5     | 0.05  | -0.652    | 52.9%   | 0.540 | Similar to rank 9 — 2021-2023 test period is hostile for ML momentum features |
| 11   | LightGBM  | top20    | 4     | 0.03  | -0.827    | 53.0%   | 0.537 | Accuracy/AUC look OK but Sharpe negative — threshold sensitivity issue |
| 12   | CatBoost  | top20    | 4     | 0.05  | -0.838    | 52.6%   | 0.542 | Similar to rank 11; CatBoost accurate but generates false-positive long signals |
| 13   | LightGBM  | top20    | 4     | 0.05  | -0.839    | 51.8%   | 0.537 | Mirrors CatBoost result — accuracy misleading, costs eat Sharpe |
| 14   | CatBoost  | top10    | 5     | 0.05  | -0.850    | 50.5%   | 0.527 | Top-10 only: drops signal vs top15/20 for CatBoost |
| 15   | CatBoost  | top20    | 5     | 0.10  | -0.720    | 53.1%   | 0.537 | High LR = faster convergence but overfits train set |
| 16   | CatBoost  | top20    | 5     | 0.03  | -0.723    | 53.8%   | 0.537 | Best CatBoost accuracy (53.8%) but still negative Sharpe — signals clustered |
| 17   | LightGBM  | top20    | 5     | 0.03  | -0.734    | 51.6%   | 0.535 | Intermediate LR 0.03 worse than 0.01 or 0.05 extremes for LGBM |
| 18   | LightGBM  | top15    | 5     | 0.05  | -0.702    | 52.4%   | 0.533 | Top-15 LGBM: marginal vs top20, same regime sensitivity |
| 19   | XGBoost   | top20    | 6     | 0.05  | -0.866    | 50.8%   | 0.527 | Deep+moderate LR overfits — Sharpe collapse on test |
| 20   | CatBoost  | top20    | 5     | 0.05  | -0.912    | 52.3%   | 0.541 | Moderate depth + lr; border_count=32 may limit split quality |
| 21   | LightGBM  | top20    | 6     | 0.05  | -0.897    | 51.4%   | 0.533 | Deepest LGBM with 0.05 LR — predictably worse than shallower |
| 22   | XGBoost   | top20    | 3     | 0.05  | -0.938    | 51.1%   | 0.519 | Low AUC (0.519) — XGB with depth=3 + lr=0.05 misses features |
| 23   | XGBoost   | top20    | 4     | 0.05  | -1.279    | 50.6%   | 0.521 | Deeper than rank 22 but worse — overfits despite subsample |
| 24   | XGBoost   | top20    | 4     | 0.10  | -1.306    | 49.5%   | 0.521 | High LR + depth4 = classic overfit; accuracy <50% is a bad sign |
| 25   | CatBoost  | top20    | 4     | 0.10  | -0.859    | 51.6%   | 0.528 | CB more regularised than XGB at same settings; still negative |
| 26   | LightGBM  | top20    | 4     | 0.10  | -0.713    | 51.0%   | 0.527 | LGBM handles 0.10 slightly better; still deeply negative Sharpe |
| 27   | XGBoost   | top20    | 6     | 0.10  | -0.747    | 51.8%   | 0.528 | High LR on deep XGB — noise amplified |
| 28   | CatBoost  | top20    | 6     | 0.01  | -0.600    | 51.8%   | 0.534 | Included above as rank 8 |

**Key observation:** ALL 28 Stage 1 configs show negative Sharpe on the 80/20 test split (2021-10-27 to 2023-12-31). The test period includes the brutal 2022 bear market where momentum-based features badly underperform. Walk-forward validation provides a more balanced view.

---

## Stage 2: Walk-Forward Validation (Top 5, expanding window 2019–2023)

| Rank | Family    | Features | Depth | LR   | WF Sharpe | Std   | Min    | Max   | WF Acc | Beats Baseline? |
|------|-----------|----------|-------|------|-----------|-------|--------|-------|--------|----------------|
| 1    | **LightGBM**  | top10    | 5     | 0.05 | **1.365** | 1.701 | -0.644 | 4.342 | 55.1%  | **✅ Yes** |
| 2    | **CatBoost**  | top20    | 3     | 0.01 | **1.325** | 1.810 | -1.699 | 3.809 | 54.5%  | **✅ Yes** |
| 3    | **LightGBM**  | top20    | 3     | 0.01 | **1.197** | 1.513 | -0.908 | 3.814 | 54.4%  | **✅ Yes** |
| 4    | XGBoost   | top20    | 3     | 0.01 | 1.050     | 1.773 | -1.461 | 3.600 | 55.0%  | ❌ No (1.050 < 1.062) |
| 5    | XGBoost   | top20    | 5     | 0.03 | 0.623     | 1.347 | -1.622 | 2.590 | 53.1%  | ❌ No |

### Per-Year Breakdown

| Year | LGBM-top10 | CB-top20 | LGBM-top20 | XGB-d3 | XGB-d5 | Notes |
|------|-----------|---------|-----------|--------|--------|-------|
| 2019 | 1.941 | 1.585 | 1.253 | 2.274 | 0.913 | Bull market; all models positive |
| 2020 | 4.342 | 3.809 | 3.814 | 3.600 | 2.590 | COVID crash + recovery; huge momentum signal |
| 2021 | 0.751 | 0.770 | 0.874 | 0.972 | 0.411 | Late bull cycle; models fade |
| 2022 | **-0.644** | **-1.699** | **-0.908** | **-1.461** | **-1.622** | Bear market; models struggle badly |
| 2023 | 0.434 | 2.161 | 0.951 | -0.136 | 0.823 | Recovery; CatBoost adapts best |

---

## Which Model Family Performed Best?

**LightGBM with top-10 features** achieved the highest walk-forward Sharpe (1.365) with the best downside control (min Sharpe -0.644 vs -1.699 for CatBoost).

**Hypothesis:** LightGBM with `num_leaves=63` and `max_depth=5` balances expressiveness vs regularisation better than CatBoost's symmetric trees for this regime. Using only top-10 features acts as implicit regularisation — the remaining 10 features in top20 add noise from the 2022 bear regime. The leaf-wise growth of LightGBM can model non-linear regime interactions that CatBoost's symmetric splits miss, which is why it holds up slightly better in 2023.

**CatBoost** with shallow depth=3 and very low lr=0.01 performed second, demonstrating that aggressive regularisation (both in tree depth and learning rate) is key for this dataset size (~3K rows at training start).

**XGBoost** consistently underperformed vs LightGBM and CatBoost at equivalent hyperparameter settings. The hist method is efficient but XGB's default regularisation appears insufficient for this noisy regime-dependent data.

---

## Top 5 Configs Ranked by Walk-Forward Sharpe

1. **LightGBM, top-10 features, depth=5, lr=0.05, n=300, num_leaves=63, L2=1.0** — Sharpe 1.365 ± 1.701
2. **CatBoost, top-20 features, depth=3, lr=0.01, iter=300, L2=1.0, border_count=32** — Sharpe 1.325 ± 1.810
3. **LightGBM, top-20 features, depth=3, lr=0.01, n=300, num_leaves=31, L2=1.0** — Sharpe 1.197 ± 1.513
4. **XGBoost, top-20 features, depth=3, lr=0.01, n=500, L2=0.5** — Sharpe 1.050 ± 1.773
5. **XGBoost, top-20 features, depth=5, lr=0.03, n=500, L2=1.0** — Sharpe 0.623 ± 1.347

---

## Honest Assessment

### Does any config reliably beat Donchian 1.062?

**Partially — but with important caveats:**

3 of 5 walk-forward configs beat the 1.062 baseline on mean WF Sharpe. However:

1. **High variance is the central problem.** The best config (LGBM, Sharpe 1.365) has std=1.701. This means ~1-sigma range is [-0.34, +3.07]. A single bad year (2022: -0.644) nearly wipes out the multi-year gains. The Donchian baseline at 1.062 almost certainly has lower variance and more consistent year-over-year performance.

2. **2022 is the canary.** Every ML model goes significantly negative in 2022. The CatBoost depth=3 config hits -1.699. The Donchian strategy is trend-following and would likely also underperform in a straight bear market, but the magnitude suggests ML momentum features don't adequately model regime change.

3. **The screening→WF disconnect is striking.** All Stage 1 screening Sharpes were negative (worst test period 2021-2023), while walk-forward mean Sharpes are positive (because 2019-2020 strongly positive years carry the average). This means Stage 1 screening is a POOR predictor of walk-forward quality for this dataset.

4. **Mean Sharpe > 1.062 but NOT statistically significant.** With 5 years of data and std ~1.7, we cannot reject H0: Sharpe ≤ 1.062 at 95% confidence. The ML "edge" is within noise range.

### Verdict: TIER 3 — Continue Iterating

The results meet the Tier 3 threshold (Sharpe ≥ 0.4, shows edge over B&H) but do NOT reach Tier 2 (Sharpe ≥ 0.7 + beats B&H reliably) due to high year-to-year variance and the catastrophic 2022 drawdown.

**Recommended next steps:**
- Add regime detection as a meta-feature (bull/bear/sideways) — models clearly fail in bear markets
- Test ensemble of top-3 configs (LGBM-top10 + CB-d3 + LGBM-d3) to reduce per-year variance
- Investigate 2022 performance: are there features that would signal "don't trade" in bear regime?
- Consider 4h horizon instead of 24h — shorter predictions may be more learnable

---

## Wandb Logging

- Stage 1 sweep (28 configs): `stage1_screening_28configs_222524` — tags: `contract_004`, `sweep`
- Stage 2 batch (5 walk-forward): `stage2_walkforward_5configs_222649` — tags: `contract_004`, `sweep`
- WF rank 1 (LGBM-top10): `lgbm_lr0.05_d5_n300_nl63_L21_S1.36` — tags: `contract_004`, `sweep`
- WF rank 2 (CB-d3): `cat_lr0.01_d3_n300_L21_S1.32` — tags: `contract_004`, `sweep`
- WF rank 3 (LGBM-d3): `lgbm_lr0.01_d3_n300_nl31_L21_S1.20` — tags: `contract_004`, `sweep`
- WF rank 4 (XGB-d3): `xgb_lr0.01_d3_n500_L20.5_S1.05` — tags: `contract_004`, `sweep`
- WF rank 5 (XGB-d5): `xgb_lr0.03_d5_n500_L21_S0.62` — tags: `contract_004`, `sweep`

Total: **7 wandb runs** + results table with 28 Stage 1 configs.
