# RESEARCH LOG — Sparky AI

Running log of all research findings. Newest entries at the top.

---

## PPY Validation Audit — 2026-02-19

**ALL cross-timeframe portfolio Sharpe values from overnight runs are inflated by sqrt(2) = 1.414x.**

Root cause: combining BTC 4h + ETH 8h returns via `index.intersection()` produces an 8h-resolution
common index (~1,096 obs/year). Agents used `ppy=2190` (4h) for annualization instead of `ppy=1095` (8h).
The `validate_periods_per_year()` guard was bypassed by passing `.values` (numpy array strips DatetimeIndex).

Individual strategy results (single-timeframe) are NOT affected.

Corrected values below. DSR is ppy-independent (uses per-period SR) and remains valid.

---

## eth_strategies directive — Session 12 — 2026-02-19

**STATUS**: TIER 1 (corrected) — Cross-timeframe portfolio
**TRIALS**: n_trials_start=627, n_trials_end=16269 (15,642 configs this session)

### Session 12 Champion (3-way portfolio)

**BTC4h(68,9)_iv(vw150,tv0.20) 30% + ETH8h(82,34)_iv(vw30,tv0.06) 57% + ETH8h(30,5)_iv(vw30,tv0.22) 13%**

| Metric | Agent Reported | Corrected (ppy=1095) |
|--------|---------------|---------------------|
| Sharpe@30bps | ~~3.434~~ | **2.428** |
| Sharpe@50bps | ~~3.129~~ | **2.212** |
| DSR | 0.993 | 0.993 (unchanged) |
| MaxDD | -3.62% | -3.62% (legitimate) |
| 2020+ Sharpe | ~~3.146~~ | **2.225** |
| 2022+ Sharpe | ~~1.651~~ | **1.167** |
| Robustness | 2000/2000 = 100% | 2000/2000 = 100% |

### Key Discoveries

1. **Short ETH8h as 3rd component**: ETH8h(30,5)_iv at 13% weight captures short-term momentum
2. **ETH8h tv=0.06**: Lower target vol for dominant ETH8h_long leg
3. **BTC4h params stable**: ep=65-70, xp=9, vw=150, tv=0.18-0.22
4. **ETH4h/ETH2h/BTC8h eliminated** as 3rd components — only short ETH8h helps

Results: `results/eth_strategies_20260218/session_012_master.json`

---

## btc_deep_20260218 Sessions 2-9 — 2026-02-19

**Individual configs VALIDATED. Portfolio results corrected for ppy.**

### BTC Individual Configs (no ppy correction needed)

| Config | S@30bps | DSR | MaxDD | S2020+ |
|--------|---------|-----|-------|--------|
| Don4h(160,25) iv(vw30,tv0.15) | **2.319** | 1.000 | -12.5% | 1.200 |
| Don8h(82,20) iv(vw30,tv0.2) | **2.220** | 1.000 | -21.6% | 1.404 |
| Don8h(80,20) iv(vw30,tv0.2) | **2.213** | 1.000 | -19.7% | 1.427 |
| Don8h(75,20) iv(vw30,tv0.2) | **2.170** | 1.000 | -19.4% | 1.452 |

### BTC Cross-Timeframe Portfolios (corrected)

| Portfolio | Agent Reported | Corrected (ppy=1095) |
|-----------|---------------|---------------------|
| Don4h(160,25)[40%]+Don8h(80,20)[60%] | ~~3.196~~ | **~2.26** |
| 3-way (40/40/20) | ~~3.203~~ | **~2.30** |
| P2@100bps | ~~2.402~~ | **~1.70** |

BTC correlation claim (0.59) is artifact of mixed-period returns. Actual at matched 8h: **0.90**.

### Walk-Forward Validation (BTC individual, from S5)
- Don8h(82,20)_iv: IS=2.220 → WF=1.708 (77% retention), WF_DSR=0.924
- Don4h(160,25)_iv: IS=2.319 → WF=1.552 (67% retention), WF_DSR=0.831

Results: `results/btc_deep_20260218/session_009_final_summary.json`

---

## eth_strategies directive — Session 7 — 2026-02-19

**STATUS**: TIER 1 confirmed (honest, 4h-grid portfolio)
**TRIALS**: n_trials_start=1781, n_trials_end=2718 (937 configs this session)

### CRITICAL: Session 5 TIER 1 was inflated

Session 5 "TIER 1" portfolio S30=2.760 was computed with **wrong periods_per_year**.
The common BTC+ETH 8h grid was evaluated with ppy=2190 (4h) instead of ppy=1095 (8h).
This inflated Sharpe by sqrt(2). Honest value: S≈1.952 (confirmed by session 6 replication).

### NEW Honest TIER 1 (4h grid, ppy=2190 everywhere)

**BTC Don4h(60,20)_iv(vw=90, tv=0.15) 75% + ETH Don4h(164,47)flat 25%**

| Metric | Value |
|--------|-------|
| Sharpe@30bps | **2.293** |
| Sharpe@50bps | 2.163 |
| DSR (n=2718) | 0.994 |
| Max Drawdown | **-13.1%** |
| Sharpe 2020+ | **1.760** |
| Sortino | 0.074 |
| Calmar | 3.54 |
| Corr(BTC,ETH) | 0.321 |
| Tier | **TIER 1** |

Sub-period validation: full S=2.293 MaxDD=-13.1%, 2020+ S=1.760 MaxDD=-11.7%.
**Artifact**: `results/eth_strategies_20260218/session_007_master.json`

### Session 7 Key Discoveries

1. **Frequency inflation bug**: Session 5 S=2.760 = artifact. Real portfolio S≈2.293 on 4h grid.
2. **BTC Don4h is better than BTC Don8h as benchmark**: BTC Don4h(50-60,20) S30=1.85-1.89 vs benchmark (30,25) S30=1.705.
3. **BTC inv_vol achieves DSR=1.000**: BTC(60,20)_iv(vw=60-90, tv=0.10-0.15) has DSR=1.000 standalone with MaxDD=-10% to -16%.
4. **Optimal 4h portfolio structure**: BTC-iv 70-75% + ETH-flat 25-30% on 4h grid. Higher BTC weight gives lower MaxDD.
5. **ETH Bollinger 4h null**: Best S30=2.156 flat — dominated by Donchian. Not a competitive standalone.
6. **MaxDD<15% with S30>2.2**: BTC inv_vol (vw≥90) reduces portfolio MaxDD without Sharpe penalty.
7. **Fine-tuning peak**: BTC(60,20)_iv(120,0.15)+ETH(164,47)_75-25 = S30=2.298, S50=2.174, MaxDD=-13.5%.

### Best Candidates Session 7

| Config | S30 | S50 | DSR | MaxDD | S2020 |
|--------|-----|-----|-----|-------|-------|
| BTC(60,20)iv(90,0.15)+ETH(164,47) 75-25 | 2.293 | 2.163 | 0.994 | -13.1% | 1.760 |
| BTC(60,20)iv(120,0.15)+ETH(164,47) 75-25 | 2.298 | 2.174 | 0.995 | -13.5% | 1.766 |
| BTC(60,20)iv(120,0.12)+ETH(164,47) 75-25 | 2.293 | 2.181 | 0.995 | -12.6% | 1.763 |
| BTC(60,20)iv(90,0.15)+ETH(138,47) 75-25 | 2.292 | 2.161 | 0.995 | -14.4% | 1.801 |
| BTC(60,20)iv(120,0.15)+ETH(138,47) 75-25 | 2.297 | 2.172 | 0.995 | -14.9% | 1.807 |

### Next Steps (requires AK decision)

- OOS evaluation — **needs AK approval** (one-shot, existing one-shot budget)
- Walk-forward validation of best 4h portfolio
- Explore 3-way: BTC-iv + ETH Don4h(138) + ETH Don4h(164) at 60-20-20

---

## btc_deep + eth_strategies directives — 2026-02-18/19 — TIER 1 RESULTS (REVISED)

**NOTE**: Session 5 "TIER 1" portfolio S=2.760 was inflated by frequency bug. See Session 7 above for corrected results.

Two parallel directives ran across multiple sessions totaling 933+ cumulative trials.

### Revised Best Result — Portfolio TIER 1

**BTC Don4h(60,20)iv(vw=90,tv=0.15) 75% + ETH Don4h(164,47)flat 25%** — S30=2.293 MaxDD=-13.1% (session 7)

### Best Individual Strategies

| Tag | S30 | S50 | DSR | MDD | S2020 |
|-----|-----|-----|-----|-----|-------|
| BTC_Don4h(160,25)_iv0.15vw30 | 2.319 | 2.107 | 1.000 | -12.5% | 1.20 |
| ETH_Don4h(138,47)_flat | 2.140 | 2.099 | 0.993 | -38.8% | 1.71 |
| BTC_Don4h(60,20)_iv(90,0.15) | 2.093 | 1.807 | 1.000 | -17.8% | 1.086 |
| ETH_Don8h(83,33)_iv30_0.15 | 2.056 | 1.971 | 0.988 | -9.5% | 1.89 |

### Key Discoveries (btc_deep, 3 sessions, 1260 unique configs)

1. **BTC Don4h large entries (120-200)**: Previous sweeps only tested entry≤80. The true optimum is entry=150-170. 198 configs with DSR≥0.999 and S30≥2.0 — massive plateau.
2. **Extreme cost robustness**: All top-5 BTC configs survive 100bps (S@100>1.0). Edge is real.
3. **BTC 2020+**: Requires long lookback (160-period = 27 days on 4h) post-2020. ETH maintains momentum at medium lookback.

### Best Individual Strategies

| Tag | S30 | S50 | DSR | MDD | S2020 |
|-----|-----|-----|-----|-----|-------|
| BTC_Don4h(160,25)_iv0.15vw30 | 2.319 | 2.107 | 1.000 | -12.5% | 1.20 |
| ETH_Don4h(138,47)_flat | 2.140 | 2.099 | 0.993 | -38.8% | 1.71 |
| ETH_Don8h(83,33)_iv30_0.15 | 2.056 | 1.971 | 0.988 | -9.5% | 1.89 |
| ETH_Don4h(140,70)_flat | 2.061 | 2.030 | 0.977 | — | 1.97 |
| ETH_Don8h(72,27)_flat | 2.081 | — | 0.986 | -39.7% | 1.64 |

### Key Discoveries (btc_deep, 3 sessions, 1260 unique configs)

1. **BTC Don4h large entries (120-200)**: Previous sweeps only tested entry≤80. The true optimum is entry=150-170. 198 configs with DSR≥0.999 and S30≥2.0 — massive plateau.
2. **Extreme cost robustness**: All top-5 BTC configs survive 100bps (S@100>1.0). Edge is real.
3. **BTC 2020+**: Requires long lookback (160-period = 27 days on 4h) post-2020. ETH maintains momentum at medium lookback.

### Key Discoveries (eth_strategies, 5 sessions, 933 cumulative trials)

1. **ETH Don4h beats Don8h**: S30=2.140 vs 2.081. Optimal: ep=138, xp=47 (xp/ep≈0.34).
2. **ETH Don8h inv_vol collapses MaxDD**: -39%→-10% at trivial Sharpe cost (2.081→2.056).
3. **Portfolio S30=2.760**: BTC+ETH cross-asset portfolio beats best individual (+0.62) AND slashes MaxDD from -38.8% to -9.8%.
4. **Robust plateaus**: ETH Don4h ep=120-160, xp=48-50 (multiple DSR=0.993 configs); BTC Don4h ep=120-200 plateau.

### Next Steps (requires AK decision)

- OOS evaluation of best portfolio — **needs AK approval** (one-shot)
- Walk-forward validation of `BTC_Don4h(160,25)_iv0.15vw30` and `ETH_Don4h(138,47)_flat`
- ETH Don4h exit=65-80 region (S2020 keeps rising with larger exits — ETH_Don4h(140,70) S2020=1.97)
- BTC+ETH portfolio weight optimization (currently coarse 50/35/15)

---

## Corrected Candidate Rankings — 2026-02-18

**NOTE**: Broad exploration inv_vol results are INVALIDATED (PR #58 sizing bug).
Only Agent A (eth_strategies) and Agent B (btc_deep) results are trustworthy.
Broad exploration inv_vol results must be revalidated before inclusion.

| # | Config | Source | S@30 | S@50 | DSR | MaxDD | 2020+ S | Status |
|---|--------|--------|------|------|-----|-------|---------|--------|
| 1 | btc_don8h(40,10) iv(vw30,tv0.2) | Agent B | 2.045 | 1.817 | 1.000 | -14.2% | 1.087 | VALIDATED |
| 2 | eth_don8h(72,27) flat | Agent A | 2.081 | 2.041 | 0.986 | -39.0% | 1.762 | VALIDATED |
| 3 | eth_don8h(83,30) flat | Agent A | 2.080 | 2.046 | 0.986 | -32.9% | 1.617 | VALIDATED |
| 4 | eth_don8h(85,30) invvol | Agent A | 2.019 | — | 0.979 | -10.4% | — | NEEDS 50bps |
| 5 | eth_don4h(140,48) flat | Agent A | 2.126 | — | 0.993 | — | — | INCOMPLETE |

Remaining in-sample. OOS requires explicit AK approval.

---

## layer4_sizing_donchian_20260218 — Session 1 — 2026-02-18

**DIRECTIVE**: layer4_sizing_donchian_20260218
**STATUS**: SUCCESS at 30bps — 4 configs pass gate. FAIL at 50bps — zero configs pass any criteria
**DATA**: ohlcv_hourly_max_coverage resampled to 4h
**COSTS**: 30 bps standard, 50 bps stress test
**ARTIFACTS**: results/layer4_sizing_donchian_20260218/
**BASE**: meta-labeled Donchian(30,25) 4k, Sharpe 1.981, MaxDD -0.490 unsized

### Sizing families tested (267 configs, 8 families)

| Family | Verdict | Best Sharpe@30 | Best MaxDD | Finding |
|--------|---------|---------------|------------|---------|
| Kelly (calibrated) | FAIL | -2.51 | — | Platt collapses proba spread (std=0.032), positions near-zero |
| Kelly (uncalibrated) | FAIL | -0.10 | -0.28 | Edge too small for Kelly to overcome costs |
| High-confidence filter | FAIL | 1.550 | -0.490 | Fewer trades but same MaxDD profile |
| **Inverse vol** | **SUCCESS** | **1.633** | **-0.248** | tv=0.15 is the boundary; 4 configs pass gate |
| Inv-vol + high conf | FAIL | 1.366 | — | Fewer trades kills Sharpe |
| Regime × inv-vol | FAIL | 1.234 | — | HMM proba near-binary, crushes position size |
| Discrete inv-vol | FAIL | 1.925 | -0.254 | Closest miss — just outside MaxDD gate |
| ATR-based | FAIL | 2.388 | -0.541 | Scales UP in bear markets — counterproductive |
| Vol threshold | FAIL | 1.868 | -0.296 | Best trade-off but fails -0.25 gate |

### Success configs (all inverse vol, tv=0.15)

| vol_window | Sharpe@30 | DSR | MaxDD@30 | Sharpe@50 | 2020-23 Sharpe | 2020-23 MaxDD |
|------------|-----------|-----|----------|-----------|----------------|---------------|
| **60** | **1.633** | **0.992** | **-0.248** | 1.092 | 1.031 | -0.221 |
| 80 | 1.559 | 0.985 | -0.248 | 1.013 | 1.039 | -0.228 |
| 100 | 1.556 | 0.985 | -0.244 | 1.005 | 1.102 | -0.223 |
| 120 | 1.540 | 0.982 | -0.240 | 0.986 | 1.109 | -0.225 |

### Pareto frontier

| Config | Sharpe@30 | MaxDD | Note |
|--------|-----------|-------|------|
| Meta unsized | 1.981 | -0.490 | No sizing |
| vw=30, tv=0.4 | 2.157 | -0.403 | Best Sharpe |
| vw=60, tv=0.20 | 1.817 | -0.273 | Near gate |
| vw=60, tv=0.18 | 1.754 | -0.263 | Near gate |
| **vw=60, tv=0.15** | **1.633** | **-0.248** | **Gate pass** |
| vw=120, tv=0.15 | 1.540 | -0.240 | Best MaxDD in success |

Continuous Pareto frontier — MaxDD=-0.25 constraint cuts at Sharpe~1.63.

### 50bps stress test: FAIL

**Zero configs pass all 3 criteria at 50bps.** Not even MaxDD > -0.25 alone at 50bps.
The 30bps success configs at 50bps:

| vol_window | Sharpe@50 | MaxDD@50 | DSR@50 |
|------------|-----------|----------|--------|
| 60 | 1.092 | -0.339 | 0.727 |
| 80 | 1.013 | -0.342 | 0.632 |
| 100 | 1.005 | -0.343 | 0.620 |
| 120 | 0.986 | -0.338 | 0.594 |

The strategy is only deployable with limit orders and reliable execution infrastructure.
50bps represents market orders, partial fills, exchange downtime forcing delayed execution.
A strategy that passes at 30bps and fails at 50bps has thin margins.

### Key insights

- **Kelly failing is the right outcome.** Meta-model accuracy is 56% — barely above 50%.
  Kelly sizes on probability, and at 56% the Kelly bet is tiny. The meta-labeling edge is
  in trade *filtering* (which trades to take), not probability-based *sizing* (how much to
  bet). Inverse vol doesn't care about the meta-model's probability — it scales with market
  conditions.
- **vw=120/tv=0.15 is most interesting for deployment** despite lowest Sharpe (1.540):
  lowest MaxDD (-0.240), best 2020-2023 Sharpe (1.109), most robust to regime change.
  Longer vol windows smooth out noise and adapt slower — better for real execution.
- **Continuous Pareto frontier** — no sizing method breaks through the MaxDD/Sharpe
  trade-off. The -0.25 constraint cuts the frontier at Sharpe ~1.63.

### Program totals (Layers 3+4)

- 3 sessions (2 meta-labeling + 1 sizing), 627 configs, ~$9 total
- Deployment candidate (30bps only): inv-vol(vw=60, tv=0.15) — Sharpe 1.633, MaxDD -0.248, DSR 0.992
- Requires limit-order execution — no margin for slippage

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
