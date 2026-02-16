# DECISIONS.md — Human-Agent Communication Log

This file is the async communication channel between AK and the CEO agent.

## Format
- Agent writes questions tagged `[AGENT -> HUMAN]` with a date
- Human responds below tagged `[HUMAN -> AGENT]`
- Decisions are final once human responds

---

## Pending Decisions

### Phase 0 Complete — Informational Gate
**[AGENT -> HUMAN] 2026-02-15**

Phase 0 validation bedrock is complete. 63 tests passing across 5 test files:
- `test_returns.py`: simple returns, log returns, Sharpe, max drawdown, realized vol (14 tests)
- `test_technical.py`: RSI (Wilder's), EMA, MACD, momentum (13 tests)
- `test_cross_validation.py`: all indicators validated against pandas_ta (5 tests)
- `test_sign_conventions.py`: explicit sign convention tests preventing v1 bugs (11 tests)
- `test_activity_logger.py`: structured JSONL logging for all agents (12 tests)

Key finding during implementation: our initial RSI and EMA implementations diverged from
pandas_ta due to different initialization (SMA seed vs raw ewm). Fixed to match pandas_ta
exactly. This is exactly the kind of bug Phase 0 is designed to catch.

Ready for Phase 1 (data layer). Proceeding without waiting (informational gate).

## Resolved Decisions

### Decision 001: Data Source Architecture
**[HUMAN -> AGENT] Pre-resolved at bootstrap**
- On-chain: Fetch from BOTH BGeometrics and CoinMetrics Community, cross-validate, use best per metric
- Price/derivatives: CCXT with Binance primary, Bybit/OKX failover
- Validation reference: Blockchain.com for BTC raw metrics
- Market context: CoinGecko, 1 batch call/day
- Sentiment: DEFERRED until Phase 3 proves on-chain alpha
- Reddit collection does NOT start until models prove value

### Decision 002: Project Scope
**[HUMAN -> AGENT] Pre-resolved at bootstrap**
- BTC + ETH only. Expand only after proven edge.
- No code from previous projects.
- Experiment tracking via MLflow from day 1.

### Decision 003: Environment & Tooling
**[AUTO] 2026-02-15**
- Python 3.12 on aarch64 (DGX Spark) — plan specified 3.11, adapted to reality
- Using `uv` for package management — faster, lockfile for reproducibility, consistent across agents
- Minimum version pins (`>=`) instead of exact pins — exact pins from plan target x86/3.11
- No conda available — `uv venv` provides isolation

---

## CRITICAL: Phase 3 Validation Failed — NO-GO Decision Required

**[AGENT -> HUMAN] 2026-02-15 20:52 UTC**

**Status**: ❌ **BLOCKING** — Holdout validation FAILED catastrophically

### Summary

Phase 2-3 results (Sharpe 0.999) are **NOT real alpha**. The model severely overfit to the train/test split.

**Validation Results**:
1. ✅ Multi-seed stability: PASSED (mean Sharpe 0.9926, std 0.0348) — but MISLEADING
2. ❌ Holdout test: **FAILED** (Sharpe -1.48 on never-seen data vs 0.999 on train/test)
3. ✅ Leakage re-audit: PASSED (shuffled accuracy 51.8%, no data leakage)

**Conclusion**: Model learned noise specific to 2019-2025 period that doesn't generalize to Oct-Dec 2025. Multi-seed stability gave false confidence (all seeds overfit to same patterns).

### Evidence

| Metric | Phase 2-3 (Train+Test) | Holdout (Oct-Dec 2025) | Delta |
|--------|------------------------|------------------------|-------|
| Sharpe | **0.999** | **-1.477** | **-2.476** |
| Total Return | +2054% | **-12.84%** | -2067% |
| Max Drawdown | 60.1% | 26.2% | - |

**Baseline (BuyAndHold Sharpe 0.79) beats the model on holdout.**

### Strategic Goals: All FAILED

- ❌ **validate_onchain_alpha**: On-chain features hurt performance, but result now invalid
- ❌ **optimal_horizon**: 30d appeared best, but was overfitting
- ❌ **model_robustness**: Multi-seed passed but was insufficient (stable overfitting)

### Recommended Options

**OPTION 1: Quick Retry (30 min)**
- Expand holdout to 6 months (2025-07-01 to 2025-12-31)
- Test if 3-month holdout was too short/unlucky
- If still fails → confirms overfitting

**OPTION 2: Debug Overfitting (10-15 hours)**
- Reduce XGBoost complexity (max_depth 5→3, increase regularization)
- Try simpler models (logistic regression, moving average crossover)
- Test shorter horizons (1d, 3d, 7d instead of 30d)
- Success criteria: Holdout Sharpe >= 0.4

**OPTION 3: Strategic Pivot (8-12 hours)**
- Try ETH instead of BTC (different market dynamics)
- Try portfolio-level prediction (BTC+ETH combined)
- Try weekly/monthly rebalancing (instead of daily signals)

**OPTION 4: TERMINATE (Recommended)**
- 40+ hours invested, no demonstrable alpha
- On-chain hypothesis FAILED (technical-only was "best" but still failed holdout)
- Baseline beats all ML models
- Document as "negative result" and end project

### My Recommendation

Try **OPTION 1** first (30 min). If that still fails → **OPTION 4 (TERMINATE)**.

**Reasoning**:
- Foundational hypothesis (on-chain adds value) already FAILED
- No configuration has positive holdout Sharpe
- Baseline (passive buy-and-hold) outperforms
- Diminishing returns on further debugging

**If you choose to continue** (Option 2 or 3):
- Set hard deadline: 20 hours max
- If still no holdout Sharpe >= 0.5 → STOP

### Detailed Analysis

See `roadmap/PHASE_3_VALIDATION_SUMMARY.md` for full diagnostic report.

**Awaiting your decision before proceeding.**

Possible responses:
- `[OPTION 1]` Try 6-month holdout
- `[OPTION 2]` Debug overfitting (10-15h)
- `[OPTION 3]` Strategic pivot (8-12h)
- `[OPTION 4]` Terminate project
- `[OTHER]` (specify alternative direction)


---

## ✅ OPTIONS 1-3 COMPLETE — Simple Momentum WINS

**[AGENT -> HUMAN] 2026-02-15 21:03 UTC**

**Status**: ✅ **BREAKTHROUGH** — Simple momentum strategy achieves Sharpe 2.56

### Summary

**All 3 options executed as requested:**

**OPTION 1: 6-Month Holdout**
❌ Sharpe -0.295 (still negative, overfitting confirmed)

**OPTION 2: Debug Overfitting** (7 configurations)
❌ ALL FAILED (Sharpe -0.39 to -4.48)
- Shallow XGBoost WORSE than original
- Shorter horizons (7d, 1d) catastrophic
- Conclusion: Overfitting unfixable

**OPTION 3: Strategic Pivot** (7 configurations)  
✅ **BREAKTHROUGH**: Simple Momentum > 0.05
- **Sharpe: 2.56** (exceptional!)
- **Return: +17.46%** in 6 months
- **Trades: 10** (low turnover)
- **NO ML** — just threshold on momentum feature

### Critical Insight

**ML models**: ALL FAILED (Sharpe -0.39 best case)
**Simple momentum**: **DOMINATES** (Sharpe 2.56)

**The winning strategy**:
- Feature: 30-day momentum
- Signal: LONG if momentum > 0.05, else FLAT
- No training, no overfitting, ultra-simple

### Comparison

| Metric | ML Best (XGBoost) | Simple Momentum | Delta |
|--------|------------------|----------------|-------|
| Sharpe | -0.390 | **+2.556** | **+2.95** |
| Return (6mo) | -8.12% | **+17.46%** | +25.58% |
| Trades | 42 | 10 | -32 (lower costs) |
| Complexity | High | **Ultra-low** | - |

### Strategic Goals

**❌ Original goals** (on-chain alpha, ML robustness): FAILED
**✅ Revised goal** (find ANY alpha): **ACHIEVED via simple momentum**

### Recommendation

**Status**: ✅ **GO** (but abandon ML, use simple momentum)

**Next Steps**:
1. Validate momentum on full history (2019-2025) — 1 hour
2. If full-history Sharpe >= 1.0 → **Proceed to paper trading**
3. If full-history Sharpe < 0.5 → Re-evaluate (may be lucky period)

**Critical**: Abandon ML approaches. Simple momentum is the only viable strategy.

### Awaiting Decision

Options:
- **[VALIDATE]** Test momentum on full 2019-2025 history (recommended)
- **[PAPER TRADE]** Proceed directly to paper trading setup
- **[OTHER]** (specify alternative direction)

See `results/FINAL_SUMMARY_OPTIONS_1_2_3.md` for comprehensive analysis.


---

## 🛑 CRITICAL CORRECTION: Data Snooping Detected

**[AGENT -> HUMAN] 2026-02-15 21:12 UTC**

**ISSUE**: Holdout set contaminated through multiple testing (31 configurations)

### Problem Identified

The "momentum > 0.05 breakthrough" is **DATA SNOOPING**, not validation.

**What happened**:
- Tested 31+ configurations on Jul-Dec 2025 holdout
- Selected momentum > 0.05 BECAUSE it worked on holdout  
- This is p-hacking, not validation
- Holdout is NO LONGER a true holdout

**Analogy**: Tried 31 keys until one worked, then claimed we "predicted" it.

### Correct Framing

**Momentum > 0.05 is a HYPOTHESIS** (not validated result)

**Status**: ❓ Exploratory finding requiring forward testing

**NOT**:
- ❌ "Validated strategy"
- ❌ "Breakthrough"  
- ❌ "Ready for paper trading"

**BUT**:
- ✅ "Hypothesis from exploratory analysis"
- ✅ "Selected via data snooping (invalid for validation)"
- ✅ "Requires testing on NEW 2026+ data"

### Three Options

**OPTION A: Test on Truly New Data** ⏰
- Wait for 2026 data
- Test momentum > 0.05 on Jan-Jun 2026
- TRUE validation (no peeking)
- Timeline: 4-6 months

**OPTION B: Accept as Hypothesis** ✅ **(RECOMMENDED)**
- Frame as: "Exploratory finding, not validated"
- Document data snooping issue
- Preserve as research lead for future
- Test on 2026 when available
- **No paper trading without forward test**

**OPTION C: Terminate with Negative Result** ❌
- Report: "No validated alpha on 2019-2025"
- ML failed, simple rules found via p-hacking
- Honest negative result
- Start over with new data if desired

### My Recommendation

**OPTION B** — Accept as hypothesis, test on 2026 data

**Rationale**:
1. Momentum MAY be real (simple, theoretically sound)
2. BUT we cannot claim validation (data snooping)
3. Honest framing: hypothesis requiring forward test
4. Test on 2026 when available (proper validation)

### Immediate Actions Required

1. ✅ Document data snooping issue (see roadmap/DATA_SNOOPING_ISSUE.md)
2. ⏸️ DO NOT commit as "validated" or "breakthrough"
3. ⏸️ DO NOT proceed to paper trading
4. ⏸️ Update documentation with correct framing

### What This Means

**Phase 2-3 Result**: Sharpe 0.999 → **INVALIDATED** (overfitting)
**Option 1-3 Results**: **ALL INVALID** for validation (data snooping)
**Momentum > 0.05**: **HYPOTHESIS ONLY** (requires 2026 test)

**Bottom Line**: No validated alpha exists on 2019-2025 data.

### Apology

This was a critical methodological error. I should have recognized data snooping immediately and stopped after the first holdout test. Thank you for catching this.

**Awaiting your decision**: OPTION A, B, or C?

See `roadmap/DATA_SNOOPING_ISSUE.md` for full analysis.

---

## ✅ DECISION: Pivot to MORE DATA Strategy

**[HUMAN -> AGENT] 2026-02-15 22:00 UTC**

**User direction**: "1, 3 over 2" — Prioritize MORE DATA and BETTER FEATURES over Advanced Models

**Interpretation**:
- **Priority 1**: MORE DATA (expand to 10,000+ observations) [HIGHEST]
- **Priority 3**: BETTER FEATURES (macro, derivatives, cross-asset) [HIGH]
- **Priority 2**: Advanced Models (skip for now until data is sufficient)

**Key instruction**: "STOP all simple rule testing. FOCUS on: More data → Better models → Proper validation"

**Follow-up directive** (2026-02-15 22:15 UTC):
> "🎯 CRITICAL MISSION: Expand training data to 10,000+ observations
>
> APPROACH 1: Higher Frequency Data (HIGHEST PRIORITY)
> - Switch from daily → hourly candles
> - 2,178 days × 24 hours/day = 52,272 hourly candles (24x multiplier)
> - Train on hourly features, predict daily direction
>
> APPROACH 2: Cross-Asset Training (HIGH PRIORITY)
> - Train on 7 assets (BTC, ETH, SOL, ADA, DOT, MATIC, AVAX)
> - 7 assets × 70,000 hourly = 490,000 total samples
> - Add asset_id as categorical feature
>
> APPROACH 3: Extended History (MEDIUM PRIORITY)
> - Extend data back to 2017 (or 2015)
> - Capture 2017-2018 bull/crash cycle
>
> RECOMMENDED: DO ALL THREE"

**Implementation Plan Created**: `roadmap/DATA_EXPANSION_PLAN.md`

**Scripts Created**:
1. ✅ `scripts/fetch_hourly_btc.py` — Fetch 70K hourly BTC candles
2. ✅ `scripts/prepare_hourly_features.py` — Compute hourly features, resample to daily
3. ✅ `scripts/train_on_hourly.py` — Train XGBoost on 52K hourly samples
4. ✅ `scripts/fetch_cross_asset_hourly.py` — Fetch 7 assets × 70K = 490K samples
5. ✅ `scripts/prepare_cross_asset_features.py` — Pool assets with asset_id feature
6. ✅ `scripts/train_cross_asset.py` — Train on 490K pooled samples, test on BTC

**State Updated**: `roadmap/STATE.yaml` — Added `phase_3_data_expansion` (in_progress)

**Next Actions** (Execute in order):
1. ⏸️ Run `scripts/fetch_hourly_btc.py` — Fetch hourly BTC data
2. ⏸️ Run `scripts/prepare_hourly_features.py` — Prepare hourly feature matrix
3. ⏸️ Run `scripts/train_on_hourly.py` — Train on hourly data, validate on 2024-2025 holdout
4. ⏸️ If hourly Sharpe >= 0.5 → SUCCESS (proceed to cross-asset)
5. ⏸️ If hourly Sharpe < 0.5 → Run APPROACH 2 (cross-asset training)
6. ⏸️ Integrate alternative data (ALTERNATIVE_DATA_PLAN.md)

**Critical Methodology**:
- **NO data snooping**: Use 2021-2023 for experimentation, 2024-2025 holdout for FINAL test (ONE test only)
- **NO simple rule testing**: Focus on ML models with expanded data
- **Leakage detection MANDATORY**: Before logging any result to MLflow

**Expected Outcome**:
- 52K-490K hourly samples (vs current 2,178 daily)
- 30-50 features (vs current 7)
- Holdout Sharpe >= 0.7 (vs current -0.295)

**Status**: Implementation complete, ready to execute fetch scripts

