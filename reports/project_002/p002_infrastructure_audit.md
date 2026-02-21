# P002 Infrastructure Audit Report

**Date**: 2026-02-20
**Auditor**: Oversight Agent (Opus)
**Branch**: `main` (commit `2f316a9`)
**Scope**: Full codebase audit for Project 002 readiness

---

## 1. Executive Summary

**Overall readiness: 5/10**

The codebase has a solid foundation from P001 — data loading, holdout enforcement, metrics computation, DSR, guardrails, block bootstrap, CPCV, and the orchestrator are all production-tested. However, P002 is a fundamentally different strategy family (on-chain regime + GARCH vol targeting + funding rate carry) and the infrastructure gaps are substantial:

**Top 3 blockers:**
1. **Missing `arch` + `statsmodels` packages** — GARCH(1,1) estimation (Layer 0) cannot even be prototyped
2. **No funding rate data pipeline** — no fetcher, no data on disk, no loader alias (Layer 2 dead on arrival)
3. **No rule-based backtest function in library** — `net_ret()` lives in `bin/infra/sweep_utils.py`, not importable by research agents via `from sparky.backtest import ...`

None of these are architectural problems. All are buildable in 2-3 oversight sessions.

---

## 2. Test Suite Health

```
Total:    951 tests
Passed:   945 (99.4%)
Failed:   6
Skipped:  1
Duration: 163.91s
```

### Failed Tests

| Test | Root Cause | Action |
|------|-----------|--------|
| `test_onchain_bgeometrics::test_sync_incremental_from_last_timestamp` | `store.load()` mock returns empty tuple | Fix mock (pre-existing) |
| `test_onchain_bgeometrics::test_sync_calls_append_for_new_data` | Same mock bug | Fix mock (pre-existing) |
| `test_onchain_data::test_coinmetrics_data_structure` | Asserts data to 2026, but holdout truncates at 2023-12-02 | Fix test assertion (pre-existing) |
| `test_onchain_data::test_blockchain_com_data_structure` | Same holdout truncation | Fix test assertion (pre-existing) |
| `test_onchain_data::test_data_completeness` | Expects 8 years but IS data is 7 years | Fix test assertion (pre-existing) |
| `test_macro_features::test_data_coverage` | DXY return coverage 63.3% < 68% threshold | Data staleness (pre-existing) |

All failures are pre-existing. Core logic tests (backtest, statistics, metrics, guardrails, costs, CPCV, bootstrap) are 100% passing.

### Missing Dependencies

| Package | Status | Required For |
|---------|--------|-------------|
| `arch` | **INSTALLED** (8.0.0) — verified on aarch64 | GARCH(1,1) volatility estimation (Layer 0) |
| `statsmodels` | **INSTALLED** (0.14.6) — verified on aarch64 | ACF/PACF, Markov switching, statistical tests |

Both installed successfully via `uv pip install arch`. Still need to be added to `pyproject.toml` for reproducibility.

---

## 3. Module Audit Results

### Category A: Data Layer

| Module | File | Status | P002 Ready | Gaps | Tests |
|--------|------|--------|-----------|------|-------|
| **loader** | `src/sparky/data/loader.py` | EXISTS_TESTED | Partial | No funding rate aliases | 13/13 |
| **price** | `src/sparky/data/price.py` | EXISTS_TESTED | Partial | No `fetch_funding_rate()` | 12/12 |
| **onchain_bgeometrics** | `src/sparky/data/onchain_bgeometrics.py` | EXISTS_TESTED | Partial | No exchange netflow (not on free tier); funding rate is Advanced-only | 23/25 |
| **onchain_coinmetrics** | `src/sparky/data/onchain_coinmetrics.py` | EXISTS_TESTED | Partial | `FlowInExNtv`/`FlowOutExNtv` not in default metric list | 7/7 |
| **onchain_blockchain_com** | `src/sparky/data/onchain_blockchain_com.py` | EXISTS_TESTED | Yes | Validation reference role, no changes needed | 13/13 |
| **source_selector** | `src/sparky/data/source_selector.py` | EXISTS_TESTED | Yes | — | 5/5 |
| **storage** | `src/sparky/data/storage.py` | EXISTS_TESTED | Yes | — | 10/10 |
| **quality** | `src/sparky/data/quality.py` | EXISTS_TESTED | Yes | — | 18/18 |
| **market_context** | `src/sparky/data/market_context.py` | EXISTS_TESTED | Partial | Snapshot-only, no historical time series | 6/6 |
| **funding_rate_fetcher** | MISSING | — | No | Entire module must be built | — |

### Category B: Feature Engineering

| Module | File | Status | P002 Ready | Gaps | Tests |
|--------|------|--------|-----------|------|-------|
| **returns** | `src/sparky/features/returns.py` | EXISTS_TESTED | Partial | No GARCH, no EWMA vol, no vol targeting | 15/15 |
| **technical** | `src/sparky/features/technical.py` | EXISTS_TESTED | Yes | General RSI/EMA/MACD, reusable | 35/35 |
| **onchain** | `src/sparky/features/onchain.py` | EXISTS_TESTED | Partial | Fixed thresholds (MVRV>7, SOPR<1), no adaptive rolling percentile, no composite regime signal, no netflow feature | 15/15 |
| **regime** | `src/sparky/features/regime.py` | EXISTS_TESTED | Partial | Price/volume only, no on-chain regime | 8/8 |
| **regime_indicators** | `src/sparky/features/regime_indicators.py` | EXISTS_TESTED | Partial | Static regime→position_size mapping, not GARCH-based vol targeting | 8/8 |
| **advanced** | `src/sparky/features/advanced.py` | EXISTS_TESTED | Yes | BB, ATR, volume, OBV all reusable | 32/32 |
| **microstructure** | `src/sparky/features/microstructure.py` | EXISTS_TESTED | Yes | Candle patterns reusable | 14/14 |
| **multi_resolution** | `src/sparky/features/multi_resolution.py` | EXISTS_TESTED | Yes | Multi-timeframe framework reusable | ~10 pass |

### Category C: Backtesting and Validation

| Module | File | Status | P002 Ready | Gaps | Tests |
|--------|------|--------|-----------|------|-------|
| **engine** | `src/sparky/backtest/engine.py` | EXISTS_TESTED | No | ML-only (fit/predict), binary positions only, no rule-based path | 12/12 |
| **costs** | `src/sparky/backtest/costs.py` | EXISTS_TESTED | Partial | No maker/taker distinction, no derivatives fee tier | 10/10 |
| **cpcv** | `src/sparky/backtest/cpcv.py` | EXISTS_TESTED | Partial | `test_size` hardcoded to `n_groups//2`, no separate embargo, simplified PBO | 8/8 |
| **statistics** | `src/sparky/backtest/statistics.py` | EXISTS_TESTED | Partial | Missing: permutation test, parameter plateau test, regime decomposition, rule-based walk-forward | ~20 pass |
| **leakage_detector** | `src/sparky/backtest/leakage_detector.py` | EXISTS_TESTED | Partial | Shuffled-label test requires ML model; temporal/index checks are generic | pass |
| **metrics** | `src/sparky/tracking/metrics.py` | EXISTS_TESTED | Yes | All 20 metrics present, DSR accepts n_trials, PPY validation | ~30 pass |
| **guardrails** | `src/sparky/tracking/guardrails.py` | EXISTS_TESTED | Yes | All pre/post checks strategy-agnostic | ~15 pass |
| **net_ret** | `bin/infra/sweep_utils.py` | EXISTS_TESTED | Partial | Works but lives in utility script, not library | 10/10 |

### Category D: Models and Strategies

| Module | File | Status | P002 Ready | Gaps |
|--------|------|--------|-----------|------|
| **simple_baselines** | `src/sparky/models/simple_baselines.py` | EXISTS_TESTED | No | P001 Donchian/SMA/ATR strategies |
| **baselines** | `src/sparky/models/baselines.py` | EXISTS_TESTED | Partial | BuyAndHold/SimpleMomentum reusable as benchmarks |
| **regime_filtered_donchian** | `src/sparky/models/regime_filtered_donchian.py` | EXISTS_TESTED | No | P001 Donchian-specific |
| **regime_hmm** | `src/sparky/models/regime_hmm.py` | EXISTS_TESTED | Partial | HMM regime detection reusable concept, but tied to Donchian |
| **regime_adaptive_lookback** | `src/sparky/models/regime_adaptive_lookback.py` | EXISTS | No | P001 Donchian-specific |
| **regime_markov_switching** | `src/sparky/models/regime_markov_switching.py` | EXISTS | No | P001 Donchian-specific |
| **regime_volatility_term_structure** | `src/sparky/models/regime_volatility_term_structure.py` | EXISTS | Partial | Vol term structure concept reusable, but tied to Donchian signals |
| **regime_weighted_ensemble** | `src/sparky/models/regime_weighted_ensemble.py` | EXISTS | No | P001 Donchian-specific |
| **signal_aggregator** | `src/sparky/models/signal_aggregator.py` | EXISTS_TESTED | Partial | Regime-aware aggregation reusable concept |
| **xgboost_model** | `src/sparky/models/xgboost_model.py` | EXISTS_TESTED | No | ML model, not relevant to P002 |
| **lstm_model** | `src/sparky/models/lstm_model.py` | EXISTS_TESTED | No | ML model, not relevant to P002 |
| **Strategy Protocol** | — | MISSING | No | No formal protocol for rule-based strategies |

**Key finding**: All 11 model files are P001-specific (Donchian breakout variants or ML models). None implement a common strategy interface beyond the ML `ModelProtocol` (fit/predict). P002 needs a `SignalStrategy` protocol that produces position signals directly from data, without fit/predict.

### Category E: Orchestration and Research Management

| Module | File | Status | P002 Ready | Gaps |
|--------|------|--------|-----------|------|
| **orchestrator** | `src/sparky/workflow/orchestrator.py` | EXISTS_TESTED | Yes | YAML directives, session management, stopping criteria, gates |
| **program** | `src/sparky/workflow/program.py` | EXISTS_TESTED | Yes | Multi-phase state machine, coverage tracking |
| **session** | `src/sparky/workflow/session.py` | EXISTS_TESTED | Yes | Claude session launcher, alerts, idle detection |
| **telemetry** | `src/sparky/workflow/telemetry.py` | EXISTS_TESTED | Yes | Stream parsing, behavioral flags |
| **runner** | `src/sparky/workflow/runner.py` | EXISTS_TESTED | Yes | YAML-driven phase execution |
| **experiment tracker** | `src/sparky/tracking/experiment.py` | EXISTS_TESTED | Yes | W&B logging, config hashing, dedup |
| **manager_log** | `src/sparky/tracking/manager_log.py` | EXISTS_TESTED | Yes | JSONL audit trail |
| **activity_logger** | `src/sparky/oversight/activity_logger.py` | EXISTS_TESTED | Yes | Per-session JSONL event logging |
| **holdout_guard** | `src/sparky/oversight/holdout_guard.py` | EXISTS_TESTED | Yes | Tamper-proof OOS boundary enforcement |
| **Pre-registration** | — | MISSING | No | No hypothesis/parameter locking system |
| **Trial counter** | — | MISSING | No | DSR accepts n_trials but no automated counting |
| **Kill protocol** | — | MISSING | No | No automated strategy kill decisions |

**Key finding**: The orchestrator, session management, and experiment tracking are mature and P002-ready. The three missing items (pre-registration, trial counting, kill protocol) are research management features that enforce P002's stricter methodology.

### Category F: Configuration

| Item | File | P002 Ready | Gaps |
|------|------|-----------|------|
| **holdout_policy** | `configs/holdout_policy.yaml` | Yes | BTC + ETH + cross_asset holdout from 2024-01-01 |
| **trading_rules** | `configs/trading_rules.yaml` | Yes | Comprehensive, immutable |
| **data_sources** | `configs/data_sources.yaml` | Partial | Documents derivatives/funding but marked "deferred" |
| **project_001 configs** | `configs/project_001/` | N/A | P001-specific, reference only |
| **Cost config** | Code-only | Partial | Fee tiers hardcoded in `costs.py`, not configurable via YAML |

---

## 4. Gap Analysis by P002 Phase

### Phase 0 — Pre-Research Setup

**Ready:**
- Data loader with holdout enforcement (`loader.py`)
- BTC/ETH OHLCV data (daily + hourly + 2h/4h/8h)
- On-chain metrics: MVRV, SOPR, NUPL, Puell Multiple (BGeometrics)
- CoinMetrics BTC on-chain (2017-2023)
- Data quality checks (`quality.py`)
- Storage with UTC enforcement (`storage.py`)
- W&B experiment tracking (`experiment.py`)
- Orchestrator + phase state machine (`orchestrator.py`, `program.py`)

**Extend:**
- Add `FlowInExNtv`/`FlowOutExNtv` to CoinMetrics default metrics (trivial)
- Add funding rate + derivatives dataset aliases to loader (trivial)
- Add `arch` and `statsmodels` to `pyproject.toml` (trivial)

**Build:**
- **Funding rate data fetcher** — CCXT `fetchFundingRateHistory` for Coinbase/Binance BTC+ETH perps (significant, ~4hr)
- **Pre-registration system** — YAML spec locking hypotheses, parameter ranges, max configs, kill criteria (moderate, ~2hr)
- **Automated trial counter** — append-only JSONL log counting every backtest run, integrated with DSR computation (moderate, ~2hr)

### Phase 1 — Signal Research

**Ready:**
- Technical indicators (RSI, EMA, MACD, momentum)
- Returns computation (simple, log, annualized Sharpe)
- Realized volatility (backward-looking)
- On-chain features: MVRV signal, SOPR signal, NUPL regime, hash ribbon, NVT z-score, Puell signal
- Regime indicators: volatility regime, trend detection, combined regime
- Activity logger for structured research event logging

**Extend:**
- Add adaptive thresholds for on-chain signals (rolling percentile instead of fixed) (moderate, ~2hr)
- Add exchange netflow feature function (`FlowInExNtv - FlowOutExNtv` with z-score) (trivial, ~30min)

**Build:**
- **GARCH(1,1) volatility estimator** — using `arch` library, rolling forecast, parameter stability (significant, ~4hr)
- **EWMA volatility** — exponentially weighted realized vol (trivial, ~30min)
- **On-chain composite regime signal** — combine MVRV + SOPR + netflow → binary long/flat (moderate, ~2hr)
- **Funding rate features** — rolling averages (8h/24h/7d), z-scores, regime classification (moderate, ~2hr)
- **Volatility targeting position sizer** — `position_size = target_vol / forecast_vol`, capped (moderate, ~1hr)

### Phase 2 — Single-Strategy Testing

**Ready:**
- Block bootstrap Monte Carlo (configurable block size, general-purpose)
- DSR calculator (accepts n_trials)
- Sharpe CI (bootstrap)
- Sharpe significance test
- Strategy vs benchmark test
- Full metrics suite (`compute_all_metrics` — 20 keys)
- Pre/post experiment guardrails (strategy-agnostic)
- Leakage detector (temporal + index checks)
- PPY validation (catches cross-timeframe inflation)

**Extend:**
- Promote `net_ret()` + `subperiod_analysis()` from `bin/infra/sweep_utils.py` to `src/sparky/backtest/` (trivial, ~1hr)
- Add maker/taker + derivatives fee tiers to `TransactionCostModel` (moderate, ~2hr)

**Build:**
- **Parameter plateau test** — check Sharpe robustness across parameter neighborhood (trivial, ~1hr)
- **Monte Carlo permutation test** — permute positions, recompute Sharpe, 1000x (moderate, ~2hr)

### Phase 3 — CPCV and Robustness

**Ready:**
- CPCV `cpcv_paths()` — configurable n_groups, computes PBO

**Extend:**
- Add configurable `test_size` (k) parameter to CPCV (moderate, ~2hr)
- Add separate `embargo_days` parameter to CPCV (moderate, ~1hr)
- Improve PBO to use IS/OOS logit comparison per de Prado (moderate, ~2hr)

**Build:**
- **Regime-conditioned performance decomposition** — returns + regime labels → per-regime metrics (moderate, ~2hr)

### Phase 4 — Walk-Forward Portfolio Validation

**Ready:**
- Walk-forward engine exists (but ML-only interface)
- Per-fold metrics computation
- Portfolio WF pattern validated in P001 (BTC 30% / ETH 70%)

**Extend:**
- Add standalone `walk_forward_rule_based()` function for rule-based strategies (moderate, ~3hr)

**Build:**
- Nothing new — the pieces exist, they just need to be assembled for rule-based strategies

---

## 5. Sequencing Recommendation

### Tier 1 — Must complete before P002 YAML is activated (blocks Phase 0-1)

| # | Item | Effort | Dependencies |
|---|------|--------|-------------|
| 1 | Install `arch` + `statsmodels` (add to pyproject.toml) | 10 min | None |
| 2 | Build funding rate data fetcher (CCXT `fetchFundingRateHistory`) | 4 hr | None |
| 3 | Build GARCH(1,1) volatility estimator | 4 hr | `arch` package |
| 4 | Build EWMA volatility + vol targeting position sizer | 1 hr | None |
| 5 | Build on-chain composite regime signal (MVRV + SOPR + netflow → long/flat) | 2 hr | CoinMetrics flow metrics |
| 6 | Add exchange netflow feature + CoinMetrics default metrics update | 30 min | None |
| 7 | Build funding rate features (rolling avg, z-score, regime classification) | 2 hr | Funding rate data |
| 8 | Add adaptive thresholds for on-chain signals (rolling percentile) | 2 hr | None |
| 9 | Promote `net_ret()` + `subperiod_analysis()` to `src/sparky/backtest/` | 1 hr | None |
| 10 | Build pre-registration system (YAML spec + validation) | 2 hr | None |
| 11 | Build automated trial counter (JSONL + DSR integration) | 2 hr | None |
| 12 | Add loader aliases for funding rate datasets | 10 min | Funding rate data |

**Estimated sessions**: 2-3 oversight sessions (parallelize items 1-6, then 7-12)

### Tier 2 — Must complete before Phase 2 starts

| # | Item | Effort | Dependencies |
|---|------|--------|-------------|
| 13 | Add maker/taker + derivatives fee tiers to cost model | 2 hr | None |
| 14 | Build parameter plateau test | 1 hr | None |
| 15 | Build Monte Carlo permutation test | 2 hr | None |

**Estimated sessions**: 1 oversight session (all parallelizable)

### Tier 3 — Must complete before Phase 3 starts

| # | Item | Effort | Dependencies |
|---|------|--------|-------------|
| 16 | Extend CPCV with configurable k + separate embargo | 3 hr | None |
| 17 | Build regime-conditioned performance decomposition | 2 hr | Regime detection |
| 18 | Build standalone walk-forward for rule-based strategies | 3 hr | `net_ret()` in library |

**Estimated sessions**: 1 oversight session

**Total estimated build effort**: 4-5 oversight sessions, ~32 hours of sub-agent work.

---

## 6. Reuse vs. Rebuild Decisions

### Reuse As-Is (no changes)
- **`storage.py`** — DataStore with UTC enforcement, parquet I/O, manifesting
- **`quality.py`** — Data quality checks
- **`source_selector.py`** — On-chain source cross-validation
- **`metrics.py`** — `compute_all_metrics()` with DSR, PPY validation
- **`guardrails.py`** — Pre/post experiment checks
- **`statistics.py::block_bootstrap_monte_carlo()`** — General-purpose, not P001-specific
- **`statistics.py::sharpe_confidence_interval()`** — General-purpose
- **`orchestrator.py` + `program.py` + `session.py`** — Full orchestration stack
- **`experiment.py`** — W&B tracking with config hashing
- **`holdout_guard.py`** — OOS boundary enforcement

### Extend (add P002 support alongside existing)
- **`loader.py`** — Add funding rate dataset aliases to `_DATASET_ALIASES` dict. Existing aliases unchanged.
  - Ref: `src/sparky/data/loader.py:50-95` (`_DATASET_ALIASES` dict)
- **`price.py`** — Add `fetch_funding_rate()` method to `CCXTPriceFetcher`. Existing `fetch_daily_ohlcv()` unchanged.
  - Ref: `src/sparky/data/price.py` class `CCXTPriceFetcher`
- **`onchain_coinmetrics.py`** — Append `FlowInExNtv`/`FlowOutExNtv` to `ASSET_METRICS["btc"]`.
  - Ref: `src/sparky/data/onchain_coinmetrics.py` `ASSET_METRICS` dict
- **`onchain.py`** — Add adaptive threshold variants (`mvrv_signal_adaptive`, `sopr_signal_adaptive`) using rolling percentile. Existing fixed-threshold functions unchanged.
  - Ref: `src/sparky/features/onchain.py:60-85` (current fixed-threshold functions)
- **`costs.py`** — Add `maker_fee_pct`/`taker_fee_pct` params + `coinbase_derivatives()` factory. Backward-compatible: existing `standard()` and `stress_test()` unchanged.
  - Ref: `src/sparky/backtest/costs.py:6-50`
- **`cpcv.py`** — Add `test_size` parameter (default `n_groups // 2` for backward compat) + `embargo_days` parameter.
  - Ref: `src/sparky/backtest/cpcv.py:15` (function signature), line 50 (`test_size = n_groups // 2`)

### Build New
- **`src/sparky/data/funding_rate.py`** — New CCXT-based fetcher for funding rate history. Pattern: follow `price.py` structure (pagination, validation, rate limiting, parquet storage).
- **`src/sparky/features/garch.py`** — New module: GARCH(1,1) estimation, rolling forecast, parameter stability. Depends on `arch` package.
- **`src/sparky/features/vol_targeting.py`** — New module: `position_size = target_vol / forecast_vol` with caps and smoothing.
- **`src/sparky/features/funding_rate_features.py`** — New module: rolling averages, z-scores, regime classification from funding rate data.
- **`src/sparky/features/onchain_regime.py`** — New module: composite regime signal combining MVRV + SOPR + netflow → binary long/flat with adaptive thresholds.
- **`src/sparky/backtest/rule_based.py`** — Promoted `net_ret()` + `subperiod_analysis()` + new `walk_forward_rule_based()`.
- **`src/sparky/tracking/pre_registration.py`** — YAML hypothesis spec, parameter locking, trial counting.

### Do NOT Modify
- **`engine.py`** (`WalkForwardBacktester`) — P002 rule-based strategies will bypass the engine entirely, using `net_ret()` (promoted to library). Modifying the engine for fractional/short positions would be high effort and risk breaking P001 tests with no P002 benefit.
- **P001 model files** — All 11 model files (`regime_filtered_donchian.py`, `regime_hmm.py`, etc.) are P001-specific. Leave them as-is for reference. P002 strategies go in new files.

---

## 7. Risk Register

| # | Risk | Probability | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | **Coinbase funding rate API lacks sufficient history** — Coinbase perps launched Oct 2023. Only ~14 months of data in-sample (pre-2024 holdout). Binance has data from Sep 2019. | High | High | Use Binance funding rate data for backtesting (5+ years). Map to Coinbase fee structure. Accept that Coinbase-specific data is limited. |
| 2 | ~~**`arch` package GPU incompatibility on DGX Spark (aarch64)**~~ — **RESOLVED**. `arch==8.0.0` and `statsmodels==0.14.6` install and import cleanly on aarch64. Verified 2026-02-20. | ~~Medium~~ None | ~~Medium~~ None | N/A — risk eliminated. |
| 3 | **On-chain data gaps during OOS period** — BGeometrics free tier has rate limits (8 req/hr, 15 req/day). Syncing 2024+ data for OOS evaluation could hit limits. | Medium | Low | Data is already cached through 2023. OOS sync can be batched across days. CoinMetrics has no daily limit for community API. |
| 4 | **Research agent time waste on missing infrastructure** — If any Phase 0-1 module is incomplete when P002 launches, research agents will write ad-hoc implementations (low quality, no tests, tech debt). | High | Medium | Complete ALL Tier 1 items before activating P002 YAML. Verify with dry-run: spawn a research agent on Phase 0, confirm it can import everything it needs. |
| 5 | **CPCV k=2 with N=12 gives C(12,2)=66 paths** — combinatorial explosion is manageable, but each path requires Sharpe computation on ~1/6 of data. With hourly data (~40k rows), this is ~7k rows per path × 66 paths. Risk: slow execution or memory issues. | Low | Low | Profile after implementation. 66 paths × 7k rows each is modest. Block bootstrap (1000 simulations) is already heavier and works fine. |
