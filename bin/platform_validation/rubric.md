# Platform Validation Rubric — Engineering QA

This rubric is used by the Platform Validation Agent to review code changes from
an **engineering correctness** perspective. It is complementary to the Research
Validation rubric (which covers statistical methodology). Focus here is on
infrastructure soundness, data plumbing, and system correctness — not quant finance theory.

---

## 0. MANDATORY: Shared Utility Auto-Pass Rule

**ANY script in `scripts/` that imports from a shared utility module (e.g.,
`from sweep_utils import ...` or any module in `bin/infra/`) is AUTOMATICALLY COMPLIANT
with sections 1.1 (signal timing), 1.2 (cost application), 3.1 (n_trials),
and 4.2 (guardrails).**

These scripts delegate all correctness-critical operations to the utility module:
- Signal timing (`positions.shift(1)`) → handled inside `net_ret()`
- Cost deduction → handled inside `net_ret()` and `evaluate()`
- DSR/n_trials → passed inside `evaluate()` to `compute_all_metrics()`
- Guardrails → `run_pre_checks()`, `run_post_checks()`, `log_results()` called inside `evaluate()`
- wandb logging → handled by wrapper functions

**You MUST NOT flag these scripts for missing signal shifting, cost deduction,
guardrails, or n_trials — regardless of whether the code is truncated or you
cannot see the full implementation.** The `from sweep_utils import ...`
statement alone is sufficient proof of compliance. Do NOT reason "I cannot
verify" or "the truncated code doesn't show" — the import IS the verification.

If you flag a script that imports from a utility module for any of these
issues, your review is WRONG and will cause a false positive CI failure.

---

## 1. Backtesting Engine Integrity

### 1.1 Signal Timing

Signals must be computed from data available at decision time (end of bar T), and
applied to the **next** bar's return (T+1). The standard pattern is:
```python
net_returns = signals.shift(1) * price_returns   # signals[T] applied to return[T+1]
```

**HIGH severity:**
- `signal * return` (elementwise) without `.shift(1)` on the signal side — pure lookahead bias
- Feature computed at bar T used as signal applied to return at bar T (same bar)
- `positions * returns` where `positions` is not shifted by at least 1 period

**NOT a violation (common false positives):**
- `target = returns.shift(-1)` — this creates the *label* for supervised learning, not a trading signal
- Shift applied inside a model's `predict()` — the model itself may handle timing internally
- Shift applied inside a shared utility function (e.g., `net_ret()` doing `positions.shift(1)`)
  or signal function (e.g., `rolling().max().shift(1)` inside signal generators). If you only
  see the call site `pos = strategy_fn(prices)` followed by `evaluate(prices, pos, ...)`,
  the shift is inside those functions — do NOT flag the call site.
- Research sweep scripts that call `evaluate()` from `sweep_utils.py` — that function
  handles signal shifting, cost deduction, guardrails, and n_trials internally.

### 1.2 Cost Application

Transaction costs must be deducted **before** any performance metrics are computed.
The `TransactionCostModel.apply(returns, positions)` method handles this.

**HIGH severity:**
- Sharpe or Sortino computed on gross returns (before cost deduction) then reported as net
- `costs.apply()` called after `compute_all_metrics()` in the same backtest loop
- `transaction_costs_bps` config key set to 0, None, or absent
- Using `cost_bps` or `costs_bps` inconsistently (the canonical key is `transaction_costs_bps`)

**NOT a violation:**
- Costs applied inside a shared utility (e.g., `net_ret(prices, positions, cf)` deducts
  costs and returns net returns, then `evaluate()` passes those net returns to
  `compute_all_metrics()`). If the call site is `evaluate(prices, pos, cfg, n, df, 30)`,
  the cost deduction happens internally — do NOT flag the call site.

**MEDIUM severity:**
- Cost model constructed with non-standard values without documented justification
- Round-trip cost claimed as one-way, or vice versa (50 bps per side = 100 bps RT)

### 1.3 Walk-Forward Splits

**HIGH severity:**
- Training set includes any data from the test/validation window
- No gap (embargo) between train and test periods — must be ≥ embargo_days from policy
- `n_splits < 3` for reported walk-forward results (too few to be meaningful)

**MEDIUM severity:**
- Walk-forward results reported without noting how many splits were used
- Different embargo values used in training vs. evaluation

---

## 2. Data Pipeline Integrity

### 2.1 Data Access Patterns

All model training and feature engineering data access MUST go through
`sparky.data.loader.load()`. Raw file access bypasses holdout enforcement.

**HIGH severity:**
- `pd.read_parquet(...)` in `src/sparky/` for anything that feeds model training
- `polars.read_parquet(...)` in `src/sparky/` for model training data
- Direct file path construction to `data/` without going through the loader
- Any reference to `data/.oos_vault/` outside of `loader.py` and `split_holdout_data.py`

**MEDIUM severity:**
- `purpose="analysis"` used in training scripts (should be `purpose="training"`)
- Dataset loaded without specifying `asset=` when asset is known

### 2.2 Timezone and Index Consistency

All DataFrames with time data must have UTC-aware DatetimeIndex.

**HIGH severity:**
- `df.index.tz is None` after data loading without explicit localization
- Mixing tz-aware and tz-naive timestamps in the same operation
- `pd.Timestamp("2024-01-01")` (tz-naive) used for date comparisons against tz-aware index

**MEDIUM severity:**
- Timezone converted to non-UTC without documented reason
- DatetimeIndex sorted descending (data is expected ascending chronologically)

### 2.3 NaN and Data Quality

**HIGH severity:**
- Features passed to model.fit() without checking for NaN (can silently corrupt XGBoost, etc.)
- Forward-filled NaN in features that crosses the train/test boundary
- Rolling window features used without accounting for the warmup period (first N rows are NaN)

---

## 3. wandb Parameter Flow

### 3.1 Required Logging Keys

The orchestrator reads specific keys from wandb run summaries. Using the wrong key names
causes silent failures where the orchestrator shows "N/A" for all metrics.

**HIGH severity:**
- Logging `best_sharpe` or `best_dsr` as the primary keys instead of `sharpe` and `dsr`
- Nesting metrics under a sub-dict (e.g., `wandb.log({"metrics": {"sharpe": 1.2}})`) —
  the orchestrator reads top-level summary keys only
- `compute_all_metrics()` result not logged to wandb at all for significant experiments

**MEDIUM severity:**
- `n_trials` not passed to `compute_all_metrics()` — disables multiple testing correction
  and makes DSR ≈ PSR (meaningless). Note: `n_trials` is a parameter of
  `compute_all_metrics()`, NOT `run_post_checks()`. These are different functions with
  different parameters. `run_post_checks()` takes `n_trades` (trade count for minimum
  trades check). Do not confuse parameters across different functions.
- Session tags (`session_NNN`) not included in wandb run tags
- Directive tags not propagated to wandb runs

### 3.2 Config Logging

wandb run configs should capture all hyperparameters needed to reproduce the experiment.

**MEDIUM severity:**
- Config logged without `transaction_costs_bps` (costs are a hyperparameter)
- Model hyperparameters not logged (prevents reproducibility)
- `config_hash` not computed for deduplication (may cause redundant experiments)

---

## 4. Infrastructure and API Correctness

### 4.1 TransactionCostModel

**HIGH severity:**
- Using `TransactionCostModel()` default constructor for research (defaults are legacy values,
  not the standard cost assumption — always use `.standard()` or explicit values)
- Accessing `cost_model.total_cost_pct` as if it is the round-trip cost — it is one-way.
  Round-trip is `cost_model.round_trip_cost` (= 2 × total_cost_pct)
- Computing costs as `n_trades * cost_bps` without accounting for position size changes
  (e.g., going from long to short is a change of 2, not 1)

**MEDIUM severity:**
- Per-trade cost vs. per-dollar cost confusion

### 4.2 Guardrails Usage

**HIGH severity:**
- `run_pre_checks()` result checked but `has_blocking_failure()` not called before proceeding
- `run_pre_checks()` not called at all in a new backtest/training script in `src/sparky/`
- `log_results()` never called (guardrail outcomes not persisted for audit)

**MEDIUM severity:**
- Guardrail results silently ignored (checked but result not acted upon)

### 4.3 Holdout Guard

**HIGH severity:**
- `HoldoutGuard` instantiated but `check_data_boundaries()` never called
- `authorize_oos_evaluation()` called without `approved_by` containing a human identifier
- `purpose="oos_evaluation"` used in training or sweep code (OOS is one-shot, human-gated)

### 4.4 Timeout Decorator

**MEDIUM severity:**
- New training functions in `src/sparky/` without `@with_timeout()` decorator
- Timeout set above 900 seconds (15 min) without documented justification

---

## 5. Testing Coverage

### 5.1 New Infrastructure Functions

**HIGH severity:**
- New public functions in `src/sparky/backtest/`, `src/sparky/data/`, or
  `src/sparky/tracking/` with zero test coverage — these are load-bearing infrastructure
- New guardrail checks added without a corresponding test that verifies the check fires

**MEDIUM severity:**
- New CLI commands in `bin/sparky` without a test
- New configuration parameters with no test verifying they are read correctly
- `try/except Exception` blocks that swallow errors without a test for the error case

### 5.2 Integration Paths

**MEDIUM severity:**
- New data loading path not exercised by at least one test (even with mock data)
- New wandb logging call not tested (even with a mocked tracker)

---

## 6. Configuration and Schema Consistency

### 6.1 YAML Config Consistency

**HIGH severity:**
- New cost values in `configs/trading_rules.yaml` without corresponding update to
  `TransactionCostModel` — the code and config must agree
- Holdout boundary changed in any config file (immutable — requires human approval)
- `oos_start` date modified in `holdout_policy.yaml`

**MEDIUM severity:**
- New directive YAML missing required fields (`name`, `objective`, `wandb_tags`,
  `stopping_criteria`)
- `wandb_tags` in directive does not include the directive name as a tag

### 6.2 API Key Consistency

**MEDIUM severity:**
- Config dict uses `costs_bps` instead of canonical `transaction_costs_bps`
- `model_type` and `strategy_family` used interchangeably without a mapping
  (orchestrator context builder reads `strategy_family` preferring over `model_type`)

---

## What NOT to Flag

- General Python code style (handled by ruff)
- Import ordering (handled by ruff isort)
- Statistical methodology correctness (handled by research-validation agent)
- Security vulnerabilities (handled by bandit)
- Performance optimizations not affecting correctness
- pandas vs. polars choice (polars is preferred but not enforced)
- Comments or docstrings (unless they document incorrect behavior)
- Test helper code that intentionally uses simplified patterns

### Parameter Name Verification Rule

Before flagging a keyword argument as incorrect, you MUST verify the actual function
signature. Different functions may use similar-sounding parameter names for different
purposes. Do NOT assume a parameter name is wrong because a different function uses a
different name for a related concept.

**Concrete API reference (authoritative):**
- `compute_all_metrics(returns, n_trials=...)` — `n_trials` = number of configs tested (for DSR correction)
- `run_post_checks(returns, metrics, config, n_trades=...)` — `n_trades` = number of trades (for minimum trades check)
- `run_pre_checks(data, config, min_samples=...)` — `min_samples` = minimum row count

These are DIFFERENT parameters for DIFFERENT purposes. `n_trades` passed to
`run_post_checks()` is correct usage, not a typo for `n_trials`.
