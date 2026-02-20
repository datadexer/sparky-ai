# Sparky AI Unified Code Review Rubric

This rubric is the knowledge base for the Sparky AI Code Review Agent.
It covers **both** quantitative research methodology and engineering correctness,
replacing the separate research-validation and platform-validation rubrics.

Severity levels:
- **HIGH (Critical)**: Blocks commit. Produces incorrect results, inflated metrics,
  data leakage, system failure, or security breach.
- **MEDIUM (Major)**: Should fix. Misleading results, fragile methodology, reliability
  risk, or maintainability concern.
- **LOW (Minor)**: Informational. Best practice suggestions that do not affect correctness.

---

## 0. MANDATORY: Shared Utility Auto-Pass Rule

**ANY script in `scripts/` that imports from a shared utility module (e.g.,
`from sweep_utils import ...` or any module in `bin/infra/`) is AUTOMATICALLY COMPLIANT
with the following checks:**

**Research checks auto-passed:**
- Section 1.1 (DSR/n_trials) — handled inside `evaluate()` via `compute_all_metrics()`
- Section 2.1 (signal timing) — handled inside `net_ret()` via `positions.shift(1)`
- Section 2.2 (transaction costs) — handled inside `net_ret()` and `evaluate()`
- Section 3.2 (sub-period reporting) — handled via `subperiod_analysis()`
- Section 7.2 (guardrails) — `run_pre_checks()`, `run_post_checks()`, `log_results()` called inside `evaluate()`

**Engineering checks auto-passed:**
- Signal timing (shift applied inside `net_ret()`)
- Cost application (inside `net_ret()` and `evaluate()`)
- DSR/n_trials (passed inside `evaluate()` to `compute_all_metrics()`)
- Guardrails (inside `evaluate()`)
- wandb logging (handled by wrapper functions)

**You MUST NOT flag these scripts for any of the above issues — regardless of whether
the code is truncated or you cannot see the full implementation.** The
`from sweep_utils import ...` statement alone is sufficient proof of compliance.
Do NOT reason "I cannot verify" or "the truncated code doesn't show" — the import IS
the verification. If you flag a utility-delegating script for these issues, your review
is WRONG and will cause a false positive CI failure.

---

## 1. STATISTICAL METHODOLOGY

### 1.1 Multiple Testing Correction (HIGH)

Any comparison of more than one strategy configuration MUST use the Deflated Sharpe
Ratio (DSR) with the correct `n_trials` parameter. The `n_trials` value must reflect
ALL trials ever run across the entire research program, not just the current batch.

- DSR < 0.95 means the result could be a fluke due to multiple testing.
- Claims of "alpha," "significant," "genuine edge," or "outperforms" without DSR > 0.95
  are HIGH severity violations.
- Reporting raw Sharpe as the primary metric without DSR context is HIGH severity.
- Using `n_trials` less than the true cumulative number of configurations tested is HIGH
  severity (underestimates the multiple testing penalty).
- Failed strategy configurations MUST be included when counting `n_trials`.

**Exemptions (NOT a HIGH violation):**
- Scripts that call `compute_all_metrics()` with DSR and correctly pass cumulative
  `n_trials` — the requirement is that DSR is computed and reported, not that DSR > 0.95.
- Single-config exploratory scripts (no sweep) running DSR with cumulative `n_trials`
  from the research program. They do not need `run_pre_checks`/`run_post_checks` if
  results are clearly labeled non-primary and the canonical sweep tool is used for final decisions.
- Scripts that delegate to shared utility functions (Section 0).

Reference: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014).

### 1.2 Cross-Validation on Time Series (HIGH)

Time-series data MUST use purged cross-validation with an embargo period. Standard
k-fold CV introduces lookahead bias because future data leaks into training folds.

- Standard `sklearn.model_selection.KFold` or `StratifiedKFold` on time-series = HIGH.
- `TimeSeriesSplit` without embargo = MEDIUM.
- `TimeSeriesSplit(gap=N)` with `gap >= label_horizon` IS a valid embargo implementation.
  The `gap` parameter excludes N samples between train and test, functionally equivalent
  to purged CV for boundary leakage prevention.
- For hourly crypto data: minimum 24–72 hour embargo between train and test folds.
- Combinatorial purged CV (CPCV) is the gold standard; `TimeSeriesSplit` with gap is acceptable.

**Small dataset / unit test exception:** When n < 1,000 samples (e.g., synthetic test
data), sklearn raises ValueError if `gap + test_size > n // n_splits`. Proportionally
scaling the gap via `min(label_horizon, fold_size // 3)` to prevent this error is
ACCEPTABLE for small-dataset contexts ONLY. For production data (n > 10,000), the
computed fold_size always satisfies `fold_size // 3 > label_horizon`, so the full
embargo is applied automatically. A gap slightly below label_horizon on test-sized data
is NOT HIGH severity if the code comment documents the scaling rationale and
production-data correctness is verified.

Reference: Lopez de Prado, "Advances in Financial Machine Learning" (2018), Ch. 7.

### 1.3 Sharpe Ratio Calculation (HIGH)

Annualization factors for crypto (24/7/365 markets):
- **Hourly data**: multiply by `sqrt(8760)` (8,760 hours per year).
- **Daily data**: multiply by `sqrt(365)` (365 days per year).
- Using `sqrt(252)` (equity trading days) for crypto is WRONG and inflates Sharpe by
  ~20% (`sqrt(365/252) = 1.203`). This is HIGH severity.

Additional requirements:
- Use `ddof=1` (sample standard deviation), not `ddof=0`.
- Risk-free rate is typically 0 for crypto.
- Sharpe MUST be computed on net-of-cost returns, not gross returns.
- If both gross and net are reported, net must be clearly labeled as primary.

Formula: `SR = (mean(r) - rf) / std(r, ddof=1) * sqrt(periods_per_year)`

### 1.4 PSR/DSR Formula Correctness (HIGH)

The PSR standard error formula uses the OBSERVED Sharpe Ratio (`SR_hat`), not the
benchmark (`SR*`):

```
SE = sqrt((1 - gamma_3 * SR_hat + (gamma_4 - 1)/4 * SR_hat^2) / (T - 1))
```

Where:
- `gamma_3` = skewness of returns
- `gamma_4` = **raw Pearson kurtosis** (normal = 3, NOT excess kurtosis where normal = 0)
- `T` = number of observations
- `SR_hat` = observed sample Sharpe ratio

Common errors (all HIGH):
- Using excess kurtosis (normal = 0) instead of raw Pearson kurtosis (normal = 3).
- Using `SR*` (benchmark) instead of `SR_hat` (observed) in the SE formula.
- Assuming Gaussian returns (ignoring skewness and kurtosis). Crypto returns have
  significant skew and fat tails — the Gaussian assumption massively underestimates SE.
- Using `T` instead of `T-1` in the denominator.

DSR haircut formula:
```
SR* = sigma_SR * ((1-gamma) * Phi_inv(1 - 1/N) + gamma * Phi_inv(1 - 1/(N*e)))
```
where `gamma` = Euler-Mascheroni constant (~0.5772), `N` = number of trials.

Reference: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014).

### 1.5 Stationarity (HIGH)

Non-stationary features used as ML inputs produce spurious correlations.

- Price levels as raw features = HIGH (non-stationary).
- Use returns, log-returns, or fractional differentiation (`d < 1` to preserve memory).
- Check with ADF or KPSS test. ADF p-value > 0.05 indicates likely non-stationarity.
- Rolling z-scores and percentage changes are acceptable stationarity transforms.
- Volume should also be normalized (e.g., relative to rolling average).

### 1.6 Survivorship and Selection Bias (MEDIUM)

- Analyzing only BTC/ETH (survivors) introduces survivorship bias if generalizing to
  all crypto strategies.
- Failed strategy configs MUST be included in `n_trials` count.
- Cherry-picking the best walk-forward window or parameter set without accounting for
  the full search space is selection bias.
- Reporting only the best fold of cross-validation instead of the mean is selection bias.

---

## 2. BACKTESTING METHODOLOGY

### 2.1 Lookahead Bias (HIGH)

Signal at time T must only use data available BEFORE time T. This is the single most
common source of inflated backtest results.

**HIGH severity patterns:**
- `signal[T] * return[T]` where signal uses `close[T]`. Correct: `signal.shift(1) * return`.
- Full-dataset normalization before train/test split (e.g., `StandardScaler().fit_transform(all_data)`).
  Must fit scaler on training data only, then transform test.
- Rolling window calculations that include future data (centered MAs instead of trailing).
- Feature engineering on the entire dataset instead of expanding/rolling window.
- Using `pd.read_parquet()` directly instead of `sparky.data.loader.load()` (bypasses holdout guard).

**NOT lookahead (common false positives):**
- `target = (returns.shift(-1) > 0).astype(int)` — This creates the forward-looking LABEL
  for supervised learning, not a trading signal. Using shift(-1) on `y` is correct ML practice.
  The signal generated at inference time uses ONLY past features. The backtest must still use
  `signal.shift(1)` when computing strategy returns.
- Donchian channel signals using `rolling(window).max()` (trailing window) — NOT lookahead
  as long as the signal is shifted before backtest. The anti-lookahead check is whether the
  BACKTEST applies `signal.shift(1)`, not whether the signal itself looks back.
- Shift applied inside a shared utility function or model's `predict()` method. If you only
  see `pos = strategy_fn(prices)` followed by `evaluate(prices, pos, ...)`, the shift is
  inside those functions — do NOT flag the call site.

Historical note: A lookahead bug (`signal[T] * return[T]`) was found in PR #12 on
2026-02-16, inflating Sharpe by 43–256%. All code must use `signal.shift(1) * return`.

### 2.2 Transaction Costs (HIGH)

Two-tier cost model:
- **Standard: 30 bps per side** (60 bps round trip) — Coinbase limit orders / DEX on L2
- **Stress test: 50 bps per side** (100 bps round trip) — Coinbase market orders worst case

The guardrail enforces minimum 30 bps per side.

- Any `transaction_costs_bps` below 30 = HIGH.
- Missing `transaction_costs_bps` in config = HIGH.
- Cost set to 0 or costs not applied = HIGH.
- Cost application AFTER Sharpe calculation = HIGH (must be BEFORE).
- Costs applied inside shared utilities are NOT violations (Section 0).
- Winners should report both 30 bps and 50 bps. Only one cost level = MEDIUM.
- `transaction_costs_bps` is the canonical key — `costs_bps` or `cost_bps` = MEDIUM.

### 2.3 Holdout Integrity (HIGH)

The OOS boundary and embargo buffer are defined in `configs/holdout_policy.yaml` and
are IMMUTABLE. Do NOT hardcode specific dates.

- Code MUST use `sparky.data.loader.load(purpose="training")`.
- Raw `pd.read_parquet()` or `polars.read_parquet()` for model training = HIGH.
- Hardcoding OOS boundary dates (e.g., `"2024-01-01"`) instead of reading from
  `HoldoutGuard` or `configs/holdout_policy.yaml` = HIGH.
- Directly accessing `data/.oos_vault/` or `data/holdout/` = HIGH. OOS data must
  ONLY be accessed via `load(purpose="evaluation")` (env-var gated) or the deprecated
  `load(purpose="oos_evaluation", oos_guard=guard)`.
- Modifying `configs/holdout_policy.yaml` = HIGH. Immutable — human approval only.
- OOS evaluation requires explicit written approval from AK. Each model gets exactly
  ONE OOS evaluation. Repeated peeking and re-tuning = data snooping.

### 2.4 Position Sizing Realism (MEDIUM)

- No unlimited position size. Positions must be bounded.
- Single position < 10% of portfolio (unless explicitly justified for concentrated strategies).
- Market impact must be considered for positions > $100K notional.
- Kelly criterion, if used, should be fractional (half-Kelly or less) due to estimation error.
- Leverage must be explicitly stated and realistic for the venue.

### 2.5 Data Snooping (HIGH)

- Selecting hyperparameters based on test-set performance = HIGH.
- Using validation performance to select features, then reporting that same validation
  performance as the result = HIGH.
- Two-stage sweep protocol: Stage 1 screens on single split, Stage 2 validates top
  candidates with walk-forward.
- Fitting, evaluating, and selecting on the same data partition = HIGH.

### 2.6 Walk-Forward Splits (HIGH/MEDIUM)

**HIGH severity:**
- Training set includes data from the test/validation window.
- No gap (embargo) between train and test — must be ≥ embargo_days from policy.
- `n_splits < 3` for reported walk-forward results.

**MEDIUM severity:**
- Walk-forward results reported without noting how many splits were used.
- Different embargo values in training vs. evaluation.

---

## 3. CRYPTO-SPECIFIC CONCERNS

### 3.1 Market Structure (MEDIUM)

- Crypto trades 24/7/365. No market open/close, no overnight gaps (except exchange
  maintenance). Code assuming market hours is wrong.
- Weekend liquidity is significantly lower. Strategies with signals concentrated on
  weekends should be flagged — they may not execute at expected prices.
- Holiday effects are minimal in crypto compared to equities.

### 3.2 Regime Changes and Sub-Period Reporting (MEDIUM)

- Pre-2020 and post-2020 crypto markets are structurally different (institutional
  adoption, derivatives growth, regulatory changes). Strategies trained only on pre-2020
  data may not generalize.
- Walk-forward validation across regime boundaries is essential.
- 2022 bear market is a critical stress test period.
- **Sub-period reporting (MANDATORY for claimed winners)**: Any strategy claimed as
  beating the baseline MUST report sub-period metrics for at least: full period and 2020+
  (post-COVID). Each sub-period: Sharpe, MaxDD, annual return, n_trades, win rate, and
  buy-and-hold Sharpe for the same window. Use `subperiod_analysis()` from `sweep_utils`.
  Missing sub-period analysis for claimed winners = MEDIUM.
- Scripts using `subperiod_analysis()` from `sweep_utils` are compliant (Section 0).

### 3.3 Exchange-Specific Risks (LOW)

- Backtest assumes perfect execution — real execution has latency and partial fills.
- Data from different exchanges may have different timestamps, prices, and volume.
- OKX data is the current source for hourly candles. Flag code using different sources
  without proper alignment.

### 3.4 Funding Rates and Basis (MEDIUM)

- Perpetual futures have funding rates (every 8h) that affect P&L.
- If the strategy uses perpetuals, funding rate drag must be included.
- Spot vs. futures basis can create apparent arbitrage that doesn't survive costs.

---

## 4. FORMULA VERIFICATION

When reviewing code that implements financial formulas, verify against these references.
Deviations are HIGH severity unless mathematically equivalent.

### 4.1 Sharpe Ratio
```python
sharpe = (returns.mean() - risk_free) / returns.std(ddof=1) * np.sqrt(periods_per_year)
# periods_per_year: 8760 (hourly), 365 (daily) — NEVER 252 for crypto
```

### 4.2 Sortino Ratio
```python
downside = returns[returns < 0]
downside_std = np.sqrt((downside ** 2).mean())  # full series zeros for positives
sortino = (returns.mean() - risk_free) / downside_std * np.sqrt(periods_per_year)
```

### 4.3 Probabilistic Sharpe Ratio (PSR)
```python
from scipy.stats import norm
skew = returns.skew()
kurt = returns.kurtosis() + 3  # Convert excess to raw Pearson (normal = 3)
T = len(returns)
se = np.sqrt((1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2) / (T - 1))
psr = norm.cdf((sr_hat - sr_star) / se)
```

### 4.4 Deflated Sharpe Ratio (DSR)
```python
gamma = 0.5772156649  # Euler-Mascheroni constant
N = n_trials           # total strategies tested (cumulative)
sr_std = returns.std(ddof=1)
sr_star = sr_std * (
    (1 - gamma) * norm.ppf(1 - 1 / N) + gamma * norm.ppf(1 - 1 / (N * np.e))
)
dsr = psr(sr_hat, sr_star, skew, kurt, T)
```

### 4.5 Kelly Criterion
```python
f_star = (p * b - q) / b      # binary: p=win_prob, q=1-p, b=win/loss ratio
f_star = mean_return / var     # continuous returns
f_practical = f_star * 0.5    # ALWAYS use fractional Kelly (half or less)
```

### 4.6 Maximum Drawdown
```python
cumulative = (1 + returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()  # most negative value
```

### 4.7 Numerical Stability (HIGH/MEDIUM)

Flag these patterns:
- Division by zero without guard (`1 / std` when std could be 0).
- `np.log()` on values that could be zero or negative.
- Overflow in exponentials (`np.exp(large_number)`).
- NaN propagation without explicit handling (`np.nanmean` vs `np.mean`).
- Float equality comparison (`==` instead of `np.isclose`).

---

## 5. DATA PIPELINE INTEGRITY

### 5.1 Data Access Patterns (HIGH)

All model training and feature engineering data access MUST go through
`sparky.data.loader.load()`. Raw file access bypasses holdout enforcement.

**HIGH severity:**
- `pd.read_parquet(...)` in `src/sparky/` for anything feeding model training.
- `polars.read_parquet(...)` in `src/sparky/` for model training data.
- Direct path construction to `data/` without the loader.
- Any reference to `data/.oos_vault/` or `data/holdout/` outside of `loader.py`,
  `split_holdout_data.py`, `oos_evaluate.py`, and `build_holdout_resampled.py`.

**MEDIUM severity:**
- `purpose="analysis"` used in training scripts (should be `purpose="training"`).
- Dataset loaded without specifying `asset=` when asset is known.

**NOT a violation:**
- Utility scripts that only read data for visualization or exploration, with a comment
  explaining why they bypass the loader.
- `pd.read_parquet()` in `scripts/` for one-off analysis (not model training).

### 5.2 Timezone and Index Consistency (HIGH)

All DataFrames with time data must have UTC-aware DatetimeIndex.

**HIGH severity:**
- `df.index.tz is None` after data loading without explicit localization.
- Mixing tz-aware and tz-naive timestamps in the same operation.
- `pd.Timestamp("2024-01-01")` (tz-naive) used for comparisons against tz-aware index.

**MEDIUM severity:**
- Timezone converted to non-UTC without documented reason.
- DatetimeIndex sorted descending (expected ascending chronologically).

### 5.3 NaN and Data Quality (HIGH)

**HIGH severity:**
- Features passed to `model.fit()` without NaN check (silently corrupts XGBoost, etc.).
- Forward-filled NaN in features crossing the train/test boundary.
- Rolling window features used without accounting for warmup period (first N rows are NaN).

---

## 6. EXPERIMENT TRACKING

### 6.1 Required wandb Logging Keys (HIGH)

The orchestrator reads specific keys from wandb run summaries. Wrong key names cause
silent failures where the orchestrator shows "N/A" for all metrics.

**HIGH severity:**
- Logging `best_sharpe` or `best_dsr` as primary keys instead of `sharpe` and `dsr`.
- Nesting metrics under a sub-dict (e.g., `{"metrics": {"sharpe": 1.2}}`) —
  orchestrator reads top-level summary keys only.
- `compute_all_metrics()` result not logged to wandb at all for significant experiments.

**MEDIUM severity:**
- `n_trials` not passed to `compute_all_metrics()` — disables multiple testing correction.
  Note: `n_trials` is a parameter of `compute_all_metrics()`, NOT `run_post_checks()`.
  These are DIFFERENT functions. `run_post_checks()` takes `n_trades` (trade count).
- Session tags (`session_NNN`) not included in wandb run tags.
- Directive tags not propagated to wandb runs.

### 6.2 Sweep vs. Individual Run Logging (HIGH)

- Sweeps MUST use `log_sweep()` (one W&B run per sweep, not one per config).
- Individual validated strategies use `log_experiment()`.
- Creating one W&B run per config = HIGH severity.

### 6.3 Config Logging (MEDIUM)

- Config logged without `transaction_costs_bps`.
- Model hyperparameters not logged (prevents reproducibility).
- `config_hash` not computed for deduplication.
- Random seeds not set and logged.

---

## 7. INFRASTRUCTURE AND API CORRECTNESS

### 7.1 TransactionCostModel (HIGH)

**HIGH severity:**
- Using `TransactionCostModel()` default constructor for research (defaults are legacy
  values — always use `.standard()` or explicit values).
- Treating `cost_model.total_cost_pct` as round-trip cost — it is one-way.
  Round-trip is `cost_model.round_trip_cost` (= 2 × total_cost_pct).
- Computing costs as `n_trades * cost_bps` without accounting for position size changes
  (going long→short is a change of 2, not 1).

### 7.2 Guardrails Usage (HIGH)

**HIGH severity:**
- `run_pre_checks()` result checked but `has_blocking_failure()` not called before proceeding.
- `run_pre_checks()` not called at all in a new backtest/training script in `src/sparky/`.
- `log_results()` never called (guardrail outcomes not persisted for audit).
- Blocking failures found but execution continues anyway.

**MEDIUM severity:**
- Guardrail results silently ignored (checked but result not acted upon).
- Missing guardrail integration in training scripts = MEDIUM (HIGH if in `src/sparky/`
  core infrastructure).

**Parameter disambiguation:** `n_trials` (config count for DSR) goes to
`compute_all_metrics()`. `n_trades` (trade count) goes to `run_post_checks()`.
These are DIFFERENT parameters for DIFFERENT functions. Do not confuse them.

### 7.3 Holdout Guard API (HIGH)

**HIGH severity:**
- `HoldoutGuard` instantiated but `check_data_boundaries()` never called.
- `authorize_oos_evaluation()` called without `approved_by` containing a human identifier.
- `purpose="oos_evaluation"` used in training or sweep code (OOS is one-shot, human-gated).

### 7.4 Timeout Decorator (MEDIUM)

- New training functions in `src/sparky/` without `@with_timeout()` decorator.
- Timeout set above 900 seconds without documented justification.

### 7.5 GPU Training (MEDIUM)

All ML training scripts must use GPU acceleration:
- XGBoost: `tree_method="hist", device="cuda"`
- CatBoost: `task_type="GPU", devices="0"`
- LightGBM: `device="gpu"`
- CPU training is not permitted per project rules.

### 7.6 Reproducibility (MEDIUM)

- Random seeds must be set for all stochastic operations.
- Seeds must be logged to W&B.
- Missing seed setting = MEDIUM.

---

## 8. CODE ARCHITECTURE

### 8.1 Interface Compliance (MEDIUM)

- Strategies must implement `StrategyBase` (from `sparky.interfaces.protocols`).
- Backtesters must implement `BacktesterBase`.

### 8.2 Testing Coverage (HIGH/MEDIUM)

**HIGH severity:**
- New public functions in `src/sparky/backtest/`, `src/sparky/data/`, or
  `src/sparky/tracking/` with zero test coverage — these are load-bearing infrastructure.
- New guardrail checks added without a test verifying the check fires.

**MEDIUM severity:**
- New CLI commands in `bin/sparky` without a test.
- New configuration parameters with no test verifying they are read correctly.
- `try/except Exception` blocks that swallow errors without a test for the error case.
- New data loading path not exercised by at least one test (even with mock data).
- New wandb logging call not tested (even with a mocked tracker).

---

## 9. CONFIGURATION AND SCHEMA CONSISTENCY

### 9.1 YAML Config (HIGH)

**HIGH severity:**
- New cost values in `configs/trading_rules.yaml` without corresponding update to
  `TransactionCostModel` — code and config must agree.
- Holdout boundary changed in any config file (immutable — requires human approval).
- `oos_start` date modified in `holdout_policy.yaml`.

**MEDIUM severity:**
- New directive YAML missing required fields (`name`, `objective`, `wandb_tags`,
  `stopping_criteria`).
- `wandb_tags` in directive does not include the directive name as a tag.

### 9.2 API Key Consistency (MEDIUM)

- `costs_bps` instead of canonical `transaction_costs_bps`.
- `model_type` and `strategy_family` used interchangeably without a mapping.

---

## 10. WHAT NOT TO FLAG

Do NOT flag these — they are handled by other tools or are intentional:
- General Python code style (handled by ruff — E, F, W, I rules).
- Import ordering (handled by ruff isort).
- Security vulnerabilities (handled by bandit HIGH checks).
- Performance optimizations without correctness impact.
- pandas vs. polars choice (polars is preferred but not enforced for research scripts).
- Comments or docstrings (unless they document incorrect behavior).
- Test helper code that intentionally uses simplified patterns.
- `pd.read_parquet()` in pure visualization or one-off exploration scripts.
- Shift(-1) on the label variable `y` — this is the correct ML label construction pattern.
- Shift(1) applied inside shared utility functions — the call site is compliant.

---

## 11. AUTHORITATIVE API REFERENCE

Before flagging any keyword argument as wrong, verify the actual function signature:

```python
# compute_all_metrics — n_trials is the number of CONFIGS TESTED (for DSR correction)
metrics = compute_all_metrics(returns, n_trials=N)

# run_post_checks — n_trades is the number of TRADES MADE (for minimum-trades check)
results = run_post_checks(returns, metrics, config, n_trades=N)

# run_pre_checks — min_samples is the minimum row count
results = run_pre_checks(data, config, min_samples=500)
```

`n_trades` in `run_post_checks()` is NOT a typo for `n_trials`. These are different
parameters for different purposes in different functions. `n_trials` is the multiple
testing correction count; `n_trades` is the trade-count threshold.

Passing `n_trials` to `compute_all_metrics()` and `n_trades` to `run_post_checks()`
in the SAME script is CORRECT and expected behaviour — do NOT flag it.

---

## 12. GENERAL REVIEW PRINCIPLES

### 12.1 Verify Before Flagging

- **Function signatures**: Check actual parameter names before claiming a keyword
  argument is wrong. See Section 11.
- **Internal APIs**: If a function call uses a keyword argument you don't recognize,
  it may be correct. Only flag if you can confirm the parameter does not exist.
- **Intentional patterns**: Code that looks unusual may be deliberately written.
  Check for comments explaining the rationale before flagging.
- **Truncated code**: If you cannot see the full implementation of a referenced function
  and the script imports from a utility module, apply Section 0 and do NOT flag for
  missing implementations.

### 12.2 Severity Calibration

When assigning severity, ask: "If this code runs as-is in production, what is the
worst-case outcome?"
- Incorrect trading signal, inflated Sharpe, data leakage = **HIGH**
- Misleading research conclusion, missed edge case, poor observability = **MEDIUM**
- Documentation gap, style inconsistency, minor optimization opportunity = **LOW**

Do not escalate MEDIUM to HIGH to be safe — false positives block commits unnecessarily
and erode trust in the review system.
