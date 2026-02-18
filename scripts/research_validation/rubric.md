# Quantitative Finance Validation Rubric

This rubric is the knowledge base for the Sparky AI Research Validation Agent.
It is used to review every PR for quantitative finance correctness before merge.

Severity levels:
- **HIGH**: Blocks merge. The issue will produce incorrect results, inflated metrics, or data leakage.
- **MEDIUM**: Should fix. The issue may produce misleading results or indicates fragile methodology.
- **LOW**: Informational. Best practice suggestion that does not affect correctness.

---

## 0. MANDATORY: Shared Utility Auto-Pass Rule

**ANY script in `scripts/` that imports from a shared utility module (e.g.,
`from sweep_utils import ...` or any module in `scripts/infra/`) is AUTOMATICALLY COMPLIANT
with sections 1.1 (DSR/n_trials), 2.1 (signal timing), 2.2 (transaction costs),
3.2 (sub-period reporting), and 5.2 (guardrails).**

These scripts delegate all correctness-critical operations to the utility module:
- DSR and `n_trials` → handled inside `evaluate()` via `compute_all_metrics()`
- Signal timing (`positions.shift(1)`) → handled inside `net_ret()`
- Cost deduction → handled inside `net_ret()`
- Guardrails → handled inside `evaluate()`

**You MUST NOT flag these scripts for missing DSR, n_trials, signal shifting,
cost deduction, or guardrails — regardless of whether the code is truncated
or you cannot see the full implementation.** The `from sweep_utils import ...`
statement alone is sufficient proof of compliance. Do NOT reason "I cannot
verify" or "the truncated code doesn't show" — the import IS the verification.

If you flag a script that imports from a utility module for any of these
issues, your review is WRONG and will cause a false positive CI failure.

---

## 1. STATISTICAL METHODOLOGY (HIGH severity)

### 1.1 Multiple Testing Correction

Any comparison of more than one strategy configuration MUST use the Deflated Sharpe
Ratio (DSR) with the correct `n_trials` parameter. The `n_trials` value must reflect
ALL trials ever run across the entire research program, not just the current batch
or sweep.

- DSR < 0.95 means the result could be a fluke due to multiple testing.
- Claims of "alpha," "significant," "genuine edge," or "outperforms" without DSR > 0.95
  are HIGH severity violations.
- Reporting raw Sharpe as the primary metric without DSR context is HIGH severity.
- Using `n_trials` less than the true number of configurations tested is HIGH severity
  (underestimates the multiple testing penalty).

**Exemptions** (NOT a HIGH severity DSR violation):
- Scripts explicitly labeled as "legacy exploration" that call `compute_all_metrics()`
  with DSR and correctly pass cumulative `n_trials` are compliant, even if DSR < 0.95.
  The requirement is that DSR is computed and reported, not that DSR must exceed 0.95.
- Secondary/exploratory scripts that run exactly ONE fixed configuration (no sweep)
  need DSR with n_trials reflecting the full cumulative program count. They do NOT
  need to use `run_pre_checks`/`run_post_checks` if their results are clearly labeled
  as non-primary and the canonical sweep tool (sweep_two_stage.py) is used for final
  decisions. However, if they DO call training and backtesting, guardrails are strongly
  recommended.
- Research sweep scripts that delegate metric computation to shared utility functions
  (e.g., `evaluate()` from `sweep_utils.py` or similar) ARE compliant if the utility
  function calls `compute_all_metrics(returns, n_trials=N)` with proper `n_trials`.
  If code truncation prevents seeing the utility implementation, and the script imports
  from a utility module, do NOT flag for missing DSR. See Section 0.

Reference: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014).

### 1.2 Cross-Validation on Time Series

Time-series data MUST use purged cross-validation with an embargo period. Standard
k-fold CV on time series data introduces lookahead bias because future data leaks
into training folds.

- Standard `sklearn.model_selection.KFold` or `StratifiedKFold` on time-series = HIGH.
- `TimeSeriesSplit` without any embargo = MEDIUM (partial fix, still leaks at boundaries).
- `TimeSeriesSplit(gap=N)` with `gap >= label_horizon` IS a valid embargo implementation.
  The `gap` parameter excludes N samples between train and test sets, which is functionally
  equivalent to purged CV for the purpose of preventing boundary leakage.
- Embargo period must be >= label horizon for PRODUCTION data (e.g., if predicting 24h
  return, gap >= 24 for hourly data with n > 10,000 samples).
- For hourly crypto data: minimum 24-72 hour embargo between train and test folds.
- Combinatorial purged CV (CPCV) is the gold standard but TimeSeriesSplit with gap is acceptable.
- **Small dataset / unit test exception**: When n < 1000 samples (e.g., synthetic test
  data), sklearn raises ValueError if `gap + test_size > n // n_splits`. Proportionally
  scaling the gap via `min(label_horizon, fold_size // 3)` to prevent this error is
  ACCEPTABLE for small-dataset contexts ONLY. For production data (n > 10,000), the
  computed fold_size always satisfies `fold_size // 3 > label_horizon`, so the full
  embargo is applied automatically. A gap slightly below label_horizon on test-sized
  data is NOT a HIGH severity issue if the code comment documents the scaling rationale
  and production-data correctness is verified.

Reference: Lopez de Prado, "Advances in Financial Machine Learning" (2018), Ch. 7.

### 1.3 Sharpe Ratio Calculation

Annualization factors for crypto (24/7/365 markets):
- **Hourly data**: multiply by `sqrt(8760)` (8760 hours per year).
- **Daily data**: multiply by `sqrt(365)` (365 days per year).
- Using `sqrt(252)` (equity trading days) for crypto is WRONG and inflates Sharpe by
  ~20% (`sqrt(365/252) = 1.203`). This is HIGH severity.

Additional requirements:
- Use `ddof=1` (sample standard deviation), not `ddof=0` (population).
- Risk-free rate is typically 0 for crypto (no consensus risk-free rate).
- Sharpe MUST be computed on returns net of transaction costs, not gross returns.
- If Sharpe is reported both gross and net, the net figure must be clearly labeled as primary.

Formula: `SR = (mean(r) - rf) / std(r, ddof=1) * sqrt(periods_per_year)`

### 1.4 PSR/DSR Formula Correctness

The Probabilistic Sharpe Ratio (PSR) standard error formula uses the OBSERVED Sharpe
Ratio (`SR_hat`), not the benchmark Sharpe (`SR*`):

```
SE = sqrt((1 - gamma_3 * SR_hat + (gamma_4 - 1)/4 * SR_hat^2) / (T - 1))
```

Where:
- `gamma_3` = skewness of returns
- `gamma_4` = kurtosis of returns (raw Pearson kurtosis, where normal = 3)
- `T` = number of observations
- `SR_hat` = observed (sample) Sharpe ratio

Common errors (all HIGH severity):
- Using excess kurtosis (normal = 0) instead of raw Pearson kurtosis (normal = 3).
  This mis-specifies the SE formula.
- Using `SR*` (benchmark) instead of `SR_hat` (observed) in the SE formula.
- Assuming Gaussian returns (ignoring skewness and kurtosis terms). Crypto returns
  have significant skew and fat tails; the Gaussian assumption massively underestimates SE.
- Using `T` instead of `T-1` in the denominator.

DSR haircut formula:
```
SR* = E[max(SR)] = sigma_SR * ((1 - gamma) * Phi_inv(1 - 1/N) + gamma * Phi_inv(1 - 1/(N*e)))
```
Where `gamma` is the Euler-Mascheroni constant (~0.5772), `N` = number of trials.

Reference: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014).

### 1.5 Stationarity

Non-stationary features used as inputs to ML models produce spurious correlations
and unreliable predictions.

- Price levels as raw features = HIGH (prices are non-stationary).
- Must use returns, log-returns, or fractional differentiation (`d < 1` to preserve memory).
- Check with ADF test or KPSS test. If p-value > 0.05 on ADF, the feature is likely
  non-stationary.
- Rolling z-scores or percentage changes are acceptable stationarity transforms.
- Volume should also be normalized (e.g., relative to rolling average).

Reference: Lopez de Prado, "Advances in Financial Machine Learning" (2018), Ch. 5.

### 1.6 Survivorship and Selection Bias

- Only analyzing BTC/ETH (which survived) introduces survivorship bias if generalizing
  to "crypto strategies." Flag claims that generalize beyond the tested assets.
- Failed strategy configurations MUST be included when counting `n_trials` for DSR.
- Cherry-picking the best walk-forward window or best parameter set without accounting
  for the full search space is selection bias.
- Reporting only the best fold of cross-validation instead of the mean is selection bias.

---

## 2. BACKTESTING METHODOLOGY (HIGH severity)

### 2.1 Lookahead Bias

Signal at time T must only use data available BEFORE time T. This is the single most
common source of inflated backtest results.

Known patterns that cause lookahead (all HIGH severity):
- `signal[T] * return[T]` where signal is computed using `close[T]`. The signal uses
  information from the same bar it trades on. Correct: `signal.shift(1) * return`.
- Full-dataset normalization (e.g., `StandardScaler().fit_transform(all_data)` before
  train/test split). Must fit scaler on training data only, then transform test.
- Rolling window calculations that include future data (e.g., centered moving averages
  instead of trailing).
- Feature engineering that uses the entire dataset (e.g., `df['feat'] = df['col'].rank()`
  on the full dataframe instead of expanding/rolling).
- Using `pd.read_parquet()` directly instead of `sparky.data.loader.load()` bypasses
  the holdout guard and may include future data.

**NOT lookahead bias (common false positives):**
- `target = (returns.shift(-1) > 0).astype(int)` — This creates a forward-looking
  LABEL (the thing being predicted). This is correct ML practice for supervised
  classification of next-period direction. The shift(-1) is on `y`, not on features.
  The signal generated at inference time uses ONLY past features to produce a prediction;
  the label is only used during training. The backtest MUST still use `signal.shift(1)`
  when computing strategy returns (which is correct).
- Donchian channel signals computed at bar T using `rolling(window).max()` (trailing
  window) are NOT lookahead as long as the signal is shifted before use in backtest.
  The correct anti-lookahead check is whether the BACKTEST applies `signal.shift(1)`,
  not whether the signal itself looks back.

Historical note: A lookahead bug (`signal[T] * return[T]`) was found in Sparky's
backtest engine on 2026-02-16 (PR #12). It inflated Sharpe by 43-256%. All code must
use the corrected `signal.shift(1) * return` pattern.

### 2.2 Transaction Costs

Two-tier cost model:
- **Standard: 30 bps per side** (60 bps round trip) — Coinbase limit orders / DEX on L2
- **Stress test: 50 bps per side** (100 bps round trip) — Coinbase market orders worst case

The guardrail enforces a minimum of 30 bps per side.

Cost validation:
- Any `transaction_costs_bps` value below 30 is HIGH severity.
- Missing `transaction_costs_bps` in config is HIGH severity.
- Costs must be applied per-trade, not as a flat annual drag.
- If `cost_bps` is set to 0 or costs are not applied, this is HIGH severity.
- Watch for cost application AFTER Sharpe calculation (must be BEFORE).
- Strategies claiming an edge smaller than 2x the round-trip cost (120 bps at standard) are suspect.
- Winners should be reported at both 30 bps and 50 bps. Results at only one cost level are MEDIUM severity.

### 2.3 Holdout Integrity

The out-of-sample (OOS) boundary and embargo buffer are defined in
`configs/holdout_policy.yaml` and are IMMUTABLE. Do NOT hardcode specific dates.

- No training data may include observations after the OOS start date.
- No features may be computed using data after the embargo boundary
  (OOS start minus embargo_days, as defined in the policy).
- Code MUST use `sparky.data.loader.load(purpose="training")` which auto-truncates.
- Using raw `pd.read_parquet()` for any model training or feature engineering is HIGH
  severity (bypasses holdout guard).
- OOS evaluation requires explicit written approval from AK (the human operator).
- Each model gets exactly ONE OOS evaluation. Repeated peeking at OOS results and
  re-tuning is data snooping.
- **Any PR that modifies `configs/holdout_policy.yaml` is HIGH severity.** This file
  is immutable and can only be changed by the human operator (AK).
- **Any code that hardcodes OOS boundary dates** (e.g., `"2024-01-01"`, `"2024-07-01"`)
  instead of reading from `HoldoutGuard` or `configs/holdout_policy.yaml` is HIGH
  severity. All holdout dates must be derived dynamically from the policy.
- **Any code that directly accesses `data/.oos_vault/`** or references the vault path
  is HIGH severity. OOS data must ONLY be accessed via
  `load(purpose="oos_evaluation", oos_guard=guard)` with proper authorization.
  The vault contains full data including the holdout period — direct reads bypass
  all holdout enforcement.

### 2.4 Position Sizing Realism

- No unlimited position size assumption. Positions must be bounded.
- Single position should be < 10% of portfolio (unless explicitly justified for
  concentrated strategies).
- Market impact must be considered for large positions (> $100K notional).
- Leverage, if used, must be explicitly stated and realistic for the venue.
- Kelly criterion, if used, should be fractional (half-Kelly or less) due to
  estimation error in parameters.

### 2.5 Data Snooping

- Selecting hyperparameters based on test-set performance is snooping (HIGH severity).
- Using validation performance to select features, then reporting that same validation
  performance as the result, is snooping.
- The two-stage sweep protocol exists to mitigate this: Stage 1 screens on a single
  split, Stage 2 validates top candidates with walk-forward.
- Any code that fits, evaluates, and selects on the same data partition is HIGH severity.

---

## 3. CRYPTO-SPECIFIC CONCERNS (MEDIUM severity)

### 3.1 Market Structure

- Crypto trades 24/7/365. There are no market open/close effects, no overnight gaps
  (except exchange maintenance windows). Code assuming market hours is wrong.
- Weekend liquidity is significantly lower. Strategies with signals concentrated on
  weekends should be flagged — they may not execute at expected prices.
- Holiday effects are minimal in crypto compared to equities.

### 3.2 Regime Changes

- Pre-2020 and post-2020 crypto markets are structurally different (institutional
  adoption, derivatives market growth, regulatory changes).
- Strategies trained only on pre-2020 data may not generalize.
- Walk-forward validation across regime boundaries is essential.
- 2022 bear market is a critical stress test period. Strategies must be evaluated on it.
- **Sub-period reporting (MANDATORY)**: Any strategy claimed as beating the baseline
  MUST report sub-period metrics for at least: full period and 2020+ (post-COVID).
  Each sub-period must include: Sharpe, MaxDD, annual return, n_trades, win rate,
  and buy-and-hold Sharpe for the same window. Use `subperiod_analysis()` from
  `sweep_utils`. Missing sub-period analysis for claimed winners is MEDIUM severity.
- Scripts using `subperiod_analysis()` from `sweep_utils` are compliant (Section 0).

### 3.3 Exchange-Specific Risks

- Exchange downtime, API rate limits, and order book depth vary.
- Backtest assumes perfect execution — real execution has latency and partial fills.
- Data from different exchanges may have different timestamps, prices, and volume.
- OKX data is the current source for hourly candles. Flag any code using different
  data sources without proper alignment.

### 3.4 Funding Rates and Basis

- Perpetual futures have funding rates (typically every 8h) that affect P&L.
- If the strategy uses perpetuals, funding rate drag must be included.
- Spot vs. futures basis can create apparent arbitrage that doesn't exist after costs.

---

## 4. FORMULA VERIFICATION (HIGH severity)

When reviewing code that implements financial formulas, verify against these reference
implementations. Any deviation is HIGH severity unless mathematically equivalent.

### 4.1 Sharpe Ratio

```python
sharpe = (returns.mean() - risk_free) / returns.std(ddof=1) * np.sqrt(periods_per_year)
# periods_per_year: 8760 (hourly), 365 (daily)
# risk_free: typically 0 for crypto
```

### 4.2 Sortino Ratio

```python
downside = returns[returns < 0]
downside_std = np.sqrt((downside ** 2).mean())  # or returns.clip(upper=0).std()
sortino = (returns.mean() - risk_free) / downside_std * np.sqrt(periods_per_year)
```

Note: downside deviation uses the full return series for the denominator calculation
(zeros for positive returns), not just the subset of negative returns.

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
import numpy as np
from scipy.stats import norm

gamma = 0.5772156649  # Euler-Mascheroni constant
sr_std = returns.std(ddof=1)  # std of Sharpe estimates (approx)
N = n_trials  # total number of strategies tested

# Expected max Sharpe under null
sr_star = sr_std * (
    (1 - gamma) * norm.ppf(1 - 1 / N)
    + gamma * norm.ppf(1 - 1 / (N * np.e))
)

# Then compute PSR with sr_star as benchmark
dsr = psr(sr_hat, sr_star, skew, kurt, T)
```

### 4.5 Kelly Criterion

```python
# Binary outcome version
f_star = (p * b - q) / b  # p=win prob, q=1-p, b=win/loss ratio

# Continuous version (for returns)
f_star = mean_return / variance_return

# ALWAYS use fractional Kelly (half-Kelly or less)
f_practical = f_star * 0.5  # half-Kelly
```

### 4.6 Maximum Drawdown

```python
cumulative = (1 + returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()  # most negative value
```

### 4.7 Numerical Stability

Flag any code with these issues:
- Division by zero without guard (e.g., `1 / std` when std could be 0).
- `np.log()` on values that could be zero or negative.
- Overflow in exponentials (`np.exp(large_number)`).
- NaN propagation without explicit handling (`np.nanmean` vs `np.mean`).
- Float comparison without tolerance (`==` instead of `np.isclose`).

---

## 5. CODE ARCHITECTURE (MEDIUM severity)

### 5.1 Interface Compliance

Strategies must implement `StrategyBase` (from `sparky.interfaces.protocols`).
Backtesters must implement `BacktesterBase`. Violation = MEDIUM.

### 5.2 Guardrail Integration

All training runs must use the guardrail framework:
- `run_pre_checks(data, config)` BEFORE training. Checks: holdout boundary, minimum
  samples, no lookahead, costs specified, parameter-data ratio.
- `run_post_checks(returns, metrics, config, n_trades=N)` AFTER backtest. Checks: Sharpe
  sanity, minimum trades, DSR threshold, max drawdown, returns distribution, consistency.
  Note: the parameter is `n_trades` (trade count), NOT `n_trials`. DSR multiple testing
  correction is handled by `compute_all_metrics(returns, n_trials=N)`, which is a
  separate function called before `run_post_checks`.
- `has_blocking_failure(results)` must be checked; blocking failures must halt execution.
- Missing guardrail integration = MEDIUM severity.

**Parameter disambiguation:** `n_trials` (config count for DSR) goes to
`compute_all_metrics()`. `n_trades` (trade count) goes to `run_post_checks()`.
Do not confuse these — they are different functions with different parameters.

### 5.3 Experiment Tracking

- All experiments must log to W&B via `ExperimentTracker`.
- Sweeps must use `log_sweep()` (one run per sweep, not one run per config).
- Random seeds must be set AND logged.
- Hyperparameters must be logged to W&B config.
- Missing experiment tracking = MEDIUM severity.

### 5.4 Data Loader Usage

- Model training code must use `sparky.data.loader.load()`.
- Using `pd.read_parquet()` directly for any data that feeds into model training,
  feature engineering, or backtesting is HIGH severity (see 2.3).
- Utility scripts that only read data for visualization or exploration may use
  `pd.read_parquet()` with a comment explaining why.

### 5.5 GPU Training

- All ML training scripts must use GPU acceleration.
- XGBoost: `tree_method="hist", device="cuda"`
- CatBoost: `task_type="GPU", devices="0"`
- LightGBM: `device="gpu"`
- CPU training is not permitted per project rules. Violation = MEDIUM severity.

### 5.6 Reproducibility

- Random seeds must be set for all stochastic operations (model training, data
  shuffling, cross-validation splits).
- Seeds must be logged to W&B or otherwise recorded.
- Results must be reproducible given the same seed and data.
- Missing seed setting = MEDIUM severity.

---

## 6. GENERAL REVIEW PRINCIPLES

### 6.1 Verify Before Flagging

Before flagging any code as incorrect, verify your assumption against the actual codebase:

- **Function signatures**: Check the actual parameter names before claiming a keyword
  argument is wrong. Different functions may use similar-sounding names for different
  purposes (e.g., `n_trials` in `compute_all_metrics()` vs `n_trades` in
  `run_post_checks()` — these are correct and intentionally different).
- **Internal APIs**: If a function call uses a keyword argument you don't recognize, it
  may be correct — the rubric may not document every parameter. Only flag if you can
  confirm the parameter does not exist on the target function.
- **Intentional patterns**: Code that looks unusual may be deliberately written that way.
  Check for comments or docstrings explaining the rationale before flagging.
