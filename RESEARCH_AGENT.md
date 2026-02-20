# Research Agent Reference

This file contains everything a research agent needs. Do NOT read CLAUDE.md
or explore source files — all required APIs and rules are here.

## Coding Standards

Terse code, minimal comments — only where logic is genuinely non-obvious.
No docstrings on experiment scripts. Use numpy, polars, pandas, scipy,
sklearn, xgboost directly. JAX, PyTorch, CUDA encouraged where they help.
If another language would help (R, Julia), create a GATE_REQUEST.

Do not write "clean code" — write code that runs fast and produces correct
results. Session time is limited. Spend it on experiments, not formatting.

Prefer polars over pandas for new data processing. Use pandas when
interfacing with sklearn, XGBoost/LightGBM/CatBoost, or existing code.
Convert at boundaries: `df.to_pandas()` or `pl.from_pandas(df)`.

## Experiment Runner (MANDATORY for all experiment scripts)

Use `experiment_runner` for ALL experiment work. Do not re-implement data loading,
cost models, metrics computation, sub-period analysis, JSON saving, or W&B logging
in experiment scripts.

### For parameter grid sweeps — use `sweep()`

```python
import sys; sys.path.insert(0, "bin/infra")
from experiment_runner import sweep

results = sweep(
    base_config={"asset": "eth", "timeframe": "8h", "signal_type": "donchian", "sizing": "flat"},
    grid={"entry_period": range(60, 90, 5), "exit_period": range(20, 40, 5)},
    n_trials_start=0,
    benchmark_returns=btc_bench,
    wandb_name="eth_don8h_sweep", wandb_tags=["broad_exploration"],
    save_path="results/eth/r1.json",
    top_k=10,
)
# Handles: grid expansion, n_trials tracking, run() per config, DSR sort,
# top-K print, JSON save, wandb sweep log. Returns {"results": [...], "n_trials": N}.
# Grid keys route automatically: signal params by default, vol_window/target_vol
# to sizing_params, asset/timeframe/signal_type/sizing to top-level.
```

### For novel signal types — use `run_custom_signal()`

```python
from experiment_runner import run_custom_signal

def my_mean_rev(prices, lookback=20, threshold=2.0):
    z = (prices - prices.rolling(lookback).mean()) / prices.rolling(lookback).std()
    return (z < -threshold).astype(float) - (z > threshold).astype(float)

result = run_custom_signal(
    signal_func=my_mean_rev, signal_kwargs={"lookback": 20, "threshold": 2.0},
    asset="btc", timeframe="4h", sizing="flat",
    n_trials=42, benchmark_returns=btc_bench,
)
# signal_func(prices, **kwargs) -> pd.Series of positions (-1 to 1)
# Handles: data loading, sizing, dual-cost eval, guardrails, sub-periods, benchmark corr.
```

### For multi-strategy portfolios — use `portfolio_combine()`

```python
from experiment_runner import portfolio_combine

result = portfolio_combine(
    strategies=[
        {"config": {"asset": "btc", "timeframe": "4h", "signal_type": "donchian",
                     "signal_params": {"entry_period": 30, "exit_period": 20}}, "weight": 0.5},
        {"config": {"asset": "btc", "timeframe": "4h", "signal_type": "bollinger",
                     "signal_params": {"period": 40, "num_std": 2.0}}, "weight": 0.5},
    ],
    weighting="specified",  # "equal" | "specified" | "inverse_vol"
    n_trials=42, benchmark_returns=btc_bench,
)
# All strategies must share same asset/timeframe. Returns metrics + correlation_matrix + weights.
```

### For single configs — use `run()`

```python
from experiment_runner import run

result = run({
    "asset": "btc", "timeframe": "4h",
    "signal_type": "donchian",
    "signal_params": {"entry_period": 30, "exit_period": 20},
    "sizing": "inverse_vol",
    "sizing_params": {"vol_window": 20, "target_vol": 0.4},
    "n_trials": 10,
    "benchmark_returns": None,
})
```

Signal types: donchian (entry_period, exit_period), bollinger (period, num_std, hold_periods),
rsi_extreme (period, entry, exit).

**Do NOT:**
- Re-implement data loading, cost computation, metrics, sub-period analysis, or W&B logging
- Write standalone for-loops over parameter grids (use `sweep()`)
- Write custom signal evaluation boilerplate (use `run_custom_signal()`)
- Write multi-strategy combination logic (use `portfolio_combine()`)

## Data Loading

```python
from sparky.data.loader import load, list_datasets

df = load("btc_daily", purpose="training")      # truncated at holdout embargo
df = load("btc_1h_features", purpose="training") # hourly features
df = load("btc_daily", purpose="analysis")       # full IS data, for plots only

# See available datasets
list_datasets()  # returns [{"name": ..., "path": ..., "asset": ...}, ...]
```

**Rules:**
- NEVER use `pd.read_parquet()`. The loader enforces holdout boundaries.
- `purpose="training"` for all model work. `purpose="analysis"` for exploration only.
- OOS data is off-limits. Do not attempt `purpose="oos_evaluation"`.

## Transaction Costs (MANDATORY — two tiers)

- **Standard: 30 bps per side** (60 bps RT) — Coinbase limit orders / DEX on L2
- **Stress test: 50 bps per side** (100 bps RT) — Coinbase market orders worst case
- Guardrail blocks anything below 30 bps.

```python
from sparky.backtest.costs import TransactionCostModel

costs_standard = TransactionCostModel.standard()      # 30 bps per side
costs_stress   = TransactionCostModel.stress_test()    # 50 bps per side
net_returns = costs_standard.apply(gross_returns, positions)
```

**Run winners at BOTH 30 bps and 50 bps and report both.** Use 30 bps as the
primary evaluation. A strategy that only works at 30 bps but not 50 bps is fragile.

Config dict must always include `"transaction_costs_bps": 30` (or 50 for stress test).
The guardrail will BLOCK any run below 30 bps.

## Metrics

```python
from sparky.tracking.metrics import compute_all_metrics

metrics = compute_all_metrics(
    returns,              # array of period returns
    n_trials=N,           # total configs tested (cumulative across sessions)
    periods_per_year=365, # daily=365, hourly=8760. NEVER use 252.
)
# Returns dict with keys: sharpe, sharpe_per_period, psr, dsr, sortino,
# max_drawdown, calmar, cvar_5pct, rolling_sharpe_std, profit_factor,
# worst_year_sharpe, n_observations, win_rate, mean_return, total_return,
# skewness, kurtosis, min_track_record, n_trials
```

**DSR is the primary metric.** DSR > 0.95 = statistically significant.
DSR < 0.95 = could be a fluke. Do not declare victory.

## Guardrails (MANDATORY)

```python
from sparky.tracking.guardrails import (
    run_pre_checks, run_post_checks, has_blocking_failure, log_results
)

# Before experiment (hourly data — default min_samples=2000)
pre_results = run_pre_checks(data, config)
# For daily data (fewer rows), lower the sample threshold:
# pre_results = run_pre_checks(data, config, min_samples=500)
if has_blocking_failure(pre_results):
    raise RuntimeError("Pre-checks failed")

# After backtest — N = cumulative configs tested this session (for DSR correction)
post_results = run_post_checks(returns, metrics, config, n_trials=N)
log_results(pre_results + post_results, run_id="my_run")
```

Pre-checks: holdout boundary, minimum samples, no lookahead, costs >= 30 bps, param-data ratio.
Post-checks: sharpe sanity (<4.0), minimum trades, DSR threshold, max drawdown, returns distribution, consistency.

## Experiment Tracking (wandb)

```python
from sparky.tracking.experiment import ExperimentTracker

tracker = ExperimentTracker(experiment_name="directive_name")

# For sweeps: collect all results, log ONCE as a single run
tracker.log_sweep("session_001_sweep", results, summary_metrics={...}, tags=[...])

# For validated results only:
tracker.log_experiment("donchian_validated", config={...}, metrics={...}, tags=[...])
```

**IMPORTANT — wandb summary keys:**
When logging to wandb, include `sharpe` and `dsr` as top-level summary keys
(not `best_sharpe`/`best_dsr`). The orchestrator reads these to track progress.

```python
wandb.log({"sharpe": best_sharpe, "dsr": best_dsr, "n_configs_tested": N, ...})
```

## Backtest Engine

```python
from sparky.backtest.engine import WalkForwardBacktester

backtester = WalkForwardBacktester(
    n_splits=5,
    test_size=0.2,
    gap=30,  # embargo days
    cost_model=TransactionCostModel.standard(),
)
wf_results = backtester.run(signals, prices)
```

## Timeouts

```python
from sparky.oversight.timeout import with_timeout

@with_timeout(seconds=900)  # 15 min per config
def train_single_config(...):
    ...
```

## GPU Training

- XGBoost: `tree_method="hist", device="cuda"`
- CatBoost: `task_type="GPU", devices="0"`
- LightGBM: `device="gpu"`

CPU training is not permitted.

## Research Code Style

Research agents are scientists, not software engineers. Experiment scripts are
disposable — write code that runs fast and produces correct results.

**Do:**
- Write terse code. Let numpy/polars/pandas expressions be self-explanatory.
- Add a comment only where the logic is genuinely non-obvious (a subtle
  mathematical invariant, a non-obvious index offset, etc.).
- Use numpy, polars, pandas, scipy, sklearn, xgboost, lightgbm, catboost
  directly. Do not wrap them in extra classes or helper layers.
- Use JAX, PyTorch, or CUDA when they offer real speedups (custom losses,
  batched GPU operations).

**Do NOT:**
- Write docstrings on experiment scripts. A one-line module docstring is enough.
- Build abstraction layers or utilities for one-off scripts. Use experiment_runner
  functions. Do not duplicate infrastructure logic.
- Refactor for cleanliness. Oversight handles that if something goes to production.
- Add type annotations unless they prevent an actual bug.

**Other languages:**
If R (statistical tests), Julia (numerical simulation), or another language
would produce better results faster, write `GATE_REQUEST.md` and exit. Do not
approximate in Python when a better tool exists.

## Saving Results

`sweep()` saves automatically via `save_path=`. For individual results:
```python
import json
from pathlib import Path
out = Path("results/<directive_name>")
out.mkdir(parents=True, exist_ok=True)
with open(out / "session_N_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
```

## CRITICAL: Exit Protocol — READ THIS

When you have completed your experiments and logged results to wandb:
1. Write final results to `results/<directive_name>/`
2. **EXIT IMMEDIATELY.** Do not look for additional work.

**After finishing research, you MUST NOT:**
- Tidy up, refactor, or reformat code
- Fix lint warnings or code style issues
- Interact with git (commit, branch, push, PR)
- Edit CI configs, rubrics, or validation scripts
- Modify CLAUDE.md, RESEARCH_AGENT.md, or any config files
- Write to `.claude/`, `src/`, `tests/`, `configs/`, or `docs/`
- Edit your own memory files or project instructions
- Rewrite documentation or roadmaps
- Comment on or interact with GitHub PRs/issues

Your sandbox enforces these restrictions. If you attempt to write outside
`results/` or `scripts/*.py`, the write will be BLOCKED.

Files in `bin/infra/` are **protected platform utilities** (e.g., `sweep_utils.py`,
`sweep_two_stage.py`). You cannot edit them. To request changes, write a
`GATE_REQUEST.md` explaining what you need.

If you find yourself repeating "session is done" or similar phrases,
the orchestrator will detect the idle loop and kill your process.

## CRITICAL: Do NOT Use Git

You must NOT create branches, commit, push, or run any git commands.
Your outputs go to wandb and `results/` only.

If you need a platform change (new feature, bug fix, data issue),
write `GATE_REQUEST.md` to the project root explaining what you need,
then exit. The orchestrator will pause for oversight.

## CRITICAL: Do NOT Run CI, Linting, or Formatting

You must NOT run ruff, black, flake8, mypy, pre-commit, pytest, or any CI-related
commands. Your job is research — write experiment scripts and run them. Code quality
and cleanup are handled separately by oversight sessions. Do not waste session time
on formatting or lint fixes.

## Code Style

Write terse code with minimal comments. Let the code be self-explaining.
No verbose docstrings, no markdown generation, no print-heavy logging.
You are expected to write directly in numpy, polars, pandas, jax, cuda, pytorch, etc.
This is where the real experimentation happens — optimize for iteration speed.

**Library requests:** If you need a library or tool that isn't installed, write a
`GATE_REQUEST.md` requesting it. Do NOT run pip, uv, conda, or any package installer.
The orchestrator will install approved packages in an oversight session.

## Python Environment

Always use `.venv/bin/python`, NOT system `python` or `python3`.

## Sub-Period Validation (MANDATORY)

Every config that beats the baseline MUST report sub-period metrics alongside
full-period results. Sub-periods: **2017+** (removes early low-liquidity era)
and **2020+** (post-COVID, includes 2022 bear).

For each period: Sharpe, MaxDD, annual return, n_trades, win rate, and
buy-and-hold Sharpe for comparison.

```python
from sweep_utils import subperiod_analysis
sp = subperiod_analysis(prices, positions, cost_frac)
# Returns: {"full": {...}, "2017+": {...}, "2020+": {...}}
```

A strategy with Sharpe 1.98 on 2013-2023 but Sharpe 0.4 on 2020-2023 is NOT
deployment-worthy. Do not declare any result as "beating baseline" without
sub-period confirmation.

## Success Tiers

| Tier | Criteria | Meaning |
|------|----------|---------|
| TIER 1 | Sharpe >= 1.0, MC > 80%, MaxDD < 50% | Deploy candidate |
| TIER 2 | Sharpe >= 0.7, MC > 70%, beats B&H | Paper trade candidate |
| TIER 3 | Sharpe >= 0.4, shows edge | Keep iterating |
| TIER 4 | Sharpe < 0.4 after 5+ configs | Pivot approach |
| TIER 5 | Sharpe < 0.2 after 10+ configs, 2+ families | Abandon |
