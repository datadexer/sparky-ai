# Research Agent Reference

This file contains everything a research agent needs. Do NOT read CLAUDE.md
or explore source files — all required APIs and rules are here.

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

## Transaction Costs — 50 bps per side (MANDATORY)

ALL backtests MUST use 50 bps per side (100 bps round trip).
This reflects Coinbase taker fees / Uniswap DEX costs. No exceptions.

```python
from sparky.backtest.costs import TransactionCostModel

costs = TransactionCostModel.standard()  # 50 bps per trade
# Apply to returns:
net_returns = costs.apply(gross_returns, positions)
```

Config dict must always include `"transaction_costs_bps": 50`.
The guardrail will BLOCK any run below 50 bps.

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

# Before experiment
pre_results = run_pre_checks(data, config)
if has_blocking_failure(pre_results):
    raise RuntimeError("Pre-checks failed")

# After backtest — N = cumulative configs tested this session (for DSR correction)
post_results = run_post_checks(returns, metrics, config, n_trials=N)
log_results(pre_results + post_results, run_id="my_run")
```

Pre-checks: holdout boundary, minimum samples, no lookahead, costs >= 50 bps, param-data ratio.
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

## Saving Results

Save results to `results/<directive_name>/`:
```python
import json
from pathlib import Path

out_dir = Path("results/regime_donchian")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "session_003_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
```

## CRITICAL: Do NOT Use Git

You must NOT create branches, commit, push, or run any git commands.
Your outputs go to wandb and `results/` only.

If you need a platform change (new feature, bug fix, data issue),
write `GATE_REQUEST.md` to the project root explaining what you need,
then exit. The orchestrator will pause for oversight.

## Python Environment

Always use `.venv/bin/python`, NOT system `python` or `python3`.

## Success Tiers

| Tier | Criteria | Meaning |
|------|----------|---------|
| TIER 1 | Sharpe >= 1.0, MC > 80%, MaxDD < 50% | Deploy candidate |
| TIER 2 | Sharpe >= 0.7, MC > 70%, beats B&H | Paper trade candidate |
| TIER 3 | Sharpe >= 0.4, shows edge | Keep iterating |
| TIER 4 | Sharpe < 0.4 after 5+ configs | Pivot approach |
| TIER 5 | Sharpe < 0.2 after 10+ configs, 2+ families | Abandon |
