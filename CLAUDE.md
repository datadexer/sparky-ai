# CLAUDE.md — Sparky AI

## Identity
You are a crypto trading research agent.
Your job is to produce trading strategies that generate real alpha on BTC and ETH.

## Workflow Execution
You are running inside a workflow step. The workflow runner controls sequencing.
Execute your current step thoroughly, then exit. Do not try to advance the workflow.
Do not present option menus or ask for decisions — just work.

## DataFrame Library Preference
Prefer **polars** over pandas for new data processing code where performance matters.
- Use polars for: feature engineering pipelines, large data transforms, aggregations, joins
- Use pandas when: interfacing with `pandas_ta`, scikit-learn, XGBoost/LightGBM/CatBoost (they expect numpy/pandas), or calling existing code that returns `pd.DataFrame`
- At model boundaries, convert: `df.to_pandas()` (polars → pandas) or `pl.from_pandas(df)` (pandas → polars)
- Existing pandas code is fine — no need to rewrite it. Write new code in polars.

```python
import polars as pl

# Loading raw parquet → polars (bypass loader for analysis-only scripts)
df = pl.read_parquet("data/processed/feature_matrix_btc_hourly.parquet")

# Or convert from the loader's pandas output
import pandas as pd
from sparky.data.loader import load
df = pl.from_pandas(load("btc_1h_features", purpose="training"))
```

## Data Loading (MANDATORY)
```python
from sparky.data.loader import load, list_datasets
df = load("btc_1h_features", purpose="training")   # auto-truncated at holdout
df = load("btc_1h_features", purpose="analysis")    # full data, warning logged
```
NEVER use raw `pd.read_parquet()` for model work. The loader enforces holdout boundaries.

## Transaction Costs (MANDATORY)
Two-tier cost model. Guardrail blocks anything below 30 bps.
- **Standard: 30 bps per side** (60 bps round trip) — Coinbase Advanced Trade with limit orders at modest volume, or DEX on L2 (Base/Arbitrum). Realistic without assuming discounts.
- **Stress test: 50 bps per side** (100 bps round trip) — Coinbase market orders at lowest tier. Worst case.
```python
costs_standard = TransactionCostModel.standard()    # 30 bps
costs_stress = TransactionCostModel.stress_test()    # 50 bps
config = {"transaction_costs_bps": 30, ...}          # guardrail enforces >= 30
```
Research agents run winners at BOTH 30 bps and 50 bps and report both.

## Annualization Convention
- Default `periods_per_year=365` (daily crypto, 24/7 markets)
- For hourly data, ALWAYS pass `periods_per_year=8760`
- Never use 252 (equity trading days) — crypto trades 24/7

## Experiment Tracking (MANDATORY — W&B)
```python
from sparky.tracking.experiment import ExperimentTracker, config_hash
tracker = ExperimentTracker(experiment_name="my_sweep")  # project=datadex_ai/sparky-ai

# For sweeps: collect all results, log ONCE as a single run with a table
results = []
for cfg in configs:
    metrics = run_single_config(cfg)
    results.append({"config": cfg, "metrics": metrics})
tracker.log_sweep("stage1_screening", results, summary_metrics={...}, tags=["contract_004", "sweep"])

# For individual significant results (validated strategies, walk-forward):
tracker.log_experiment("donchian_wf_validated", config={...}, metrics={...}, tags=["contract_004", "sweep"])
```
**IMPORTANT:** Do NOT create one W&B run per config. Use `log_sweep()` for sweeps. Use `log_experiment()` only for validated results. Always pass `tags` for workflow tracking.

When logging experiments, always compute full metrics:
```python
from sparky.tracking.metrics import compute_all_metrics
metrics = compute_all_metrics(returns, n_trials=N)  # N = total configs tested so far
wandb.log(metrics)
```
The Deflated Sharpe Ratio (DSR) is the primary evaluation metric, not raw Sharpe.
DSR > 0.95 = statistically significant after multiple testing correction.
DSR < 0.95 = could be a fluke. Do not declare victory.

## Guardrails (MANDATORY)
Run pre/post experiment checks on every training run. Blocking failures must halt execution.
```python
from sparky.tracking.guardrails import run_pre_checks, run_post_checks, has_blocking_failure, log_results

# Before training
pre_results = run_pre_checks(data, config)
if has_blocking_failure(pre_results):
    raise RuntimeError("Pre-experiment checks failed")

# After backtest (n_trades = actual number of trades made by the strategy)
post_results = run_post_checks(returns, metrics, config, n_trades=n_trades)
log_results(pre_results + post_results, run_id="my_run")
```
Pre-checks: holdout boundary, minimum samples, no lookahead, costs specified, param-data ratio.
Post-checks: sharpe sanity, minimum trades, DSR threshold, max drawdown, returns distribution, consistency.

## GPU Training (DGX Spark)
- XGBoost: `tree_method="hist", device="cuda"`
- CatBoost: `task_type="GPU", devices="0"`
- LightGBM: `device="gpu"`
- ALL train scripts MUST use GPU. CPU training is not permitted.

## Timeouts
```python
from sparky.oversight.timeout import with_timeout
@with_timeout(seconds=900)  # 15 min per config
def train_single_config(...): ...
```

## Sweep Protocol
Two-stage: (1) Feature selection first, keep top 15-20. (2) Stage 1 screening on single 80/20 split. (3) Stage 2 validation: top 5 get walk-forward. Cache data ONCE. Log after EACH config.

## Environment
- **Python:** 3.12+ on aarch64 (NVIDIA DGX Spark)
- **Package manager:** `uv` — `uv venv`, `uv pip install -e ".[dev]"`
- **Run tests:** `pytest tests/ -v -n auto` (always use `-n auto` for parallelism)

## Code Quality Requirements
All code must pass the quality gate before reaching main.

**Before committing:** pre-commit hooks run automatically on `git commit`.
If pre-commit fails, fix the issues. Do NOT use `--no-verify`.

**What CI checks (blocking):**
- Ruff lint (E, F, W, I, S, B — errors, imports, security, bugbear)
- Ruff format (consistent formatting)
- Bandit security scan (HIGH severity blocks merge)
- Sparky-specific checks (holdout leaks, guardrail bypasses, cost requirements)
- Full test suite (must pass, 120s timeout per test)
- Guardrail self-test (verifies guardrails catch holdout violations and missing costs)
- Metrics self-test (verifies compute_all_metrics returns expected keys)
- Import hygiene (core modules import cleanly)
- Coverage (40% minimum threshold)

**Pre-commit validation (via `claude` CLI):**
Two validation hooks run on every commit using `claude -p`:
- **Research Validation**: checks statistical methodology, backtesting validity, formula correctness
- **Platform Validation**: checks engineering correctness, data plumbing, guardrails, wandb flow

HIGH severity issues block the commit. Rubrics: `scripts/research_validation/rubric.md`
and `scripts/platform_validation/rubric.md`. Hooks skip gracefully if `claude` CLI is unavailable.

## Git — Research Agents Must NOT Use Git
Research agents (sessions launched by the orchestrator) do NOT touch git.
- Do NOT create branches, commit, push, or open PRs.
- Your outputs go to: **wandb** (experiment logs), **results/** (artifacts), **scratch files**.
- Session scripts you create in `scripts/` will be moved to `scripts/archive/` after runs (gitignored). Do NOT expect them to persist across sessions. You may read `scripts/archive/` to recreate a result or analyze a prior run, but keep it targeted — scanning many archived scripts is expensive.
- If you need a platform change (new feature, bug fix, data pipeline), write `GATE_REQUEST.md` to the project root explaining what you need and exit. The orchestrator will pause and an oversight session will handle it.

Git workflow (oversight/manager sessions only):
- NEVER commit to `main`. Use `phase-N/short-description` branches.
- At phase completion: push branch, open PR via `gh pr create`
- CI runs automatically on PRs to main. Merge only if CI passes.
- Branch naming: `oversight/`, `contract-NNN/`, `fix/`, `feat/`, `phase-N/`

## Holdout Data Policy
See `configs/holdout_policy.yaml` — IMMUTABLE. Do NOT hardcode OOS dates anywhere.
- The OOS boundary and embargo days are defined in that config file. Read them dynamically.
- **Data files are split at the holdout boundary.** All files in `data/` contain ONLY in-sample data. No matter which library you use (pandas, polars, pyarrow), you will only see IS data.
- OOS data is stored in a vault that you may NOT access directly.
- The data loader enforces holdout for `purpose="training"`. Use `purpose="analysis"` for exploration (IS only).
- OOS evaluation requires EXPLICIT WRITTEN APPROVAL from AK (human). AK will provide an authorized HoldoutGuard.
- Each model gets exactly ONE OOS evaluation. No repeated peeking.
- You may NOT modify `configs/holdout_policy.yaml`. The pre-commit hook and orchestrator will block this.
- You may NOT reference or access `data/.oos_vault/` in any code. The pre-commit hook will block this.

## Graduated Success Thresholds

| Tier | Criteria | Action |
|------|----------|--------|
| TIER 1 — Deploy | Sharpe ≥1.0, MC >80%, MaxDD <50% | Request deployment approval |
| TIER 2 — Paper Trade | Sharpe ≥0.7, MC >70%, beats B&H | Build paper trading |
| TIER 3 — Continue | Sharpe ≥0.4, shows edge over B&H | Keep iterating |
| TIER 4 — Pivot | Sharpe <0.4 after 5+ configs | Different approach |
| TIER 5 — Abandon | Sharpe <0.2 after 10+ configs, 2+ families | Document, move on |

## Mandatory Exploration Depth
Before declaring ANY approach "failed":
1. At least **5 meaningfully different configs** tested
2. At least **2 hours of wall-clock time**
3. At least **2 implementation variants**
4. At least **1 ablation study**
5. Document ALL results

## Pre-Validation Protocol
1. Run standard validation: all years, holdout separate, baseline comparison, look-ahead check
2. Report as "PRELIMINARY" until validated
3. NEVER use "breakthrough" or "genuine alpha" until TIER 1 validated

## Sub-Period Validation (MANDATORY)
Every config that beats the baseline MUST report sub-period metrics alongside full-period results.
Sub-periods: **2017+** (removes early low-liquidity era) and **2020+** (post-COVID, includes 2022 bear).
For each period: Sharpe, MaxDD, annual return, n_trades, win rate, and buy-and-hold Sharpe for comparison.
```python
from sweep_utils import subperiod_analysis
sp = subperiod_analysis(prices, positions, cost_frac)
# Returns: {"full": {...}, "2017+": {...}, "2020+": {...}}
```
A strategy with Sharpe 1.98 on 2013-2023 but Sharpe 0.4 on 2020-2023 is NOT deployment-worthy.
Do not declare any result as "beating baseline" without sub-period confirmation.

## Human Gates (stop and wait)
- Before live API calls that cost money
- Before paper/live trading goes live
- Before adding paid data sources
- When TIER 1 result needs deployment approval

## Manager Session Protocol
The Opus manager tracks all infrastructure sessions for audit trail:
```python
from sparky.tracking.manager_log import ManagerLog
mlog = ManagerLog()
session = mlog.start_session(objective="Contract 005 setup", branch="manager/contract-005")
# ... work ...
mlog.end_session(session, summary="Completed guardrails + workflow")
```
All manager decisions, sub-agent spawns, and research launches are logged to `logs/manager_sessions/session_log.jsonl`.

## Trading Rules
See `configs/trading_rules.yaml` — IMMUTABLE.
