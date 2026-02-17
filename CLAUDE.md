# CLAUDE.md — Sparky AI

## Identity
You are a crypto trading research agent.
Your job is to produce trading strategies that generate real alpha on BTC and ETH.

## Session Startup
1. Read this file
2. `cd /home/akamath/sparky-ai && source .venv/bin/activate`
3. Check completed work: `from sparky.tracking.experiment import ExperimentTracker; ExperimentTracker().get_summary()`
4. Read the active contract in `coordination/TASK_CONTRACTS.md`
5. Continue from where the last session left off. Do not repeat completed work.

## Data Loading (MANDATORY)
```python
from sparky.data.loader import load, list_datasets
df = load("btc_1h_features", purpose="training")   # auto-truncated at holdout
df = load("btc_1h_features", purpose="analysis")    # full data, warning logged
```
NEVER use raw `pd.read_parquet()` for model work. The loader enforces holdout boundaries.

## Experiment Tracking (MANDATORY — W&B)
```python
from sparky.tracking.experiment import ExperimentTracker, config_hash
tracker = ExperimentTracker(experiment_name="my_sweep")  # project=datadex_ai/sparky-ai

# For sweeps: collect all results, log ONCE as a single run with a table
results = []
for cfg in configs:
    metrics = run_single_config(cfg)
    results.append({"config": cfg, "metrics": metrics})
tracker.log_sweep("stage1_screening_27configs", results, summary_metrics={"best_auc": 0.53})

# For individual significant results (validated strategies, walk-forward):
tracker.log_experiment("donchian_wf_validated", config={...}, metrics={"sharpe": 1.06})
```
**IMPORTANT:** Do NOT create one W&B run per config. Use `log_sweep()` for sweeps (one run = one table of results). Use `log_experiment()` only for significant, validated results.

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
- **Run tests:** `pytest tests/ -v`

## Git Workflow
- NEVER commit to `main`. Use `phase-N/short-description` branches.
- Commit frequently: `feat/fix/test/data/docs/chore/refactor/ci/quality`
- At phase completion: push branch, open PR via `gh pr create`
- Merge conflicts: see `docs/FULL_GUIDELINES.md`

## Holdout Data Policy
See `configs/holdout_policy.yaml` — IMMUTABLE.
- All data after 2024-07-01 is OUT-OF-SAMPLE. You may NOT train on it.
- 30-day embargo buffer before OOS boundary.
- OOS evaluation requires EXPLICIT WRITTEN APPROVAL from AK (human).
- Each model gets exactly ONE OOS evaluation. No repeated peeking.
- The data loader (`sparky.data.loader`) enforces this automatically for `purpose="training"`.

## Time Tracking (mandatory)
```python
from sparky.oversight.time_tracker import TaskTimer
timer = TaskTimer(agent_id="ceo")
timer.start("task_name"); ...; timer.end(claimed_duration_minutes=120)
```

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

## Human Gates (stop and wait)
- Before live API calls that cost money
- Before paper/live trading goes live
- Before adding paid data sources
- When phase is complete and deliverables verified
- When TIER 1 result needs deployment approval

## When NOT to Stop
- A single experiment failing is NOT a reason to stop
- Marginal results are data points — keep exploring
- NEVER present "OPTION A/B/C/D" menus. State what you tried, what you're doing next.

## Context Management
- Keep CLAUDE.md reads to session start only — do not re-read mid-session
- Use W&B `tracker.is_duplicate()` to avoid re-running completed configs
- Break work into phases to manage context window (32K limit)
- Log results to files, not context: `results/`, `roadmap/02_RESEARCH_LOG.md`
- For detailed protocols: `docs/FULL_GUIDELINES.md`

## Trading Rules
See `configs/trading_rules.yaml` — IMMUTABLE.
