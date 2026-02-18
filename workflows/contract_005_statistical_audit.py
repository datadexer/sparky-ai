"""CONTRACT #005: Statistical Audit workflow.

3 steps:
1. audit_existing_runs — Run DSR audit on all Contract 004 wandb runs
2. validate_metrics_integration — End-to-end pipeline test with guardrails on top configs
3. summary_report — Final verdict and Contract 006 recommendation
"""

from pathlib import Path

from sparky.workflow.engine import Step, Workflow

TAG_REMINDER = (
    "\n\nCRITICAL: You must tag every wandb run with both 'contract_005' and the step tag "
    "(shown below). The workflow cannot verify your work without these tags."
)

METRICS_REMINDER = (
    "\n\nMETRICS (MANDATORY): When logging results, always use:\n"
    "```python\n"
    "from sparky.tracking.metrics import compute_all_metrics\n"
    "metrics = compute_all_metrics(returns, n_trials=TOTAL_RUNS_SO_FAR)\n"
    "```\n"
    "The primary success metric is Deflated Sharpe Ratio (DSR) > 0.95, NOT raw Sharpe.\n"
    "A high Sharpe with low DSR means the result is likely a statistical fluke from multiple testing.\n\n"
    "GUARDRAILS USAGE:\n"
    "```python\n"
    "from sparky.tracking.guardrails import run_pre_checks, run_post_checks, has_blocking_failure, log_results\n"
    "from sparky.tracking.metrics import compute_all_metrics\n"
    "```\n"
    "Always run pre_checks before backtest and post_checks after. If has_blocking_failure() returns True, "
    "do not log results as valid — log the failure reason instead."
)


def _audit_done() -> bool:
    return Path("results/contract_005_audit.md").exists()


def _validation_done() -> bool:
    return Path("results/contract_005_validation.md").exists()


def _summary_done() -> bool:
    return Path("results/contract_005_summary.md").exists()


def build_workflow() -> Workflow:
    """Build the CONTRACT #005 workflow."""
    steps = [
        Step(
            name="audit_existing_runs",
            prompt=(
                "Run the Contract 004 DSR audit script and analyze the results.\n\n"
                "STEP 1 — Run the audit script:\n"
                "```\n"
                "python scripts/audit_contract_004.py\n"
                "```\n"
                "This will:\n"
                "- Pull all contract_004 wandb runs grouped by step tag (sweep, regime, ensemble, novel)\n"
                "- Compute expected_max_sharpe and DSR for each group\n"
                "- Write `results/contract_005_audit.md` with a per-group table\n"
                "- Generate `results/sharpe_distribution.png` (histogram of Sharpe distributions)\n"
                "- Save `results/contract_004_dsr_analysis.json` with raw analysis data\n\n"
                "STEP 2 — Analyze the results:\n"
                "Read the generated `results/contract_005_audit.md` and answer:\n"
                "1. How many total experiments were run across all contract_004 steps?\n"
                "2. What is the expected_max_sharpe for 187 experiments (the known total from contract_004)?\n"
                "   Use: `from sparky.tracking.metrics import expected_max_sharpe`\n"
                "   `T = 8760 * 5  # 5 years of hourly data`\n"
                "   `exp_max = expected_max_sharpe(187, T)`\n"
                "3. Which step(s) produced results that EXCEED the expected max Sharpe from noise?\n"
                "4. Does ANY run have DSR > 0.95 (truly significant after multiple testing correction)?\n"
                "5. For each step, is the best result signal or noise?\n\n"
                "STEP 3 — Augment the report if needed:\n"
                "If the script fails or produces incomplete results (e.g. no wandb runs found), "
                "manually compute the DSR analysis from what is available in `results/` directory:\n"
                "- Check `results/sweep_summary.md`, `results/regime_summary.md`, "
                "`results/ensemble_summary.md`, `results/novel_exploration_summary.md`\n"
                "- Extract all Sharpe values mentioned in those files\n"
                "- Compute DSR manually using:\n"
                "  ```python\n"
                "  from sparky.tracking.metrics import deflated_sharpe_ratio, expected_max_sharpe\n"
                "  import numpy as np\n"
                "  # Reconstruct returns from Sharpe: returns ~ Normal(sr/sqrt(T), 1/sqrt(T))\n"
                "  T = 8760 * 5\n"
                "  np.random.seed(42)\n"
                "  sr_best = 1.28  # best known result\n"
                "  returns = np.random.normal(sr_best / np.sqrt(T), 1.0 / np.sqrt(T), T)\n"
                "  dsr = deflated_sharpe_ratio(returns, n_trials=187)\n"
                "  ```\n"
                "- Write the augmented findings to `results/contract_005_audit.md`\n\n"
                "The file `results/contract_005_audit.md` MUST exist when this step completes.\n"
                "Log this audit run to wandb with tags=['contract_005', 'audit']."
                + TAG_REMINDER + METRICS_REMINDER + " Step tag: 'audit'."
            ),
            done_when=_audit_done,
            max_duration_minutes=60,
            max_retries=3,
            tags=["contract_005", "audit"],
        ),
        Step(
            name="validate_metrics_integration",
            prompt=(
                "Run the full metrics pipeline with guardrails on the TOP 3 configs from Contract 004.\n\n"
                "CONTEXT:\n"
                "The best results from contract_004 (from memory and summary files) are:\n"
                "1. mom_vol_adx_OR: Sharpe 1.28 — momentum + regime filter\n"
                "2. breakout_profitability_cat: Sharpe 1.24 — target re-engineering with CatBoost\n"
                "3. majority_vote_2of3: Sharpe 1.20 — majority vote ensemble\n\n"
                "Read `results/contract_005_audit.md` first to confirm these are still the top 3, "
                "or update this list if the audit found different results.\n\n"
                "FOR EACH of the top 3 configs, run the COMPLETE pipeline:\n\n"
                "```python\n"
                "from sparky.data.loader import load\n"
                "from sparky.tracking.metrics import compute_all_metrics\n"
                "from sparky.tracking.experiment import ExperimentTracker\n\n"
                "# Try to import guardrails — if not available, implement basic checks inline\n"
                "try:\n"
                "    from sparky.tracking.guardrails import (\n"
                "        run_pre_checks, run_post_checks, has_blocking_failure, log_results\n"
                "    )\n"
                "    GUARDRAILS_AVAILABLE = True\n"
                "except ImportError:\n"
                "    GUARDRAILS_AVAILABLE = False\n\n"
                "df = load('btc_1h_features', purpose='training')\n"
                "tracker = ExperimentTracker(experiment_name='contract_005')\n\n"
                "N_TOTAL_TRIALS = 187  # total contract_004 experiments\n\n"
                "for config_name, config in top_configs.items():\n"
                "    # 1. Pre-checks (look-ahead bias, data leakage)\n"
                "    if GUARDRAILS_AVAILABLE:\n"
                "        pre_result = run_pre_checks(df, config)\n"
                "        if has_blocking_failure(pre_result):\n"
                "            print(f'BLOCKED: {config_name} failed pre-checks: {pre_result}')\n"
                "            continue\n"
                "    else:\n"
                "        # Manual pre-checks: verify signal shift\n"
                "        assert 'signal' not in df.columns or True  # placeholder\n\n"
                "    # 2. Recreate the backtest for this config\n"
                "    returns = run_backtest(df, config)  # implement per config\n\n"
                "    # 3. Compute full metrics with n_trials=187\n"
                "    metrics = compute_all_metrics(returns, n_trials=N_TOTAL_TRIALS)\n\n"
                "    # 4. Post-checks\n"
                "    if GUARDRAILS_AVAILABLE:\n"
                "        post_result = run_post_checks(returns, metrics, config)\n"
                "        if has_blocking_failure(post_result):\n"
                "            print(f'BLOCKED: {config_name} failed post-checks: {post_result}')\n"
                "```\n\n"
                "If guardrails are not yet implemented in `src/sparky/tracking/guardrails.py`, "
                "implement basic versions inline:\n"
                "- Pre-check: verify signals are shifted (no look-ahead bias)\n"
                "- Pre-check: verify no NaN/Inf in returns\n"
                "- Post-check: verify Sharpe matches stored value within 0.05 tolerance\n"
                "- Post-check: verify DSR matches stored value within 0.02 tolerance\n\n"
                "CRITICAL — n_trials must be 187 for ALL metric computations in this step. "
                "This is the total number of configs tested across all contract_004 steps.\n\n"
                "Write `results/contract_005_validation.md` containing:\n"
                "- For each of the top 3 configs:\n"
                "  - Full metrics table (Sharpe, DSR, PSR, Sortino, MaxDD, Calmar, CVaR)\n"
                "  - Whether DSR > 0.95 (statistically significant)\n"
                "  - Pre-check and post-check pass/fail status\n"
                "  - Comparison to Donchian baseline (Sharpe 1.062)\n"
                "- Summary: how many configs passed all checks?\n"
                "- Are any configs deployment candidates (TIER 1: Sharpe >= 1.0, DSR > 0.95)?\n\n"
                "Log validation results to wandb with tags=['contract_005', 'validation'].\n"
                "Use `tracker.log_experiment()` for each validated config.\n"
                "The file `results/contract_005_validation.md` MUST exist when this step completes."
                + TAG_REMINDER + METRICS_REMINDER + " Step tag: 'validation'."
            ),
            done_when=_validation_done,
            max_duration_minutes=90,
            max_retries=3,
            tags=["contract_005", "validation"],
        ),
        Step(
            name="summary_report",
            prompt=(
                "Write the final Contract 005 summary report.\n\n"
                "Read ALL of the following files before writing:\n"
                "- `results/contract_005_audit.md` — DSR audit results by step\n"
                "- `results/contract_005_validation.md` — full pipeline validation results\n"
                "- `results/contract_004_dsr_analysis.json` — raw DSR analysis data\n"
                "- `results/sweep_summary.md`, `results/regime_summary.md`, "
                "`results/ensemble_summary.md`, `results/novel_exploration_summary.md`\n\n"
                "Write `results/contract_005_summary.md` containing:\n\n"
                "## 1. Contract 004 Statistical Verdict\n"
                "- Total experiments run: N\n"
                "- Expected max Sharpe from noise alone (N trials): X.XXX\n"
                "- Actual best Sharpe: X.XXX\n"
                "- Best DSR: X.XXX (significant if > 0.95)\n"
                "- VERDICT: [SIGNAL / NOISE / BORDERLINE]\n"
                "- Which step(s) produced statistically convincing results?\n"
                "- Which step(s) produced results indistinguishable from noise?\n\n"
                "## 2. Deployment Candidate Assessment\n"
                "For each config validated in contract_005 step 2:\n"
                "- Config name, Sharpe, DSR, MaxDD\n"
                "- Tier assignment (TIER 1-5 per CLAUDE.md thresholds)\n"
                "- Recommended action (deploy / paper trade / continue / pivot / abandon)\n\n"
                "TIER thresholds (from CLAUDE.md):\n"
                "- TIER 1 — Deploy: Sharpe >= 1.0, DSR > 0.95, MaxDD < 50%\n"
                "- TIER 2 — Paper Trade: Sharpe >= 0.7, DSR > 0.80, beats B&H\n"
                "- TIER 3 — Continue: Sharpe >= 0.4, shows edge over B&H\n"
                "- TIER 4 — Pivot: Sharpe < 0.4 after 5+ configs\n"
                "- TIER 5 — Abandon: Sharpe < 0.2 after 10+ configs, 2+ families\n\n"
                "## 3. Contract 006 Recommendations\n"
                "Based on the statistical audit and validation, recommend the focus for Contract 006.\n"
                "Choose ONE of:\n\n"
                "A) REQUEST OOS EVALUATION — If a config achieves TIER 1 (DSR > 0.95, Sharpe >= 1.0). "
                "Stop. Write: 'Requesting OOS evaluation approval from AK for [config_name].'\n\n"
                "B) FOCUSED OPTIMIZATION — If results are TIER 2-3 with DSR 0.80-0.95. "
                "Specify: which configs to deepen, which parameter regions to explore, "
                "minimum runs needed to reach DSR > 0.95.\n\n"
                "C) STRATEGY PIVOT — If all results have DSR < 0.80 across 187+ experiments. "
                "Specify: what fundamentally different approach to try next "
                "(e.g. options strategies, on-chain data, market microstructure, order flow).\n\n"
                "## 4. Decision\n"
                "End with a single, unambiguous decision:\n"
                "- 'PROCEED TO OOS EVALUATION: [config_name]' (requires AK approval per CLAUDE.md)\n"
                "- 'CONTINUE IN-SAMPLE: Contract 006 focuses on [specific approach]'\n"
                "- 'PIVOT STRATEGY: [new approach] because [reason]'\n\n"
                "Log the summary to wandb with tags=['contract_005', 'summary'].\n"
                "The file `results/contract_005_summary.md` MUST exist when this step completes."
                + TAG_REMINDER + METRICS_REMINDER + " Step tag: 'summary'."
            ),
            done_when=_summary_done,
            max_duration_minutes=30,
            max_retries=3,
            tags=["contract_005", "summary"],
        ),
    ]

    return Workflow(
        name="contract-005-statistical-audit",
        steps=steps,
        max_hours=6.0,
    )
