"""CONTRACT #004: Feature-First ML workflow.

5 steps:
1. Feature analysis — importance ranking, select top features
2. Two-stage sweep — screen configs, walk-forward validate top 5
3. Regime-aware hybrid — Donchian + ML regime filter
4. Ensemble — combine best models (skip if best Sharpe < 0.4)
5. Novel exploration — creative strategies beyond standard approaches
"""

from pathlib import Path

from sparky.workflow.engine import Step, Workflow

TAG_REMINDER = (
    "\n\nCRITICAL: You must tag every wandb run with both 'contract_004' and the step tag "
    "(shown below). The workflow cannot verify your work without these tags."
)

NAMING_REMINDER = (
    "\n\nRUN NAMING & GROUPING: When calling log_experiment(), pass `name=None` to auto-generate "
    "a descriptive name, or use format like 'xgb_lr0.05_d6_S0.83'. Do NOT use generic names "
    "like 'config1'. Pass `job_type='<step_tag>'` and `group='<approach>'` to organize runs. "
    "The group creates collapsible parent/child rows in wandb (e.g. group='sweep_xgboost' "
    "for all XGBoost configs, group='regime_vol' for volatility regime runs, "
    "group='regime_adx' for ADX runs). Runs sharing a group collapse into one row."
)

SUBPERIOD_REMINDER = (
    "\n\nSUB-PERIOD VALIDATION (MANDATORY): Every config that beats the baseline MUST "
    "report sub-period metrics. Use `from sweep_utils import subperiod_analysis` and call "
    "`subperiod_analysis(prices, positions, cost_frac)`. This returns metrics for full period, "
    "2017+ (removes early low-liquidity era), and 2020+ (post-COVID, includes 2022 bear). "
    "Each sub-period includes Sharpe, MaxDD, annual return, n_trades, win rate, and buy-and-hold "
    "Sharpe. Report all sub-periods for any result claimed as beating baseline."
)

METRICS_REMINDER = (
    "\n\nMETRICS (MANDATORY): When logging results, always use:\n"
    "```python\n"
    "from sparky.tracking.metrics import compute_all_metrics\n"
    "metrics = compute_all_metrics(returns, n_trials=TOTAL_RUNS_SO_FAR)\n"
    "```\n"
    "The primary success metric is Deflated Sharpe Ratio (DSR) > 0.95, NOT raw Sharpe.\n"
    "A high Sharpe with low DSR means the result is likely a statistical fluke from multiple testing."
)


def _get_tracker():
    """Lazy import to avoid wandb init at module load."""
    from sparky.tracking.experiment import ExperimentTracker

    return ExperimentTracker(experiment_name="contract_004")


def _feature_analysis_done() -> bool:
    return Path("results/feature_importance.json").exists()


def _sweep_done() -> bool:
    return _get_tracker().count_runs(tags=["contract_004", "sweep"]) >= 20


def _regime_done() -> bool:
    return _get_tracker().count_runs(tags=["contract_004", "regime"]) >= 8


def _ensemble_done() -> bool:
    return _get_tracker().count_runs(tags=["contract_004", "ensemble"]) >= 30


def _novel_done() -> bool:
    return _get_tracker().count_runs(tags=["contract_004", "novel"]) >= 15


def _skip_ensemble() -> bool:
    best = _get_tracker().best_metric("sharpe", tags=["contract_004"])
    if best is None:
        return False
    return best < 0.4


def build_workflow() -> Workflow:
    """Build the CONTRACT #004 workflow."""
    steps = [
        Step(
            name="feature_analysis",
            prompt=(
                "Perform feature importance analysis on the BTC hourly features dataset.\n\n"
                "1. Load data: `from sparky.data.loader import load; df = load('btc_1h_features', purpose='training')`\n"
                "2. Compute feature importance using XGBoost (GPU) with `tree_method='hist', device='cuda'`\n"
                "3. Use mutual information, correlation analysis, and tree-based importance\n"
                "4. Select top 15-20 features, save ranking to `results/feature_importance.json`\n"
                "5. Use `@with_timeout(seconds=900)` for any training calls\n"
                "6. Log results to wandb with tags=['contract_004', 'feature_analysis']\n\n"
                "Output: `results/feature_importance.json` with ranked features and scores."
                + TAG_REMINDER
                + NAMING_REMINDER
                + METRICS_REMINDER
                + SUBPERIOD_REMINDER
                + " Step tag: 'feature_analysis'."
            ),
            done_when=_feature_analysis_done,
            max_duration_minutes=120,
            max_retries=3,
            tags=["contract_004", "feature_analysis"],
        ),
        Step(
            name="two_stage_sweep",
            prompt=(
                "Run a rigorous two-stage hyperparameter sweep. Your goal is to find ML configs "
                "that beat the Donchian baseline (Sharpe 1.062 in-sample).\n\n"
                "BEFORE YOU START:\n"
                "- Check wandb for existing contract_004/sweep runs. Do NOT repeat configs that "
                "already exist. Build on what's been tried.\n"
                "- Read `results/feature_importance.json` for the top features.\n\n"
                "STAGE 1 — Screening (single 80/20 split):\n"
                "Test at least 20 configs total across ALL THREE model families:\n"
                "- XGBoost (`tree_method='hist', device='cuda'`) — at least 6 configs\n"
                "- LightGBM (`device='gpu'`) — at least 6 configs\n"
                "- CatBoost (`task_type='GPU', devices='0'`) — at least 6 configs\n\n"
                "For each family, vary meaningfully: learning rate (0.01-0.3), max_depth (3-8), "
                "regularization (L1/L2), n_estimators (100-1000), feature subsets (top 10 vs 15 vs 20), "
                "and any family-specific params (e.g. CatBoost border_count, LightGBM num_leaves).\n\n"
                "For EVERY config, log a 1-sentence interpretation in the wandb notes field explaining "
                "WHY you think it performed the way it did (e.g. 'High LR + deep trees overfits — "
                "train Sharpe 2.1 but val only 0.3').\n\n"
                "STAGE 2 — Walk-forward validation:\n"
                "Take top 5 configs from Stage 1, run expanding-window walk-forward validation. "
                "This is the real test. Report walk-forward Sharpe, not just the 80/20 split.\n\n"
                "AFTER YOU FINISH:\n"
                "Write `results/sweep_summary.md` with:\n"
                "- Table of all configs tested + Sharpe results (Stage 1 and Stage 2)\n"
                "- Which model family performed best and your hypothesis about why\n"
                "- Top 5 configs ranked by walk-forward Sharpe\n"
                "- Honest assessment: does any config reliably beat Donchian 1.062?\n\n"
                "Use `@with_timeout(seconds=900)` per config.\n"
                "Log each batch with `log_sweep()`, tags=['contract_004', 'sweep'].\n"
                "Log walk-forward validated results with `log_experiment()`, tags=['contract_004', 'sweep'].\n"
                "Target: at least 20 wandb runs tagged ['contract_004', 'sweep']."
                + TAG_REMINDER
                + NAMING_REMINDER
                + METRICS_REMINDER
                + SUBPERIOD_REMINDER
                + " Step tag: 'sweep'."
            ),
            done_when=_sweep_done,
            max_duration_minutes=180,
            max_retries=3,
            tags=["contract_004", "sweep"],
        ),
        Step(
            name="regime_aware_hybrid",
            prompt=(
                "CONTEXT: Direct ML prediction beat baseline slightly (Sharpe 1.365 vs 1.062) "
                "but with catastrophic 2022 failure and +/-1.7 variance. The hypothesis: if we "
                "can DETECT bear/choppy regimes and go flat during them, we eliminate the "
                "catastrophic drawdowns while keeping the upside.\n\n"
                "Read results/sweep_summary.md before coding anything.\n\n"
                "You must test AT LEAST 4 DISTINCT regime detection approaches. 'Distinct' means "
                "different logic, not different parameters of the same method. Each approach "
                "below is REQUIRED:\n\n"
                "APPROACH 1 — Volatility regime (simple baseline):\n"
                "- Rolling realized vol (try 10, 20, 50 day windows)\n"
                "- Above median vol = trending -> trade Donchian. Below = choppy -> flat.\n"
                "- This is your simplest baseline. If this works, complexity isn't needed.\n\n"
                "APPROACH 2 — Trend strength (ADX-based):\n"
                "- ADX indicator with threshold (try 20, 25, 30)\n"
                "- ADX > threshold = trending -> trade. Below = flat.\n"
                "- Different from vol: a market can be volatile but directionless (whipsaw).\n\n"
                "APPROACH 3 — ML regime classifier:\n"
                "- Label historical periods: 'trending' = periods where Donchian would have "
                "profited, 'choppy' = where it lost\n"
                "- Train classifier (CatBoost, GPU) on top features to predict regime label\n"
                "- Trade Donchian only when classifier says 'trending'\n"
                "- This is META-LEARNING: you're predicting strategy performance, not price.\n\n"
                "APPROACH 4 — Multi-signal regime (combine 2+ of the above):\n"
                "- Trade only when BOTH vol AND trend say 'trending'\n"
                "- Or: use ML classifier with vol and ADX as additional features\n"
                "- Test at least 2 combinations\n\n"
                "APPROACH 5 — Rolling drawdown filter:\n"
                "- If Donchian strategy has drawn down >X% in the last N days, go flat\n"
                "- Simple but powerful: directly cuts off losing streaks\n"
                "- Try multiple thresholds (5%, 10%, 15% drawdown over 20, 40, 60 days)\n\n"
                "For EACH approach:\n"
                "- Run Donchian ONLY in 'trade' regime, go flat in 'no-trade' regime\n"
                "- Walk-forward validate (expanding window, 2019-2023, yearly folds)\n"
                "- Compare to: unfiltered Donchian (1.062), buy-and-hold, best ML from sweep\n"
                "- Log to wandb with tags 'contract_004', 'regime', and a descriptive run name "
                "like 'vol_regime_20d' or 'adx_regime_25'\n"
                "- In wandb notes: 1-sentence interpretation of WHY this result happened\n\n"
                "After all 5 approaches, write results/regime_summary.md answering:\n"
                "1. Which regime method reduced 2022 drawdown the most?\n"
                "2. Which maintained the highest Sharpe during trending periods?\n"
                "3. Is the combined approach better than individuals?\n"
                "4. Does the ML meta-learner add value over simple indicators?\n"
                "5. What is the BEST regime-filtered Donchian config, and is it deployment-worthy?\n\n"
                "IMPORTANT:\n"
                "- 8 wandb runs minimum, but quality > quantity. Each run should test a genuinely "
                "different idea.\n"
                "- If an approach fails, explain WHY in the wandb notes, then move on. Don't "
                "repeat failures with minor tweaks.\n"
                "- If an approach WORKS (Sharpe > 1.0 with lower variance than unfiltered), test "
                "2-3 parameter variations to find the robust configuration.\n"
                "- The 2022 bear market is your key test. Any regime method that doesn't reduce "
                "2022 losses is useless.\n"
                "- Use GPU for all training. Use data loader. Use 15-min timeout per config.\n\n"
                "Do NOT stop after finding one thing that works. Test ALL 5 approaches. The "
                "done_when requires 8 runs and you need all 5 approaches represented."
                + TAG_REMINDER
                + NAMING_REMINDER
                + METRICS_REMINDER
                + SUBPERIOD_REMINDER
                + " Step tag: 'regime'."
            ),
            done_when=_regime_done,
            max_duration_minutes=180,
            max_retries=3,
            tags=["contract_004", "regime"],
        ),
        Step(
            name="ensemble",
            prompt=(
                "Build on the SPECIFIC findings from previous steps. Do NOT start from scratch.\n"
                "Check wandb for all existing contract_004/ensemble runs before coding.\n\n"
                "LATEST FINDINGS (from 22 ensemble runs already done):\n"
                "- mom_vol_adx_OR: Sharpe 1.28 — momentum + regime filter is the BEST combo so far\n"
                "- mom_adx30: Sharpe 1.14 — ADX-only filter on momentum also strong\n"
                "- don_lgbm_avg_adx30: Sharpe 1.10 — signal averaging works\n"
                "- stacking_lgbm_meta: Sharpe 1.03 — stacking modest\n"
                "- mom_lb40_t0: Sharpe 1.02, mom_lb40_t10: 1.00 — raw momentum decent\n"
                "- ML meta-learner on momentum: Sharpe 0.64-0.72 — ML meta adds no value\n"
                "- LightGBM + regime combos: 0.73-0.75 — disappointing, regime kills ML signal\n\n"
                "EARLIER REGIME FINDINGS:\n"
                "- vol_adx_AND: Sharpe 1.05, 2022=+0.58, std=0.50 (only config positive in 2022)\n"
                "- vol_adx_OR: Sharpe 1.23, 2022=-0.45\n"
                "- ADX(14,30): Sharpe 1.21, 2022=-0.28\n"
                "- Unfiltered Donchian baseline: Sharpe 1.062, 2022=-0.807\n\n"
                "Read results/sweep_summary.md, results/regime_summary.md, and any "
                "results/ensemble_summary.md.\n\n"
                "THE BEST LEAD: Momentum + regime filtering (Sharpe 1.28). Explore this deeply:\n\n"
                "1. MOMENTUM + vol_adx_OR parameter sweep: The mom_vol_adx_OR run at 1.28 used "
                "default params. Now vary: momentum lookback (10, 20, 40, 60), threshold (0, 0.02, "
                "0.05, 0.10), and vol/ADX window sizes. Find the robust optimum.\n\n"
                "2. MOMENTUM + vol_adx_AND: The AND filter was positive in 2022. Test momentum "
                "with the stricter filter. May sacrifice some Sharpe but gain 2022 protection.\n\n"
                "3. ADAPTIVE MOMENTUM SIZING: Instead of binary momentum signal, use momentum "
                "magnitude for position sizing. Strong momentum = full position, weak = quarter.\n\n"
                "4. MOMENTUM + DONCHIAN CONFIRMATION: Only trade momentum when Donchian also "
                "confirms (breakout in same direction). Dual confirmation should reduce whipsaws.\n\n"
                "5. REGIME-CONDITIONAL STRATEGY SELECTION: When ADX > 30 use momentum (better in "
                "strong trends), when 20 < ADX < 30 use Donchian (better in mild trends), "
                "when ADX < 20 go flat.\n\n"
                "6. WALK-FORWARD the top 3 ensemble results from previous session (mom_vol_adx_OR, "
                "mom_adx30, don_lgbm_avg_adx30) if not already walk-forward validated.\n\n"
                "For each:\n"
                "- Walk-forward validate (expanding window, 2019-2023, yearly folds)\n"
                "- Report yearly Sharpe breakdown (especially 2022)\n"
                "- Compare vs Donchian 1.062 AND vs mom_vol_adx_OR 1.28\n"
                "- Report: mean Sharpe, std, max DD, 2022 Sharpe specifically\n\n"
                "Update results/ensemble_summary.md with ALL results.\n\n"
                "Use GPU, data loader, 15-min timeout per config.\n"
                "Log to wandb with tags=['contract_004', 'ensemble'].\n"
                "Target: at least 30 total wandb runs tagged ['contract_004', 'ensemble']."
                + TAG_REMINDER
                + NAMING_REMINDER
                + METRICS_REMINDER
                + SUBPERIOD_REMINDER
                + " Step tag: 'ensemble'."
            ),
            done_when=_ensemble_done,
            skip_if=_skip_ensemble,
            max_duration_minutes=180,
            max_retries=3,
            tags=["contract_004", "ensemble"],
        ),
        Step(
            name="novel_exploration",
            prompt=(
                "Read ALL summary files: results/sweep_summary.md, results/regime_summary.md, "
                "results/ensemble_summary.md, results/novel_exploration_summary.md.\n"
                "Check wandb for all existing contract_004/novel runs before coding.\n\n"
                "LATEST FINDINGS (from 100+ experiments across all steps):\n"
                "- BEST OVERALL: mom_vol_adx_OR Sharpe 1.28 (momentum + regime filter)\n"
                "- breakout_profitability_cat: Sharpe 1.24 (target re-engineering with CatBoost)\n"
                "- majority_vote_2of3: Sharpe 1.20\n"
                "- mom_adx30: Sharpe 1.14\n"
                "- asym_adx30_dd12pct: Sharpe 1.11\n"
                "- regime_mom_lb40_t0_adx30: Sharpe 1.04\n"
                "- stability_opt (std=0.46): Sharpe 1.03 — most stable config found\n"
                "- vol_adx_AND: Sharpe 1.05, 2022=+0.58 — only config positive in 2022\n"
                "- LightGBM + regime: 0.73-0.75 — ML signal killed by regime filter\n"
                "- ML meta-learner: adds no value (0.64-0.72)\n\n"
                "Your job: DEEPEN the most promising leads, not add more breadth.\n\n"
                "PRIORITY 1 — Momentum + regime parameter optimization:\n"
                "The mom_vol_adx_OR combo at 1.28 is the best result. But was it tested across "
                "enough parameter combinations? Sweep:\n"
                "- Momentum lookback: 10, 20, 30, 40, 60 periods\n"
                "- Momentum threshold: -0.02, 0, 0.02, 0.05, 0.10\n"
                "- Vol window: 10, 20, 50 days\n"
                "- ADX period: 10, 14, 20; threshold: 20, 25, 30\n"
                "Find the robust optimal region, not just a single good point.\n\n"
                "PRIORITY 2 — Target re-engineering deep dive:\n"
                "breakout_profitability_cat at 1.24 is the second-best novel result. Expand:\n"
                "- Try XGBoost and LightGBM (not just CatBoost) for profitability prediction\n"
                "- Vary the definition of 'profitable breakout' (next 5, 10, 20 candles)\n"
                "- Add regime features (ADX, vol) to the profitability classifier\n\n"
                "PRIORITY 3 — Stability optimization:\n"
                "stability_opt at 1.03 with std=0.46 is the most deployable. Can we improve it?\n"
                "- Explicitly optimize for Sharpe / std ratio across yearly folds\n"
                "- Test weighting 2022 more heavily in the optimization\n"
                "- Combine stability_opt with momentum confirmation\n\n"
                "PRIORITY 4 — Robustness testing on top 3:\n"
                "Take the top 3 configs (mom_vol_adx_OR 1.28, breakout_cat 1.24, majority_vote "
                "1.20) and stress-test:\n"
                "- Perturb all parameters by +/-20% — does Sharpe stay above 1.0?\n"
                "- Test on different train window sizes (2 yr, 3 yr, 4 yr)\n"
                "- If robust to perturbation, it's real. If not, it's overfit.\n\n"
                "For each:\n"
                "- Walk-forward validate (2019-2023, yearly folds)\n"
                "- Report: mean Sharpe, std, 2022 Sharpe, max DD\n"
                "- Compare to mom_vol_adx_OR 1.28 and stability_opt 1.03 as benchmarks\n"
                "- Log to wandb with tags 'contract_004', 'novel'\n\n"
                "Update results/novel_exploration_summary.md with:\n"
                "- Parameter sensitivity analysis for momentum+regime\n"
                "- Robustness test results (do top configs survive perturbation?)\n"
                "- Final ranked list of deployment candidates\n\n"
                "Use GPU, data loader, 15-min timeout per config.\n"
                "Target: at least 15 total wandb runs tagged ['contract_004', 'novel']."
                + TAG_REMINDER
                + NAMING_REMINDER
                + METRICS_REMINDER
                + SUBPERIOD_REMINDER
                + " Step tag: 'novel'."
            ),
            done_when=_novel_done,
            max_duration_minutes=240,
            max_retries=3,
            tags=["contract_004", "novel"],
        ),
    ]

    return Workflow(
        name="contract-004",
        steps=steps,
        max_hours=12.0,
    )
