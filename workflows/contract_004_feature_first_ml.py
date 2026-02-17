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
    return _get_tracker().count_runs(tags=["contract_004", "ensemble"]) >= 3


def _novel_done() -> bool:
    return _get_tracker().count_runs(tags=["contract_004", "novel"]) >= 4


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
                + TAG_REMINDER + NAMING_REMINDER + " Step tag: 'feature_analysis'."
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
                + TAG_REMINDER + NAMING_REMINDER + " Step tag: 'sweep'."
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
                + TAG_REMINDER + NAMING_REMINDER + " Step tag: 'regime'."
            ),
            done_when=_regime_done,
            max_duration_minutes=180,
            max_retries=3,
            tags=["contract_004", "regime"],
        ),
        Step(
            name="ensemble",
            prompt=(
                "Build ensemble strategies combining the best approaches from previous steps.\n\n"
                "BEFORE YOU START:\n"
                "- Read `results/sweep_summary.md` — what were the top ML configs?\n"
                "- Read `results/regime_summary.md` — what were the best regime filters?\n"
                "- Query wandb for top 5 configs across all contract_004 runs (by walk-forward Sharpe).\n"
                "  These are your ensemble candidates.\n\n"
                "TEST THREE ENSEMBLE APPROACHES:\n"
                "1. Signal averaging: Equal-weight average of top 3-5 model signals. Simple but "
                "   often surprisingly effective. Also test inverse-volatility weighting.\n"
                "2. Regime-conditional ensemble: Use the best regime detector from step 3 to "
                "   switch between strategies. Trending regime → Donchian or trend-following ML. "
                "   Ranging regime → mean-reversion or flat.\n"
                "3. Stacking: Train a meta-learner (LightGBM GPU) on base model predictions. "
                "   Use walk-forward to avoid leakage — the meta-learner must only train on "
                "   out-of-fold predictions from the base models.\n\n"
                "For each ensemble:\n"
                "- Walk-forward validate (expanding window)\n"
                "- Compare vs Donchian baseline (Sharpe 1.062)\n"
                "- Compare vs best individual model from previous steps\n"
                "- Report: Sharpe, max drawdown, correlation with Donchian signals\n\n"
                "AFTER YOU FINISH:\n"
                "Write `results/ensemble_summary.md` with:\n"
                "- All ensemble approaches tested + results\n"
                "- Whether ensembling improved over best individual model\n"
                "- The best overall strategy found across ALL steps\n"
                "- Recommendation: which strategy (if any) merits OOS evaluation?\n\n"
                "Use `@with_timeout(seconds=900)` per config. GPU for all training.\n"
                "Log to wandb with tags=['contract_004', 'ensemble'].\n"
                "Target: at least 3 wandb runs tagged ['contract_004', 'ensemble']."
                + TAG_REMINDER + NAMING_REMINDER + " Step tag: 'ensemble'."
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
                "results/ensemble_summary.md (if exists).\n"
                "Query wandb for ALL contract_004 runs. You now have 40-60+ experiments of "
                "evidence.\n\n"
                "Your job: SYNTHESIZE what you've learned and test ideas that COMBINE insights "
                "from previous steps in ways they weren't tested individually.\n\n"
                "You must test AT LEAST 4 genuinely novel ideas. Here are candidates — pick at "
                "least 4, or invent your own if you have a better hypothesis:\n\n"
                "1. ASYMMETRIC REGIME RESPONSE: Use one regime method for entries, a different "
                "one for exits. Maybe vol-regime is better at knowing when to GET IN, but "
                "drawdown-filter is better at knowing when to GET OUT.\n\n"
                "2. ADAPTIVE DONCHIAN PARAMETERS: Instead of fixed Donchian lookback, adjust "
                "the channel width based on regime. Tight channels in trending markets (capture "
                "moves faster), wide channels in uncertain markets (avoid whipsaws). The regime "
                "detector controls the strategy parameters, not just on/off.\n\n"
                "3. MULTI-TIMEFRAME REGIME: Classify regime on WEEKLY data, execute Donchian on "
                "DAILY data. Higher timeframe regime is likely smoother and more predictable "
                "than daily.\n\n"
                "4. ML AS POSITION SIZER, NOT SIGNAL: Don't use ML to decide WHEN to trade "
                "(Donchian does that). Use ML to decide HOW MUCH. High ML confidence in "
                "trending regime = full position. Low confidence = quarter position. This "
                "reduces drawdown without missing trades entirely.\n\n"
                "5. TARGET RE-ENGINEERING: Instead of predicting next-day direction, predict: "
                "'will the Donchian breakout currently in progress result in a profitable "
                "trade?' This is a much more tractable question than price direction.\n\n"
                "6. WALK-FORWARD STABILITY FILTER: From all configs tested, find the one with "
                "the MOST CONSISTENT performance (lowest variance of yearly Sharpe) rather "
                "than highest mean. A strategy that does Sharpe 0.6 every year is more "
                "deployable than one that does 2.0 and -1.0.\n\n"
                "7. REGIME ENSEMBLE: Take the top 3 regime methods. Trade only when 2 of 3 "
                "agree. Majority-vote regime filter should be more robust than any "
                "individual.\n\n"
                "For each idea:\n"
                "- State the hypothesis BEFORE running anything\n"
                "- Walk-forward validate against the same 2019-2023 period\n"
                "- Compare to best previous result\n"
                "- Log to wandb with tags 'contract_004', 'novel'\n"
                "- Write interpretation in wandb notes\n\n"
                "Write results/novel_exploration_summary.md with:\n"
                "- Which ideas worked and which didn't\n"
                "- The single best overall configuration across ALL steps\n"
                "- Whether it's deployment-worthy (Sharpe > 0.8, variance < 1.0, 2022 not "
                "catastrophic)\n"
                "- Recommended next steps\n\n"
                "This step is about CREATIVE SYNTHESIS, not brute-force parameter search. Think "
                "about what the data has been telling you across 50+ experiments."
                + TAG_REMINDER + NAMING_REMINDER + " Step tag: 'novel'."
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
