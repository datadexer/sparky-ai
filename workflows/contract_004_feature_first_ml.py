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
                + TAG_REMINDER + " Step tag: 'feature_analysis'."
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
                + TAG_REMINDER + " Step tag: 'sweep'."
            ),
            done_when=_sweep_done,
            max_duration_minutes=180,
            max_retries=3,
            tags=["contract_004", "sweep"],
        ),
        Step(
            name="regime_aware_hybrid",
            prompt=(
                "Build and test regime-aware Donchian hybrid strategies.\n\n"
                "CONTEXT — Why you're doing this:\n"
                "Direct ML prediction of price direction has shown AUC ~0.57 and Sharpe below "
                "the Donchian baseline (1.062). The hypothesis: ML can't reliably predict direction, "
                "but it CAN identify market regimes. Donchian works well in trending markets but "
                "gets chopped up in ranging markets. If we can filter out the bad periods, we "
                "might improve Donchian's risk-adjusted returns.\n\n"
                "BEFORE YOU START:\n"
                "- Read `results/sweep_summary.md` to understand what the sweep step found.\n"
                "- Think about WHY regime detection might work where direct prediction didn't.\n"
                "- The key insight: regime classification is an easier problem than direction prediction.\n\n"
                "TEST THREE DISTINCT REGIME METHODS:\n"
                "1. Volatility-based (rules, no ML): ATR ratio, Bollinger bandwidth, rolling "
                "   std changes. At least 2-3 parameter variants.\n"
                "2. Trend strength indicators: ADX thresholds, moving average slope/alignment, "
                "   price vs MA distance. At least 2-3 variants.\n"
                "3. ML regime classifier: Train XGBoost/LightGBM (GPU) to classify regime using "
                "   features. Use unsupervised labels (e.g. k-means on returns/volatility). "
                "   At least 2-3 variants.\n\n"
                "AFTER individual methods, test AT LEAST 2 combinations:\n"
                "- Voting ensemble of regime detectors\n"
                "- Hierarchical: ML confirms volatility-based regime\n\n"
                "For each regime method:\n"
                "- Apply as filter to Donchian: only trade when regime='trending'\n"
                "- Walk-forward validate the combined system\n"
                "- Report: Sharpe, max drawdown, % of time in market, # of trades\n\n"
                "IMPORTANT: Do NOT declare this approach failed after only 2-3 attempts. "
                "The mandatory exploration depth requires at least 5 meaningfully different configs, "
                "2+ implementation variants, and 1 ablation study. Exhaust the search space.\n\n"
                "AFTER YOU FINISH:\n"
                "Write `results/regime_summary.md` with:\n"
                "- All regime methods tested + results vs plain Donchian (1.062)\n"
                "- Which regime detection approach worked best and why\n"
                "- The key tradeoff: filtering improves Sharpe but reduces exposure\n"
                "- Honest assessment of regime-aware vs plain Donchian\n\n"
                "Use `@with_timeout(seconds=900)` per config.\n"
                "Log to wandb with tags=['contract_004', 'regime'].\n"
                "Target: at least 8 wandb runs tagged ['contract_004', 'regime']."
                + TAG_REMINDER + " Step tag: 'regime'."
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
                + TAG_REMINDER + " Step tag: 'ensemble'."
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
                "Explore genuinely novel strategy ideas that go beyond the standard approaches "
                "tested in previous steps.\n\n"
                "BEFORE YOU START:\n"
                "- Read ALL previous summary files:\n"
                "  - `results/sweep_summary.md`\n"
                "  - `results/regime_summary.md`\n"
                "  - `results/ensemble_summary.md`\n"
                "- Query wandb for ALL contract_004 runs. Understand the full landscape of what's "
                "  been tried and what worked/didn't.\n"
                "- Synthesize: What patterns emerge? Where are the gaps?\n\n"
                "TEST AT LEAST 2 genuinely novel ideas from this list (or your own):\n\n"
                "1. Asymmetric strategies: Different rules for long vs short. BTC has structural "
                "   long bias — maybe long entries need less confirmation than shorts.\n"
                "2. Multi-timeframe fusion: Combine hourly signals with daily trend direction. "
                "   Trade hourly only when daily trend agrees.\n"
                "3. Target engineering: Instead of predicting next-bar return, predict max "
                "   favorable excursion (MFE), or 4h/8h/24h forward returns, or risk-adjusted "
                "   returns. The prediction target matters as much as the features.\n"
                "4. Adaptive position sizing: Kelly criterion or volatility-targeted sizing "
                "   applied to the best signal from previous steps.\n"
                "5. Feature interactions: Create polynomial or ratio features from the top "
                "   predictors. Sometimes the interaction (e.g. momentum/volatility) matters "
                "   more than individual features.\n"
                "6. Alternative loss functions: Train with asymmetric loss (penalize drawdowns "
                "   more than missed gains), or directly optimize Sharpe via custom objective.\n\n"
                "For each novel approach:\n"
                "- Explain the hypothesis BEFORE implementing\n"
                "- Walk-forward validate\n"
                "- Compare vs Donchian baseline (1.062) and best result from previous steps\n"
                "- Log 1-sentence interpretation in wandb notes\n\n"
                "AFTER YOU FINISH:\n"
                "Write `results/novel_exploration_summary.md` with:\n"
                "- Each novel idea tested, the hypothesis, and the result\n"
                "- Surprising findings (positive or negative)\n"
                "- The single best strategy found across ALL 5 workflow steps\n"
                "- Final recommendation for the contract\n\n"
                "Use `@with_timeout(seconds=900)` per config. GPU for all training.\n"
                "Log to wandb with tags=['contract_004', 'novel'].\n"
                "Target: at least 4 wandb runs tagged ['contract_004', 'novel']."
                + TAG_REMINDER + " Step tag: 'novel'."
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
