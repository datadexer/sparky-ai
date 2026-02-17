"""CONTRACT #004: Feature-First ML workflow.

4 steps:
1. Feature analysis — importance ranking, select top features
2. Two-stage sweep — screen configs, walk-forward validate top 5
3. Regime-aware hybrid — Donchian + ML regime filter
4. Ensemble — combine best models (skip if best Sharpe < 0.4)
"""

from pathlib import Path

from sparky.workflow.engine import Step, Workflow


def _get_tracker():
    """Lazy import to avoid wandb init at module load."""
    from sparky.tracking.experiment import ExperimentTracker
    return ExperimentTracker(experiment_name="contract_004")


def _feature_analysis_done() -> bool:
    return Path("results/feature_importance.json").exists()


def _sweep_done() -> bool:
    return _get_tracker().count_runs(tags=["contract_004", "sweep"]) >= 10


def _regime_done() -> bool:
    return _get_tracker().count_runs(tags=["contract_004", "regime"]) >= 2


def _ensemble_done() -> bool:
    return _get_tracker().count_runs(tags=["contract_004", "ensemble"]) >= 1


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
            ),
            done_when=_feature_analysis_done,
            max_duration_minutes=120,
            max_retries=3,
            tags=["contract_004", "feature_analysis"],
        ),
        Step(
            name="two_stage_sweep",
            prompt=(
                "Run a two-stage hyperparameter sweep using the top features from "
                "`results/feature_importance.json`.\n\n"
                "1. Load features: read `results/feature_importance.json`, take top 15-20\n"
                "2. Load data: `from sparky.data.loader import load; df = load('btc_1h_features', purpose='training')`\n"
                "3. Stage 1: Screen configs on single 80/20 split. Test at least 10 configs across:\n"
                "   - XGBoost (`tree_method='hist', device='cuda'`)\n"
                "   - LightGBM (`device='gpu'`)\n"
                "   - CatBoost (`task_type='GPU', devices='0'`)\n"
                "   Vary: learning rate, depth, regularization, n_estimators\n"
                "4. Stage 2: Top 5 configs get walk-forward validation (expanding window)\n"
                "5. Use `@with_timeout(seconds=900)` per config\n"
                "6. Log each sweep batch to wandb with `log_sweep()` and tags=['contract_004', 'sweep']\n"
                "7. Log validated results with `log_experiment()` and tags=['contract_004', 'sweep']\n\n"
                "Baseline to beat: Donchian Sharpe 1.062. Report all results vs baseline."
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
                "The hypothesis: ML classifies market regime (trending/choppy), "
                "Donchian only trades in trending regimes.\n\n"
                "1. Load data: `from sparky.data.loader import load; df = load('btc_1h_features', purpose='training')`\n"
                "2. Implement at least 2 regime detection methods:\n"
                "   a. Volatility-based (ATR threshold, rolling std)\n"
                "   b. ML classifier (XGBoost GPU) trained on regime labels\n"
                "3. Combine: Donchian signals active only when regime='trending'\n"
                "4. Walk-forward validate the combined system\n"
                "5. Use `@with_timeout(seconds=900)` per config\n"
                "6. Log to wandb with tags=['contract_004', 'regime']\n"
                "7. Use `scripts/regime_aware_donchian.py` as reference if it exists\n\n"
                "Baseline: plain Donchian Sharpe 1.062. Can regime filtering improve it?"
            ),
            done_when=_regime_done,
            max_duration_minutes=180,
            max_retries=3,
            tags=["contract_004", "regime"],
        ),
        Step(
            name="ensemble",
            prompt=(
                "Build ensemble strategies combining the best models from previous steps.\n\n"
                "1. Review wandb results: find top performers from sweep and regime steps\n"
                "2. Build at least 2 ensemble approaches:\n"
                "   a. Signal averaging (equal weight)\n"
                "   b. Stacked ensemble (meta-learner on base model predictions)\n"
                "3. Walk-forward validate ensembles\n"
                "4. Use GPU for all training, `@with_timeout(seconds=900)` per config\n"
                "5. Log to wandb with tags=['contract_004', 'ensemble']\n"
                "6. Compare all results vs Donchian baseline (Sharpe 1.062)\n\n"
                "Write final summary to `results/contract_004_summary.json`."
            ),
            done_when=_ensemble_done,
            skip_if=_skip_ensemble,
            max_duration_minutes=180,
            max_retries=3,
            tags=["contract_004", "ensemble"],
        ),
    ]

    return Workflow(
        name="contract-004",
        steps=steps,
        max_hours=24.0,
    )
