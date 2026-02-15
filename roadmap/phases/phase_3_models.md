# Phase 3: ML Models & Alpha

## Purpose
Build ML models, run systematic experiments across features/horizons/seeds,
and identify strategies that beat baselines with statistical significance on
held-out data. Produce a validated alpha signal or an honest null result.

## Tasks

| Task | Description |
|------|-------------|
| `xgboost_model` | Gradient boosted tree model with walk-forward training pipeline |
| `lstm_model` | LSTM sequence model for multi-day feature windows |
| `feature_ablation_experiments` | Measure marginal value of each feature group |
| `horizon_experiments` | Test prediction horizons: 1d, 3d, 7d, 14d, 30d |
| `model_comparison` | Head-to-head model evaluation with paired statistical tests |
| `multi_seed_stability` | Run each config across 5+ seeds; report mean and variance |
| `holdout_validation` | Final evaluation on never-touched holdout period |
| `ensemble_exploration` | Test simple ensembles (average, stacking) of top models |
| `result_validator` | Automated sanity checks: Sharpe too high? Suspiciously smooth equity? |
| `analyst_report_generator` | Auto-generate formatted report from MLflow experiment results |
| `phase3_results_report` | Final report: best models, statistical evidence, go/no-go recommendation |

## Completion Criteria
- At least 2 model architectures evaluated across multiple horizons
- All results stable across 5+ random seeds (variance reported)
- Holdout performance consistent with walk-forward estimates (no overfit)
- Statistical tests confirm significance (or honestly report null result)
- Result validator flags no anomalies
- Final report includes explicit go/no-go recommendation for paper trading

## Human Gate
**Type: Go/No-Go Decision**
Human reviews phase 3 results report and makes the call on whether alpha is
real enough to proceed to paper trading.
