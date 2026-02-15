# Phase 2: Features & Baselines

## Purpose
Build the feature engineering pipeline, walk-forward backtester, and establish
baseline strategies. This phase sets the performance floor that ML models in
Phase 3 must beat with statistical significance.

## Tasks

| Task | Description |
|------|-------------|
| `onchain_features` | Engineer features from on-chain data (NVT, MVRV, SOPR, etc.) |
| `feature_registry` | Central catalog of all features with metadata, lineage, and versioning |
| `feature_matrix_builder` | Assemble point-in-time correct feature matrices for any date range |
| `feature_selection` | Correlation filtering, mutual information, and stability-based selection |
| `backtest_engine` | Walk-forward backtester with configurable train/val/test windows |
| `transaction_costs` | Realistic cost model: spread, slippage, exchange fees |
| `statistical_tests` | Sharpe ratio confidence intervals, bootstrap hypothesis tests |
| `leakage_detector` | Automated checks for look-ahead bias and target leakage |
| `mlflow_integration` | Experiment tracking: params, metrics, artifacts logged to MLflow |
| `baseline_strategies` | Buy-and-hold, SMA crossover, mean-reversion, momentum baselines |
| `baseline_results_report` | Full performance report for all baselines with statistical significance |
| `sonnet_handoff` | Document handoff criteria so Sonnet can run Phase 3 experiments |

## Completion Criteria
- Feature matrix is point-in-time correct (leakage detector passes)
- Backtester reproduces known strategy results on synthetic data
- Transaction cost model validated against real exchange fee schedules
- At least 4 baseline strategies evaluated with walk-forward backtest
- All results include confidence intervals and significance tests
- MLflow tracks every experiment reproducibly

## Human Gate
**Type: Approve**
Human reviews baseline results report and approves the experimental framework
before ML modeling begins.
