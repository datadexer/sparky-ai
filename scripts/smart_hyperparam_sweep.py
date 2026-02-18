#!/usr/bin/env python3
"""
SMART hyperparameter sweep - focused on promising configurations.

Based on prior ML experiments showing:
- Shallow trees (depth 3-4) work better than deep
- Lower learning rates (0.01-0.05) more stable
- Moderate regularization helps

Strategy:
1. Test 20-30 focused configs (not exhaustive 192)
2. Use same yearly walk-forward as baseline
3. Focus on models that showed promise: CatBoost, LightGBM
4. Report ALL results to avoid cherry-picking
"""

import sys

sys.path.insert(0, "src")

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from sparky.backtest.costs import TransactionCostModel
from sparky.data.loader import load
from sparky.tracking.experiment import ExperimentTracker
from sparky.tracking.guardrails import has_blocking_failure, log_results, run_post_checks, run_pre_checks
from sparky.tracking.metrics import compute_all_metrics


def load_data():
    """Load hourly features and targets via holdout-enforced loader."""
    print("Loading data...")

    features = load("features_hourly_full", purpose="training")
    prices = load("btc_hourly", purpose="training")

    # Align
    features = features.loc[features.index.intersection(prices.index)]
    prices = prices.loc[features.index]

    # Create 1h ahead target: shift(-1) looks one period forward for label only.
    # The loader truncates at holdout boundary, so the last row's label is safe.
    returns_1h = prices["close"].pct_change(1).shift(-1)
    target = (returns_1h > 0).astype(int)

    # Clean
    features = features.dropna()
    target = target.loc[features.index].dropna()
    common_idx = features.index.intersection(target.index)

    features = features.loc[common_idx]
    target = target.loc[common_idx]
    prices = prices.loc[common_idx]

    print(f"Data: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Date range: {features.index.min()} to {features.index.max()}")
    print(f"Target balance: {target.value_counts(normalize=True).to_dict()}")

    return features, target, prices


def validate_config(features, target, prices, model_name, params, n_trials=1):
    """Yearly walk-forward validation (2020-2023).

    Note: 2019 skipped because no training data before it (hourly data starts 2019-01-01).
    Returns net return series alongside Sharpe metrics for DSR computation.
    """
    # Filter to in-sample period only (up to 2023-12-31)
    features_in = features.loc[:"2023"]
    target_in = target.loc[:"2023"]
    prices_in = prices.loc[:"2023"]

    # Pre-experiment guardrail checks
    config = {"model": model_name, "params": params, "transaction_costs_bps": 30}
    pre_results = run_pre_checks(features_in, config)
    if has_blocking_failure(pre_results):
        print(f"    PRE-CHECK BLOCKED: {[r for r in pre_results if r.get('level') == 'BLOCK']}")
        return None

    years = [2020, 2021, 2022, 2023]  # Skip 2019 (no prior training data)
    results = []
    all_net_returns = []
    total_trades = 0  # accumulate actual trade count across all folds
    cost_model = TransactionCostModel.for_btc()

    for year in years:
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        X_train = features_in.loc[:train_end]
        y_train = target_in.loc[:train_end]
        X_test = features_in.loc[test_start:test_end]
        y_test = target_in.loc[test_start:test_end]

        print(f"    Year {year}: Train={len(X_train)}/{len(y_train)}, Test={len(X_test)}/{len(y_test)}")

        if len(X_test) == 0 or len(X_train) == 0 or len(y_train) == 0:
            print(f"    Year {year}: Skipping (no data)")
            continue

        # Train
        if model_name == "CatBoost":
            model = CatBoostClassifier(**params, verbose=0, random_state=42)
        else:  # LightGBM
            model = LGBMClassifier(**params, verbose=-1, random_state=42)

        model.fit(X_train, y_train)

        # Predict
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        # Backtest: aggregate hourly to daily
        test_prices = prices_in.loc[test_start:test_end]
        hourly_preds = pd.Series(y_proba, index=X_test.index)

        # LONG if >=60% hourly confidence
        daily_signals = hourly_preds.resample("D").apply(lambda x: (x > 0.5).mean() >= 0.6).astype(int)

        daily_prices = test_prices.resample("D").last()
        daily_signals = daily_signals.reindex(daily_prices.index, fill_value=0)
        daily_returns = daily_prices["close"].pct_change()

        # Strategy returns with costs (no look-ahead: shift(1))
        strategy_returns = daily_signals.shift(1) * daily_returns
        trades = daily_signals.diff().abs()
        total_trades += int(trades.sum())
        costs = trades * cost_model.round_trip_cost
        strategy_returns = (strategy_returns - costs).dropna()
        all_net_returns.append(strategy_returns)

        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(365)
            total_return = (1 + strategy_returns).prod() - 1
        else:
            sharpe = 0.0
            total_return = 0.0

        results.append({"year": year, "sharpe": sharpe, "return": total_return, "auc": auc})

    if not results:
        return None

    sharpes = [r["sharpe"] for r in results]
    net_returns = pd.concat(all_net_returns) if all_net_returns else pd.Series(dtype=float)

    # Post-experiment guardrail checks — compute actual drawdown from net returns
    equity_curve = (1 + net_returns).cumprod()
    running_max = equity_curve.cummax()
    actual_drawdown = float(((equity_curve - running_max) / running_max).min()) if len(equity_curve) > 0 else -1.0
    metrics_dict = {"sharpe": float(np.mean(sharpes)), "max_drawdown": actual_drawdown}
    post_results = run_post_checks(net_returns.values, metrics_dict, config, n_trades=total_trades)
    log_results(pre_results + post_results, run_id=f"{model_name}_{params.get('depth', params.get('max_depth'))}")

    return {
        "model": model_name,
        "params": params,
        "yearly": results,
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
        "min_sharpe": float(min(sharpes)),
        "max_sharpe": float(max(sharpes)),
        "net_returns": net_returns,
    }


def main():
    print("=" * 80)
    print("SMART HYPERPARAMETER SWEEP")
    print("=" * 80)
    print(f"Started: {datetime.now()}\n")

    features, target, prices = load_data()

    # Define focused configs
    configs = []

    # CatBoost: focus on shallow + moderate LR
    for depth in [3, 4, 5]:
        for lr in [0.01, 0.03, 0.05]:
            for l2 in [1, 3, 5]:
                configs.append(
                    {
                        "model": "CatBoost",
                        "params": {
                            "depth": depth,
                            "learning_rate": lr,
                            "l2_leaf_reg": l2,
                            "iterations": 500,
                            "task_type": "GPU",
                        },
                    }
                )

    # LightGBM: similar focused search
    for depth in [3, 4, 5]:
        for lr in [0.01, 0.03, 0.05]:
            for l1 in [0.0, 0.1, 1.0]:
                configs.append(
                    {
                        "model": "LightGBM",
                        "params": {
                            "max_depth": depth,
                            "learning_rate": lr,
                            "reg_alpha": l1,
                            "n_estimators": 500,
                            "device": "gpu",
                        },
                    }
                )

    print(f"Testing {len(configs)} configurations")
    print(f"Estimated time: ~{len(configs) * 1.5:.0f} minutes\n")

    all_results = []
    baseline_sharpe = 1.062
    beats_baseline = 0
    # Cumulative n_trials: count all trials ever tested in this program.
    # Prior sweeps (contract 004: ~187 runs, prior sessions: ~63) = ~250+.
    cumulative_prior_trials = 250
    n_trials_cumulative = len(configs) + cumulative_prior_trials

    tracker = ExperimentTracker(experiment_name="smart_hyperparam_sweep")

    for i, cfg in enumerate(configs):
        print(
            f"[{i + 1}/{len(configs)}] {cfg['model']} depth={cfg['params'].get('depth', cfg['params'].get('max_depth'))} lr={cfg['params']['learning_rate']}"
        )

        result = validate_config(features, target, prices, cfg["model"], cfg["params"], n_trials=n_trials_cumulative)
        if result is None:
            print("  → SKIPPED (blocked by pre-checks)")
            continue

        # Compute DSR with cumulative n_trials
        net_rets = result.pop("net_returns", pd.Series(dtype=float))
        if len(net_rets) > 30:
            dsr_metrics = compute_all_metrics(net_rets.values, n_trials=n_trials_cumulative, periods_per_year=365)
            result["dsr"] = dsr_metrics["dsr"]
            result["psr"] = dsr_metrics["psr"]
        else:
            result["dsr"] = 0.0
            result["psr"] = 0.0

        all_results.append(result)

        mean_sharpe = result["mean_sharpe"]
        print(
            f"  → Mean Sharpe: {mean_sharpe:.3f} (min={result['min_sharpe']:.3f}, max={result['max_sharpe']:.3f})"
            f" DSR={result['dsr']:.3f}"
        )

        if mean_sharpe > baseline_sharpe:
            beats_baseline += 1
            print(f"  BEATS BASELINE ({baseline_sharpe:.3f})! (DSR must also be > 0.95 to confirm)")

        # Save intermediate
        if (i + 1) % 5 == 0:
            Path("results/validation").mkdir(parents=True, exist_ok=True)
            with open("results/validation/smart_sweep_intermediate.json", "w") as f:
                json.dump({"configs": all_results, "baseline": baseline_sharpe}, f, indent=2)

    # Final summary
    all_results.sort(key=lambda x: x["mean_sharpe"], reverse=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_configs": len(all_results),
        "n_trials_cumulative": n_trials_cumulative,
        "baseline_sharpe": baseline_sharpe,
        "configs_beat_baseline": beats_baseline,
        "top_10": all_results[:10],
        "all_results": all_results,
    }

    output_path = Path("results/validation/smart_hyperparam_sweep.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Log to W&B as a single sweep run
    summary_metrics = {
        "n_configs": len(all_results),
        "configs_beat_baseline": beats_baseline,
        "best_sharpe": all_results[0]["mean_sharpe"] if all_results else 0.0,
        "best_dsr": all_results[0]["dsr"] if all_results else 0.0,
    }
    tracker.log_sweep(
        "smart_hyperparam_sweep",
        all_results,
        summary_metrics=summary_metrics,
        tags=["smart_sweep", "ml_baseline"],
    )

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"Total configs tested: {len(all_results)} (cumulative program total: {n_trials_cumulative})")
    print(f"Configs beating baseline ({baseline_sharpe:.3f}): {beats_baseline}")
    print("\nTop 3 configs:")
    for i, r in enumerate(all_results[:3], 1):
        print(
            f"{i}. {r['model']} (depth={r['params'].get('depth', r['params'].get('max_depth'))}, lr={r['params']['learning_rate']:.3f})"
        )
        print(f"   Mean Sharpe: {r['mean_sharpe']:.3f} ± {r['std_sharpe']:.3f}  DSR={r['dsr']:.3f}")

    print(f"\nResults saved: {output_path}")
    print(f"Completed: {datetime.now()}")


if __name__ == "__main__":
    main()
