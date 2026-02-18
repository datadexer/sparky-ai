#!/usr/bin/env python3
"""Two-stage hyperparameter sweep with feature selection.

Stage 0: Feature selection — keep top 15-20 by importance
Stage 1: Screening — single 80/20 split, all configs, ~2 min each
Stage 2: Validation — top 5 configs, full walk-forward

Writes incremental results to results/sweep_progress.csv after each config.

PROTECTED FILE — research agents cannot edit this file directly.
To request changes, write a GATE_REQUEST.md in the project root describing
what you need and why. An oversight session will review and apply the change.
"""

import sys

sys.path.insert(0, "src")

import os

os.environ["PYTHONUNBUFFERED"] = "1"

import pandas as pd
import numpy as np
import json
import csv
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score, roc_auc_score

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sparky.backtest.costs import TransactionCostModel
from sparky.data.loader import load
from sparky.tracking.experiment import ExperimentTracker
from sparky.tracking.guardrails import has_blocking_failure, log_results, run_post_checks, run_pre_checks
from sparky.tracking.metrics import compute_all_metrics

BASELINE_SHARPE = 1.062
PROGRESS_FILE = Path("results/sweep_progress.csv")
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_and_cache():
    """Load features, targets, and daily prices ONCE via holdout-enforced loader."""
    print("=" * 80, flush=True)
    print("LOADING DATA (cached for all configs)", flush=True)
    print("=" * 80, flush=True)

    features = load("feature_matrix_btc_hourly", purpose="training")
    target_df = load("targets_btc_hourly_1d", purpose="training")
    if isinstance(target_df, pd.DataFrame):
        target = target_df["target"]
    else:
        target = target_df

    # Prices used for Sharpe calculation during walk-forward validation
    prices_hourly = load("ohlcv_hourly_max_coverage", purpose="training")
    prices_daily = prices_hourly.resample("D").last()
    del prices_hourly  # free memory

    # Align everything
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    # Drop NaN rows
    mask = features.notna().all(axis=1) & target.notna()
    features = features.loc[mask]
    target = target.loc[mask]

    # Replace inf with NaN then drop
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    target = target.loc[features.index]

    print(f"Features: {features.shape[0]} samples × {features.shape[1]} features", flush=True)
    print(f"Date range: {features.index.min()} to {features.index.max()}", flush=True)
    print(f"Target balance: {target.mean():.1%} positive", flush=True)

    return features, target, prices_daily


def _get_net_returns(signals: pd.Series, prices_daily: pd.DataFrame, cost_model: TransactionCostModel) -> pd.Series:
    """Extract net return series from signals and daily prices."""
    common_idx = signals.index.intersection(prices_daily.index)
    signals = signals.loc[common_idx]
    prices = prices_daily.loc[common_idx]

    returns = prices["close"].pct_change()
    strategy_returns = signals.shift(1) * returns  # no look-ahead
    costs = signals.diff().abs() * cost_model.total_cost_pct
    return (strategy_returns - costs).dropna()


def compute_sharpe(signals: pd.Series, prices_daily: pd.DataFrame, cost_model: TransactionCostModel) -> float:
    """Compute Sharpe from signals and daily prices. Signals pre-aligned."""
    net_returns = _get_net_returns(signals, prices_daily, cost_model)
    if len(net_returns) < 30 or net_returns.std() == 0:
        return 0.0
    return float(net_returns.mean() / net_returns.std() * np.sqrt(365))


def log_progress(stage, config_id, model_name, sharpe, acc, params_str):
    """Append one line to sweep_progress.csv."""
    with open(PROGRESS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                stage,
                config_id,
                model_name,
                f"{sharpe:.4f}",
                f"{acc:.4f}",
                params_str,
            ]
        )


# ============================================================
# STAGE 0: Feature Selection
# ============================================================


def feature_selection(features, target, top_n=20):
    """Select top N features by XGBoost importance."""
    print("\n" + "=" * 80, flush=True)
    print(f"STAGE 0: FEATURE SELECTION (keeping top {top_n})", flush=True)
    print("=" * 80, flush=True)

    # Train a default XGBoost on 80% of data
    split_idx = int(len(features) * 0.8)
    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Rank features
    importances = pd.Series(model.feature_importances_, index=features.columns)
    importances = importances.sort_values(ascending=False)

    print("\nFeature importance ranking:", flush=True)
    for i, (feat, imp) in enumerate(importances.items(), 1):
        marker = " <-- KEPT" if i <= top_n else ""
        print(f"  {i:2d}. {feat:<35s} {imp:.4f}{marker}", flush=True)

    selected = importances.head(top_n).index.tolist()
    print(f"\nSelected {len(selected)} features (was {features.shape[1]})", flush=True)

    return selected


# ============================================================
# STAGE 1: Screening (single 80/20 split)
# ============================================================


def screen_config(X_train, y_train, X_test, y_test, prices_daily, cost_model, model_name, params):
    """Quick screen: single train/test, return Sharpe + accuracy."""
    config = {"model": model_name, "params": params, "transaction_costs_bps": 30}
    pre_results = run_pre_checks(X_train, config)
    if has_blocking_failure(pre_results):
        return 0.0, 0.5

    if model_name == "CatBoost":
        model = CatBoostClassifier(**params, verbose=0, random_state=42)
    elif model_name == "LightGBM":
        model = LGBMClassifier(**params, verbose=-1, random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(**params, eval_metric="logloss", verbosity=0, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    signals = pd.Series((y_proba > 0.52).astype(int), index=X_test.index)
    sharpe = compute_sharpe(signals, prices_daily, cost_model)

    return sharpe, acc


def run_stage1(features, target, prices_daily, configs):
    """Screen all configs with single 80/20 temporal split."""
    print("\n" + "=" * 80, flush=True)
    print(f"STAGE 1: SCREENING ({len(configs)} configs, single split)", flush=True)
    print("=" * 80, flush=True)

    cost_model = TransactionCostModel.for_btc()
    split_idx = int(len(features) * 0.8)

    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = target.iloc[split_idx:]

    print(f"Train: {len(X_train)} samples ({features.index[0]} to {features.index[split_idx - 1]})")
    print(f"Test:  {len(X_test)} samples ({features.index[split_idx]} to {features.index[-1]})")
    print(flush=True)

    results = []
    for i, cfg in enumerate(configs, 1):
        try:
            sharpe, acc = screen_config(
                X_train,
                y_train,
                X_test,
                y_test,
                prices_daily,
                cost_model,
                cfg["model"],
                cfg["params"],
            )
        except Exception as e:
            print(f"  [{i}/{len(configs)}] {cfg['model']} ERROR: {e}", flush=True)
            sharpe, acc = 0.0, 0.5

        results.append({"config": cfg, "sharpe": sharpe, "acc": acc})
        beat = " ** BEATS BASELINE **" if sharpe > BASELINE_SHARPE else ""
        print(
            f"  [{i}/{len(configs)}] {cfg['model']} d={cfg['params'].get('depth', cfg['params'].get('max_depth'))}"
            f" lr={cfg['params']['learning_rate']}"
            f" → Sharpe={sharpe:.3f} Acc={acc:.3f}{beat}",
            flush=True,
        )

        log_progress("screen", i, cfg["model"], sharpe, acc, str(cfg["params"]))

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


# ============================================================
# STAGE 2: Full walk-forward validation on top N
# ============================================================


def validate_walkforward(features, target, prices_daily, model_name, params, years=None):
    """Full yearly walk-forward validation."""
    if years is None:
        years = [2019, 2020, 2021, 2022, 2023]

    # Pre-experiment guardrail checks on full feature set before walk-forward
    wf_config = {"model": model_name, "params": params, "transaction_costs_bps": 30}
    pre_results = run_pre_checks(features, wf_config)
    if has_blocking_failure(pre_results):
        print(f"  Walk-forward PRE-CHECK BLOCKED for {model_name}", flush=True)
        return {"mean_sharpe": 0.0, "mean_acc": 0.5, "yearly": [], "net_returns": pd.Series(dtype=float)}

    cost_model = TransactionCostModel.for_btc()
    yearly_results = []
    all_net_returns = []
    all_signals = []

    for year in years:
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        X_train = features.loc[:train_end]
        y_train = target.loc[:train_end]
        X_test = features.loc[test_start:test_end]
        y_test = target.loc[test_start:test_end]

        if len(X_test) == 0 or len(X_train) < 100:
            continue

        if model_name == "CatBoost":
            model = CatBoostClassifier(**params, verbose=0, random_state=42)
        elif model_name == "LightGBM":
            model = LGBMClassifier(**params, verbose=-1, random_state=42)
        elif model_name == "XGBoost":
            model = XGBClassifier(**params, eval_metric="logloss", verbosity=0, random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
        signals = pd.Series((y_proba > 0.52).astype(int), index=X_test.index)
        sharpe = compute_sharpe(signals, prices_daily, cost_model)
        year_net_returns = _get_net_returns(signals, prices_daily, cost_model)

        yearly_results.append(
            {
                "year": year,
                "sharpe": sharpe,
                "acc": acc,
                "auc": auc,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )
        all_net_returns.append(year_net_returns)
        all_signals.append(signals)

    if not yearly_results:
        return {"mean_sharpe": 0.0, "mean_acc": 0.5, "yearly": [], "net_returns": pd.Series(dtype=float)}

    combined_net_returns = pd.concat(all_net_returns) if all_net_returns else pd.Series(dtype=float)

    # Count actual trades (position changes across all walk-forward years)
    combined_signals = pd.concat(all_signals) if all_signals else pd.Series(dtype=int)
    n_trades = int(combined_signals.diff().abs().sum())

    # Post-experiment guardrail checks
    mean_sh = float(np.mean([r["sharpe"] for r in yearly_results]))
    config = {"model": model_name, "params": params, "transaction_costs_bps": 30}
    post_results = run_post_checks(combined_net_returns.values, {"sharpe": mean_sh}, config, n_trades=n_trades)
    log_results(post_results, run_id=f"wf_{model_name}_{params.get('depth', params.get('max_depth', '?'))}")

    return {
        "mean_sharpe": np.mean([r["sharpe"] for r in yearly_results]),
        "std_sharpe": np.std([r["sharpe"] for r in yearly_results]),
        "mean_acc": np.mean([r["acc"] for r in yearly_results]),
        "mean_auc": np.mean([r["auc"] for r in yearly_results]),
        "min_sharpe": min(r["sharpe"] for r in yearly_results),
        "max_sharpe": max(r["sharpe"] for r in yearly_results),
        "yearly": yearly_results,
        "net_returns": combined_net_returns,
    }


def run_stage2(features, target, prices_daily, top_configs, top_n=5):
    """Full walk-forward on top N screening configs."""
    print("\n" + "=" * 80, flush=True)
    print(f"STAGE 2: FULL WALK-FORWARD VALIDATION (top {top_n} configs)", flush=True)
    print("=" * 80, flush=True)

    validated = []
    for i, entry in enumerate(top_configs[:top_n], 1):
        cfg = entry["config"]
        print(f"\n  [{i}/{top_n}] {cfg['model']} (screening Sharpe={entry['sharpe']:.3f})", flush=True)

        result = validate_walkforward(
            features,
            target,
            prices_daily,
            cfg["model"],
            cfg["params"],
        )

        print(
            f"    Walk-forward: Mean Sharpe={result['mean_sharpe']:.3f} "
            f"± {result.get('std_sharpe', 0):.3f}, "
            f"Acc={result['mean_acc']:.3f}",
            flush=True,
        )
        for yr in result["yearly"]:
            print(f"      {yr['year']}: Sharpe={yr['sharpe']:.3f} Acc={yr['acc']:.3f}", flush=True)

        log_progress("validate", i, cfg["model"], result["mean_sharpe"], result["mean_acc"], str(cfg["params"]))

        validated.append(
            {
                "config": cfg,
                "screening_sharpe": entry["sharpe"],
                **result,
            }
        )

    validated.sort(key=lambda x: x["mean_sharpe"], reverse=True)
    return validated


# ============================================================
# Config grid
# ============================================================


def build_configs():
    """Build 54 configs: 18 CatBoost + 18 LightGBM + 18 XGBoost."""
    configs = []

    for depth in [3, 4, 5]:
        for lr in [0.01, 0.03, 0.05]:
            for l2 in [1.0, 3.0]:
                configs.append(
                    {
                        "model": "CatBoost",
                        "params": {
                            "iterations": 200,
                            "depth": depth,
                            "learning_rate": lr,
                            "l2_leaf_reg": l2,
                            "task_type": "GPU",
                            "devices": "0",
                        },
                    }
                )

    for depth in [3, 4, 5]:
        for lr in [0.01, 0.03, 0.05]:
            for l1 in [0.0, 0.5]:
                configs.append(
                    {
                        "model": "LightGBM",
                        "params": {
                            "n_estimators": 200,
                            "max_depth": depth,
                            "learning_rate": lr,
                            "reg_lambda": 1.0,
                            "reg_alpha": l1,
                            "device": "gpu",
                            "gpu_platform_id": 0,
                            "gpu_device_id": 0,
                        },
                    }
                )

    for depth in [3, 4, 5]:
        for lr in [0.01, 0.03, 0.05]:
            for l2 in [0.0, 1.0]:
                configs.append(
                    {
                        "model": "XGBoost",
                        "params": {
                            "n_estimators": 200,
                            "max_depth": depth,
                            "learning_rate": lr,
                            "reg_lambda": l2,
                            "tree_method": "hist",
                            "device": "cuda",
                        },
                    }
                )

    return configs


# ============================================================
# Main
# ============================================================


def main():
    print("=" * 80, flush=True)
    print("TWO-STAGE HYPERPARAMETER SWEEP", flush=True)
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", flush=True)
    print("=" * 80, flush=True)

    # Write CSV header
    if not PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "stage", "config_id", "model", "sharpe", "acc", "params"])

    # Load data once
    features, target, prices_daily = load_and_cache()

    # Stage 0: Feature selection
    selected_features = feature_selection(features, target, top_n=20)
    features_reduced = features[selected_features]
    print(f"\nReduced: {features_reduced.shape[1]} features (from {features.shape[1]})", flush=True)

    # Build configs
    configs = build_configs()
    print(f"\nTotal configs: {len(configs)}", flush=True)

    # Stage 1: Screening
    screen_results = run_stage1(features_reduced, target, prices_daily, configs)

    print("\n--- STAGE 1 TOP 10 ---", flush=True)
    for i, r in enumerate(screen_results[:10], 1):
        cfg = r["config"]
        print(f"  {i}. {cfg['model']} Sharpe={r['sharpe']:.3f} Acc={r['acc']:.3f}", flush=True)

    # Stage 2: Validate top 5
    validated = run_stage2(features_reduced, target, prices_daily, screen_results, top_n=5)

    # Final report
    print("\n" + "=" * 80, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"Baseline (Donchian): Sharpe {BASELINE_SHARPE:.3f}", flush=True)
    print(flush=True)

    # n_trials must reflect ALL configs tested in this research program, not just
    # the current sweep. Prior sweeps (contract 004, smart_hyperparam_sweep) tested
    # ~250+ configs. Use cumulative total to avoid underestimating DSR penalty.
    n_total_configs = max(len(configs) + 250, len(configs))  # cumulative program total
    for i, v in enumerate(validated, 1):
        cfg = v["config"]
        beat = "BEATS BASELINE" if v["mean_sharpe"] > BASELINE_SHARPE else "below baseline"
        # Compute DSR from walk-forward net returns
        net_rets = v.get("net_returns", pd.Series(dtype=float))
        if len(net_rets) > 30:
            all_metrics = compute_all_metrics(net_rets.values, n_trials=n_total_configs, periods_per_year=365)
            dsr_str = f"DSR={all_metrics['dsr']:.3f}"
        else:
            dsr_str = "DSR=N/A"
        print(
            f"  {i}. {cfg['model']} → Walk-forward Sharpe={v['mean_sharpe']:.3f} "
            f"± {v.get('std_sharpe', 0):.3f} {dsr_str} ({beat})",
            flush=True,
        )
        print(f"     Params: {cfg['params']}", flush=True)
        print(f"     Screening Sharpe: {v['screening_sharpe']:.3f}", flush=True)
        for yr in v["yearly"]:
            print(f"       {yr['year']}: Sharpe={yr['sharpe']:.3f}", flush=True)

    best = validated[0] if validated else None
    if best and best["mean_sharpe"] > BASELINE_SHARPE:
        print(f"\n✅ ML BEATS BASELINE: {best['mean_sharpe']:.3f} > {BASELINE_SHARPE:.3f}", flush=True)
    elif best and best["mean_sharpe"] > 0.7:
        print(f"\n⚠️  ML shows promise: {best['mean_sharpe']:.3f} (baseline: {BASELINE_SHARPE:.3f})", flush=True)
    else:
        sharpe_val = best["mean_sharpe"] if best else 0.0
        print(f"\n❌ ML does not beat baseline: {sharpe_val:.3f} vs {BASELINE_SHARPE:.3f}", flush=True)

    # Log sweep results to W&B
    tracker = ExperimentTracker(experiment_name="sweep_two_stage")
    wb_results = []
    for v in validated:
        net_rets = v.get("net_returns", pd.Series(dtype=float))
        dsr = 0.0
        if len(net_rets) > 30:
            m = compute_all_metrics(net_rets.values, n_trials=n_total_configs, periods_per_year=365)
            dsr = m.get("dsr", 0.0)
        wb_results.append(
            {
                "config": v["config"],
                "metrics": {"sharpe": v["mean_sharpe"], "dsr": dsr, "std_sharpe": v.get("std_sharpe", 0)},
            }
        )
    tracker.log_sweep(
        "two_stage_validated",
        wb_results,
        summary_metrics={
            "best_sharpe": validated[0]["mean_sharpe"] if validated else 0.0,
            "n_screened": len(screen_results),
            "n_validated": len(validated),
        },
        tags=["contract_004", "sweep", "walk_forward"],
    )

    # Save full results
    output_path = Path("results/validation/sweep_two_stage.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "baseline_sharpe": BASELINE_SHARPE,
                "selected_features": selected_features,
                "n_configs_screened": len(screen_results),
                "top_5_validated": validated,
                "screening_results": screen_results[:20],  # top 20 for reference
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nResults saved: {output_path}", flush=True)
    print(f"Progress log: {PROGRESS_FILE}", flush=True)
    print(f"Completed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", flush=True)


if __name__ == "__main__":
    main()
