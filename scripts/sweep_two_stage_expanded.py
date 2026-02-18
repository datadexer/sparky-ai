#!/usr/bin/env python3
"""Two-stage sweep on expanded feature set (88 features)."""

import sys

sys.path.insert(0, "src")

import os

os.environ["PYTHONUNBUFFERED"] = "1"

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sparky.data.loader import load
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from sparky.backtest.costs import TransactionCostModel

BASELINE_SHARPE = 1.062
PROGRESS_FILE = Path("results/sweep_expanded_progress.csv")
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Initialize CSV header
if not PROGRESS_FILE.exists():
    with open(PROGRESS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "stage", "config_id", "model", "sharpe", "acc", "params"])


def load_expanded():
    """Load expanded feature set (88 features)."""
    print("=" * 80, flush=True)
    print("LOADING EXPANDED FEATURE SET (88 features)", flush=True)
    print("=" * 80, flush=True)

    features = load("feature_matrix_btc_hourly_expanded", purpose="training")
    target = load("targets_btc_hourly_1d", purpose="training")
    if isinstance(target, pd.DataFrame):
        target = target["target"]

    prices_hourly = load("ohlcv_hourly_max_coverage", purpose="analysis")
    prices_daily = prices_hourly.resample("D").last()
    del prices_hourly

    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    mask = features.notna().all(axis=1) & target.notna()
    features = features.loc[mask]
    target = target.loc[mask]

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    target = target.loc[features.index]

    print(f"Features: {features.shape[0]} samples × {features.shape[1]} features", flush=True)
    print(f"Date range: {features.index.min()} to {features.index.max()}", flush=True)
    print(f"Target balance: {target.mean():.1%} positive", flush=True)

    return features, target, prices_daily


def compute_sharpe(signals, prices_daily, cost_model):
    """Compute Sharpe from signals."""
    common_idx = signals.index.intersection(prices_daily.index)
    signals = signals.loc[common_idx]
    prices = prices_daily.loc[common_idx]

    returns = prices["close"].pct_change()
    strategy_returns = signals.shift(1) * returns
    costs = signals.diff().abs() * cost_model.total_cost_pct
    net_returns = (strategy_returns - costs).dropna()

    if len(net_returns) < 30 or net_returns.std() == 0:
        return 0.0
    return float(net_returns.mean() / net_returns.std() * np.sqrt(365))


def log_progress(stage, config_id, model_name, sharpe, acc, params_str):
    """Append one line to progress CSV."""
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


def feature_selection(features, target, top_n=25):
    """Select top N features by XGBoost importance."""
    print("\n" + "=" * 80, flush=True)
    print(f"STAGE 0: FEATURE SELECTION (keeping top {top_n})", flush=True)
    print("=" * 80, flush=True)

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

    importances = pd.Series(model.feature_importances_, index=features.columns)
    importances = importances.sort_values(ascending=False)

    print(f"\nTop {top_n} features:", flush=True)
    for i, (feat, imp) in enumerate(importances.head(top_n).items(), 1):
        print(f"  {i:2d}. {feat:<40s} {imp:.4f}", flush=True)

    selected = importances.head(top_n).index.tolist()
    return selected


def screen_config(X_train, y_train, X_test, y_test, prices_daily, cost_model, model_name, params):
    """Quick screen: single train/test."""
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


def run_stage1(features, target, prices_daily):
    """Screen configs (CatBoost only for speed)."""
    print("\n" + "=" * 80, flush=True)
    print("STAGE 1: SCREENING (CatBoost only, expanded features)", flush=True)
    print("=" * 80, flush=True)

    # CatBoost configs
    configs = []
    for depth in [3, 4, 5]:
        for lr in [0.01, 0.03]:
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

    cost_model = TransactionCostModel.for_btc()
    split_idx = int(len(features) * 0.8)

    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = target.iloc[split_idx:]

    print(f"Train: {len(X_train)} samples", flush=True)
    print(f"Test:  {len(X_test)} samples", flush=True)
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
            print(f"  [{i}/{len(configs)}] ERROR: {e}", flush=True)
            sharpe, acc = 0.0, 0.5

        results.append({"config": cfg, "sharpe": sharpe, "acc": acc})
        beat = " ** BEATS BASELINE **" if sharpe > BASELINE_SHARPE else ""
        print(
            f"  [{i}/{len(configs)}] d={cfg['params']['depth']} lr={cfg['params']['learning_rate']}"
            f" → Sharpe={sharpe:.3f} Acc={acc:.3f}{beat}",
            flush=True,
        )

        log_progress("screen", i, cfg["model"], sharpe, acc, str(cfg["params"]))

    # Top 5
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print("\n--- STAGE 1 TOP 5 ---", flush=True)
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. {r['config']['model']} Sharpe={r['sharpe']:.3f} Acc={r['acc']:.3f}", flush=True)

    return [r["config"] for r in results[:5]]


def walk_forward_validate(features, target, prices_daily, config):
    """5-fold walk-forward validation."""
    years = [2019, 2020, 2021, 2022, 2023]
    cost_model = TransactionCostModel.for_btc()
    sharpes, accs = [], []

    for year in years:
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        X_train = features.loc[:train_end]
        y_train = target.loc[:train_end]
        X_test = features.loc[test_start:test_end]
        y_test = target.loc[test_start:test_end]

        if len(X_test) < 100:
            continue

        try:
            if config["model"] == "CatBoost":
                model = CatBoostClassifier(**config["params"], verbose=0, random_state=42)
            elif config["model"] == "XGBoost":
                model = XGBClassifier(**config["params"], eval_metric="logloss", verbosity=0, random_state=42)
            else:
                raise ValueError(f"Unknown model: {config['model']}")

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba > 0.5).astype(int)

            acc = accuracy_score(y_test, y_pred)
            signals = pd.Series((y_proba > 0.52).astype(int), index=X_test.index)
            sharpe = compute_sharpe(signals, prices_daily, cost_model)

            sharpes.append(sharpe)
            accs.append(acc)
        except Exception as e:
            print(f"    {year}: ERROR {e}", flush=True)
            continue

    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)
    mean_acc = np.mean(accs)

    return mean_sharpe, std_sharpe, mean_acc


def run_stage2(features, target, prices_daily, top_configs):
    """Full walk-forward on top 5."""
    print("\n" + "=" * 80, flush=True)
    print(f"STAGE 2: WALK-FORWARD VALIDATION (top {len(top_configs)})", flush=True)
    print("=" * 80, flush=True)

    for i, cfg in enumerate(top_configs, 1):
        print(
            f"\n  [{i}/{len(top_configs)}] {cfg['model']} d={cfg['params'].get('depth', cfg['params'].get('max_depth'))}"
            f" lr={cfg['params']['learning_rate']}",
            flush=True,
        )

        mean_sharpe, std_sharpe, mean_acc = walk_forward_validate(features, target, prices_daily, cfg)

        print(f"    Mean Sharpe={mean_sharpe:.3f} ± {std_sharpe:.3f}, Acc={mean_acc:.3f}", flush=True)

        log_progress("validate", i, cfg["model"], mean_sharpe, mean_acc, str(cfg["params"]))


def main():
    """Run full two-stage sweep on expanded features."""
    features, target, prices_daily = load_expanded()

    # Stage 0: Feature selection
    selected_features = feature_selection(features, target, top_n=25)
    features = features[selected_features]
    print(f"\nReduced: {features.shape[1]} features (from 88)", flush=True)

    # Stage 1: Screen
    top_configs = run_stage1(features, target, prices_daily)

    # Stage 2: Validate
    run_stage2(features, target, prices_daily, top_configs)

    print("\n" + "=" * 80, flush=True)
    print("SWEEP COMPLETE", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
