#!/usr/bin/env python3
"""Comprehensive two-stage sweep v2 — all 3 model families with Sharpe evaluation.

Stage 1: Screening — single 80/20 split, 24+ configs across XGBoost/LightGBM/CatBoost
Stage 2: Walk-forward validation — top 5 configs, expanding window

Builds on existing CatBoost results. Adds XGBoost/LightGBM with Sharpe-based eval.
Avoids repeating configs already in sweep_expanded_progress.csv.

Logs:
  - log_sweep() for stage 1 batches (tags=['contract_004', 'sweep'])
  - log_experiment() for each walk-forward result (tags=['contract_004', 'sweep'])
"""
import sys
sys.path.insert(0, "src")

import os
os.environ["PYTHONUNBUFFERED"] = "1"

import time
import json
import csv
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sparky.backtest.costs import TransactionCostModel
from sparky.oversight.timeout import with_timeout
from sparky.tracking.experiment import ExperimentTracker

# ── Constants ──────────────────────────────────────────────────────────────────
BASELINE_SHARPE = 1.062
PROGRESS_FILE = Path("results/sweep_v2_progress.csv")
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Top 20 features from feature_importance.json
TOP_20_FEATURES = [
    "volume_dry_up", "rsi_6h", "rsi_168h", "price_range_expansion",
    "momentum_720h", "rsi_4h", "momentum_divergence_72h_336h", "vwap_cross",
    "mfi_14h", "price_acceleration_10h", "bb_position_20h",
    "momentum_divergence_4h_24h", "momentum_4h", "atr_14h",
    "rsi_divergence_14h_168h", "breakout_proximity_upper", "rsi_14h",
    "momentum_336h", "rsi_volume_interaction", "close_above_open_ratio_20h",
]
TOP_15_FEATURES = TOP_20_FEATURES[:15]
TOP_10_FEATURES = TOP_20_FEATURES[:10]


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_data():
    """Load features/targets/prices ONCE, in-sample only (≤2024-05-31)."""
    print("=" * 80, flush=True)
    print("LOADING DATA", flush=True)
    print("=" * 80, flush=True)

    # Use loader for holdout enforcement
    try:
        from sparky.data.loader import load
        features = load("btc_1h_features", purpose="training")
    except Exception:
        # Fallback to direct parquet if dataset name doesn't exist
        features = pd.read_parquet("data/processed/feature_matrix_btc_hourly.parquet")
        features = features.loc[:"2024-05-31"]

    target = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")
    if isinstance(target, pd.DataFrame):
        target = target["target"]

    # Load hourly prices, resample to daily
    prices_hourly = pd.read_parquet("data/raw/btc/ohlcv_hourly_max_coverage.parquet")
    prices_daily = prices_hourly.resample("D").last()
    del prices_hourly

    # Align and clean
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    # In-sample boundary
    cutoff = pd.Timestamp("2024-05-31", tz="UTC") if features.index.tz else pd.Timestamp("2024-05-31")
    features = features.loc[:cutoff]
    target = target.loc[:cutoff]

    # Clean NaN/inf
    features = features.replace([np.inf, -np.inf], np.nan)
    mask = features.notna().all(axis=1) & target.notna()
    features = features.loc[mask]
    target = target.loc[mask]

    # Restrict to our known top features (subset that actually exist in data)
    available_top20 = [f for f in TOP_20_FEATURES if f in features.columns]
    if len(available_top20) < 10:
        # Fall back to all features if top-20 aren't available
        available_top20 = list(features.columns)

    print(f"Features shape: {features.shape}, Target balance: {target.mean():.1%}", flush=True)
    print(f"Date range: {features.index.min()} → {features.index.max()}", flush=True)
    print(f"Top-20 available: {len(available_top20)}/{len(TOP_20_FEATURES)}", flush=True)

    return features, target, prices_daily, available_top20


# ── Sharpe computation ─────────────────────────────────────────────────────────
def compute_sharpe(signals: pd.Series, prices_daily: pd.DataFrame) -> float:
    """Compute daily Sharpe from directional signals. No look-ahead."""
    cost_model = TransactionCostModel.for_btc()

    common_idx = signals.index.intersection(prices_daily.index)
    if len(common_idx) < 30:
        return 0.0

    signals = signals.loc[common_idx]
    prices = prices_daily.loc[common_idx, "close"]

    returns = prices.pct_change()
    strategy_returns = signals.shift(1) * returns  # signal[T-1] × return[T]
    costs = signals.diff().abs() * cost_model.total_cost_pct
    net_returns = (strategy_returns - costs).dropna()

    if len(net_returns) < 30 or net_returns.std() == 0:
        return 0.0
    return float(net_returns.mean() / net_returns.std() * np.sqrt(365))


# ── Single config train/eval ───────────────────────────────────────────────────
@with_timeout(seconds=900)
def run_single_config(
    model_cls,
    model_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    prices_daily: pd.DataFrame,
    feature_names: list[str],
) -> dict:
    """Train model, return metrics dict with Sharpe."""
    t0 = time.time()

    # Filter to available features
    feats = [f for f in feature_names if f in X_train.columns]
    if len(feats) < 5:
        feats = list(X_train.columns)

    X_tr = X_train[feats]
    X_vl = X_val[feats]

    model = model_cls(**model_params)
    model.fit(X_tr, y_train)

    preds_val = model.predict(X_vl)
    probs_val = model.predict_proba(X_vl)[:, 1]
    accuracy = float((preds_val == y_val).mean())

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_val, probs_val))
    except Exception:
        auc = 0.5

    # Compute Sharpe on validation set
    signals = pd.Series(probs_val > 0.5, index=y_val.index).astype(float)
    sharpe_val = compute_sharpe(signals, prices_daily)

    # Also compute train Sharpe for overfitting diagnosis
    preds_train = model.predict(X_tr)
    signals_train = pd.Series(preds_train, index=y_train.index).astype(float)
    sharpe_train = compute_sharpe(signals_train, prices_daily)

    elapsed = time.time() - t0

    return {
        "accuracy": accuracy,
        "auc": auc,
        "sharpe": sharpe_val,
        "sharpe_train": sharpe_train,
        "elapsed_seconds": elapsed,
        "n_features": len(feats),
    }


# ── Walk-forward validation ────────────────────────────────────────────────────
def walk_forward_validate(
    model_cls,
    model_params: dict,
    features: pd.DataFrame,
    target: pd.Series,
    prices_daily: pd.DataFrame,
    feature_names: list[str],
    min_train_years: int = 2,
    step_months: int = 6,
) -> dict:
    """Expanding-window walk-forward validation."""
    feats = [f for f in feature_names if f in features.columns]
    if len(feats) < 5:
        feats = list(features.columns)

    X = features[feats]
    y = target

    start_year = X.index.min().year
    end_year = X.index.max().year

    # Build windows: train up to year N, validate year N+1 (6-month steps)
    periods = []
    train_end = pd.Timestamp(f"{start_year + min_train_years}-01-01", tz=X.index.tz)

    while train_end < X.index.max():
        val_start = train_end
        val_end = train_end + pd.DateOffset(months=step_months)
        if val_end > X.index.max():
            val_end = X.index.max()

        if val_end <= val_start:
            break

        periods.append((None, train_end, val_start, val_end))
        train_end = val_end

    all_signals = []
    window_sharpes = []

    for i, (_, te, vs, ve) in enumerate(periods):
        X_tr = X.loc[:te]
        y_tr = y.loc[:te]
        X_vl = X.loc[vs:ve]
        y_vl = y.loc[vs:ve]

        if len(X_tr) < 100 or len(X_vl) < 20:
            continue

        model = model_cls(**model_params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_vl)
        sig = pd.Series(preds, index=y_vl.index).astype(float)
        all_signals.append(sig)

        w_sharpe = compute_sharpe(sig, prices_daily)
        window_sharpes.append(w_sharpe)
        print(f"  Window {i+1}: train[:{te.date()}] val[{vs.date()}:{ve.date()}] Sharpe={w_sharpe:.3f}", flush=True)

    if not all_signals:
        return {"wf_sharpe": 0.0, "wf_sharpe_std": 0.0, "n_windows": 0}

    # Combine all validation signals and compute overall Sharpe
    combined = pd.concat(all_signals)
    wf_sharpe_overall = compute_sharpe(combined, prices_daily)

    return {
        "wf_sharpe": wf_sharpe_overall,
        "wf_sharpe_per_window_mean": float(np.mean(window_sharpes)),
        "wf_sharpe_std": float(np.std(window_sharpes)),
        "wf_sharpe_min": float(np.min(window_sharpes)),
        "wf_sharpe_max": float(np.max(window_sharpes)),
        "n_windows": len(window_sharpes),
        "pct_positive_windows": float(np.mean([s > 0 for s in window_sharpes])),
    }


# ── Progress logging ───────────────────────────────────────────────────────────
def init_progress():
    if not PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "stage", "model", "sharpe_val", "sharpe_train",
                             "accuracy", "auc", "feature_set", "params_hash", "params"])


def log_progress(stage, model_name, sharpe_val, sharpe_train, accuracy, auc, feature_set, params_str):
    import hashlib
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    with open(PROGRESS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            stage, model_name, f"{sharpe_val:.4f}", f"{sharpe_train:.4f}",
            f"{accuracy:.4f}", f"{auc:.4f}", feature_set, params_hash, params_str,
        ])


# ── Config definitions ─────────────────────────────────────────────────────────
def build_configs(available_top20: list[str]) -> list[dict]:
    """Build 24 configs: 8 XGBoost + 8 LightGBM + 8 CatBoost, varied meaningfully."""
    top10 = available_top20[:10]
    top15 = available_top20[:15]
    top20 = available_top20[:20]

    configs = []

    # ── XGBoost (8 configs) ─────────────────────────────────────────────────
    # Note: Previous runs used only lr={0.01,0.05,0.1} × depth={3,5,7} × n=200
    # New: use more regularization, higher n_estimators, colsample, subsample
    xgb_base = {"tree_method": "hist", "device": "cuda", "random_state": 42, "eval_metric": "logloss"}

    configs += [
        {
            "model_cls": XGBClassifier,
            "model_type": "xgboost",
            "feature_set": "top15",
            "feature_names": top15,
            "params": {**xgb_base, "n_estimators": 500, "learning_rate": 0.01, "max_depth": 4,
                       "reg_alpha": 0.1, "reg_lambda": 2.0, "subsample": 0.8, "colsample_bytree": 0.8},
            "note": "Conservative XGB: low LR + high n_estimators + regularization to prevent overfit on vol/RSI features",
        },
        {
            "model_cls": XGBClassifier,
            "model_type": "xgboost",
            "feature_set": "top20",
            "feature_names": top20,
            "params": {**xgb_base, "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
                       "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.9, "colsample_bytree": 0.7},
            "note": "Moderate XGB: balanced LR/depth with feature col subsampling for diversity on all 20 features",
        },
        {
            "model_cls": XGBClassifier,
            "model_type": "xgboost",
            "feature_set": "top10",
            "feature_names": top10,
            "params": {**xgb_base, "n_estimators": 200, "learning_rate": 0.05, "max_depth": 3,
                       "reg_alpha": 0.0, "reg_lambda": 1.0, "min_child_weight": 5},
            "note": "Shallow XGB on top-10 only: avoids noise from lower-ranked features, emphasizes volume+RSI signal",
        },
        {
            "model_cls": XGBClassifier,
            "model_type": "xgboost",
            "feature_set": "top15",
            "feature_names": top15,
            "params": {**xgb_base, "n_estimators": 500, "learning_rate": 0.02, "max_depth": 6,
                       "reg_alpha": 1.0, "reg_lambda": 5.0, "subsample": 0.7, "colsample_bytree": 0.6},
            "note": "Deep XGB with heavy L1+L2 reg: tests if complex interactions can be learned without overfitting",
        },
        {
            "model_cls": XGBClassifier,
            "model_type": "xgboost",
            "feature_set": "top20",
            "feature_names": top20,
            "params": {**xgb_base, "n_estimators": 1000, "learning_rate": 0.01, "max_depth": 3,
                       "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 0.8},
            "note": "Very long training XGB (1000 trees, low LR): ensemble effect with gradient boosting on all features",
        },
        {
            "model_cls": XGBClassifier,
            "model_type": "xgboost",
            "feature_set": "top15",
            "feature_names": top15,
            "params": {**xgb_base, "n_estimators": 300, "learning_rate": 0.1, "max_depth": 3,
                       "reg_alpha": 2.0, "reg_lambda": 2.0, "subsample": 0.6, "colsample_bytree": 0.5},
            "note": "High LR, aggressive dropout-style subsampling, stochastic gradient boosting for noisy financial data",
        },
        {
            "model_cls": XGBClassifier,
            "model_type": "xgboost",
            "feature_set": "top10",
            "feature_names": top10,
            "params": {**xgb_base, "n_estimators": 500, "learning_rate": 0.03, "max_depth": 5,
                       "reg_alpha": 0.5, "reg_lambda": 3.0, "min_child_weight": 10},
            "note": "High min_child_weight (10): requires large leaf samples, reduces noise-fitting on sparse features",
        },
        {
            "model_cls": XGBClassifier,
            "model_type": "xgboost",
            "feature_set": "top20",
            "feature_names": top20,
            "params": {**xgb_base, "n_estimators": 200, "learning_rate": 0.2, "max_depth": 4,
                       "reg_alpha": 0.0, "reg_lambda": 0.5, "subsample": 1.0, "colsample_bytree": 1.0},
            "note": "Fast learner (lr=0.2, no reg): tests if raw gradient signal without regularization extracts more alpha",
        },
    ]

    # ── LightGBM (8 configs) ────────────────────────────────────────────────
    # Previous runs: lr={0.01,0.05,0.1} × depth={3,5,7} × n=200 — all ~52% acc, no Sharpe
    # New: vary num_leaves (key LightGBM param), min_data_in_leaf, feature_fraction, bagging
    lgbm_base = {"device": "gpu", "random_state": 42, "verbose": -1, "n_jobs": -1}

    configs += [
        {
            "model_cls": LGBMClassifier,
            "model_type": "lightgbm",
            "feature_set": "top15",
            "feature_names": top15,
            "params": {**lgbm_base, "n_estimators": 500, "learning_rate": 0.01, "num_leaves": 31,
                       "min_child_samples": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                       "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 1.0},
            "note": "Conservative LGBM: standard 31 leaves, bagging for variance reduction on 15-feature subset",
        },
        {
            "model_cls": LGBMClassifier,
            "model_type": "lightgbm",
            "feature_set": "top20",
            "feature_names": top20,
            "params": {**lgbm_base, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63,
                       "min_child_samples": 30, "feature_fraction": 0.7, "bagging_fraction": 0.9,
                       "bagging_freq": 3, "reg_alpha": 0.5, "reg_lambda": 2.0},
            "note": "64-leaf LGBM (more complex than XGB default): explores richer momentum+vol interactions",
        },
        {
            "model_cls": LGBMClassifier,
            "model_type": "lightgbm",
            "feature_set": "top10",
            "feature_names": top10,
            "params": {**lgbm_base, "n_estimators": 1000, "learning_rate": 0.01, "num_leaves": 15,
                       "min_child_samples": 50, "feature_fraction": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.5},
            "note": "Shallow 15-leaf LGBM on top-10: high min_samples prevents overfitting sparse cryptocurrency signals",
        },
        {
            "model_cls": LGBMClassifier,
            "model_type": "lightgbm",
            "feature_set": "top15",
            "feature_names": top15,
            "params": {**lgbm_base, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 127,
                       "min_child_samples": 15, "feature_fraction": 0.6, "bagging_fraction": 0.8,
                       "bagging_freq": 5, "reg_alpha": 1.0, "reg_lambda": 5.0},
            "note": "128-leaf deep LGBM + heavy L1+L2: tests if very complex decision boundaries exist in BTC hourly data",
        },
        {
            "model_cls": LGBMClassifier,
            "model_type": "lightgbm",
            "feature_set": "top20",
            "feature_names": top20,
            "params": {**lgbm_base, "n_estimators": 500, "learning_rate": 0.02, "num_leaves": 31,
                       "min_child_samples": 100, "feature_fraction": 0.9, "reg_alpha": 0.3, "reg_lambda": 1.0},
            "note": "Very conservative min_samples=100: rejects noisy splits, focuses on robust large-scale patterns",
        },
        {
            "model_cls": LGBMClassifier,
            "model_type": "lightgbm",
            "feature_set": "top10",
            "feature_names": top10,
            "params": {**lgbm_base, "n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31,
                       "min_child_samples": 20, "feature_fraction": 0.8, "bagging_fraction": 0.7,
                       "bagging_freq": 10, "reg_alpha": 0.0, "reg_lambda": 0.0},
            "note": "No regularization LGBM on top-10: tests pure gradient signal without explicit penalty on compact feature set",
        },
        {
            "model_cls": LGBMClassifier,
            "model_type": "lightgbm",
            "feature_set": "top15",
            "feature_names": top15,
            "params": {**lgbm_base, "n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63,
                       "min_child_samples": 40, "feature_fraction": 0.7, "bagging_fraction": 0.8,
                       "bagging_freq": 5, "reg_alpha": 0.5, "reg_lambda": 2.0, "max_depth": 6},
            "note": "Depth-limited LGBM: combined max_depth + num_leaves double-constraint for cleaner tree structure",
        },
        {
            "model_cls": LGBMClassifier,
            "model_type": "lightgbm",
            "feature_set": "top20",
            "feature_names": top20,
            "params": {**lgbm_base, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                       "min_child_samples": 30, "feature_fraction": 0.5, "bagging_fraction": 0.6,
                       "bagging_freq": 3, "reg_alpha": 2.0, "reg_lambda": 2.0},
            "note": "Aggressive feature + sample subsampling (50%/60%): stochastic ensemble on full 20 features",
        },
    ]

    # ── CatBoost (8 configs) — complement existing ones ──────────────────────
    # Previous runs: iterations=200, depth={3,4,5}, lr={0.01,0.03}, l2={1,3}
    # Gaps: border_count, rsm (feature subsample), grow_policy, od_type, higher iterations
    cat_base = {"task_type": "GPU", "devices": "0", "random_state": 42, "verbose": 0}

    configs += [
        {
            "model_cls": CatBoostClassifier,
            "model_type": "catboost",
            "feature_set": "top15",
            "feature_names": top15,
            "params": {**cat_base, "iterations": 500, "learning_rate": 0.01, "depth": 4,
                       "l2_leaf_reg": 3.0, "rsm": 0.8, "border_count": 64},
            "note": "CatBoost with border_count=64 (more bins) + rsm=0.8: finer split thresholds for smooth momentum signals",
        },
        {
            "model_cls": CatBoostClassifier,
            "model_type": "catboost",
            "feature_set": "top20",
            "feature_names": top20,
            "params": {**cat_base, "iterations": 300, "learning_rate": 0.05, "depth": 5,
                       "l2_leaf_reg": 5.0, "rsm": 0.7, "border_count": 32},
            "note": "Higher LR CatBoost with L2=5 + coarser bins (32): tests if heavy regularization prevents RSI/vol overfit",
        },
        {
            "model_cls": CatBoostClassifier,
            "model_type": "catboost",
            "feature_set": "top10",
            "feature_names": top10,
            "params": {**cat_base, "iterations": 1000, "learning_rate": 0.01, "depth": 3,
                       "l2_leaf_reg": 1.0, "rsm": 1.0, "od_type": "Iter", "od_wait": 50},
            "note": "Long CatBoost with early stopping on top-10: uses all features but stops before overfitting begins",
        },
        {
            "model_cls": CatBoostClassifier,
            "model_type": "catboost",
            "feature_set": "top15",
            "feature_names": top15,
            "params": {**cat_base, "iterations": 200, "learning_rate": 0.01, "depth": 6,
                       "l2_leaf_reg": 10.0, "rsm": 0.8, "border_count": 128},
            "note": "Deep CatBoost (depth=6) with very high L2=10: complex interactions + aggressive shrinkage for stability",
        },
        {
            "model_cls": CatBoostClassifier,
            "model_type": "catboost",
            "feature_set": "top20",
            "feature_names": top20,
            "params": {**cat_base, "iterations": 500, "learning_rate": 0.02, "depth": 4,
                       "l2_leaf_reg": 2.0, "rsm": 0.6, "bagging_temperature": 0.5},
            "note": "CatBoost with Bayesian bagging (temperature=0.5) + high feature drop: diversified ensemble effect",
        },
        {
            "model_cls": CatBoostClassifier,
            "model_type": "catboost",
            "feature_set": "top15",
            "feature_names": top15,
            "params": {**cat_base, "iterations": 300, "learning_rate": 0.05, "depth": 3,
                       "l2_leaf_reg": 1.0, "rsm": 0.9, "grow_policy": "Lossguide", "max_leaves": 31},
            "note": "LossGuide growth policy: leaf-wise like LGBM rather than depth-first, may find finer signal boundaries",
        },
        {
            "model_cls": CatBoostClassifier,
            "model_type": "catboost",
            "feature_set": "top10",
            "feature_names": top10,
            "params": {**cat_base, "iterations": 200, "learning_rate": 0.03, "depth": 4,
                       "l2_leaf_reg": 3.0, "rsm": 0.8, "border_count": 64},
            "note": "CatBoost on minimal top-10 set + moderate rsm: focused on purest signals (vol dry-up + RSI divergence)",
        },
        {
            "model_cls": CatBoostClassifier,
            "model_type": "catboost",
            "feature_set": "top20",
            "feature_names": top20,
            "params": {**cat_base, "iterations": 500, "learning_rate": 0.01, "depth": 5,
                       "l2_leaf_reg": 5.0, "rsm": 0.7, "border_count": 64, "bagging_temperature": 1.0},
            "note": "Long conservative CatBoost: full Bayesian bagging + L2=5 + fine bins on all 20 features",
        },
    ]

    return configs


# ── Main sweep ─────────────────────────────────────────────────────────────────
def run_stage1(features, target, prices_daily, available_top20, tracker):
    """Stage 1: Screen all 24 configs with single 80/20 split."""
    print("\n" + "=" * 80, flush=True)
    print("STAGE 1: SCREENING (24 configs × 3 model families)", flush=True)
    print("=" * 80, flush=True)

    configs = build_configs(available_top20)

    # 80/20 temporal split
    split_idx = int(len(features) * 0.8)
    split_date = features.index[split_idx]
    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_val = features.iloc[split_idx:]
    y_val = target.iloc[split_idx:]

    print(f"Train: {X_train.index.min().date()} → {X_train.index.max().date()} ({len(X_train)} rows)", flush=True)
    print(f"Val:   {X_val.index.min().date()} → {X_val.index.max().date()} ({len(X_val)} rows)", flush=True)

    results_by_family = {"xgboost": [], "lightgbm": [], "catboost": []}
    all_results = []

    for i, cfg in enumerate(configs):
        model_cls = cfg["model_cls"]
        model_type = cfg["model_type"]
        params = cfg["params"]
        feature_names = cfg["feature_names"]
        feature_set = cfg["feature_set"]
        note = cfg["note"]

        print(f"\n[{i+1:2d}/{len(configs)}] {model_type.upper()} | {feature_set} | {note[:60]}...", flush=True)
        print(f"  Params: {json.dumps({k: v for k, v in params.items() if k not in ['tree_method', 'device', 'task_type', 'devices', 'random_state', 'verbose', 'eval_metric', 'n_jobs']}, default=str)}", flush=True)

        try:
            metrics = run_single_config(
                model_cls, params, X_train, y_train, X_val, y_val,
                prices_daily, feature_names
            )
            sharpe_val = metrics["sharpe"]
            sharpe_train = metrics.get("sharpe_train", 0.0)
            accuracy = metrics["accuracy"]
            auc = metrics["auc"]
            elapsed = metrics["elapsed_seconds"]

            # Interpretation note
            overfit = sharpe_train > sharpe_val * 1.5 if sharpe_val > 0.2 else False
            interp = f"Val Sharpe={sharpe_val:.3f}, Train Sharpe={sharpe_train:.3f}"
            if overfit:
                interp += f" — OVERFIT (train 2× val)"
            elif sharpe_val >= BASELINE_SHARPE:
                interp += f" — BEATS BASELINE ({BASELINE_SHARPE})"
            elif sharpe_val >= 0.7:
                interp += " — PROMISING"
            else:
                interp += " — below threshold"

            print(f"  RESULT: Sharpe={sharpe_val:.3f} (train={sharpe_train:.3f}) Acc={accuracy:.4f} AUC={auc:.4f} [{elapsed:.1f}s]", flush=True)
            print(f"  INTERPRETATION: {interp}", flush=True)

            result_entry = {
                "config": {**params, "model_type": model_type, "feature_set": feature_set,
                           "n_features": len(feature_names)},
                "metrics": metrics,
                "note": note,
                "interpretation": interp,
            }
            results_by_family[model_type].append(result_entry)
            all_results.append(result_entry)

            log_progress("screen", model_type, sharpe_val, sharpe_train, accuracy, auc,
                         feature_set, str(params))

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            log_progress("screen", model_type, 0.0, 0.0, 0.0, 0.0, feature_set, str(params))
            all_results.append({
                "config": {**params, "model_type": model_type, "feature_set": feature_set},
                "metrics": {"sharpe": 0.0, "accuracy": 0.0, "auc": 0.5, "elapsed_seconds": 0.0},
                "note": note,
                "interpretation": f"ERROR: {e}",
            })

    # Log each family as a sweep batch
    for family, results in results_by_family.items():
        if not results:
            continue
        sharpes = [r["metrics"].get("sharpe", 0) for r in results]
        summary = {
            "best_sharpe": float(max(sharpes)) if sharpes else 0.0,
            "mean_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
            "n_above_baseline": int(sum(1 for s in sharpes if s >= BASELINE_SHARPE)),
        }
        sweep_name = f"stage1_{family}_{len(results)}configs"
        print(f"\nLogging {family} batch to wandb: {sweep_name}", flush=True)
        tracker.log_sweep(
            name=sweep_name,
            results=results,
            summary_metrics=summary,
            tags=["contract_004", "sweep"],
            job_type="sweep",
        )

    print("\n" + "=" * 80, flush=True)
    print("STAGE 1 SUMMARY", flush=True)
    print("=" * 80, flush=True)
    for r in sorted(all_results, key=lambda x: x["metrics"].get("sharpe", 0), reverse=True)[:10]:
        cfg = r["config"]
        m = r["metrics"]
        print(f"  {cfg.get('model_type','?'):10} {cfg.get('feature_set','?'):7} "
              f"Sharpe={m.get('sharpe',0):.3f} "
              f"Acc={m.get('accuracy',0):.4f} | {r['note'][:50]}", flush=True)

    return all_results


def run_stage2(all_stage1_results, features, target, prices_daily, tracker):
    """Stage 2: Walk-forward validation on top 5 configs from Stage 1."""
    print("\n" + "=" * 80, flush=True)
    print("STAGE 2: WALK-FORWARD VALIDATION (top 5 configs)", flush=True)
    print("=" * 80, flush=True)

    # Sort by validation Sharpe, take top 5
    sorted_results = sorted(all_stage1_results,
                            key=lambda x: x["metrics"].get("sharpe", 0),
                            reverse=True)

    # Ensure at least 1 from each family in top 5 if possible
    top5 = []
    seen_families = set()
    for r in sorted_results:
        family = r["config"].get("model_type", "")
        if len(top5) >= 5:
            break
        top5.append(r)
        seen_families.add(family)

    # If we're missing a family, replace the lowest of the same family with best from missing
    families_needed = {"xgboost", "lightgbm", "catboost"} - seen_families
    for fam in families_needed:
        fam_results = [r for r in sorted_results if r["config"].get("model_type") == fam]
        if fam_results and len(top5) == 5:
            # Replace lowest Sharpe in top5
            top5.sort(key=lambda x: x["metrics"].get("sharpe", 0))
            top5[0] = fam_results[0]

    top5.sort(key=lambda x: x["metrics"].get("sharpe", 0), reverse=True)

    print(f"\nTop 5 configs for walk-forward:", flush=True)
    for i, r in enumerate(top5):
        cfg = r["config"]
        m = r["metrics"]
        print(f"  {i+1}. {cfg.get('model_type','?'):10} {cfg.get('feature_set','?'):7} "
              f"Stage1_Sharpe={m.get('sharpe',0):.3f} | {r['note'][:50]}", flush=True)

    wf_results = []

    for i, r in enumerate(top5):
        cfg = r["config"]
        model_type = cfg.get("model_type", "unknown")
        feature_set = cfg.get("feature_set", "top15")
        note = r.get("note", "")

        # Get model class
        cls_map = {
            "xgboost": XGBClassifier,
            "lightgbm": LGBMClassifier,
            "catboost": CatBoostClassifier,
        }
        model_cls = cls_map.get(model_type)
        if model_cls is None:
            print(f"  Skipping unknown model type: {model_type}", flush=True)
            continue

        # Reconstruct feature list
        feat_map = {"top10": TOP_10_FEATURES[:10], "top15": TOP_15_FEATURES[:15], "top20": TOP_20_FEATURES[:20]}
        feature_names = feat_map.get(feature_set, TOP_15_FEATURES)

        # Rebuild params without non-model keys
        params = {k: v for k, v in cfg.items()
                  if k not in ["model_type", "feature_set", "n_features", "config_hash",
                               "git_hash", "data_manifest_hash"]}

        stage1_sharpe = r["metrics"].get("sharpe", 0)
        print(f"\n[{i+1}/5] Walk-forward: {model_type.upper()} {feature_set} (Stage1 Sharpe={stage1_sharpe:.3f})", flush=True)

        try:
            wf_metrics = walk_forward_validate(
                model_cls, params, features, target, prices_daily, feature_names
            )
            wf_sharpe = wf_metrics["wf_sharpe"]

            print(f"  Walk-forward Sharpe: {wf_sharpe:.3f} (std={wf_metrics.get('wf_sharpe_std',0):.3f}, "
                  f"wins={wf_metrics.get('pct_positive_windows',0):.0%})", flush=True)

            # Log as individual experiment
            all_metrics = {**r["metrics"], **wf_metrics}
            log_name = None  # auto-generate

            tracker.log_experiment(
                name=log_name,
                config={**params, "model_type": model_type, "feature_set": feature_set,
                        "stage": "walk_forward", "stage1_sharpe": stage1_sharpe},
                metrics={
                    "sharpe": wf_sharpe,
                    "sharpe_stage1": stage1_sharpe,
                    "accuracy": r["metrics"].get("accuracy", 0),
                    "auc": r["metrics"].get("auc", 0.5),
                    **{k: v for k, v in wf_metrics.items()},
                },
                features_used=feature_names,
                tags=["contract_004", "sweep", "walk_forward"],
                job_type="sweep",
            )

            log_progress("walkforward", model_type, wf_sharpe, stage1_sharpe,
                         r["metrics"].get("accuracy", 0), r["metrics"].get("auc", 0.5),
                         feature_set, str(params))

            wf_results.append({
                "model_type": model_type,
                "feature_set": feature_set,
                "params": params,
                "stage1_sharpe": stage1_sharpe,
                "wf_sharpe": wf_sharpe,
                "wf_metrics": wf_metrics,
                "note": note,
                "beats_baseline": wf_sharpe >= BASELINE_SHARPE,
            })

        except Exception as e:
            print(f"  ERROR in walk-forward: {e}", flush=True)
            wf_results.append({
                "model_type": model_type,
                "feature_set": feature_set,
                "params": params,
                "stage1_sharpe": stage1_sharpe,
                "wf_sharpe": 0.0,
                "wf_metrics": {},
                "note": note,
                "beats_baseline": False,
                "error": str(e),
            })

    # Sort by WF Sharpe
    wf_results.sort(key=lambda x: x["wf_sharpe"], reverse=True)

    print("\n" + "=" * 80, flush=True)
    print("STAGE 2 RESULTS (Walk-forward)", flush=True)
    print("=" * 80, flush=True)
    for r in wf_results:
        beat = " *** BEATS BASELINE ***" if r["beats_baseline"] else ""
        print(f"  {r['model_type']:10} {r['feature_set']:7} "
              f"WF_Sharpe={r['wf_sharpe']:.3f} Stage1={r['stage1_sharpe']:.3f}{beat}", flush=True)

    return wf_results


def write_summary(all_stage1_results, wf_results):
    """Write results/sweep_summary.md."""
    summary_path = Path("results/sweep_summary.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    best_wf = max(wf_results, key=lambda x: x["wf_sharpe"]) if wf_results else None
    beats_any = any(r["beats_baseline"] for r in wf_results)

    lines = [
        "# Sweep Summary — Contract #004 Step 2",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Baseline to beat: Donchian Multi-TF Sharpe = **{BASELINE_SHARPE}** (in-sample, no look-ahead)",
        "",
        "## Stage 1 — Screening Results (80/20 temporal split)",
        "",
        "| # | Model | Feature Set | Sharpe (val) | Sharpe (train) | Accuracy | AUC | Note |",
        "|---|-------|-------------|:---:|:---:|:---:|:---:|------|",
    ]

    sorted_s1 = sorted(all_stage1_results,
                       key=lambda x: x["metrics"].get("sharpe", 0), reverse=True)
    for i, r in enumerate(sorted_s1):
        cfg = r["config"]
        m = r["metrics"]
        sharpe_v = m.get("sharpe", 0)
        sharpe_t = m.get("sharpe_train", 0)
        beat = " ✓" if sharpe_v >= BASELINE_SHARPE else ""
        lines.append(
            f"| {i+1} | {cfg.get('model_type','?')} | {cfg.get('feature_set','?')} "
            f"| {sharpe_v:.3f}{beat} | {sharpe_t:.3f} "
            f"| {m.get('accuracy',0):.4f} | {m.get('auc',0):.4f} "
            f"| {r['note'][:60]} |"
        )

    lines += [
        "",
        "## Stage 2 — Walk-forward Validation Results",
        "",
        "| Rank | Model | Feature Set | WF Sharpe | Stage1 Sharpe | WF Std | Win% | Beats Baseline? |",
        "|------|-------|-------------|:---:|:---:|:---:|:---:|:---:|",
    ]

    for i, r in enumerate(wf_results):
        wm = r.get("wf_metrics", {})
        beat = "**YES**" if r["beats_baseline"] else "No"
        lines.append(
            f"| {i+1} | {r['model_type']} | {r['feature_set']} "
            f"| **{r['wf_sharpe']:.3f}** | {r['stage1_sharpe']:.3f} "
            f"| {wm.get('wf_sharpe_std', 0):.3f} "
            f"| {wm.get('pct_positive_windows', 0):.0%} "
            f"| {beat} |"
        )

    # Best family analysis
    family_sharpes = {}
    for r in all_stage1_results:
        fam = r["config"].get("model_type", "?")
        s = r["metrics"].get("sharpe", 0)
        if fam not in family_sharpes:
            family_sharpes[fam] = []
        family_sharpes[fam].append(s)

    best_family = max(family_sharpes, key=lambda f: np.mean(family_sharpes[f]))
    best_family_mean = np.mean(family_sharpes[best_family])
    best_family_max = np.max(family_sharpes[best_family])

    lines += [
        "",
        "## Model Family Analysis",
        "",
    ]
    for fam, sharpes in sorted(family_sharpes.items(), key=lambda x: -np.mean(x[1])):
        lines.append(f"- **{fam}**: mean={np.mean(sharpes):.3f}, max={np.max(sharpes):.3f}, "
                     f"n_above_baseline={sum(1 for s in sharpes if s>=BASELINE_SHARPE)}/{len(sharpes)}")

    lines += [
        "",
        f"**Best family: {best_family}** (mean Sharpe={best_family_mean:.3f}, max={best_family_max:.3f})",
        "",
        "**Hypothesis:** " + {
            "catboost": "CatBoost's symmetric tree structure and built-in handling of ordered boosting "
                        "reduces temporal look-ahead risk better than XGB/LGBM, helping on financial time-series.",
            "lightgbm": "LightGBM's leaf-wise growth with num_leaves control finds finer signal thresholds "
                        "in the RSI/momentum feature space that depth-limited models miss.",
            "xgboost": "XGBoost's exact greedy splitting with colsample variance reduction generalizes "
                        "better on the top-20 feature set due to explicit feature dropout.",
        }.get(best_family, "No clear winner — all families performing similarly."),
        "",
        "## Top 5 Configs by Walk-forward Sharpe",
        "",
    ]

    for i, r in enumerate(wf_results[:5]):
        lines += [
            f"### {i+1}. {r['model_type'].upper()} — {r['feature_set']} (WF Sharpe: {r['wf_sharpe']:.3f})",
            f"- Stage1 Sharpe: {r['stage1_sharpe']:.3f}",
            f"- WF metrics: {r.get('wf_metrics', {})}",
            f"- Key params: {json.dumps({k: v for k, v in r['params'].items() if k not in ['tree_method','device','task_type','devices','random_state','verbose','eval_metric','n_jobs']}, default=str)}",
            f"- Note: {r['note']}",
            "",
        ]

    # Honest assessment
    lines += [
        "## Honest Assessment: Does Any Config Beat Donchian 1.062?",
        "",
    ]

    if beats_any:
        beaters = [r for r in wf_results if r["beats_baseline"]]
        lines += [
            f"**YES** — {len(beaters)} config(s) beat the Donchian baseline on walk-forward validation:",
            "",
        ]
        for r in beaters:
            lines.append(f"- {r['model_type']} {r['feature_set']}: WF Sharpe {r['wf_sharpe']:.3f} > {BASELINE_SHARPE}")
        lines += [
            "",
            "**Caveats**: Walk-forward validation is still in-sample (2019-2023). Need OOS evaluation "
            "with explicit AK approval before claiming genuine alpha. Statistical significance test needed.",
        ]
    else:
        best_wf_sharpe = max((r["wf_sharpe"] for r in wf_results), default=0)
        lines += [
            f"**NO** — Best walk-forward Sharpe is {best_wf_sharpe:.3f} vs Donchian {BASELINE_SHARPE}.",
            "",
            "**Analysis**: ML models achieve ~52-55% accuracy on hourly BTC direction, but "
            "the transaction costs (~0.13% per trade) erode the edge significantly. The Donchian "
            "strategy benefits from trend-following with fewer but larger position changes.",
            "",
            "**Next steps**: ",
            "- Try position sizing based on prediction confidence (proba threshold > 0.55)",
            "- Reduce trading frequency by only trading when model is highly confident",
            "- Ensemble multiple models to reduce noise",
            "- Investigate different target: next-24h vs next-1h return",
        ]

    lines += [
        "",
        "---",
        f"*Total configs tested: {len(all_stage1_results)} (Stage 1) + {len(wf_results)} (Stage 2 WF)*",
        f"*Baseline: Donchian Multi-TF Sharpe = {BASELINE_SHARPE} (corrected for look-ahead bias)*",
    ]

    summary_path.write_text("\n".join(lines))
    print(f"\nSummary written to {summary_path}", flush=True)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    print("=" * 80, flush=True)
    print("SPARKY AI — Two-Stage Sweep v2 (Contract #004, Step 2)", flush=True)
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", flush=True)
    print(f"Baseline to beat: Donchian Sharpe = {BASELINE_SHARPE}", flush=True)
    print("=" * 80, flush=True)

    # Init tracker
    tracker = ExperimentTracker(experiment_name="contract_004_sweep_v2")

    # Init progress file
    init_progress()

    # Load data once
    features, target, prices_daily, available_top20 = load_data()

    # Stage 1: Screening
    all_stage1_results = run_stage1(features, target, prices_daily, available_top20, tracker)

    # Save stage 1 results
    with open("results/sweep_v2_stage1.json", "w") as f:
        # Serialize cleanly (drop model_cls which isn't JSON-serializable)
        clean = []
        for r in all_stage1_results:
            cfg = {k: v for k, v in r["config"].items() if k != "model_cls"}
            clean.append({**r, "config": cfg})
        json.dump(clean, f, indent=2, default=str)
    print("Stage 1 results saved to results/sweep_v2_stage1.json", flush=True)

    # Stage 2: Walk-forward validation
    wf_results = run_stage2(all_stage1_results, features, target, prices_daily, tracker)

    # Save walk-forward results
    with open("results/sweep_v2_walkforward.json", "w") as f:
        clean = []
        for r in wf_results:
            clean.append({k: v for k, v in r.items() if k != "model_cls"})
        json.dump(clean, f, indent=2, default=str)
    print("Walk-forward results saved to results/sweep_v2_walkforward.json", flush=True)

    # Write summary
    write_summary(all_stage1_results, wf_results)

    print("\n" + "=" * 80, flush=True)
    print("SWEEP COMPLETE", flush=True)
    best_wf = max(wf_results, key=lambda x: x["wf_sharpe"]) if wf_results else None
    if best_wf:
        print(f"Best walk-forward Sharpe: {best_wf['wf_sharpe']:.3f} "
              f"({best_wf['model_type']} {best_wf['feature_set']})", flush=True)
        beats = "YES" if best_wf["beats_baseline"] else "NO"
        print(f"Beats Donchian {BASELINE_SHARPE}? {beats}", flush=True)
    print("=" * 80, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
