#!/usr/bin/env python3
"""Two-stage hyperparameter sweep — Contract #004, Step 2.

Uses daily-resampled hourly feature matrix (btc_hourly_expanded).
Sharpe is computed on daily signals vs daily price returns, no look-ahead.
Baseline: Donchian Sharpe 1.062 (in-sample).

Stage 1: Screening — single 80/20 temporal split, 24+ configs across 3 families.
Stage 2: Validation — expanding-window walk-forward on top 5.
"""

import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparky.data.loader import load
from sparky.oversight.timeout import ExperimentTimeout, with_timeout
from sparky.tracking.experiment import ExperimentTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASELINE_SHARPE = 1.062
PROGRESS_FILE = Path("results/sweep_v2_progress.csv")
RESULTS_DIR = Path("results/validation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Top 20 features from feature_importance.json ──────────────────────────────
TOP20_FEATURES = [
    "volume_dry_up",
    "rsi_6h",
    "rsi_168h",
    "price_range_expansion",
    "momentum_720h",
    "rsi_4h",
    "momentum_divergence_72h_336h",
    "vwap_cross",
    "mfi_14h",
    "price_acceleration_10h",
    "bb_position_20h",
    "momentum_divergence_4h_24h",
    "momentum_4h",
    "atr_14h",
    "rsi_divergence_14h_168h",
    "breakout_proximity_upper",
    "rsi_14h",
    "momentum_336h",
    "rsi_volume_interaction",
    "close_above_open_ratio_20h",
]
TOP15_FEATURES = TOP20_FEATURES[:15]
TOP10_FEATURES = TOP20_FEATURES[:10]


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data():
    """Load and prepare in-sample data. Returns (features, target, prices_daily)."""
    logger.info("Loading data...")
    features = load("feature_matrix_btc_hourly_expanded", purpose="training")
    target_df = load("targets_btc_hourly_1d", purpose="training")
    target = target_df["target"]
    prices_raw = load("ohlcv_hourly_max_coverage", purpose="analysis")

    # Resample hourly prices to daily
    prices_daily = prices_raw.resample("D").last()

    # Align features and target
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    # In-sample only: through end of 2023
    features = features.loc[:"2023-12-31"]
    target = target.loc[:"2023-12-31"]

    # Drop NaN/inf
    features = features.replace([np.inf, -np.inf], np.nan)
    mask = features.notna().all(axis=1) & target.notna()
    features = features.loc[mask]
    target = target.loc[features.index]

    logger.info(
        f"Data: {features.shape[0]} rows × {features.shape[1]} features | "
        f"{features.index.min().date()} → {features.index.max().date()}"
    )
    logger.info(f"Target balance: {target.mean():.1%} positive")
    return features, target, prices_daily


# ── Sharpe computation ────────────────────────────────────────────────────────


def compute_sharpe(signals: pd.Series, prices_daily: pd.DataFrame) -> float:
    """Annualised Sharpe from daily signals and prices. Signals shifted to avoid look-ahead."""
    cost_pct = 0.001  # 10 bps round-trip

    common = signals.index.intersection(prices_daily.index)
    if len(common) < 30:
        return 0.0

    sig = signals.loc[common]
    price_close = prices_daily.loc[common, "close"]

    daily_ret = price_close.pct_change()
    # No look-ahead: signal[T] is applied on return[T+1]
    strat_ret = sig.shift(1) * daily_ret
    # Transaction costs on signal changes
    costs = sig.diff().abs() * cost_pct
    net_ret = (strat_ret - costs).dropna()

    if len(net_ret) < 30 or net_ret.std() == 0:
        return 0.0

    return float(net_ret.mean() / net_ret.std() * np.sqrt(365))


def signals_from_proba(y_proba: np.ndarray, index: pd.Index, threshold: float = 0.52) -> pd.Series:
    """Convert probability predictions to long/flat signals."""
    return pd.Series((y_proba > threshold).astype(int), index=index)


# ── Model factory ─────────────────────────────────────────────────────────────


def make_model(cfg: dict):
    family = cfg["family"]
    params = cfg["params"].copy()
    if family == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(**params, eval_metric="logloss", verbosity=0, random_state=42)
    elif family == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params, verbose=-1, random_state=42)
    elif family == "CatBoost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params, verbose=0, random_seed=42)
    raise ValueError(f"Unknown family: {family}")


# ── Config grid ───────────────────────────────────────────────────────────────


def build_configs() -> list[dict]:
    """Build 24+ configs: 8 CatBoost + 8 LightGBM + 8 XGBoost."""
    configs = []

    # ── CatBoost (8 configs) ──
    for depth, lr, l2, border_count in [
        (4, 0.05, 1.0, 32),
        (5, 0.05, 2.0, 32),
        (6, 0.05, 3.0, 64),
        (4, 0.10, 1.0, 32),
        (5, 0.10, 2.0, 64),
        (3, 0.01, 1.0, 32),
        (6, 0.01, 5.0, 64),
        (5, 0.03, 3.0, 32),
    ]:
        configs.append(
            {
                "family": "CatBoost",
                "features": "top20",
                "params": {
                    "depth": depth,
                    "learning_rate": lr,
                    "iterations": 300,
                    "l2_leaf_reg": l2,
                    "border_count": border_count,
                    "task_type": "GPU",
                    "devices": "0",
                },
            }
        )

    # ── LightGBM (8 configs) ──
    for max_depth, num_leaves, lr, reg_l in [
        (4, 31, 0.05, 1.0),
        (5, 63, 0.05, 1.0),
        (6, 63, 0.10, 2.0),
        (4, 15, 0.10, 0.5),
        (3, 31, 0.01, 1.0),
        (5, 127, 0.03, 2.0),
        (6, 31, 0.05, 3.0),
        (4, 63, 0.03, 0.1),
    ]:
        configs.append(
            {
                "family": "LightGBM",
                "features": "top20",
                "params": {
                    "max_depth": max_depth,
                    "num_leaves": num_leaves,
                    "learning_rate": lr,
                    "n_estimators": 300,
                    "reg_lambda": reg_l,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "device": "gpu",
                    "gpu_platform_id": 0,
                    "gpu_device_id": 0,
                },
            }
        )

    # ── XGBoost (8 configs) ──
    for max_depth, lr, reg_l, n_est in [
        (4, 0.05, 1.0, 300),
        (5, 0.05, 2.0, 300),
        (6, 0.10, 1.0, 200),
        (3, 0.01, 0.5, 500),
        (4, 0.10, 3.0, 200),
        (5, 0.03, 1.0, 500),
        (6, 0.05, 0.1, 300),
        (3, 0.05, 2.0, 300),
    ]:
        configs.append(
            {
                "family": "XGBoost",
                "features": "top20",
                "params": {
                    "max_depth": max_depth,
                    "learning_rate": lr,
                    "n_estimators": n_est,
                    "reg_lambda": reg_l,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "tree_method": "hist",
                    "device": "cuda",
                },
            }
        )

    # ── Extra configs with different feature subsets (top10 / top15) ──
    # CatBoost on top10 and top15
    for feat_set in ["top10", "top15"]:
        configs.append(
            {
                "family": "CatBoost",
                "features": feat_set,
                "params": {
                    "depth": 5,
                    "learning_rate": 0.05,
                    "iterations": 300,
                    "l2_leaf_reg": 2.0,
                    "task_type": "GPU",
                    "devices": "0",
                },
            }
        )

    # LightGBM on top10 and top15
    for feat_set in ["top10", "top15"]:
        configs.append(
            {
                "family": "LightGBM",
                "features": feat_set,
                "params": {
                    "max_depth": 5,
                    "num_leaves": 63,
                    "learning_rate": 0.05,
                    "n_estimators": 300,
                    "reg_lambda": 1.0,
                    "device": "gpu",
                    "gpu_platform_id": 0,
                    "gpu_device_id": 0,
                },
            }
        )

    return configs


def get_features(features_all: pd.DataFrame, feat_set: str) -> pd.DataFrame:
    """Select feature subset, filtering to available columns."""
    if feat_set == "top20":
        cols = [c for c in TOP20_FEATURES if c in features_all.columns]
    elif feat_set == "top15":
        cols = [c for c in TOP15_FEATURES if c in features_all.columns]
    elif feat_set == "top10":
        cols = [c for c in TOP10_FEATURES if c in features_all.columns]
    else:
        cols = features_all.columns.tolist()
    return features_all[cols]


# ── Progress logging ──────────────────────────────────────────────────────────


def log_progress(stage: str, cfg: dict, sharpe: float, acc: float, elapsed: float):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not PROGRESS_FILE.exists()
    with open(PROGRESS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["ts", "stage", "family", "features", "sharpe", "acc", "elapsed_s", "params"])
        w.writerow(
            [
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                stage,
                cfg["family"],
                cfg["features"],
                f"{sharpe:.4f}",
                f"{acc:.4f}",
                f"{elapsed:.1f}",
                json.dumps(cfg["params"]),
            ]
        )


# ── Stage 1: Screen ───────────────────────────────────────────────────────────


@with_timeout(seconds=900)
def screen_config(cfg: dict, X_tr, y_tr, X_te, y_te, prices_daily) -> dict:
    """Train on 80%, evaluate on 20%. Returns sharpe, acc, auc, elapsed_s."""
    t0 = time.time()
    model = make_model(cfg)
    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    acc = float(accuracy_score(y_te, y_pred))
    auc = float(roc_auc_score(y_te, y_proba)) if len(np.unique(y_te)) > 1 else 0.5
    signals = signals_from_proba(y_proba, X_te.index)
    sharpe = compute_sharpe(signals, prices_daily)
    elapsed = time.time() - t0

    return {"sharpe": sharpe, "acc": acc, "auc": auc, "elapsed_s": elapsed}


def run_stage1(features_all, target, prices_daily, configs) -> list[dict]:
    """Run all configs on 80/20 split. Returns results sorted by sharpe desc."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STAGE 1: SCREENING ({len(configs)} configs, single 80/20 split)")
    logger.info("=" * 80)

    split_idx = int(len(features_all) * 0.8)
    logger.info(
        f"Train: {split_idx} rows ({features_all.index[0].date()} → {features_all.index[split_idx - 1].date()})"
    )
    logger.info(
        f"Test:  {len(features_all) - split_idx} rows ({features_all.index[split_idx].date()} → {features_all.index[-1].date()})\n"
    )

    results = []
    for i, cfg in enumerate(configs, 1):
        X = get_features(features_all, cfg["features"])
        X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_te = target.iloc[:split_idx], target.iloc[split_idx:]

        try:
            r = screen_config(cfg, X_tr, y_tr, X_te, y_te, prices_daily)
        except ExperimentTimeout:
            logger.warning(f"  [{i}/{len(configs)}] TIMEOUT: {cfg['family']} {cfg['features']}")
            r = {"sharpe": 0.0, "acc": 0.5, "auc": 0.5, "elapsed_s": 900.0}
        except Exception as e:
            logger.error(f"  [{i}/{len(configs)}] ERROR: {cfg['family']}: {e}")
            r = {"sharpe": 0.0, "acc": 0.5, "auc": 0.5, "elapsed_s": 0.0}

        beat = " ⭐ BEATS BASELINE" if r["sharpe"] > BASELINE_SHARPE else ""
        depth = cfg["params"].get("depth", cfg["params"].get("max_depth", "?"))
        lr = cfg["params"].get("learning_rate", "?")
        logger.info(
            f"  [{i:2d}/{len(configs)}] {cfg['family']:10s} {cfg['features']:6s} "
            f"d={depth} lr={lr} → Sharpe={r['sharpe']:.3f} Acc={r['acc']:.3f} AUC={r['auc']:.3f} "
            f"({r['elapsed_s']:.0f}s){beat}"
        )
        log_progress("screen", cfg, r["sharpe"], r["acc"], r["elapsed_s"])
        results.append({"config": cfg, **r})

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


# ── Stage 2: Walk-forward ─────────────────────────────────────────────────────


def walkforward_config(cfg: dict, features_all, target, prices_daily) -> dict:
    """Expanding-window walk-forward. Train on all years before test year."""
    # Walk-forward years: test on each year 2019-2023
    # Min 2 years of training data required
    test_years = [2019, 2020, 2021, 2022, 2023]
    yearly = []

    for year in test_years:
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        X = get_features(features_all, cfg["features"])
        X_tr = X.loc[:train_end]
        y_tr = target.loc[:train_end]
        X_te = X.loc[test_start:test_end]
        y_te = target.loc[test_start:test_end]

        if len(X_tr) < 200 or len(X_te) < 30:
            logger.warning(f"    {year}: skipped (train={len(X_tr)}, test={len(X_te)})")
            continue

        try:
            model = make_model(cfg)
            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_te)[:, 1]
            y_pred = (y_proba > 0.5).astype(int)

            acc = float(accuracy_score(y_te, y_pred))
            auc = float(roc_auc_score(y_te, y_proba)) if len(np.unique(y_te)) > 1 else 0.5
            signals = signals_from_proba(y_proba, X_te.index)
            sharpe = compute_sharpe(signals, prices_daily)
            yearly.append(
                {"year": year, "sharpe": sharpe, "acc": acc, "auc": auc, "train_n": len(X_tr), "test_n": len(X_te)}
            )
        except Exception as e:
            logger.error(f"    {year} ERROR: {e}")

    if not yearly:
        return {"mean_sharpe": 0.0, "std_sharpe": 0.0, "mean_acc": 0.5, "mean_auc": 0.5, "yearly": []}

    sharpes = [r["sharpe"] for r in yearly]
    return {
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "min_sharpe": float(np.min(sharpes)),
        "max_sharpe": float(np.max(sharpes)),
        "mean_acc": float(np.mean([r["acc"] for r in yearly])),
        "mean_auc": float(np.mean([r["auc"] for r in yearly])),
        "n_years": len(yearly),
        "yearly": yearly,
    }


def run_stage2(features_all, target, prices_daily, top_configs: list[dict], top_n: int = 5) -> list[dict]:
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STAGE 2: WALK-FORWARD VALIDATION (top {top_n} configs)")
    logger.info("=" * 80)

    validated = []
    for i, entry in enumerate(top_configs[:top_n], 1):
        cfg = entry["config"]
        logger.info(f"\n  [{i}/{top_n}] {cfg['family']} {cfg['features']} (screening Sharpe={entry['sharpe']:.3f})")

        t0 = time.time()
        wf = walkforward_config(cfg, features_all, target, prices_daily)
        elapsed = time.time() - t0

        beat = " ⭐ BEATS BASELINE" if wf["mean_sharpe"] > BASELINE_SHARPE else ""
        logger.info(
            f"    Walk-forward: Sharpe={wf['mean_sharpe']:.3f} ± {wf['std_sharpe']:.3f} "
            f"min={wf.get('min_sharpe', 0):.3f} max={wf.get('max_sharpe', 0):.3f} "
            f"Acc={wf['mean_acc']:.3f}{beat}"
        )
        for yr in wf["yearly"]:
            logger.info(
                f"      {yr['year']}: Sharpe={yr['sharpe']:.3f} "
                f"Acc={yr['acc']:.3f} AUC={yr['auc']:.3f} "
                f"(train={yr['train_n']}, test={yr['test_n']})"
            )

        log_progress("wf_validate", cfg, wf["mean_sharpe"], wf["mean_acc"], elapsed)

        validated.append(
            {
                "config": cfg,
                "screening_sharpe": entry["sharpe"],
                **wf,
            }
        )

    validated.sort(key=lambda x: x["mean_sharpe"], reverse=True)
    return validated


# ── Wandb logging ─────────────────────────────────────────────────────────────


def log_sweep_to_wandb(tracker: ExperimentTracker, stage: str, results: list[dict], summary: dict):
    """Log a batch of results as a single wandb sweep run."""
    sweep_results = []
    for r in results:
        cfg = r["config"]
        sweep_results.append(
            {
                "config": {
                    "model_type": cfg["family"].lower(),
                    "features": cfg["features"],
                    **cfg["params"],
                },
                "metrics": {
                    "sharpe": r.get("sharpe", r.get("mean_sharpe", 0)),
                    "accuracy": r.get("acc", r.get("mean_acc", 0.5)),
                    "auc": r.get("auc", r.get("mean_auc", 0.5)),
                },
            }
        )

    run_id = tracker.log_sweep(
        name=f"{stage}_{len(results)}configs_{datetime.now().strftime('%H%M%S')}",
        results=sweep_results,
        summary_metrics=summary,
        tags=["contract_004", "sweep"],
        job_type="sweep",
    )
    return run_id


def log_wf_to_wandb(tracker: ExperimentTracker, entry: dict, rank: int):
    """Log individual walk-forward validated result."""
    cfg = entry["config"]
    params = cfg["params"].copy()
    params["model_type"] = cfg["family"].lower()
    params["features"] = cfg["features"]

    run_id = tracker.log_experiment(
        name=None,  # auto-generate descriptive name
        config=params,
        metrics={
            "sharpe": entry["mean_sharpe"],
            "sharpe_std": entry["std_sharpe"],
            "sharpe_min": entry.get("min_sharpe", 0),
            "sharpe_max": entry.get("max_sharpe", 0),
            "accuracy": entry["mean_acc"],
            "auc": entry.get("mean_auc", 0.5),
            "beats_baseline": int(entry["mean_sharpe"] > BASELINE_SHARPE),
            "screening_sharpe": entry["screening_sharpe"],
            "wf_rank": rank,
        },
        tags=["contract_004", "sweep"],
        job_type="sweep",
    )
    return run_id


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    logger.info("=" * 80)
    logger.info("TWO-STAGE SWEEP — Contract #004 Step 2")
    logger.info(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info(f"Baseline Donchian Sharpe: {BASELINE_SHARPE}")
    logger.info("=" * 80)

    tracker = ExperimentTracker(experiment_name="contract_004_sweep")

    # ── Load data ──
    features_all, target, prices_daily = load_data()

    # ── Build configs ──
    configs = build_configs()
    logger.info(
        f"\nTotal configs: {len(configs)} "
        f"({sum(c['family'] == 'CatBoost' for c in configs)} CB, "
        f"{sum(c['family'] == 'LightGBM' for c in configs)} LGBM, "
        f"{sum(c['family'] == 'XGBoost' for c in configs)} XGB)"
    )

    # ── Stage 1 ──
    stage1_results = run_stage1(features_all, target, prices_daily, configs)

    logger.info("\n--- STAGE 1 TOP 10 ---")
    for i, r in enumerate(stage1_results[:10], 1):
        cfg = r["config"]
        logger.info(
            f"  {i:2d}. {cfg['family']:10s} {cfg['features']:6s} "
            f"d={cfg['params'].get('depth', cfg['params'].get('max_depth', '?'))} "
            f"lr={cfg['params']['learning_rate']} "
            f"→ Sharpe={r['sharpe']:.3f} AUC={r['auc']:.3f}"
        )

    # Log Stage 1 batch to wandb
    best_s1 = stage1_results[0]["sharpe"] if stage1_results else 0
    best_s1_family = stage1_results[0]["config"]["family"] if stage1_results else "none"
    log_sweep_to_wandb(
        tracker,
        "stage1_screening",
        stage1_results,
        {
            "best_sharpe_stage1": best_s1,
            "best_family": 0.0,  # wandb needs floats
            "n_configs": float(len(stage1_results)),
            "n_beats_baseline": float(sum(r["sharpe"] > BASELINE_SHARPE for r in stage1_results)),
        },
    )
    logger.info(f"[wandb] Logged Stage 1 sweep ({len(stage1_results)} configs)")

    # ── Stage 2 ──
    validated = run_stage2(features_all, target, prices_daily, stage1_results, top_n=5)

    # Log Stage 2 batch to wandb
    log_sweep_to_wandb(
        tracker,
        "stage2_walkforward",
        validated,
        {
            "best_wf_sharpe": validated[0]["mean_sharpe"] if validated else 0,
            "n_wf_configs": float(len(validated)),
            "n_wf_beats_baseline": float(sum(v["mean_sharpe"] > BASELINE_SHARPE for v in validated)),
        },
    )
    logger.info(f"[wandb] Logged Stage 2 sweep ({len(validated)} walk-forward configs)")

    # Log individual walk-forward results
    for rank, v in enumerate(validated, 1):
        run_id = log_wf_to_wandb(tracker, v, rank)
        logger.info(f"[wandb] Logged WF rank {rank}: {v['config']['family']} Sharpe={v['mean_sharpe']:.3f} → {run_id}")

    # ── Final report ──
    logger.info("\n" + "=" * 80)
    logger.info("FINAL WALK-FORWARD RESULTS")
    logger.info("=" * 80)
    logger.info(f"Baseline (Donchian): {BASELINE_SHARPE:.3f}")

    for i, v in enumerate(validated, 1):
        cfg = v["config"]
        verdict = "BEATS BASELINE ⭐" if v["mean_sharpe"] > BASELINE_SHARPE else "below baseline"
        logger.info(
            f"\n  {i}. {cfg['family']} ({cfg['features']}) → "
            f"WF Sharpe={v['mean_sharpe']:.3f} ± {v['std_sharpe']:.3f} ({verdict})"
        )
        logger.info(f"     params: {cfg['params']}")
        logger.info(f"     Screening Sharpe: {v['screening_sharpe']:.3f}")
        for yr in v["yearly"]:
            logger.info(f"       {yr['year']}: Sharpe={yr['sharpe']:.3f}")

    # ── Save results JSON ──
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_sharpe": BASELINE_SHARPE,
        "n_stage1_configs": len(stage1_results),
        "n_stage2_validated": len(validated),
        "stage1_top10": [
            {
                "rank": i + 1,
                "family": r["config"]["family"],
                "features": r["config"]["features"],
                "params": r["config"]["params"],
                "sharpe": r["sharpe"],
                "acc": r["acc"],
                "auc": r["auc"],
            }
            for i, r in enumerate(stage1_results[:10])
        ],
        "stage2_walkforward": [
            {
                "rank": i + 1,
                "family": v["config"]["family"],
                "features": v["config"]["features"],
                "params": v["config"]["params"],
                "screening_sharpe": v["screening_sharpe"],
                "mean_sharpe": v["mean_sharpe"],
                "std_sharpe": v["std_sharpe"],
                "min_sharpe": v.get("min_sharpe", 0),
                "max_sharpe": v.get("max_sharpe", 0),
                "mean_acc": v["mean_acc"],
                "yearly": v["yearly"],
                "beats_baseline": v["mean_sharpe"] > BASELINE_SHARPE,
            }
            for i, v in enumerate(validated)
        ],
    }
    out_path = RESULTS_DIR / "sweep_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved: {out_path}")
    logger.info(f"Progress log: {PROGRESS_FILE}")
    logger.info(f"Completed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    return output


if __name__ == "__main__":
    results = main()
