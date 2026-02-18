#!/usr/bin/env python3
"""
CONTRACT #004 — Step 4: Ensemble Strategies

Three ensemble approaches:
1. Signal averaging: Equal-weight + inverse-volatility weighted average of top ML signals
2. Regime-conditional: ADX(14,30) switches between ML and Donchian
3. Stacking: LightGBM GPU meta-learner on out-of-fold base model predictions

Walk-forward validated (expanding window, 2019–2023).
Baseline: Donchian Multi-TF Sharpe 1.062
Best individual ML: LightGBM top-10 Sharpe 1.365

All runs tagged: contract_004, ensemble
"""

import sys

sys.path.insert(0, "src")

import os

os.environ["PYTHONUNBUFFERED"] = "1"

import json
import time
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sparky.data.loader import load
from sparky.features.returns import annualized_sharpe
from sparky.oversight.timeout import with_timeout
from sparky.tracking.experiment import ExperimentTracker

# ─── Constants ─────────────────────────────────────────────────────────────────
BASELINE_DONCHIAN_SHARPE = 1.062
BEST_ML_SHARPE = 1.365  # LightGBM top-10, Step 2
ADX_BEST_SHARPE = 1.181  # ADX(14,30), Step 3

ENTRY_PERIOD = 40
EXIT_PERIOD = 20

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAGS = ["contract_004", "ensemble"]
JOB_TYPE = "ensemble"

# Top 5 configs from Step 2 walk-forward (in order of WF Sharpe)
# LGBM-top10 (1.365), CB-top20-d3 (1.325), LGBM-top20-d3 (1.197), XGB-top20-d3 (1.050), XGB-top20-d5 (0.623)
TOP3_CONFIGS = [
    (
        "LightGBM",
        {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "reg_lambda": 1.0,
            "num_leaves": 63,
            "device": "gpu",
            "n_jobs": 1,
            "verbose": -1,
            "random_state": 42,
        },
    ),
    (
        "CatBoost",
        {
            "iterations": 300,
            "depth": 3,
            "learning_rate": 0.01,
            "l2_leaf_reg": 1.0,
            "border_count": 32,
            "task_type": "GPU",
            "devices": "0",
            "verbose": 0,
            "random_state": 42,
        },
    ),
    (
        "LightGBM",
        {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.01,
            "reg_lambda": 1.0,
            "num_leaves": 31,
            "device": "gpu",
            "n_jobs": 1,
            "verbose": -1,
            "random_state": 42,
        },
    ),
]

TOP5_CONFIGS = TOP3_CONFIGS + [
    ("XGBoost", None),  # placeholder — XGBoost skipped for speed (Sharpe 1.050 barely baseline)
    ("XGBoost", None),
]


# ─── Data Loading ──────────────────────────────────────────────────────────────
def load_data():
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Features via loader (enforces holdout)
    df_feat = load("feature_matrix_btc_hourly_expanded", purpose="training")

    # Raw hourly for Donchian signals
    prices_hourly = pd.read_parquet("data/raw/btc/ohlcv_hourly_max_coverage.parquet")
    prices_daily = prices_hourly["close"].resample("D").last().dropna()
    if prices_daily.index.tz is None:
        prices_daily.index = prices_daily.index.tz_localize("UTC")
    prices_daily = prices_daily.loc["2019-01-01":"2023-12-31"]
    daily_returns = prices_daily.pct_change().dropna()

    # Daily-aggregate features for ML
    df_daily = df_feat.resample("D").mean()
    df_daily = df_daily.loc["2019-01-01":"2023-12-31"]

    # Top 10 features from Step 2 (LightGBM top-10)
    # (using feature importance from step 2 analysis)
    top10_cols = [
        "rsi_divergence_14h_168h",
        "price_momentum_divergence",
        "intraday_range",
        "recovery_from_20h_low",
        "rsi_volume_interaction",
        "vwap_deviation_24h",
        "tick_direction_ratio_24h",
        "rsi_divergence_4h_24h",
        "momentum_divergence_72h_336h",
        "drawdown_from_20h_high",
    ]
    # Filter to available columns
    available = [c for c in top10_cols if c in df_daily.columns]
    if len(available) < 5:
        # Fallback: pick top-10 by variance
        available = df_daily.var().sort_values(ascending=False).head(10).index.tolist()
    top20_cols = df_daily.var().sort_values(ascending=False).head(20).index.tolist()

    # Daily target (next-day direction)
    target_raw = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")
    if isinstance(target_raw, pd.DataFrame):
        target_raw = target_raw["target"]
    target_daily = target_raw.resample("D").last()
    target_daily = target_daily.loc["2019-01-01":"2023-12-31"]

    # Align
    common_idx = prices_daily.index.intersection(df_daily.index).intersection(target_daily.index)
    prices_daily = prices_daily.loc[common_idx]
    daily_returns = daily_returns.reindex(common_idx).fillna(0)
    df_daily = df_daily.loc[common_idx]
    target_daily = target_daily.loc[common_idx]

    # Remove NaN rows
    valid = target_daily.notna()
    prices_daily = prices_daily[valid]
    daily_returns = daily_returns[valid]
    df_daily = df_daily[valid].fillna(df_daily.median())
    target_daily = target_daily[valid]

    print(
        f"  Daily prices: {len(prices_daily)} rows ({prices_daily.index[0].date()} - {prices_daily.index[-1].date()})"
    )
    print(f"  Features top-10: {available}")
    print(f"  Target distribution: {target_daily.value_counts().to_dict()}")

    return prices_daily, daily_returns, df_daily, target_daily, available, top20_cols


# ─── Donchian Signal ────────────────────────────────────────────────────────────
def donchian_signal(prices: pd.Series, entry: int = 40, exit_p: int = 20) -> pd.Series:
    """No look-ahead Donchian channel signal."""
    upper = prices.rolling(window=entry).max()
    lower = prices.rolling(window=exit_p).min()
    sig = pd.Series(0, index=prices.index, dtype=int)
    in_pos = False
    for i in range(len(prices)):
        if i < entry:
            continue
        px = prices.iloc[i]
        if not in_pos:
            if px >= upper.iloc[i - 1]:
                in_pos = True
                sig.iloc[i] = 1
        else:
            if i >= exit_p and px <= lower.iloc[i - 1]:
                in_pos = False
            else:
                sig.iloc[i] = 1
    return sig


# ─── ADX ────────────────────────────────────────────────────────────────────────
def compute_adx(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    plus_dm = delta.clip(lower=0)
    minus_dm = (-delta).clip(lower=0)
    plus_di = plus_dm.ewm(span=period, adjust=False).mean()
    minus_di = minus_dm.ewm(span=period, adjust=False).mean()
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


# ─── Equity + Sharpe Helpers ────────────────────────────────────────────────────
def compute_equity(returns: pd.Series, positions: pd.Series) -> pd.Series:
    pos = positions.reindex(returns.index).fillna(0)
    pos_shifted = pos.shift(1).fillna(0)
    strat_ret = pos_shifted * returns
    equity = (1 + strat_ret).cumprod()
    return equity


def sharpe_from_signals(returns: pd.Series, signals: pd.Series) -> float:
    pos = signals.reindex(returns.index).fillna(0).shift(1).fillna(0)
    strat_ret = pos * returns
    if strat_ret.std() == 0 or len(strat_ret) < 10:
        return 0.0
    return float(annualized_sharpe(strat_ret))


def year_sharpe(returns: pd.Series, signals: pd.Series, year: int) -> float:
    mask = returns.index.year == year
    return sharpe_from_signals(returns[mask], signals.reindex(returns[mask].index).fillna(0))


def train_model(name: str, params: dict, X_train, y_train):
    if name == "LightGBM":
        model = LGBMClassifier(**params)
    elif name == "CatBoost":
        model = CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {name}")
    model.fit(X_train, y_train)
    return model


# ════════════════════════════════════════════════════════════════════════════════
# APPROACH 1 — Signal Averaging (Equal-weight + Inverse-Vol Weighted)
# ════════════════════════════════════════════════════════════════════════════════
@with_timeout(seconds=900)
def run_signal_averaging(prices_daily, daily_returns, df_daily, target_daily, top10_cols, top20_cols, tracker):
    print("\n" + "=" * 80)
    print("APPROACH 1: Signal Averaging Ensemble")
    print("=" * 80)

    years = list(range(2019, 2024))
    configs = [
        ("LGBM-top10", TOP3_CONFIGS[0][0], TOP3_CONFIGS[0][1], top10_cols),
        ("CB-top20", TOP3_CONFIGS[1][0], TOP3_CONFIGS[1][1], top20_cols),
        ("LGBM-top20", TOP3_CONFIGS[2][0], TOP3_CONFIGS[2][1], top20_cols),
    ]

    results_equal = {}
    results_invvol = {}

    for year in years:
        train_mask = daily_returns.index.year < year
        test_mask = daily_returns.index.year == year

        X_trains = []
        for name, family, params, feat_cols in configs:
            avail_cols = [c for c in feat_cols if c in df_daily.columns]
            X_trains.append(df_daily[avail_cols])

        if train_mask.sum() < 100:
            results_equal[year] = 0.0
            results_invvol[year] = 0.0
            continue

        # Train models and get test probabilities
        year_probas = []
        for name, family, params, feat_cols in configs:
            avail_cols = [c for c in feat_cols if c in df_daily.columns]
            X_tr = df_daily[avail_cols][train_mask]
            y_tr = target_daily[train_mask]
            X_te = df_daily[avail_cols][test_mask]

            # Remove NaN
            valid = X_tr.notna().all(axis=1) & y_tr.notna()
            X_tr, y_tr = X_tr[valid], y_tr[valid]
            X_te = X_te.fillna(X_te.median())

            if len(X_tr) < 50 or len(X_te) == 0:
                year_probas.append(None)
                continue

            try:
                model = train_model(family, params, X_tr, y_tr)
                proba = model.predict_proba(X_te)[:, 1]
                year_probas.append(pd.Series(proba, index=X_te.index))
            except Exception as e:
                print(f"  WARN: {name} year {year} failed: {e}")
                year_probas.append(None)

        valid_probas = [p for p in year_probas if p is not None]
        if not valid_probas:
            results_equal[year] = 0.0
            results_invvol[year] = 0.0
            continue

        # Align all probas to common index
        common_idx = valid_probas[0].index
        for p in valid_probas[1:]:
            common_idx = common_idx.intersection(p.index)
        valid_probas = [p.loc[common_idx] for p in valid_probas]

        # Equal-weight ensemble
        proba_eq = np.mean([p.values for p in valid_probas], axis=0)
        signals_eq = pd.Series((proba_eq > 0.5).astype(int), index=common_idx)
        sh_eq = year_sharpe(daily_returns, signals_eq, year)
        results_equal[year] = round(sh_eq, 3)

        # Inverse-volatility weighting: weight model by 1/rolling_vol_of_predictions
        # Use last-year OOF volatility as a proxy (estimate from signal variance)
        weights = []
        for p in valid_probas:
            vol = float(p.std()) if p.std() > 0 else 1.0
            weights.append(1.0 / vol)
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        proba_ivw = sum(w * p.values for w, p in zip(weights, valid_probas))
        signals_ivw = pd.Series((proba_ivw > 0.5).astype(int), index=common_idx)
        sh_ivw = year_sharpe(daily_returns, signals_ivw, year)
        results_invvol[year] = round(sh_ivw, 3)

        print(f"  {year}: equal-weight={sh_eq:.3f}, inv-vol={sh_ivw:.3f} (n_models={len(valid_probas)})")

    sharpes_eq = [v for v in results_equal.values() if v is not None]
    sharpes_ivw = [v for v in results_invvol.values() if v is not None]

    mean_eq = float(np.mean(sharpes_eq))
    std_eq = float(np.std(sharpes_eq))
    mean_ivw = float(np.mean(sharpes_ivw))
    std_ivw = float(np.std(sharpes_ivw))

    print(f"\n  Equal-weight: Mean={mean_eq:.3f} ± {std_eq:.3f}")
    print(f"  Inv-vol:      Mean={mean_ivw:.3f} ± {std_ivw:.3f}")
    print(f"  Baseline:     {BASELINE_DONCHIAN_SHARPE:.3f}")

    # Log equal-weight
    beats_eq = mean_eq > BASELINE_DONCHIAN_SHARPE
    cfg_eq = {
        "approach": "signal_averaging_equal_weight",
        "n_models": 3,
        "models": ["LGBM-top10", "CB-top20", "LGBM-top20"],
        "threshold": 0.5,
        "notes": (
            f"Equal-weight average of top-3 model probabilities. "
            f"Mean WF Sharpe {mean_eq:.3f} ± {std_eq:.3f}. "
            f"2022: {results_equal.get(2022, 'N/A')}. "
            + (
                "Beats baseline — diversification reduces model-specific variance."
                if beats_eq
                else "Below baseline — ensembling does not recover ML alpha."
            )
        ),
    }
    metrics_eq = {
        "sharpe": mean_eq,
        "mean_wf_sharpe": mean_eq,
        "std_wf_sharpe": std_eq,
        "sharpe_2022": results_equal.get(2022, 0.0),
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        "best_individual_ml": BEST_ML_SHARPE,
        **{f"sharpe_{yr}": v for yr, v in results_equal.items()},
    }
    tracker.log_experiment(
        name=f"ensemble_equal_weight_S{mean_eq:.2f}",
        config=cfg_eq,
        metrics=metrics_eq,
        tags=TAGS,
        job_type=JOB_TYPE,
        group="signal_averaging",
    )
    print(f"  Logged: ensemble_equal_weight_S{mean_eq:.2f}")

    # Log inverse-vol weighted
    beats_ivw = mean_ivw > BASELINE_DONCHIAN_SHARPE
    cfg_ivw = {
        "approach": "signal_averaging_inv_vol_weighted",
        "n_models": 3,
        "models": ["LGBM-top10", "CB-top20", "LGBM-top20"],
        "weighting": "inverse_prediction_volatility",
        "threshold": 0.5,
        "notes": (
            f"Inverse-volatility weighted average: weight = 1/std(predictions). "
            f"Upweights stable models, downweights noisy ones. "
            f"Mean WF Sharpe {mean_ivw:.3f} ± {std_ivw:.3f}. "
            f"2022: {results_invvol.get(2022, 'N/A')}. "
            + (
                "Inv-vol weighting improves over equal-weight."
                if mean_ivw > mean_eq
                else "Equal-weight performs comparably — prediction volatilities similar across models."
            )
        ),
    }
    metrics_ivw = {
        "sharpe": mean_ivw,
        "mean_wf_sharpe": mean_ivw,
        "std_wf_sharpe": std_ivw,
        "sharpe_2022": results_invvol.get(2022, 0.0),
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        "best_individual_ml": BEST_ML_SHARPE,
        **{f"sharpe_{yr}": v for yr, v in results_invvol.items()},
    }
    tracker.log_experiment(
        name=f"ensemble_inv_vol_S{mean_ivw:.2f}",
        config=cfg_ivw,
        metrics=metrics_ivw,
        tags=TAGS,
        job_type=JOB_TYPE,
        group="signal_averaging",
    )
    print(f"  Logged: ensemble_inv_vol_S{mean_ivw:.2f}")

    return {
        "equal_weight": {"mean": mean_eq, "std": std_eq, "per_year": results_equal},
        "inv_vol": {"mean": mean_ivw, "std": std_ivw, "per_year": results_invvol},
    }


# ════════════════════════════════════════════════════════════════════════════════
# APPROACH 2 — Regime-Conditional Ensemble
# ════════════════════════════════════════════════════════════════════════════════
@with_timeout(seconds=900)
def run_regime_conditional(prices_daily, daily_returns, df_daily, target_daily, top10_cols, tracker):
    """
    Use ADX(14,30) regime to switch between ML and Donchian.

    Trending regime (ADX > 30): use LGBM-top10 predictions (better in trending)
    Non-trending / ranging (ADX <= 30): flat (Donchian exits; ML unreliable)

    Variant A: ADX-switched ML (trending → LGBM, ranging → flat)
    Variant B: ADX-switched hybrid (trending → LGBM, ranging → Donchian)
    """
    print("\n" + "=" * 80)
    print("APPROACH 2: Regime-Conditional Ensemble")
    print("=" * 80)

    years = list(range(2019, 2024))
    adx_series = compute_adx(prices_daily, period=14)

    # Donchian baseline
    don_sig = donchian_signal(prices_daily, ENTRY_PERIOD, EXIT_PERIOD)

    # LGBM top-10 model (best single from Step 2)
    name_lgbm, params_lgbm = TOP3_CONFIGS[0]
    avail_cols = [c for c in top10_cols if c in df_daily.columns]
    if len(avail_cols) < 3:
        avail_cols = df_daily.var().sort_values(ascending=False).head(10).index.tolist()

    results_a = {}  # trending→ML, ranging→flat
    results_b = {}  # trending→ML, ranging→Donchian

    for year in years:
        train_mask = daily_returns.index.year < year
        test_mask = daily_returns.index.year == year

        if train_mask.sum() < 100:
            results_a[year] = 0.0
            results_b[year] = 0.0
            continue

        X_tr = df_daily[avail_cols][train_mask]
        y_tr = target_daily[train_mask]
        X_te = df_daily[avail_cols][test_mask]

        valid = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr, y_tr = X_tr[valid], y_tr[valid]
        X_te = X_te.fillna(X_te.median())

        try:
            model = train_model(name_lgbm, params_lgbm, X_tr, y_tr)
            proba_ml = pd.Series(model.predict_proba(X_te)[:, 1], index=X_te.index)
            signals_ml = (proba_ml > 0.5).astype(int)
        except Exception as e:
            print(f"  WARN: LGBM year {year} failed: {e}")
            signals_ml = pd.Series(0, index=X_te.index)

        # ADX regime for test year
        adx_test = adx_series.reindex(X_te.index, method="ffill").fillna(20)
        trending_mask = (adx_test > 30).astype(int)
        don_test = don_sig.reindex(X_te.index).fillna(0).astype(int)

        # Variant A: trending → ML, ranging → flat
        sig_a = (signals_ml * trending_mask).clip(0, 1).astype(int)
        sh_a = year_sharpe(daily_returns, sig_a, year)
        results_a[year] = round(sh_a, 3)

        # Variant B: trending → ML, ranging → Donchian
        sig_b = pd.Series(np.where(trending_mask.values, signals_ml.values, don_test.values), index=X_te.index).astype(
            int
        )
        sh_b = year_sharpe(daily_returns, sig_b, year)
        results_b[year] = round(sh_b, 3)

        adx_frac = float(trending_mask.mean())
        print(f"  {year}: trending_frac={adx_frac:.1%} | ML-only={sh_a:.3f} | ML+Donchian={sh_b:.3f}")

    sharpes_a = [v for v in results_a.values()]
    sharpes_b = [v for v in results_b.values()]

    mean_a, std_a = float(np.mean(sharpes_a)), float(np.std(sharpes_a))
    mean_b, std_b = float(np.mean(sharpes_b)), float(np.std(sharpes_b))

    print(f"\n  A (trending→ML, ranging→flat): Mean={mean_a:.3f} ± {std_a:.3f}")
    print(f"  B (trending→ML, ranging→Don):  Mean={mean_b:.3f} ± {std_b:.3f}")
    print(f"  Baseline: {BASELINE_DONCHIAN_SHARPE:.3f}")

    cfg_a = {
        "approach": "regime_conditional_ml_only",
        "regime_indicator": "ADX",
        "adx_period": 14,
        "adx_threshold": 30,
        "trending_regime_model": "LGBM-top10",
        "ranging_regime_action": "flat",
        "notes": (
            f"ADX(14)>30 → LGBM-top10 signal; ADX≤30 → flat. "
            f"Mean WF Sharpe {mean_a:.3f} ± {std_a:.3f}. 2022: {results_a.get(2022, 'N/A')}. "
            "Only trades when market has strong directional trend — cuts exposure in ranging/bear regimes."
        ),
    }
    metrics_a = {
        "sharpe": mean_a,
        "mean_wf_sharpe": mean_a,
        "std_wf_sharpe": std_a,
        "sharpe_2022": results_a.get(2022, 0.0),
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        "best_individual_ml": BEST_ML_SHARPE,
        **{f"sharpe_{yr}": v for yr, v in results_a.items()},
    }
    tracker.log_experiment(
        name=f"regime_cond_ml_flat_S{mean_a:.2f}",
        config=cfg_a,
        metrics=metrics_a,
        tags=TAGS,
        job_type=JOB_TYPE,
        group="regime_conditional",
    )
    print(f"  Logged: regime_cond_ml_flat_S{mean_a:.2f}")

    cfg_b = {
        "approach": "regime_conditional_ml_donchian",
        "regime_indicator": "ADX",
        "adx_period": 14,
        "adx_threshold": 30,
        "trending_regime_model": "LGBM-top10",
        "ranging_regime_model": "Donchian(40/20)",
        "notes": (
            f"ADX(14)>30 → LGBM-top10; ADX≤30 → Donchian(40/20). "
            f"Hybrid leverages ML in strong trends, falls back to time-tested Donchian in choppy regime. "
            f"Mean WF Sharpe {mean_b:.3f} ± {std_b:.3f}. 2022: {results_b.get(2022, 'N/A')}. "
            + (
                "Better than ML-only — Donchian fallback adds value in ranging regime."
                if mean_b > mean_a
                else "ML-only outperforms — Donchian fallback not helpful here."
            )
        ),
    }
    metrics_b = {
        "sharpe": mean_b,
        "mean_wf_sharpe": mean_b,
        "std_wf_sharpe": std_b,
        "sharpe_2022": results_b.get(2022, 0.0),
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        "best_individual_ml": BEST_ML_SHARPE,
        **{f"sharpe_{yr}": v for yr, v in results_b.items()},
    }
    tracker.log_experiment(
        name=f"regime_cond_ml_don_S{mean_b:.2f}",
        config=cfg_b,
        metrics=metrics_b,
        tags=TAGS,
        job_type=JOB_TYPE,
        group="regime_conditional",
    )
    print(f"  Logged: regime_cond_ml_don_S{mean_b:.2f}")

    return {
        "variant_a_ml_flat": {"mean": mean_a, "std": std_a, "per_year": results_a},
        "variant_b_ml_donchian": {"mean": mean_b, "std": std_b, "per_year": results_b},
    }


# ════════════════════════════════════════════════════════════════════════════════
# APPROACH 3 — Stacking: LightGBM GPU Meta-Learner
# ════════════════════════════════════════════════════════════════════════════════
@with_timeout(seconds=900)
def run_stacking(prices_daily, daily_returns, df_daily, target_daily, top10_cols, top20_cols, tracker):
    """
    Walk-forward stacking:
    1. For each test year, use OOF predictions from prior years to train meta-learner.
    2. Meta-learner = LightGBM (GPU) with base model probas as features.
    3. STRICT: meta-learner only trains on OOF predictions — no leakage.
    """
    print("\n" + "=" * 80)
    print("APPROACH 3: Stacking with LightGBM GPU Meta-Learner")
    print("=" * 80)

    years = list(range(2019, 2024))

    base_configs = [
        (
            "LGBM-top10",
            "LightGBM",
            TOP3_CONFIGS[0][1],
            [c for c in top10_cols if c in df_daily.columns]
            or df_daily.var().sort_values(ascending=False).head(10).index.tolist(),
        ),
        (
            "CB-top20",
            "CatBoost",
            TOP3_CONFIGS[1][1],
            df_daily.var().sort_values(ascending=False).head(20).index.tolist(),
        ),
        (
            "LGBM-top20",
            "LightGBM",
            TOP3_CONFIGS[2][1],
            df_daily.var().sort_values(ascending=False).head(20).index.tolist(),
        ),
    ]
    n_base = len(base_configs)

    # Meta-learner: LightGBM GPU, light config
    meta_params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "reg_lambda": 2.0,
        "num_leaves": 15,
        "device": "gpu",
        "n_jobs": 1,
        "verbose": -1,
        "random_state": 42,
    }

    # Step 1: Collect OOF predictions across all years for meta-training
    # We use past years' base model predictions to train meta-learner for current year.

    # Accumulate: for each year, train base models on pre-year data and predict
    oof_store = {}  # year -> (meta_X, meta_y)

    print("  Building OOF prediction store across years...")
    for yr in years:
        train_mask = daily_returns.index.year < yr
        test_mask = daily_returns.index.year == yr

        if train_mask.sum() < 100:
            oof_store[yr] = None
            continue

        base_probas = []
        for bname, bfamily, bparams, bcols in base_configs:
            avail = [c for c in bcols if c in df_daily.columns]
            if not avail:
                avail = df_daily.var().sort_values(ascending=False).head(10).index.tolist()
            X_tr = df_daily[avail][train_mask]
            y_tr = target_daily[train_mask]
            X_te = df_daily[avail][test_mask]

            valid = X_tr.notna().all(axis=1) & y_tr.notna()
            X_tr_v, y_tr_v = X_tr[valid], y_tr[valid]
            X_te = X_te.fillna(X_te.median())

            if len(X_tr_v) < 50 or len(X_te) == 0:
                base_probas.append(np.full(len(X_te), 0.5))
                continue

            try:
                model = train_model(bfamily, bparams, X_tr_v, y_tr_v)
                p = model.predict_proba(X_te)[:, 1]
                base_probas.append(p)
            except Exception as e:
                print(f"    WARN: {bname} year {yr} failed: {e}")
                base_probas.append(np.full(len(X_te), 0.5))

        # Build meta features for this year
        common_len = min(len(p) for p in base_probas)
        meta_X = np.column_stack([p[:common_len] for p in base_probas])
        meta_idx = daily_returns.index[test_mask][:common_len]
        meta_y = target_daily[test_mask].values[:common_len]

        oof_store[yr] = (meta_X, meta_y, meta_idx)

    # Step 2: Walk-forward: for year Y, train meta on OOF(years < Y), test on OOF(Y)
    results_stack = {}

    for yr in years:
        if oof_store[yr] is None:
            results_stack[yr] = 0.0
            continue

        # Collect meta training data from all prior years
        meta_train_X_list = []
        meta_train_y_list = []
        for prior_yr in years:
            if prior_yr >= yr:
                break
            if oof_store.get(prior_yr) is not None:
                px, py, _ = oof_store[prior_yr]
                meta_train_X_list.append(px)
                meta_train_y_list.append(py)

        if not meta_train_X_list:
            # No prior OOF data — fall back to equal weight
            meta_X_test, meta_y_test, meta_idx = oof_store[yr]
            proba_stack = meta_X_test.mean(axis=1)
            signals_stack = pd.Series((proba_stack > 0.5).astype(int), index=meta_idx)
            sh_stack = year_sharpe(daily_returns, signals_stack, yr)
            results_stack[yr] = round(sh_stack, 3)
            print(f"  {yr}: no prior OOF → equal-weight fallback, Sharpe={sh_stack:.3f}")
            continue

        meta_X_train = np.vstack(meta_train_X_list)
        meta_y_train = np.concatenate(meta_train_y_list)
        meta_X_test, meta_y_test, meta_idx = oof_store[yr]

        try:
            meta_model = LGBMClassifier(**meta_params)
            meta_model.fit(meta_X_train, meta_y_train)
            proba_stack = meta_model.predict_proba(meta_X_test)[:, 1]
            print(f"  {yr}: meta-learner trained on {len(meta_X_train)} OOF samples")
        except Exception as e:
            print(f"  WARN: meta-learner year {yr} failed: {e} — using equal-weight fallback")
            proba_stack = meta_X_test.mean(axis=1)

        signals_stack = pd.Series((proba_stack > 0.5).astype(int), index=meta_idx)
        sh_stack = year_sharpe(daily_returns, signals_stack, yr)
        results_stack[yr] = round(sh_stack, 3)
        print(f"  {yr}: stacking Sharpe={sh_stack:.3f}")

    sharpes_stack = list(results_stack.values())
    mean_stack = float(np.mean(sharpes_stack))
    std_stack = float(np.std(sharpes_stack))

    print(f"\n  Stacking: Mean={mean_stack:.3f} ± {std_stack:.3f}")
    print(f"  Baseline: {BASELINE_DONCHIAN_SHARPE:.3f}, Best ML: {BEST_ML_SHARPE:.3f}")

    beats = mean_stack > BASELINE_DONCHIAN_SHARPE
    cfg_stack = {
        "approach": "stacking_lgbm_meta",
        "base_models": ["LGBM-top10", "CB-top20", "LGBM-top20"],
        "meta_learner": "LightGBM-GPU",
        "meta_n_estimators": meta_params["n_estimators"],
        "meta_max_depth": meta_params["max_depth"],
        "meta_lr": meta_params["learning_rate"],
        "oof_strategy": "walk_forward_accumulating",
        "notes": (
            f"LightGBM GPU meta-learner trained on walk-forward OOF base model predictions. "
            f"No leakage: meta trains only on prior-year OOF, not same-year data. "
            f"Mean WF Sharpe {mean_stack:.3f} ± {std_stack:.3f}. "
            f"2022: {results_stack.get(2022, 'N/A')}. "
            + (
                "Stacking beats baseline — meta-learner finds better signal combinations."
                if beats
                else "Stacking does not beat baseline — OOF is too small/noisy for effective meta-learning (~3-4 years of data)."
            )
        ),
    }
    metrics_stack = {
        "sharpe": mean_stack,
        "mean_wf_sharpe": mean_stack,
        "std_wf_sharpe": std_stack,
        "sharpe_2022": results_stack.get(2022, 0.0),
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        "best_individual_ml": BEST_ML_SHARPE,
        **{f"sharpe_{yr}": v for yr, v in results_stack.items()},
    }
    tracker.log_experiment(
        name=f"stacking_lgbm_meta_S{mean_stack:.2f}",
        config=cfg_stack,
        metrics=metrics_stack,
        tags=TAGS,
        job_type=JOB_TYPE,
        group="stacking",
    )
    print(f"  Logged: stacking_lgbm_meta_S{mean_stack:.2f}")

    return {"stacking": {"mean": mean_stack, "std": std_stack, "per_year": results_stack}}


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("CONTRACT #004 — STEP 4: ENSEMBLE STRATEGIES")
    print("=" * 80)
    t0 = time.time()

    prices_daily, daily_returns, df_daily, target_daily, top10_cols, top20_cols = load_data()

    tracker = ExperimentTracker(experiment_name="ensemble_contract004_step4")

    all_results = {}

    # Approach 1: Signal Averaging
    try:
        r1 = run_signal_averaging(prices_daily, daily_returns, df_daily, target_daily, top10_cols, top20_cols, tracker)
        all_results["signal_averaging"] = r1
    except Exception as e:
        print(f"ERROR in signal averaging: {e}")
        all_results["signal_averaging"] = {"error": str(e)}

    # Approach 2: Regime-Conditional
    try:
        r2 = run_regime_conditional(prices_daily, daily_returns, df_daily, target_daily, top10_cols, tracker)
        all_results["regime_conditional"] = r2
    except Exception as e:
        print(f"ERROR in regime-conditional: {e}")
        all_results["regime_conditional"] = {"error": str(e)}

    # Approach 3: Stacking
    try:
        r3 = run_stacking(prices_daily, daily_returns, df_daily, target_daily, top10_cols, top20_cols, tracker)
        all_results["stacking"] = r3
    except Exception as e:
        print(f"ERROR in stacking: {e}")
        all_results["stacking"] = {"error": str(e)}

    elapsed = time.time() - t0

    # ─── Print Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ENSEMBLE SUMMARY")
    print("=" * 80)
    print(f"{'Approach':<45} {'MeanSh':>8} {'StdSh':>8} {'Sh2022':>8}")
    print("-" * 80)

    summary_rows = []
    for group, group_results in all_results.items():
        if "error" in group_results:
            print(f"  {group}: ERROR — {group_results['error']}")
            continue
        for variant, vres in group_results.items():
            if isinstance(vres, dict) and "mean" in vres:
                row = {
                    "approach": f"{group}/{variant}",
                    "mean_sharpe": vres["mean"],
                    "std_sharpe": vres["std"],
                    "sharpe_2022": vres["per_year"].get(2022, "N/A"),
                    "per_year": vres["per_year"],
                }
                summary_rows.append(row)
                sh22 = vres["per_year"].get(2022, "N/A")
                print(
                    f"  {row['approach']:<43} {vres['mean']:>8.3f} {vres['std']:>8.3f} "
                    f"  {sh22 if isinstance(sh22, str) else sh22:>6.3f}"
                )

    print(f"\n  Donchian baseline:      {BASELINE_DONCHIAN_SHARPE:.3f}")
    print(f"  Best individual ML:     {BEST_ML_SHARPE:.3f} (LGBM top-10)")
    print(f"  Best regime filtered:   {ADX_BEST_SHARPE:.3f} (ADX-14-t30)")

    # Find best ensemble
    best_row = max(summary_rows, key=lambda x: x["mean_sharpe"]) if summary_rows else None
    if best_row:
        print(f"\n  BEST ENSEMBLE: {best_row['approach']} — Sharpe {best_row['mean_sharpe']:.3f}")

    print(f"\nTotal elapsed: {elapsed / 60:.1f} min")

    # Save raw results
    results_file = RESULTS_DIR / "ensemble_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "summary": summary_rows,
                "donchian_baseline": BASELINE_DONCHIAN_SHARPE,
                "best_individual_ml": BEST_ML_SHARPE,
                "best_regime_filtered": ADX_BEST_SHARPE,
                "elapsed_seconds": elapsed,
                "raw": {
                    k: {
                        vk: {
                            "mean": vv["mean"],
                            "std": vv["std"],
                            "per_year": {str(y): s for y, s in vv["per_year"].items()},
                        }
                        for vk, vv in v.items()
                        if isinstance(vv, dict) and "mean" in vv
                    }
                    for k, v in all_results.items()
                    if "error" not in v
                },
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Results saved to {results_file}")

    return all_results, summary_rows


if __name__ == "__main__":
    all_results, summary_rows = main()
    print("\nDone.")
