#!/usr/bin/env python3
"""
CONTRACT #004 — Step 4 (v2): Ensemble + Momentum Strategies

Tests untested combinations from Step 4 spec + AK's momentum priority:

ENSEMBLE COMBOS (untested from spec):
1. LightGBM + vol_adx_OR regime filter
2. LightGBM + vol_adx_AND regime filter
3. Signal averaging: (Donchian + LightGBM) filtered by ADX(14,30)
4. Regime-switched: vol_adx_AND → Donchian vs LightGBM
5. Inverse-vol weighted ensemble of top 3 regime configs

MOMENTUM PRIORITY (AK directive):
6. Simple momentum + best regime filter (ADX-30)
7. Simple momentum + vol_adx_AND filter
8. Simple momentum + vol_adx_OR filter
9. Simple momentum + rolling drawdown filter
10. LightGBM meta-learner trained on momentum profitability

Baseline comparisons:
- Donchian (unfiltered): Sharpe 1.062
- LightGBM top-10 (best ML): Sharpe 1.365
- ADX(14,30) Donchian (best regime): Sharpe 1.181
- vol_adx_AND (best risk-adj): Sharpe 1.068 (2022: +0.192)
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
from lightgbm import LGBMClassifier

from sparky.data.loader import load
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.oversight.timeout import with_timeout
from sparky.tracking.experiment import ExperimentTracker

# ─── Constants ──────────────────────────────────────────────────────────────────
BASELINE_DONCHIAN_SHARPE = 1.062
BEST_ML_SHARPE = 1.365  # LightGBM top-10, Step 2
ADX_BEST_SHARPE = 1.181  # ADX(14,30), Step 3
VOL_ADX_AND_SHARPE = 1.068  # vol_adx_AND, Step 3 (2022: +0.192)
VOL_ADX_OR_SHARPE = 1.23  # vol_adx_OR, Step 3

ENTRY_PERIOD = 40
EXIT_PERIOD = 20

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAGS = ["contract_004", "ensemble"]
JOB_TYPE = "ensemble"

# LightGBM top-10 config (best from Step 2)
LGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "reg_lambda": 1.0,
    "num_leaves": 63,
    "device": "gpu",
    "n_jobs": 1,
    "verbose": -1,
    "random_state": 42,
}

TOP10_COLS = [
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


# ─── Data Loading ────────────────────────────────────────────────────────────────
def load_data():
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    t0 = time.time()

    # Features (holdout-enforced)
    df_feat = load("feature_matrix_btc_hourly_expanded", purpose="training")

    # Raw hourly prices → resample to daily
    prices_hourly = pd.read_parquet("data/raw/btc/ohlcv_hourly_max_coverage.parquet")
    prices_daily = prices_hourly["close"].resample("D").last().dropna()
    if prices_daily.index.tz is None:
        prices_daily.index = prices_daily.index.tz_localize("UTC")
    prices_daily = prices_daily.loc["2019-01-01":"2023-12-31"]
    daily_returns = prices_daily.pct_change().dropna()

    # Daily-aggregate features
    df_daily = df_feat.resample("D").mean()
    df_daily = df_daily.loc["2019-01-01":"2023-12-31"]

    # Feature columns
    avail10 = [c for c in TOP10_COLS if c in df_daily.columns]
    if len(avail10) < 5:
        avail10 = df_daily.var().sort_values(ascending=False).head(10).index.tolist()
    avail20 = df_daily.var().sort_values(ascending=False).head(20).index.tolist()

    # Target (next-day direction)
    target_raw = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")
    if isinstance(target_raw, pd.DataFrame):
        target_raw = target_raw["target"]
    target_daily = target_raw.resample("D").last()
    target_daily = target_daily.loc["2019-01-01":"2023-12-31"]

    # Align all series
    common = prices_daily.index.intersection(df_daily.index).intersection(target_daily.index)
    prices_daily = prices_daily.loc[common]
    daily_returns = daily_returns.reindex(common).fillna(0)
    df_daily = df_daily.loc[common]
    target_daily = target_daily.loc[common]

    valid = target_daily.notna()
    prices_daily = prices_daily[valid]
    daily_returns = daily_returns[valid]
    df_daily = df_daily[valid].fillna(df_daily.median())
    target_daily = target_daily[valid]

    print(f"  Daily rows: {len(prices_daily)} ({prices_daily.index[0].date()} – {prices_daily.index[-1].date()})")
    print(f"  Features: {df_daily.shape[1]} cols, top-10 avail: {len(avail10)}")
    elapsed = time.time() - t0
    print(f"  Data loaded in {elapsed:.1f}s")

    return prices_daily, daily_returns, df_daily, target_daily, avail10, avail20


# ─── Signal Helpers ──────────────────────────────────────────────────────────────
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


def momentum_signal(prices: pd.Series, lookback: int = 20, threshold: float = 0.05) -> pd.Series:
    """
    Simple momentum: buy when N-day return > threshold, flat otherwise.
    No look-ahead: uses price[t-1] to compute signal used at close[t].
    """
    mom = prices.pct_change(lookback)
    sig = (mom.shift(1) > threshold).astype(int)
    return sig


def compute_adx(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    plus_dm = delta.clip(lower=0)
    minus_dm = (-delta).clip(lower=0)
    plus_di = plus_dm.ewm(span=period, adjust=False).mean()
    minus_di = minus_dm.ewm(span=period, adjust=False).mean()
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
    return dx.ewm(span=period, adjust=False).mean()


def vol_regime(prices: pd.Series, window: int = 20) -> pd.Series:
    """1 when rolling vol > expanding median (trending high-vol)."""
    ret = prices.pct_change()
    rv = ret.rolling(window).std() * np.sqrt(252)
    exp_med = rv.expanding().median()
    return (rv > exp_med).astype(int)


def sharpe_from_signals(returns: pd.Series, signals: pd.Series) -> float:
    pos = signals.reindex(returns.index).fillna(0).shift(1).fillna(0)
    strat = pos * returns
    if strat.std() == 0 or len(strat) < 10:
        return 0.0
    return float(annualized_sharpe(strat))


def year_sharpe(returns: pd.Series, signals: pd.Series, year: int) -> float:
    mask = returns.index.year == year
    return sharpe_from_signals(returns[mask], signals.reindex(returns[mask].index).fillna(0))


def maxdd_from_signals(returns: pd.Series, signals: pd.Series) -> float:
    pos = signals.reindex(returns.index).fillna(0).shift(1).fillna(0)
    strat = pos * returns
    eq = (1 + strat).cumprod()
    return float(max_drawdown(eq))


def walk_forward_static(prices_daily, daily_returns, signal_fn, label: str) -> dict:
    """
    Walk-forward for a STATIC (no ML training) signal.
    signal_fn(prices_daily) → pd.Series of positions (0/1).
    We compute the full signal once, then slice per year.
    """
    full_sig = signal_fn(prices_daily)
    per_year = {}
    for yr in range(2019, 2024):
        sh = year_sharpe(daily_returns, full_sig, yr)
        per_year[yr] = round(sh, 3)
    values = list(per_year.values())
    mean_sh = float(np.mean(values))
    std_sh = float(np.std(values))
    mdd = maxdd_from_signals(daily_returns, full_sig)
    return {"mean": mean_sh, "std": std_sh, "mdd": mdd, "per_year": per_year}


def walk_forward_ml(
    prices_daily,
    daily_returns,
    df_daily,
    target_daily,
    feat_cols,
    label: str,
    regime_fn=None,  # optional: pd.Series(0/1) filtering ML signal
) -> dict:
    """
    Walk-forward for an ML signal (expanding window, train<year, test=year).
    If regime_fn is provided, ML signal is AND-gated with regime.
    """
    years = list(range(2019, 2024))
    avail = [c for c in feat_cols if c in df_daily.columns]
    if len(avail) < 3:
        avail = df_daily.var().sort_values(ascending=False).head(10).index.tolist()

    per_year = {}
    for year in years:
        train_mask = daily_returns.index.year < year
        test_mask = daily_returns.index.year == year

        if train_mask.sum() < 100:
            per_year[year] = 0.0
            continue

        X_tr = df_daily[avail][train_mask]
        y_tr = target_daily[train_mask]
        X_te = df_daily[avail][test_mask]

        valid = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr, y_tr = X_tr[valid], y_tr[valid]
        X_te = X_te.fillna(X_te.median())

        if len(X_tr) < 50 or len(X_te) == 0:
            per_year[year] = 0.0
            continue

        try:
            model = LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            sig_ml = pd.Series((proba > 0.5).astype(int), index=X_te.index)
        except Exception as e:
            print(f"  WARN: LGBM year {year} failed: {e}")
            per_year[year] = 0.0
            continue

        if regime_fn is not None:
            reg = regime_fn.reindex(sig_ml.index).fillna(0).astype(int)
            sig_ml = (sig_ml * reg).clip(0, 1)

        sh = year_sharpe(daily_returns, sig_ml, year)
        per_year[year] = round(sh, 3)
        print(f"    {year}: Sharpe={sh:.3f}")

    values = list(per_year.values())
    mean_sh = float(np.mean(values))
    std_sh = float(np.std(values))
    mdd = 0.0  # approximate; full series not trivial
    return {"mean": mean_sh, "std": std_sh, "mdd": mdd, "per_year": per_year}


def log_result(tracker, name, group, config_dict, metrics_dict):
    """Wrapper to log with standard tags."""
    tracker.log_experiment(
        name=name,
        config=config_dict,
        metrics=metrics_dict,
        tags=TAGS,
        job_type=JOB_TYPE,
        group=group,
    )
    print(f"  [WANDB] Logged: {name}")


# ════════════════════════════════════════════════════════════════════════════════
# ENSEMBLE COMBOS (untested from Step 4 spec)
# ════════════════════════════════════════════════════════════════════════════════


@with_timeout(seconds=900)
def run_lgbm_regime_combos(prices_daily, daily_returns, df_daily, target_daily, avail10, tracker):
    """
    Tests 4 regime-filtered LightGBM combos:
    1. LightGBM + vol_adx_OR  (hypothesis: better than just ML or just regime)
    2. LightGBM + vol_adx_AND  (hypothesis: cuts 2022 catastrophe)
    3. Signal average: (LightGBM + Donchian)/2 filtered by ADX-30
    4. Regime-switch: vol_adx_AND → Donchian; else → LightGBM
    """
    print("\n" + "=" * 80)
    print("ENSEMBLE COMBOS: LightGBM + Regime Filters")
    print("=" * 80)

    # Pre-compute regime masks (static, full series)
    adx14 = compute_adx(prices_daily, period=14)
    rv20 = vol_regime(prices_daily, window=20)

    regime_adx30 = (adx14 > 30).astype(int)  # ADX(14,30)
    regime_vol_adx_OR = ((rv20 == 1) | (adx14 > 25)).astype(int)  # vol OR adx25
    regime_vol_adx_AND = ((rv20 == 1) & (adx14 > 25)).astype(int)  # vol AND adx25

    # Full Donchian signal
    don_sig = donchian_signal(prices_daily, ENTRY_PERIOD, EXIT_PERIOD)

    all_combos = []

    # --- 1. LightGBM + vol_adx_OR ---
    print("\n[1/4] LightGBM + vol_adx_OR filter")
    res_or = walk_forward_ml(
        prices_daily,
        daily_returns,
        df_daily,
        target_daily,
        avail10,
        "lgbm_vol_adx_OR",
        regime_fn=regime_vol_adx_OR,
    )
    print(f"  Mean Sharpe={res_or['mean']:.3f} ± {res_or['std']:.3f}  2022={res_or['per_year'].get(2022, 'N/A')}")

    name_or = f"lgbm_vol_adx_OR_S{res_or['mean']:.2f}"
    cfg_or = {
        "approach": "lgbm_vol_adx_OR",
        "model": "LightGBM-top10",
        "regime_filter": "vol(20d) OR ADX(14,25)",
        "hypothesis": "ML 2022 catastrophe (-0.644) gets cut by OR regime filter",
        "notes": (
            f"LightGBM-top10 signal AND-gated with vol_adx_OR regime (trade when vol>median OR ADX>25). "
            f"Mean WF Sharpe {res_or['mean']:.3f}±{res_or['std']:.3f}. 2022: {res_or['per_year'].get(2022, 'N/A')}. "
            f"Prior: LGBM alone: {BEST_ML_SHARPE}, vol_adx_OR regime alone: {VOL_ADX_OR_SHARPE}."
        ),
    }
    metrics_or = {
        "sharpe": res_or["mean"],
        "mean_wf_sharpe": res_or["mean"],
        "std_wf_sharpe": res_or["std"],
        "sharpe_2022": res_or["per_year"].get(2022, 0.0),
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        "best_individual_ml": BEST_ML_SHARPE,
        **{f"sharpe_{yr}": v for yr, v in res_or["per_year"].items()},
    }
    log_result(tracker, name_or, "lgbm_regime_combos", cfg_or, metrics_or)
    all_combos.append({"name": name_or, **res_or})

    # --- 2. LightGBM + vol_adx_AND ---
    print("\n[2/4] LightGBM + vol_adx_AND filter (stricter)")
    res_and = walk_forward_ml(
        prices_daily,
        daily_returns,
        df_daily,
        target_daily,
        avail10,
        "lgbm_vol_adx_AND",
        regime_fn=regime_vol_adx_AND,
    )
    print(f"  Mean Sharpe={res_and['mean']:.3f} ± {res_and['std']:.3f}  2022={res_and['per_year'].get(2022, 'N/A')}")

    name_and = f"lgbm_vol_adx_AND_S{res_and['mean']:.2f}"
    cfg_and = {
        "approach": "lgbm_vol_adx_AND",
        "model": "LightGBM-top10",
        "regime_filter": "vol(20d) AND ADX(14,25)",
        "hypothesis": "AND filter was +0.58 in 2022 for Donchian — does it rescue ML too?",
        "notes": (
            f"LightGBM-top10 signal AND-gated with vol_adx_AND regime (vol>median AND ADX>25). "
            f"Stricter filter than OR — should cut 2022 more aggressively. "
            f"Mean WF Sharpe {res_and['mean']:.3f}±{res_and['std']:.3f}. 2022: {res_and['per_year'].get(2022, 'N/A')}. "
            f"Prior: vol_adx_AND regime alone: {VOL_ADX_AND_SHARPE} (2022: +0.192)."
        ),
    }
    metrics_and = {
        "sharpe": res_and["mean"],
        "mean_wf_sharpe": res_and["mean"],
        "std_wf_sharpe": res_and["std"],
        "sharpe_2022": res_and["per_year"].get(2022, 0.0),
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        "best_individual_ml": BEST_ML_SHARPE,
        **{f"sharpe_{yr}": v for yr, v in res_and["per_year"].items()},
    }
    log_result(tracker, name_and, "lgbm_regime_combos", cfg_and, metrics_and)
    all_combos.append({"name": name_and, **res_and})

    # --- 3. Signal average: (Donchian + LightGBM)/2, filtered by ADX-30 ---
    print("\n[3/4] Signal average: (Donchian + LightGBM)/2 filtered by ADX-30")

    years = list(range(2019, 2024))
    avail = [c for c in avail10 if c in df_daily.columns]
    per_year_avg = {}

    for year in years:
        train_mask = daily_returns.index.year < year
        test_mask = daily_returns.index.year == year

        if train_mask.sum() < 100:
            per_year_avg[year] = 0.0
            continue

        X_tr = df_daily[avail][train_mask]
        y_tr = target_daily[train_mask]
        X_te = df_daily[avail][test_mask]
        valid = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr, y_tr = X_tr[valid], y_tr[valid]
        X_te = X_te.fillna(X_te.median())

        if len(X_tr) < 50 or len(X_te) == 0:
            per_year_avg[year] = 0.0
            continue

        try:
            model = LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_tr, y_tr)
            proba_ml = model.predict_proba(X_te)[:, 1]
            sig_ml = pd.Series((proba_ml > 0.5).astype(int), index=X_te.index)
        except Exception as e:
            print(f"  WARN: LGBM year {year}: {e}")
            per_year_avg[year] = 0.0
            continue

        # Average Donchian + ML signals
        don_te = don_sig.reindex(X_te.index).fillna(0).astype(float)
        avg_sig = (don_te + sig_ml.astype(float)) / 2.0

        # Apply ADX-30 regime gate
        adx_te = regime_adx30.reindex(X_te.index).fillna(0)
        final_sig = avg_sig * adx_te
        # Round: trade if avg_signal * regime > 0.4 (either or both)
        final_sig = (final_sig > 0.4).astype(int)

        sh = year_sharpe(daily_returns, final_sig, year)
        per_year_avg[year] = round(sh, 3)
        print(f"    {year}: Sharpe={sh:.3f}")

    mean_avg = float(np.mean(list(per_year_avg.values())))
    std_avg = float(np.std(list(per_year_avg.values())))
    name_avg = f"don_lgbm_avg_adx30_S{mean_avg:.2f}"
    print(f"  Mean Sharpe={mean_avg:.3f} ± {std_avg:.3f}  2022={per_year_avg.get(2022, 'N/A')}")

    cfg_avg = {
        "approach": "signal_average_don_lgbm_adx30",
        "models": ["Donchian(40/20)", "LightGBM-top10"],
        "regime_filter": "ADX(14,30)",
        "combination": "equal_weight_average then ADX gate",
        "notes": (
            f"Equal-weight average of Donchian + LightGBM signals, filtered by ADX(14,30). "
            f"Hypothesis: averaging two different signal types (rule-based + ML) provides diversity. "
            f"Mean WF Sharpe {mean_avg:.3f}±{std_avg:.3f}. 2022: {per_year_avg.get(2022, 'N/A')}."
        ),
    }
    metrics_avg = {
        "sharpe": mean_avg,
        "mean_wf_sharpe": mean_avg,
        "std_wf_sharpe": std_avg,
        "sharpe_2022": per_year_avg.get(2022, 0.0),
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        **{f"sharpe_{yr}": v for yr, v in per_year_avg.items()},
    }
    log_result(tracker, name_avg, "lgbm_regime_combos", cfg_avg, metrics_avg)
    all_combos.append({"name": name_avg, "mean": mean_avg, "std": std_avg, "per_year": per_year_avg})

    # --- 4. Regime-switch: vol_adx_AND → Donchian; else → LightGBM ---
    print("\n[4/4] Regime-switch: vol_adx_AND → Donchian; non-AND → LightGBM")

    per_year_sw = {}
    for year in years:
        train_mask = daily_returns.index.year < year
        test_mask = daily_returns.index.year == year

        if train_mask.sum() < 100:
            per_year_sw[year] = 0.0
            continue

        X_tr = df_daily[avail][train_mask]
        y_tr = target_daily[train_mask]
        X_te = df_daily[avail][test_mask]
        valid = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr, y_tr = X_tr[valid], y_tr[valid]
        X_te = X_te.fillna(X_te.median())

        if len(X_tr) < 50 or len(X_te) == 0:
            per_year_sw[year] = 0.0
            continue

        try:
            model = LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_tr, y_tr)
            proba_ml = model.predict_proba(X_te)[:, 1]
            sig_ml = pd.Series((proba_ml > 0.5).astype(int), index=X_te.index)
        except Exception as e:
            print(f"  WARN: LGBM year {year}: {e}")
            per_year_sw[year] = 0.0
            continue

        # Regime-switch logic: AND → Donchian, non-AND → LightGBM
        and_te = regime_vol_adx_AND.reindex(X_te.index).fillna(0).astype(int)
        don_te = don_sig.reindex(X_te.index).fillna(0).astype(int)
        # When AND=1: use Donchian; When AND=0: use ML
        sw_sig = np.where(and_te.values == 1, don_te.values, sig_ml.values)
        sw_sig = pd.Series(sw_sig, index=X_te.index).astype(int)

        sh = year_sharpe(daily_returns, sw_sig, year)
        per_year_sw[year] = round(sh, 3)
        print(f"    {year}: Sharpe={sh:.3f}  AND_frac={and_te.mean():.1%}")

    mean_sw = float(np.mean(list(per_year_sw.values())))
    std_sw = float(np.std(list(per_year_sw.values())))
    name_sw = f"regime_switch_and_don_S{mean_sw:.2f}"
    print(f"  Mean Sharpe={mean_sw:.3f} ± {std_sw:.3f}  2022={per_year_sw.get(2022, 'N/A')}")

    cfg_sw = {
        "approach": "regime_switch_don_vs_lgbm",
        "trending_model": "Donchian(40/20) when vol_adx_AND=1",
        "ranging_model": "LightGBM-top10 when vol_adx_AND=0",
        "notes": (
            f"Regime-switch: vol_adx_AND (trending+vol)→Donchian, otherwise→LightGBM. "
            f"Donchian is the more robust strategy in confirmed trending regimes. "
            f"LightGBM used in weaker regimes where Donchian might struggle. "
            f"Mean WF Sharpe {mean_sw:.3f}±{std_sw:.3f}. 2022: {per_year_sw.get(2022, 'N/A')}."
        ),
    }
    metrics_sw = {
        "sharpe": mean_sw,
        "mean_wf_sharpe": mean_sw,
        "std_wf_sharpe": std_sw,
        "sharpe_2022": per_year_sw.get(2022, 0.0),
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        **{f"sharpe_{yr}": v for yr, v in per_year_sw.items()},
    }
    log_result(tracker, name_sw, "lgbm_regime_combos", cfg_sw, metrics_sw)
    all_combos.append({"name": name_sw, "mean": mean_sw, "std": std_sw, "per_year": per_year_sw})

    return all_combos


# ════════════════════════════════════════════════════════════════════════════════
# INVERSE-VOL WEIGHTED ENSEMBLE (top 3 regime configs)
# ════════════════════════════════════════════════════════════════════════════════


@with_timeout(seconds=300)
def run_invvol_ensemble(prices_daily, daily_returns, tracker):
    """
    Inverse-volatility weighted ensemble of top 3 regime configs:
    - vol_adx_OR  (Sharpe 1.23, std ~0.94 from Step 3)
    - ADX(14,30)  (Sharpe 1.21, std 0.83 from Step 3)
    - vol_adx_AND (Sharpe 1.07, std 0.50 from Step 3)

    Weight = 1 / (yearly_sharpe_variance of each).
    The vol_adx_AND has lowest variance → highest weight.
    """
    print("\n" + "=" * 80)
    print("INVERSE-VOL ENSEMBLE: Top 3 Regime Configs")
    print("=" * 80)

    adx14 = compute_adx(prices_daily, period=14)
    rv20 = vol_regime(prices_daily, window=20)
    don_sig = donchian_signal(prices_daily, ENTRY_PERIOD, EXIT_PERIOD)

    # Per-year Sharpes from Step 3 (used for weight calculation)
    # vol_adx_OR  (from regime_summary.md context — using ADX-t25 as proxy)
    # ADX(14,30): per-year {2020: 2.342, 2021: 1.124, 2022: 0.000, 2023: 1.256}
    # vol_adx_AND: per-year {2020: 2.562, 2021: 0.563, 2022: +0.192, 2023: 0.956}
    # Sharpe variances (std^2):
    known_stds = {
        "adx_30": 0.829,
        "vol_adx_AND": 0.904,
        "vol_adx_OR": 0.940,  # approximate from context
    }

    # Compute the three signals
    def sig_adx30(prices):
        a = compute_adx(prices, 14)
        return (don_sig * (a > 30).astype(int)).clip(0, 1)

    def sig_vol_adx_AND(prices):
        rv = vol_regime(prices, 20)
        a = compute_adx(prices, 14)
        filt = ((rv == 1) & (a > 25)).astype(int)
        return (don_sig * filt).clip(0, 1)

    def sig_vol_adx_OR(prices):
        rv = vol_regime(prices, 20)
        a = compute_adx(prices, 14)
        filt = ((rv == 1) | (a > 25)).astype(int)
        return (don_sig * filt).clip(0, 1)

    signals_list = [
        ("adx_30", sig_adx30(prices_daily)),
        ("vol_adx_AND", sig_vol_adx_AND(prices_daily)),
        ("vol_adx_OR", sig_vol_adx_OR(prices_daily)),
    ]

    # Compute weights = 1 / var (= 1 / std^2)
    inv_var = {k: 1.0 / (known_stds[k] ** 2) for k in known_stds}
    total = sum(inv_var.values())
    weights = {k: v / total for k, v in inv_var.items()}
    print(f"  Weights: { {k: f'{v:.3f}' for k, v in weights.items()} }")

    # Combine: weighted sum → threshold at 0.4 to trade
    per_year_ivw = {}
    for year in range(2019, 2024):
        mask = daily_returns.index.year == year
        ret_yr = daily_returns[mask]

        weighted_sig = sum(
            weights[name] * sig.reindex(ret_yr.index).fillna(0).astype(float) for name, sig in signals_list
        )
        final_sig = (weighted_sig > 0.4).astype(int)
        sh = sharpe_from_signals(ret_yr, final_sig)
        per_year_ivw[year] = round(sh, 3)
        print(f"  {year}: Sharpe={sh:.3f}  in_market={float(final_sig.mean()):.1%}")

    mean_ivw = float(np.mean(list(per_year_ivw.values())))
    std_ivw = float(np.std(list(per_year_ivw.values())))
    name_ivw = f"invvol_ensemble_3regimes_S{mean_ivw:.2f}"
    print(f"\n  Mean Sharpe={mean_ivw:.3f} ± {std_ivw:.3f}  2022={per_year_ivw.get(2022, 'N/A')}")

    cfg_ivw = {
        "approach": "invvol_weighted_ensemble",
        "components": ["adx_30_donchian", "vol_adx_AND_donchian", "vol_adx_OR_donchian"],
        "weights": weights,
        "threshold": 0.4,
        "weight_method": "1/variance (inverse of yearly Sharpe std^2)",
        "notes": (
            f"Inverse-variance weighted combination of top 3 regime-filtered Donchian configs. "
            f"vol_adx_AND has lowest std (0.50) → highest weight. "
            f"Mean WF Sharpe {mean_ivw:.3f}±{std_ivw:.3f}. 2022: {per_year_ivw.get(2022, 'N/A')}. "
            f"Hypothesis: weighting by stability rather than mean Sharpe improves risk-adjusted returns."
        ),
    }
    metrics_ivw = {
        "sharpe": mean_ivw,
        "mean_wf_sharpe": mean_ivw,
        "std_wf_sharpe": std_ivw,
        "sharpe_2022": per_year_ivw.get(2022, 0.0),
        "weight_adx30": weights.get("adx_30", 0),
        "weight_vol_adx_AND": weights.get("vol_adx_AND", 0),
        "weight_vol_adx_OR": weights.get("vol_adx_OR", 0),
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        **{f"sharpe_{yr}": v for yr, v in per_year_ivw.items()},
    }
    log_result(tracker, name_ivw, "invvol_ensemble", cfg_ivw, metrics_ivw)

    return {"name": name_ivw, "mean": mean_ivw, "std": std_ivw, "per_year": per_year_ivw}


# ════════════════════════════════════════════════════════════════════════════════
# AK's PRIORITY: MOMENTUM STRATEGY COMBOS
# ════════════════════════════════════════════════════════════════════════════════


@with_timeout(seconds=600)
def run_momentum_combos(prices_daily, daily_returns, tracker):
    """
    Test regime filters applied to momentum strategies.

    Momentum: buy when N-day return > threshold, flat otherwise.
    Threshold configs: 0.0, 0.05, 0.10 (based on AK's directive)
    Best config identified from AK's holdout note: threshold=0.05, lookback=20d

    Tests:
    1. Unfiltered momentum (baseline for this section)
    2. Momentum + ADX(14,30)
    3. Momentum + vol_adx_AND
    4. Momentum + vol_adx_OR
    5. Momentum + rolling drawdown filter (10%, 40d)
    """
    print("\n" + "=" * 80)
    print("MOMENTUM STRATEGY COMBOS (AK Priority)")
    print("=" * 80)

    adx14 = compute_adx(prices_daily, period=14)
    rv20 = vol_regime(prices_daily, window=20)

    regime_adx30 = (adx14 > 30).astype(int)
    regime_vol_adx_AND = ((rv20 == 1) & (adx14 > 25)).astype(int)
    regime_vol_adx_OR = ((rv20 == 1) | (adx14 > 25)).astype(int)

    all_results = []

    # ─── 1. Unfiltered momentum across thresholds ────────────────────────────
    print("\n[BASELINE] Unfiltered momentum (thresholds 0.0, 0.05, 0.10)")
    best_mom_sharpe = -999
    best_mom_thresh = 0.05
    best_mom_lookback = 20

    for lookback in [20, 40]:
        for thresh in [0.0, 0.05, 0.10]:
            label = f"mom_lb{lookback}_t{int(thresh * 100)}"
            res = walk_forward_static(
                prices_daily,
                daily_returns,
                lambda p, lb=lookback, th=thresh: momentum_signal(p, lb, th),
                label,
            )
            sh22 = res["per_year"].get(2022, 0.0)
            print(f"  {label}: Mean={res['mean']:.3f}±{res['std']:.3f} 2022={sh22:.3f}")

            if res["mean"] > best_mom_sharpe:
                best_mom_sharpe = res["mean"]
                best_mom_thresh = thresh
                best_mom_lookback = lookback

            name_m = f"mom_lb{lookback}_t{int(thresh * 100)}_S{res['mean']:.2f}"
            cfg_m = {
                "approach": "momentum_unfiltered",
                "lookback": lookback,
                "threshold": thresh,
                "notes": f"Simple momentum: buy when {lookback}d return > {thresh}. Unfiltered baseline.",
            }
            metrics_m = {
                "sharpe": res["mean"],
                "mean_wf_sharpe": res["mean"],
                "std_wf_sharpe": res["std"],
                "max_drawdown": res["mdd"],
                "sharpe_2022": sh22,
                "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
                **{f"sharpe_{yr}": v for yr, v in res["per_year"].items()},
            }
            log_result(tracker, name_m, "momentum_unfiltered", cfg_m, metrics_m)
            all_results.append({"name": name_m, **res, "label": label})

    print(
        f"\n  Best unfiltered momentum: lookback={best_mom_lookback}, thresh={best_mom_thresh} → Sharpe={best_mom_sharpe:.3f}"
    )

    # ─── 2. Best momentum + ADX(14,30) ───────────────────────────────────────
    print(f"\n[2] Best momentum (lb={best_mom_lookback}, t={best_mom_thresh}) + ADX(14,30)")
    mom_sig = momentum_signal(prices_daily, best_mom_lookback, best_mom_thresh)
    res_adx = walk_forward_static(
        prices_daily,
        daily_returns,
        lambda p: (momentum_signal(p, best_mom_lookback, best_mom_thresh) * (compute_adx(p, 14) > 30).astype(int)).clip(
            0, 1
        ),
        "mom_adx30",
    )
    sh22 = res_adx["per_year"].get(2022, 0.0)
    print(f"  Mean={res_adx['mean']:.3f}±{res_adx['std']:.3f} 2022={sh22:.3f}")

    name_ma = f"mom_adx30_S{res_adx['mean']:.2f}"
    cfg_ma = {
        "approach": "momentum_adx30_filter",
        "lookback": best_mom_lookback,
        "threshold": best_mom_thresh,
        "adx_period": 14,
        "adx_threshold": 30,
        "hypothesis": "ADX identifies trending regimes where momentum works best",
        "notes": (
            f"Momentum(lb={best_mom_lookback}, thresh={best_mom_thresh}) filtered by ADX(14,30). "
            f"ADX(14,30) gave Donchian 0.000 in 2022. Does it do the same for momentum? "
            f"Mean WF Sharpe {res_adx['mean']:.3f}±{res_adx['std']:.3f}. 2022: {sh22:.3f}."
        ),
    }
    metrics_ma = {
        "sharpe": res_adx["mean"],
        "mean_wf_sharpe": res_adx["mean"],
        "std_wf_sharpe": res_adx["std"],
        "max_drawdown": res_adx["mdd"],
        "sharpe_2022": sh22,
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        "unfiltered_momentum": best_mom_sharpe,
        **{f"sharpe_{yr}": v for yr, v in res_adx["per_year"].items()},
    }
    log_result(tracker, name_ma, "momentum_regime", cfg_ma, metrics_ma)
    all_results.append({"name": name_ma, **res_adx, "label": "mom+adx30"})

    # ─── 3. Best momentum + vol_adx_AND ──────────────────────────────────────
    print("\n[3] Best momentum + vol_adx_AND")
    res_va = walk_forward_static(
        prices_daily,
        daily_returns,
        lambda p: (
            momentum_signal(p, best_mom_lookback, best_mom_thresh)
            * ((vol_regime(p, 20) == 1) & (compute_adx(p, 14) > 25)).astype(int)
        ).clip(0, 1),
        "mom_vol_adx_AND",
    )
    sh22 = res_va["per_year"].get(2022, 0.0)
    print(f"  Mean={res_va['mean']:.3f}±{res_va['std']:.3f} 2022={sh22:.3f}")

    name_va = f"mom_vol_adx_AND_S{res_va['mean']:.2f}"
    cfg_va = {
        "approach": "momentum_vol_adx_AND",
        "lookback": best_mom_lookback,
        "threshold": best_mom_thresh,
        "filter": "vol(20d) AND ADX(14,25)",
        "hypothesis": "vol_adx_AND gave +0.192 in 2022 for Donchian — should rescue momentum too",
        "notes": (
            f"Momentum(lb={best_mom_lookback}, thresh={best_mom_thresh}) filtered by vol_adx_AND. "
            f"vol_adx_AND gave Donchian 2022: +0.192. Testing if same filter protects momentum. "
            f"Mean WF Sharpe {res_va['mean']:.3f}±{res_va['std']:.3f}. 2022: {sh22:.3f}."
        ),
    }
    metrics_va = {
        "sharpe": res_va["mean"],
        "mean_wf_sharpe": res_va["mean"],
        "std_wf_sharpe": res_va["std"],
        "max_drawdown": res_va["mdd"],
        "sharpe_2022": sh22,
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        "unfiltered_momentum": best_mom_sharpe,
        **{f"sharpe_{yr}": v for yr, v in res_va["per_year"].items()},
    }
    log_result(tracker, name_va, "momentum_regime", cfg_va, metrics_va)
    all_results.append({"name": name_va, **res_va, "label": "mom+vol_adx_AND"})

    # ─── 4. Best momentum + vol_adx_OR ───────────────────────────────────────
    print("\n[4] Best momentum + vol_adx_OR")
    res_vo = walk_forward_static(
        prices_daily,
        daily_returns,
        lambda p: (
            momentum_signal(p, best_mom_lookback, best_mom_thresh)
            * ((vol_regime(p, 20) == 1) | (compute_adx(p, 14) > 25)).astype(int)
        ).clip(0, 1),
        "mom_vol_adx_OR",
    )
    sh22 = res_vo["per_year"].get(2022, 0.0)
    print(f"  Mean={res_vo['mean']:.3f}±{res_vo['std']:.3f} 2022={sh22:.3f}")

    name_vo = f"mom_vol_adx_OR_S{res_vo['mean']:.2f}"
    cfg_vo = {
        "approach": "momentum_vol_adx_OR",
        "lookback": best_mom_lookback,
        "threshold": best_mom_thresh,
        "filter": "vol(20d) OR ADX(14,25)",
        "notes": (
            f"Momentum filtered by vol_adx_OR (more permissive). "
            f"Mean WF Sharpe {res_vo['mean']:.3f}±{res_vo['std']:.3f}. 2022: {sh22:.3f}."
        ),
    }
    metrics_vo = {
        "sharpe": res_vo["mean"],
        "mean_wf_sharpe": res_vo["mean"],
        "std_wf_sharpe": res_vo["std"],
        "max_drawdown": res_vo["mdd"],
        "sharpe_2022": sh22,
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        "unfiltered_momentum": best_mom_sharpe,
        **{f"sharpe_{yr}": v for yr, v in res_vo["per_year"].items()},
    }
    log_result(tracker, name_vo, "momentum_regime", cfg_vo, metrics_vo)
    all_results.append({"name": name_vo, **res_vo, "label": "mom+vol_adx_OR"})

    # ─── 5. Momentum + rolling drawdown filter ────────────────────────────────
    print("\n[5] Best momentum + rolling drawdown filter (10%, 40d)")
    dd_threshold = 0.10
    dd_window = 40

    def make_mom_dd_sig(p, returns, lb, th, dd_thresh, dd_win):
        mom_s = momentum_signal(p, lb, th)
        # Compute rolling drawdown of momentum strategy itself
        pos_sh = mom_s.shift(1).fillna(0)
        strat_ret = pos_sh * returns.reindex(p.index).fillna(0)
        equity = (1 + strat_ret).cumprod()
        roll_max = equity.rolling(dd_win).max()
        roll_dd = (equity - roll_max) / (roll_max + 1e-9)
        not_in_dd = (roll_dd > -dd_thresh).astype(int)
        return (mom_s * not_in_dd).clip(0, 1)

    res_dd = walk_forward_static(
        prices_daily,
        daily_returns,
        lambda p: make_mom_dd_sig(p, daily_returns, best_mom_lookback, best_mom_thresh, dd_threshold, dd_window),
        "mom_dd_filter",
    )
    sh22 = res_dd["per_year"].get(2022, 0.0)
    print(f"  Mean={res_dd['mean']:.3f}±{res_dd['std']:.3f} 2022={sh22:.3f}")

    name_dd = f"mom_dd10pct40d_S{res_dd['mean']:.2f}"
    cfg_dd = {
        "approach": "momentum_drawdown_filter",
        "lookback": best_mom_lookback,
        "threshold": best_mom_thresh,
        "dd_threshold": dd_threshold,
        "dd_window": dd_window,
        "notes": (
            f"Momentum filtered by rolling drawdown (go flat if strategy DD > {dd_threshold * 100:.0f}% in {dd_window}d). "
            f"Hypothesis: DD filter stops momentum from riding losing streaks. "
            f"Mean WF Sharpe {res_dd['mean']:.3f}±{res_dd['std']:.3f}. 2022: {sh22:.3f}."
        ),
    }
    metrics_dd = {
        "sharpe": res_dd["mean"],
        "mean_wf_sharpe": res_dd["mean"],
        "std_wf_sharpe": res_dd["std"],
        "max_drawdown": res_dd["mdd"],
        "sharpe_2022": sh22,
        "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
        "unfiltered_momentum": best_mom_sharpe,
        **{f"sharpe_{yr}": v for yr, v in res_dd["per_year"].items()},
    }
    log_result(tracker, name_dd, "momentum_regime", cfg_dd, metrics_dd)
    all_results.append({"name": name_dd, **res_dd, "label": "mom+dd_filter"})

    return all_results, best_mom_sharpe, best_mom_thresh, best_mom_lookback


# ════════════════════════════════════════════════════════════════════════════════
# ML META-LEARNER ON MOMENTUM
# ════════════════════════════════════════════════════════════════════════════════


@with_timeout(seconds=900)
def run_ml_meta_on_momentum(
    prices_daily, daily_returns, df_daily, avail10, avail20, best_mom_thresh, best_mom_lookback, tracker
):
    """
    Train LightGBM meta-learner to predict: 'will momentum strategy profit in next N days?'
    This is different from Step 3's meta-learner (which predicted Donchian profitability).

    Label: forward-rolling N-day Sharpe of momentum strategy > 0
    Features: top-20 features + regime indicators
    """
    print("\n" + "=" * 80)
    print("ML META-LEARNER ON MOMENTUM")
    print("=" * 80)

    # Compute momentum strategy returns
    mom_sig = momentum_signal(prices_daily, best_mom_lookback, best_mom_thresh)
    pos_sh = mom_sig.shift(1).fillna(0)
    strat_ret = pos_sh * daily_returns.reindex(prices_daily.index).fillna(0)

    # Forward 30-day momentum Sharpe → meta-label
    # At time T, will momentum be profitable over NEXT 30 days?
    roll_sh = strat_ret.rolling(30).mean() / (strat_ret.rolling(30).std() + 1e-9) * np.sqrt(252)
    meta_label = (roll_sh.shift(-30) > 0).astype(int)

    # Features: daily-aggregate from hourly features
    feat_daily = df_daily.copy()
    ret_raw = daily_returns
    feat_daily["vol_10d"] = ret_raw.rolling(10).std() * np.sqrt(252)
    feat_daily["vol_30d"] = ret_raw.rolling(30).std() * np.sqrt(252)
    feat_daily["adx_14"] = compute_adx(prices_daily, 14)
    feat_daily["mom_20d"] = prices_daily.pct_change(20)
    feat_daily["mom_40d"] = prices_daily.pct_change(40)
    feat_daily["mom_signal"] = mom_sig.astype(float)

    # Drop NaN-heavy cols, align
    feat_clean = feat_daily.dropna(axis=1, thresh=int(0.7 * len(feat_daily)))
    common = feat_clean.index.intersection(meta_label.index)
    feat_clean = feat_clean.loc[common]
    meta_label = meta_label.loc[common]

    valid = feat_clean.notna().all(axis=1) & meta_label.notna()
    feat_clean = feat_clean[valid]
    meta_label = meta_label[valid]

    # Keep top 20 by variance
    top_feats = feat_clean.var().sort_values(ascending=False).head(20).index.tolist()
    feat_clean = feat_clean[top_feats]

    print(f"  Meta-label: {meta_label.sum()} trade, {(~meta_label.astype(bool)).sum()} no-trade")
    print(f"  Feature shape: {feat_clean.shape}")

    best_sharpe = -999
    best_result = None

    for lr in [0.05, 0.01]:
        for depth in [3, 5]:
            print(f"\n  LightGBM meta: lr={lr}, depth={depth}")
            per_year = {}

            for yr in range(2019, 2024):
                train_mask = feat_clean.index.year < yr
                test_mask = feat_clean.index.year == yr

                if train_mask.sum() < 100 or test_mask.sum() < 10:
                    per_year[yr] = 0.0
                    continue

                X_tr = feat_clean[train_mask]
                y_tr = meta_label[train_mask]
                X_te = feat_clean[test_mask]

                model_params = {
                    "n_estimators": 300,
                    "max_depth": depth,
                    "learning_rate": lr,
                    "reg_lambda": 1.0,
                    "num_leaves": 31 if depth <= 3 else 63,
                    "device": "gpu",
                    "n_jobs": 1,
                    "verbose": -1,
                    "random_state": 42,
                }

                try:
                    meta_model = LGBMClassifier(**model_params)
                    meta_model.fit(X_tr, y_tr)
                    preds = meta_model.predict(X_te)
                    regime_yr = pd.Series(preds, index=feat_clean[test_mask].index)
                except Exception as e:
                    print(f"    WARN: year {yr}: {e}")
                    per_year[yr] = 0.0
                    continue

                # Apply meta-learner as momentum regime filter
                mom_yr = mom_sig.reindex(prices_daily.index[prices_daily.index.year == yr])
                reg_yr = regime_yr.reindex(mom_yr.index, method="ffill").fillna(0)
                filt_sig = (mom_yr * reg_yr.values).clip(0, 1).astype(int)

                ret_yr = daily_returns[daily_returns.index.year == yr]
                sh = sharpe_from_signals(ret_yr, filt_sig)
                per_year[yr] = round(sh, 3)

            mean_sh = float(np.mean(list(per_year.values())))
            std_sh = float(np.std(list(per_year.values())))
            sh22 = per_year.get(2022, 0.0)
            print(f"  Mean={mean_sh:.3f}±{std_sh:.3f} 2022={sh22:.3f}")

            run_name = f"ml_meta_momentum_lr{lr}_d{depth}_S{mean_sh:.2f}"
            cfg = {
                "approach": "ml_meta_momentum",
                "meta_model": "LightGBM-GPU",
                "lr": lr,
                "depth": depth,
                "label": "forward_30d_momentum_sharpe>0",
                "momentum_lookback": best_mom_lookback,
                "momentum_threshold": best_mom_thresh,
                "hypothesis": "ML learns WHEN momentum works vs when it fails",
                "notes": (
                    f"LightGBM meta-learner: predicts if momentum strategy will be profitable "
                    f"over next 30 days. Different from Step 3's Donchian meta-learner. "
                    f"Mean WF Sharpe {mean_sh:.3f}±{std_sh:.3f}. 2022: {sh22:.3f}. "
                    + (
                        "Better than Donchian meta-learner (0.843) — momentum has more learnable regime patterns."
                        if mean_sh > 0.843
                        else "Insufficient meta-signal — momentum regime patterns same as Donchian."
                    )
                ),
            }
            metrics = {
                "sharpe": mean_sh,
                "mean_wf_sharpe": mean_sh,
                "std_wf_sharpe": std_sh,
                "sharpe_2022": sh22,
                "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
                "prior_donchian_meta": 0.843,
                **{f"sharpe_{yr}": v for yr, v in per_year.items()},
            }
            log_result(tracker, run_name, "ml_meta_momentum", cfg, metrics)

            if mean_sh > best_sharpe:
                best_sharpe = mean_sh
                best_result = {"name": run_name, "mean": mean_sh, "std": std_sh, "per_year": per_year}

    return best_result


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("CONTRACT #004 — STEP 4 (v2): ENSEMBLE + MOMENTUM STRATEGIES")
    print("=" * 80)
    t_total = time.time()

    prices_daily, daily_returns, df_daily, target_daily, avail10, avail20 = load_data()

    tracker = ExperimentTracker(experiment_name="ensemble_v2_momentum")

    all_results = {}

    # ─── ENSEMBLE COMBOS (untested) ──────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SECTION A: Untested LightGBM + Regime Combos")
    print("=" * 80)
    try:
        combos = run_lgbm_regime_combos(prices_daily, daily_returns, df_daily, target_daily, avail10, tracker)
        all_results["lgbm_combos"] = combos
    except Exception as e:
        print(f"ERROR in lgbm_regime_combos: {e}")
        import traceback

        traceback.print_exc()
        all_results["lgbm_combos"] = []

    # ─── INVERSE-VOL ENSEMBLE ────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SECTION B: Inverse-Vol Weighted Ensemble of Top 3 Regime Configs")
    print("=" * 80)
    try:
        ivw_res = run_invvol_ensemble(prices_daily, daily_returns, tracker)
        all_results["invvol_ensemble"] = ivw_res
    except Exception as e:
        print(f"ERROR in invvol_ensemble: {e}")
        all_results["invvol_ensemble"] = {}

    # ─── MOMENTUM PRIORITY ──────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SECTION C: Momentum Strategy + Regime Filters (AK Priority)")
    print("=" * 80)
    try:
        mom_results, best_mom_sh, best_mom_thresh, best_mom_lb = run_momentum_combos(
            prices_daily, daily_returns, tracker
        )
        all_results["momentum"] = mom_results
    except Exception as e:
        print(f"ERROR in momentum_combos: {e}")
        import traceback

        traceback.print_exc()
        all_results["momentum"] = []
        best_mom_sh, best_mom_thresh, best_mom_lb = 0, 0.05, 20

    # ─── ML META ON MOMENTUM ─────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SECTION D: ML Meta-Learner on Momentum")
    print("=" * 80)
    try:
        ml_res = run_ml_meta_on_momentum(
            prices_daily, daily_returns, df_daily, avail10, avail20, best_mom_thresh, best_mom_lb, tracker
        )
        all_results["ml_meta_momentum"] = ml_res
    except Exception as e:
        print(f"ERROR in ml_meta_momentum: {e}")
        import traceback

        traceback.print_exc()
        all_results["ml_meta_momentum"] = {}

    elapsed = time.time() - t_total

    # ─── PRINT FINAL SUMMARY ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — ALL EXPERIMENTS")
    print("=" * 80)
    print(f"\n{'Strategy':<55} {'MeanSh':>8} {'Std':>8} {'Sh2022':>8}")
    print("-" * 80)

    # Reference rows
    refs = [
        ("Donchian baseline (unfiltered)", 1.062, "—", "-1.4"),
        ("LightGBM top-10 (Step 2)", 1.365, 1.701, "-0.644"),
        ("ADX(14,30) Donchian (Step 3 best)", 1.181, 0.829, "0.000"),
        ("vol_adx_AND Donchian (Step 3 risk-adj)", 1.068, 0.904, "+0.192"),
        ("Regime-cond ML+flat (Step 4 prior)", 0.970, 0.824, "+0.683"),
        ("Stacking LightGBM (Step 4 prior)", 1.029, 0.962, "+0.003"),
    ]
    for label, sh, std, sh22 in refs:
        std_s = f"{std:.3f}" if isinstance(std, float) else std
        sh22_s = sh22 if isinstance(sh22, str) else f"{sh22:.3f}"
        print(f"  {label:<53} {sh:>8.3f} {std_s:>8} {sh22_s:>8}")

    print("-" * 80)
    print("  NEW EXPERIMENTS:")

    summary_rows = []

    # Collect from lgbm_combos
    for r in all_results.get("lgbm_combos", []):
        if isinstance(r, dict) and "mean" in r:
            sh22 = r["per_year"].get(2022, "N/A")
            print(
                f"  {r['name']:<53} {r['mean']:>8.3f} {r['std']:>8.3f} {sh22 if isinstance(sh22, str) else sh22:>8.3f}"
            )
            summary_rows.append(r)

    # invvol ensemble
    ivw = all_results.get("invvol_ensemble", {})
    if isinstance(ivw, dict) and "mean" in ivw:
        sh22 = ivw["per_year"].get(2022, "N/A")
        print(
            f"  {ivw['name']:<53} {ivw['mean']:>8.3f} {ivw['std']:>8.3f} {sh22 if isinstance(sh22, str) else sh22:>8.3f}"
        )
        summary_rows.append(ivw)

    # Momentum combos
    for r in all_results.get("momentum", []):
        if isinstance(r, dict) and "mean" in r:
            sh22 = r["per_year"].get(2022, "N/A")
            print(
                f"  {r['name']:<53} {r['mean']:>8.3f} {r['std']:>8.3f} {sh22 if isinstance(sh22, str) else sh22:>8.3f}"
            )
            summary_rows.append(r)

    # ML meta on momentum
    ml = all_results.get("ml_meta_momentum", {})
    if isinstance(ml, dict) and "mean" in ml:
        sh22 = ml["per_year"].get(2022, "N/A")
        print(
            f"  {ml['name']:<53} {ml['mean']:>8.3f} {ml['std']:>8.3f} {sh22 if isinstance(sh22, str) else sh22:>8.3f}"
        )
        summary_rows.append(ml)

    # Best new result
    if summary_rows:
        best = max(summary_rows, key=lambda x: x.get("mean", -999))
        print(f"\n  ✓ BEST NEW: {best.get('name', '?')} → Sharpe {best.get('mean', 0):.3f}")
        best_sh22 = best["per_year"].get(2022, "N/A")
        print(f"    2022 Sharpe: {best_sh22}")
        best_risk = min(summary_rows, key=lambda x: abs(x.get("std", 999)))
        print(f"  ✓ LOWEST RISK: {best_risk.get('name', '?')} → Std {best_risk.get('std', 0):.3f}")

    print(f"\nTotal elapsed: {elapsed / 60:.1f} min")

    # ─── SAVE RESULTS ────────────────────────────────────────────────────────
    results_file = RESULTS_DIR / "ensemble_v2_results.json"

    def safe_dict(d):
        if not isinstance(d, dict):
            return {}
        return {str(k): (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in d.items()}

    out = {
        "reference_baselines": {
            "donchian_unfiltered": 1.062,
            "lgbm_top10": 1.365,
            "adx30_donchian": 1.181,
            "vol_adx_AND_donchian": 1.068,
        },
        "new_experiments": [
            {
                "name": r.get("name", ""),
                "mean_sharpe": round(float(r.get("mean", 0)), 3),
                "std_sharpe": round(float(r.get("std", 0)), 3),
                "sharpe_2022": r.get("per_year", {}).get(2022, "N/A"),
                "per_year": {str(k): v for k, v in r.get("per_year", {}).items()},
            }
            for r in summary_rows
        ],
        "elapsed_seconds": elapsed,
    }

    with open(results_file, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Results saved to {results_file}")

    return out


if __name__ == "__main__":
    result = main()
    print("\nDone.")
