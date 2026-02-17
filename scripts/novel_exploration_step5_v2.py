#!/usr/bin/env python3
"""
CONTRACT #004 — Step 5 (v2): Novel Exploration — Deep Dive

GROUND TRUTH FROM 100+ EXPERIMENTS:
  - BEST OVERALL: regime_cond ADX(25/15) Sharpe 1.326 (momentum→Donchian→flat 3-zone)
  - mom_vol_adx_OR: Sharpe 1.279, std 1.465, 2022=-1.317
  - breakout_profitability_cat: Sharpe 1.243, std 1.177, 2022=-0.807
  - majority_vote_2of3: Sharpe 1.197, std 0.828, 2022=-0.277
  - ADX(14,30) Donchian: Sharpe 1.181, std 0.829, 2022=0.000 (BEST 2022 PROTECTION)
  - ADX(21,30) mom+OR: Sharpe 1.135, std 0.846, 2022=-0.260 (smoother regime)
  - stability_opt (triple gate): Sharpe 1.032, std 0.465, 2022=+0.583 (MOST STABLE)
  - Donchian baseline: Sharpe 1.062

THIS SCRIPT PRIORITIES:
  P1: Momentum+regime deep sweep — find robust optimal region (not just a peak point)
      Key insight: ADX(21) >> ADX(14) for std reduction.
  P2: Target re-engineering — XGB and LGBM for profitability prediction + regime features
  P3: Stability optimization — improve stability_opt while maintaining Sharpe > 1.0
  P4: Robustness testing — perturb top 3 configs +/-20%, stress-test train windows

Target: 15+ wandb runs tagged ['contract_004', 'novel']
"""
import sys
sys.path.insert(0, "src")

import os
os.environ["PYTHONUNBUFFERED"] = "1"

import time
import warnings
import json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sparky.data.loader import load
from sparky.tracking.experiment import ExperimentTracker
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.oversight.timeout import with_timeout

# ─── Constants ─────────────────────────────────────────────────────────────────
BASELINE_SHARPE     = 1.062   # Donchian Multi-TF baseline
BEST_REGIME_SHARPE  = 1.181   # ADX(14,30) Donchian, Step 3
BEST_NOVEL_SHARPE   = 1.326   # regime_cond ADX(25/15), Step 4v3
BEST_STABLE_SHARPE  = 1.032   # stability_opt triple gate

ENTRY_PERIOD = 40
EXIT_PERIOD  = 20

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAGS     = ["contract_004", "novel"]
JOB_TYPE = "novel"


# ─── Data Loading ───────────────────────────────────────────────────────────────
def load_data():
    print("=" * 80)
    print("LOADING DATA (once, cached)")
    print("=" * 80)

    df_feat = load("feature_matrix_btc_hourly_expanded", purpose="training")

    prices_hourly = pd.read_parquet("data/raw/btc/ohlcv_hourly_max_coverage.parquet")
    prices_daily = prices_hourly["close"].resample("D").last().dropna()
    if prices_daily.index.tz is None:
        prices_daily.index = prices_daily.index.tz_localize("UTC")
    prices_daily = prices_daily.loc["2019-01-01":"2023-12-31"]
    daily_returns = prices_daily.pct_change().dropna()

    # Daily-aggregate features
    df_daily = df_feat.resample("D").mean()
    df_daily = df_daily.loc["2019-01-01":"2023-12-31"]

    # Top 10 features from Step 2 LightGBM walk-forward best
    top10_cols = [
        "rsi_divergence_14h_168h", "price_momentum_divergence", "intraday_range",
        "recovery_from_20h_low", "rsi_volume_interaction", "vwap_deviation_24h",
        "tick_direction_ratio_24h", "rsi_divergence_4h_24h", "momentum_divergence_72h_336h",
        "drawdown_from_20h_high",
    ]
    available10 = [c for c in top10_cols if c in df_daily.columns]
    if len(available10) < 5:
        available10 = df_daily.var().sort_values(ascending=False).head(10).index.tolist()
    available20 = df_daily.var().sort_values(ascending=False).head(20).index.tolist()

    # Daily target (next-day direction)
    target_raw = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")
    if isinstance(target_raw, pd.DataFrame):
        target_raw = target_raw["target"]
    target_daily = target_raw.resample("D").last().loc["2019-01-01":"2023-12-31"]

    # Align
    common_idx = prices_daily.index.intersection(df_daily.index).intersection(target_daily.index)
    prices_daily  = prices_daily.loc[common_idx]
    daily_returns  = daily_returns.reindex(common_idx).fillna(0)
    df_daily       = df_daily.loc[common_idx]
    target_daily   = target_daily.loc[common_idx]

    valid = target_daily.notna()
    prices_daily  = prices_daily[valid]
    daily_returns  = daily_returns[valid]
    df_daily       = df_daily[valid].fillna(df_daily.median())
    target_daily   = target_daily[valid]

    print(f"  Daily prices: {len(prices_daily)} rows ({prices_daily.index[0].date()} - {prices_daily.index[-1].date()})")
    print(f"  Features top-10: {len(available10)}, top-20: {len(available20)}")
    return prices_daily, daily_returns, df_daily, target_daily, available10, available20


# ─── Core signal utilities ───────────────────────────────────────────────────────
def donchian_signal(prices: pd.Series, entry: int = 40, exit_p: int = 20) -> pd.Series:
    """Donchian channel breakout — no look-ahead, stateful."""
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


def compute_adx(prices: pd.Series, period: int = 14) -> pd.Series:
    """Approximate ADX from close prices (Wilder EWM smoothing)."""
    delta = prices.diff()
    plus_dm  = delta.clip(lower=0)
    minus_dm = (-delta).clip(lower=0)
    plus_di  = plus_dm.ewm(span=period, adjust=False).mean()
    minus_di = minus_dm.ewm(span=period, adjust=False).mean()
    dx  = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
    return dx.ewm(span=period, adjust=False).mean()


def momentum_signal(prices: pd.Series, lookback: int = 40, threshold: float = 0.0) -> pd.Series:
    """Binary momentum: 1 if momentum > threshold, else 0."""
    mom = prices.pct_change(lookback)
    return (mom > threshold).astype(int)


def sharpe_from_signals(returns: pd.Series, signals: pd.Series) -> float:
    pos = signals.reindex(returns.index).fillna(0).shift(1).fillna(0)
    strat_ret = pos * returns
    if strat_ret.std() == 0 or len(strat_ret) < 10:
        return 0.0
    return float(annualized_sharpe(strat_ret))


def year_sharpe(returns: pd.Series, signals: pd.Series, year: int) -> float:
    mask = returns.index.year == year
    return sharpe_from_signals(returns[mask], signals.reindex(returns[mask].index).fillna(0))


def run_walk_forward(prices, daily_returns, signal_fn, years=None):
    """Walk-forward: signal_fn(prices, daily_returns, year) -> pd.Series[positions for that year]."""
    if years is None:
        years = list(range(2019, 2024))
    per_year = {}
    for yr in years:
        try:
            positions_yr = signal_fn(prices, daily_returns, yr)
            sh = year_sharpe(daily_returns, positions_yr, yr)
            per_year[str(yr)] = round(sh, 3)
        except Exception as e:
            print(f"  WARN: year {yr} failed: {e}")
            per_year[str(yr)] = 0.0

    vals = list(per_year.values())
    return {
        "mean_sharpe":  round(float(np.mean(vals)), 3),
        "std_sharpe":   round(float(np.std(vals)), 3),
        "min_sharpe":   round(float(min(vals)), 3),
        "max_sharpe":   round(float(max(vals)), 3),
        "per_year":     per_year,
    }


def log_run(tracker, name, config, metrics, group, notes=""):
    """Standardized logging."""
    config_logged = {**config, "notes": notes}
    metrics_logged = {
        **metrics,
        "baseline_donchian": BASELINE_SHARPE,
        "best_regime_sharpe": BEST_REGIME_SHARPE,
        "best_novel_sharpe": BEST_NOVEL_SHARPE,
    }
    tracker.log_experiment(
        name=name,
        config=config_logged,
        metrics=metrics_logged,
        tags=TAGS,
        job_type=JOB_TYPE,
        group=group,
    )
    beats = "✅" if metrics.get("mean_sharpe", 0) > BEST_REGIME_SHARPE else "❌"
    print(f"  Logged: {name} | Sharpe={metrics.get('mean_sharpe', 0):.3f} "
          f"std={metrics.get('std_sharpe', 0):.3f} "
          f"2022={metrics.get('sharpe_2022', 0):.3f} {beats}")


# ════════════════════════════════════════════════════════════════════════════════
# PRIORITY 1 — Momentum + Regime Deep Sweep
# ════════════════════════════════════════════════════════════════════════════════
def p1_momentum_regime_sweep(prices, daily_returns, tracker):
    """
    Deep parameter sweep for Momentum + vol_adx_OR.

    Key known facts:
    - Default lb=40, t=0, adx14, adx_thresh=30, v20: Sharpe 1.279 (best raw mom)
    - ADX(21) reduces std from 1.465 → 0.846 and improves 2022: -1.317 → -0.260
    - lb=60 helps mean Sharpe (1.191) but worse 2022 (-0.811)
    - lb=10, t=5%+ kills performance

    New hypothesis: Systematic grid to find the ROBUST region where Sharpe > 1.0
    consistently across different parameter combinations.
    """
    print("\n" + "=" * 80)
    print("PRIORITY 1: Momentum + Regime Deep Sweep")
    print("=" * 80)

    # Focused grid around the known-good region
    configs = [
        # (mom_lb, mom_thresh, vol_window, adx_period, adx_thresh)
        # Row 1: ADX(21) variations — known to smooth regime dramatically
        (40,  0.00, 20, 21, 25),  # ADX(21,25) — relax threshold
        (40,  0.00, 20, 21, 20),  # ADX(21,20) — more permissive
        (60,  0.00, 20, 21, 30),  # lb=60 + ADX(21) — combine insights
        (40,  0.02, 20, 21, 30),  # lb=40, t=2%, ADX(21)
        # Row 2: vol window exploration
        (40,  0.00, 30, 14, 30),  # vol30 — slower vol filter
        (40,  0.00, 50, 14, 30),  # vol50 — very slow vol filter
        (40,  0.00, 20, 14, 25),  # ADX threshold=25 (known from v3: 1.218 Sharpe)
        # Row 3: lookback variations
        (30,  0.00, 20, 21, 30),  # lb=30 + ADX(21)
        (60,  0.00, 20, 21, 25),  # lb=60 + ADX(21,25)
        (40, -0.02, 20, 21, 30),  # Negative threshold (allow weak negative mom to still trade)
    ]

    results = []
    for lb, thresh, vol_w, adx_p, adx_t in configs:
        cfg_name = f"mom_OR_lb{lb}_t{int(thresh*100)}_adx{adx_p}t{adx_t}_v{vol_w}"
        print(f"\n  Running: {cfg_name}")

        def make_fn(lb_, thresh_, vol_w_, adx_p_, adx_t_):
            def signal_fn(prices_, returns_, yr):
                # Momentum signal
                mom = prices_.pct_change(lb_)
                mom_sig = (mom > thresh_).astype(int)
                # Vol regime: rolling vol > median
                ret_ = prices_.pct_change()
                roll_vol = ret_.rolling(vol_w_).std() * np.sqrt(252)
                cutoff = pd.Timestamp(f"{yr}-01-01", tz="UTC")
                train_vol = roll_vol[roll_vol.index < cutoff]
                vol_med = train_vol.median() if len(train_vol) > 10 else roll_vol.median()
                vol_reg = (roll_vol > vol_med).astype(int)
                # ADX regime
                adx = compute_adx(prices_, period=adx_p_)
                adx_reg = (adx > adx_t_).astype(int)
                # OR combination
                combined_regime = ((vol_reg == 1) | (adx_reg == 1)).astype(int)
                # Momentum + regime
                final_sig = (mom_sig * combined_regime)
                # Return test year slice
                mask = prices_.index.year == yr
                return final_sig[mask]
            return signal_fn

        fn = make_fn(lb, thresh, vol_w, adx_p, adx_t)
        wf = run_walk_forward(prices, daily_returns, fn)
        sharpe_2022 = wf["per_year"].get("2022", 0.0)

        metrics = {
            "mean_sharpe": wf["mean_sharpe"],
            "std_sharpe":  wf["std_sharpe"],
            "min_sharpe":  wf["min_sharpe"],
            "max_sharpe":  wf["max_sharpe"],
            "sharpe_2022": sharpe_2022,
            **{f"sharpe_{yr}": wf["per_year"].get(str(yr), 0.0) for yr in range(2019, 2024)},
        }
        config = {
            "mom_lookback": lb, "mom_threshold": thresh, "vol_window": vol_w,
            "adx_period": adx_p, "adx_threshold": adx_t, "combo": "vol_OR_adx",
            "strategy": "momentum_regime",
        }

        run_name = f"mom_OR_lb{lb}_t{int(thresh*100)}_adx{adx_p}t{adx_t}_v{vol_w}_S{wf['mean_sharpe']:.2f}"
        log_run(tracker, run_name, config, metrics, group="mom_regime_sweep",
                notes=f"P1: momentum+vol_OR_adx sweep; lb={lb},t={thresh},adx{adx_p}>{adx_t},vol{vol_w}")
        results.append((cfg_name, wf["mean_sharpe"], wf["std_sharpe"], sharpe_2022))

    print("\n  P1 Summary (top by Sharpe):")
    for name, sh, std, s22 in sorted(results, key=lambda x: -x[1])[:5]:
        print(f"    {name}: Sharpe={sh:.3f}, std={std:.3f}, 2022={s22:.3f}")
    return results


# ════════════════════════════════════════════════════════════════════════════════
# PRIORITY 2 — Target Re-Engineering Deep Dive
# ════════════════════════════════════════════════════════════════════════════════
def p2_target_reengineering(prices, daily_returns, df_daily, tracker, feature_cols):
    """
    Expand breakout_profitability beyond CatBoost.
    - XGBoost and LightGBM for profitability prediction
    - Vary profitable breakout window (5, 10, 20 candles)
    - Add regime features (ADX, vol) to the classifier
    """
    print("\n" + "=" * 80)
    print("PRIORITY 2: Target Re-Engineering Deep Dive")
    print("=" * 80)

    # Precompute Donchian signal
    don_sig = donchian_signal(prices, ENTRY_PERIOD, EXIT_PERIOD)
    adx_14 = compute_adx(prices, period=14)
    daily_returns_aligned = daily_returns.reindex(prices.index).fillna(0)

    # Build breakout dataset: when does a new entry occur?
    entries_idx = []
    prev_sig = 0
    for idx, sig in don_sig.items():
        if sig == 1 and prev_sig == 0:
            entries_idx.append(idx)
        prev_sig = sig

    print(f"  Total Donchian breakouts: {len(entries_idx)}")

    results = []

    for fwd_window in [5, 10, 20]:
        # Build target: was breakout profitable over next fwd_window days?
        profitable = []
        for entry_date in entries_idx:
            try:
                loc = daily_returns_aligned.index.get_loc(entry_date)
                if loc + fwd_window >= len(daily_returns_aligned):
                    continue
                fwd_returns = daily_returns_aligned.iloc[loc:loc + fwd_window]
                cum_ret = (1 + fwd_returns).prod() - 1
                profitable.append((entry_date, int(cum_ret > 0)))
            except Exception:
                continue

        if len(profitable) < 10:
            print(f"  SKIP fwd_window={fwd_window}: only {len(profitable)} samples")
            continue

        entry_dates = [p[0] for p in profitable]
        labels = [p[1] for p in profitable]
        print(f"\n  fwd_window={fwd_window}: {len(labels)} breakouts, {sum(labels)}/{len(labels)} profitable ({100*sum(labels)/len(labels):.0f}%)")

        # Build feature matrix at entry dates
        # Base features + regime features
        base_feats = feature_cols[:10]
        adx_feat = adx_14.reindex(prices.index)
        vol_feat = prices.pct_change().rolling(20).std() * np.sqrt(252)

        for model_name, model_cls, model_kwargs in [
            ("xgb",  XGBClassifier,  dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                                          tree_method="hist", device="cuda",
                                          eval_metric="logloss", verbosity=0,
                                          use_label_encoder=False)),
            ("lgbm", LGBMClassifier, dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                                          device="gpu", verbose=-1, random_state=42)),
            ("cat",  CatBoostClassifier, dict(iterations=100, depth=3, learning_rate=0.05,
                                              task_type="GPU", devices="0",
                                              verbose=0, random_state=42)),
        ]:
            cfg_name = f"breakout_{model_name}_fwd{fwd_window}"
            print(f"    Training {cfg_name}...")

            # Walk-forward: train on all breakouts BEFORE test year, test on test year
            per_year_sharpe = {}
            for yr in range(2019, 2024):
                # Filter breakouts for training (before test year)
                train_entries = [i for i, d in enumerate(entry_dates) if d.year < yr]
                test_entries  = [i for i, d in enumerate(entry_dates) if d.year == yr]

                if len(train_entries) < 8 or len(test_entries) < 1:
                    per_year_sharpe[str(yr)] = 0.0
                    continue

                # Build feature matrix: base features + ADX + vol
                def build_X(idxs):
                    rows = []
                    for i in idxs:
                        d = entry_dates[i]
                        try:
                            feat_row = df_daily.loc[d, base_feats].values.tolist() if d in df_daily.index else [0.0] * len(base_feats)
                            adx_val = float(adx_feat.get(d, 0.0))
                            vol_val = float(vol_feat.get(d, 0.0))
                            rows.append(feat_row + [adx_val, vol_val])
                        except Exception:
                            rows.append([0.0] * (len(base_feats) + 2))
                    return np.array(rows, dtype=np.float32)

                X_train = build_X(train_entries)
                y_train = np.array([labels[i] for i in train_entries])
                X_test  = build_X(test_entries)
                test_labels = [labels[i] for i in test_entries]
                test_dates  = [entry_dates[i] for i in test_entries]

                # Handle degenerate training set
                if len(np.unique(y_train)) < 2:
                    per_year_sharpe[str(yr)] = 0.0
                    continue

                try:
                    mdl = model_cls(**model_kwargs)
                    mdl.fit(X_train, y_train)
                    preds = mdl.predict(X_test)
                except Exception as e:
                    print(f"      WARN: {yr} fit failed: {e}")
                    per_year_sharpe[str(yr)] = 0.0
                    continue

                # Build position signal for test year: trade only on ML-approved breakouts
                yr_mask = daily_returns_aligned.index.year == yr
                yr_returns = daily_returns_aligned[yr_mask]
                positions = pd.Series(0, index=yr_returns.index)

                don_yr = don_sig[yr_mask]
                in_trade_dates = set()
                for k, (d, pred) in enumerate(zip(test_dates, preds)):
                    if pred == 1:
                        # Find the Donchian signal window from this entry
                        d_loc = don_sig.index.get_loc(d) if d in don_sig.index else None
                        if d_loc is not None:
                            # Hold while signal=1 (Donchian exit)
                            j = d_loc
                            while j < len(don_sig) and don_sig.iloc[j] == 1 and don_sig.index[j].year == yr:
                                in_trade_dates.add(don_sig.index[j])
                                j += 1

                for dt in yr_returns.index:
                    if dt in in_trade_dates:
                        positions[dt] = 1

                sh = sharpe_from_signals(yr_returns, positions)
                per_year_sharpe[str(yr)] = round(sh, 3)

            vals = list(per_year_sharpe.values())
            wf_sharpe = np.mean(vals)
            wf_std    = np.std(vals)
            sharpe_2022 = per_year_sharpe.get("2022", 0.0)

            metrics = {
                "mean_sharpe": round(float(wf_sharpe), 3),
                "std_sharpe":  round(float(wf_std), 3),
                "sharpe_2022": round(float(sharpe_2022), 3),
                "min_sharpe":  round(float(min(vals)), 3),
                "max_sharpe":  round(float(max(vals)), 3),
                **{f"sharpe_{yr}": per_year_sharpe.get(str(yr), 0.0) for yr in range(2019, 2024)},
            }
            config = {
                "strategy": "breakout_profitability",
                "model": model_name,
                "fwd_window": fwd_window,
                "n_features": len(base_feats) + 2,
                "features": "top10+adx+vol",
            }
            run_name = f"breakout_{model_name}_fwd{fwd_window}_S{wf_sharpe:.2f}"
            log_run(tracker, run_name, config, metrics, group="target_reengineering",
                    notes=f"P2: {model_name} predicts profitable breakout over {fwd_window}d horizon")
            results.append((cfg_name, wf_sharpe, wf_std, sharpe_2022))

    return results


# ════════════════════════════════════════════════════════════════════════════════
# PRIORITY 3 — Stability Optimization
# ════════════════════════════════════════════════════════════════════════════════
def p3_stability_optimization(prices, daily_returns, tracker):
    """
    Build on stability_opt (triple gate: adx30 + vol_adx_AND).
    Sharpe 1.032, std 0.465, 2022=+0.583 — most deployable config found.

    Try:
    1. ADX period variations on the triple gate
    2. Stability-opt + momentum confirmation
    3. Optimize Sharpe/Std ratio (varies ADX threshold + vol window)
    """
    print("\n" + "=" * 80)
    print("PRIORITY 3: Stability Optimization")
    print("=" * 80)

    don_sig = donchian_signal(prices, ENTRY_PERIOD, EXIT_PERIOD)
    results = []

    stability_configs = [
        # (adx_period_outer, adx_thresh_outer, vol_window, adx_period_inner, adx_thresh_inner, name)
        # The original triple gate: ADX(14,30) AND vol>med AND ADX(14,25)
        (14, 30, 20, 14, 25, "adx14t30_AND_vol20_AND_adx14t25"),
        # Try ADX(21) for outer gate — smoother
        (21, 30, 20, 21, 25, "adx21t30_AND_vol20_AND_adx21t25"),
        # Relax thresholds slightly
        (14, 25, 20, 14, 20, "adx14t25_AND_vol20_AND_adx14t20"),
        # Vol window variation
        (14, 30, 30, 14, 25, "adx14t30_AND_vol30_AND_adx14t25"),
        # Stability + momentum confirmation
        (14, 30, 20, 14, 25, "triple_gate_plus_mom40"),  # adds mom_lb=40 AND
    ]

    for i, cfg_tuple in enumerate(stability_configs):
        adx_p_out, adx_t_out, vol_w, adx_p_in, adx_t_in, cfg_name = cfg_tuple
        add_mom = ("plus_mom" in cfg_name)
        print(f"\n  Running: {cfg_name}")

        def make_fn(adx_po, adx_to, vol_w_, adx_pi, adx_ti, use_mom):
            def signal_fn(prices_, returns_, yr):
                # ADX outer filter
                adx_out = compute_adx(prices_, period=adx_po)
                gate_adx_out = (adx_out > adx_to).astype(int)
                # Vol filter (above median using train data causal)
                ret_ = prices_.pct_change()
                roll_vol = ret_.rolling(vol_w_).std() * np.sqrt(252)
                cutoff = pd.Timestamp(f"{yr}-01-01", tz="UTC")
                train_vol = roll_vol[roll_vol.index < cutoff]
                vol_med = train_vol.median() if len(train_vol) > 10 else roll_vol.median()
                gate_vol = (roll_vol > vol_med).astype(int)
                # ADX inner filter (same period, lower threshold)
                adx_in = compute_adx(prices_, period=adx_pi)
                gate_adx_in = (adx_in > adx_ti).astype(int)
                # Triple gate
                triple = (gate_adx_out & gate_vol & gate_adx_in).astype(int)
                # Optional: also require momentum confirmation
                if use_mom:
                    mom_sig = (prices_.pct_change(40) > 0.0).astype(int)
                    triple = (triple & mom_sig).astype(int)
                # Apply to Donchian
                don_all = donchian_signal(prices_, ENTRY_PERIOD, EXIT_PERIOD)
                mask = prices_.index.year == yr
                final = (don_all[mask] * triple.reindex(prices_[mask].index).fillna(0)).clip(0, 1).astype(int)
                return final
            return signal_fn

        fn = make_fn(adx_p_out, adx_t_out, vol_w, adx_p_in, adx_t_in, add_mom)
        wf = run_walk_forward(prices, daily_returns, fn)
        sharpe_2022 = wf["per_year"].get("2022", 0.0)

        # Sharpe/Std ratio (stability metric)
        sharpe_per_std = round(wf["mean_sharpe"] / max(wf["std_sharpe"], 0.001), 3)

        metrics = {
            "mean_sharpe":   wf["mean_sharpe"],
            "std_sharpe":    wf["std_sharpe"],
            "min_sharpe":    wf["min_sharpe"],
            "max_sharpe":    wf["max_sharpe"],
            "sharpe_2022":   sharpe_2022,
            "sharpe_per_std": sharpe_per_std,
            **{f"sharpe_{yr}": wf["per_year"].get(str(yr), 0.0) for yr in range(2019, 2024)},
        }
        config = {
            "strategy": "stability_optimized",
            "adx_period_outer": adx_p_out, "adx_thresh_outer": adx_t_out,
            "vol_window": vol_w, "adx_period_inner": adx_p_in,
            "adx_thresh_inner": adx_t_in, "add_momentum": add_mom,
        }
        run_name = f"stability_{cfg_name}_S{wf['mean_sharpe']:.2f}_std{wf['std_sharpe']:.2f}"
        log_run(tracker, run_name, config, metrics, group="stability_optimized",
                notes=f"P3: triple-gate stability opt; {cfg_name}; sharpe/std={sharpe_per_std}")
        results.append((cfg_name, wf["mean_sharpe"], wf["std_sharpe"], sharpe_2022, sharpe_per_std))

    print("\n  P3 Summary (by Sharpe/Std ratio):")
    for name, sh, std, s22, sps in sorted(results, key=lambda x: -x[4])[:5]:
        print(f"    {name}: Sharpe={sh:.3f}, std={std:.3f}, 2022={s22:.3f}, S/std={sps:.3f}")
    return results


# ════════════════════════════════════════════════════════════════════════════════
# PRIORITY 4 — Robustness Testing on Top 3
# ════════════════════════════════════════════════════════════════════════════════
def p4_robustness_testing(prices, daily_returns, tracker):
    """
    Stress-test top 3 configs:
    - mom_vol_adx_OR (Sharpe 1.279): lb=40, t=0, adx14, thresh=30, vol20
    - breakout_cat (Sharpe 1.243): Donchian(40/20) + ADX filter
    - majority_vote_2of3 (Sharpe 1.197): ADX>30, 2-of-3

    Tests:
    1. Perturb all key parameters by +/-20% — does Sharpe stay above 1.0?
    2. Test on different train window sizes (pure in-sample: 2yr, 3yr, 4yr full data)
    """
    print("\n" + "=" * 80)
    print("PRIORITY 4: Robustness Testing — Perturb Top Configs")
    print("=" * 80)

    results = []

    # ── 4A: mom_vol_adx_OR perturbation ────────────────────────────────────────
    print("\n  4A: mom_vol_adx_OR parameter perturbation (+/-20%)")
    # Baseline: lb=40, adx_period=14, adx_thresh=30, vol=20
    # +20%: lb=48, adx_p=17, adx_t=36, vol=24 → round to integers
    # -20%: lb=32, adx_p=11, adx_t=24, vol=16
    mom_perturb_configs = [
        # (lb, adx_p, adx_t, vol_w, label)
        (40, 14, 30, 20, "baseline"),
        (48, 17, 36, 24, "+20pct"),
        (32, 11, 24, 16, "-20pct"),
        (40, 14, 27, 20, "adx_t-10pct"),  # just threshold down
        (40, 14, 33, 20, "adx_t+10pct"),  # just threshold up
        (36, 14, 30, 18, "lb_vol-10pct"), # lb and vol down
        (44, 14, 30, 22, "lb_vol+10pct"), # lb and vol up
    ]
    for lb, adx_p, adx_t, vol_w, label in mom_perturb_configs:
        cfg_name = f"mom_OR_robust_{label}"
        print(f"    {cfg_name}: lb={lb}, adx{adx_p}>{adx_t}, vol={vol_w}")

        def make_mom_fn(lb_, adx_p_, adx_t_, vol_w_):
            def signal_fn(prices_, returns_, yr):
                mom = prices_.pct_change(lb_)
                mom_sig = (mom > 0.0).astype(int)
                ret_ = prices_.pct_change()
                roll_vol = ret_.rolling(vol_w_).std() * np.sqrt(252)
                cutoff = pd.Timestamp(f"{yr}-01-01", tz="UTC")
                train_vol = roll_vol[roll_vol.index < cutoff]
                vol_med = train_vol.median() if len(train_vol) > 10 else roll_vol.median()
                vol_reg = (roll_vol > vol_med).astype(int)
                adx = compute_adx(prices_, period=adx_p_)
                adx_reg = (adx > adx_t_).astype(int)
                combined = ((vol_reg == 1) | (adx_reg == 1)).astype(int)
                final = (mom_sig * combined)
                mask = prices_.index.year == yr
                return final[mask]
            return signal_fn

        fn = make_mom_fn(lb, adx_p, adx_t, vol_w)
        wf = run_walk_forward(prices, daily_returns, fn)
        sharpe_2022 = wf["per_year"].get("2022", 0.0)
        robust = "ROBUST" if wf["mean_sharpe"] > 1.0 else "FRAGILE"

        metrics = {
            "mean_sharpe": wf["mean_sharpe"],
            "std_sharpe":  wf["std_sharpe"],
            "min_sharpe":  wf["min_sharpe"],
            "sharpe_2022": sharpe_2022,
            "is_robust": int(wf["mean_sharpe"] > 1.0),
            **{f"sharpe_{yr}": wf["per_year"].get(str(yr), 0.0) for yr in range(2019, 2024)},
        }
        config = {
            "strategy": "mom_vol_adx_OR_robustness",
            "perturbation": label, "mom_lookback": lb,
            "adx_period": adx_p, "adx_threshold": adx_t, "vol_window": vol_w,
        }
        run_name = f"robust_mom_OR_{label}_S{wf['mean_sharpe']:.2f}"
        log_run(tracker, run_name, config, metrics, group="robustness_testing",
                notes=f"P4: robustness test mom_vol_adx_OR; {label}; {robust}")
        results.append((cfg_name, wf["mean_sharpe"], wf["std_sharpe"], robust))

    # ── 4B: ADX(14,30) Donchian perturbation ───────────────────────────────────
    print("\n  4B: ADX(14,30) Donchian perturbation — best 2022 protector")
    # Baseline: adx_period=14, adx_thresh=30, entry=40, exit=20
    don_perturb_configs = [
        # (adx_p, adx_t, entry, exit, label)
        (14, 30, 40, 20, "baseline"),
        (17, 36, 48, 24, "+20pct_all"),
        (11, 24, 32, 16, "-20pct_all"),
        (14, 30, 48, 24, "don_only+20"),  # only Donchian params
        (14, 30, 32, 16, "don_only-20"),
        (17, 30, 40, 20, "adx_p+20"),    # only ADX period
        (14, 36, 40, 20, "adx_t+20"),    # only ADX threshold
    ]
    for adx_p, adx_t, entry, exit_p, label in don_perturb_configs:
        cfg_name = f"don_adx_robust_{label}"
        print(f"    {cfg_name}: adx{adx_p}>{adx_t}, don({entry}/{exit_p})")

        def make_don_fn(adx_p_, adx_t_, entry_, exit_p_):
            def signal_fn(prices_, returns_, yr):
                don = donchian_signal(prices_, entry_, exit_p_)
                adx = compute_adx(prices_, period=adx_p_)
                adx_reg = (adx > adx_t_).astype(int)
                mask = prices_.index.year == yr
                final = (don[mask] * adx_reg.reindex(prices_[mask].index).fillna(0)).clip(0, 1).astype(int)
                return final
            return signal_fn

        fn = make_don_fn(adx_p, adx_t, entry, exit_p)
        wf = run_walk_forward(prices, daily_returns, fn)
        sharpe_2022 = wf["per_year"].get("2022", 0.0)
        robust = "ROBUST" if wf["mean_sharpe"] > 1.0 else "FRAGILE"

        metrics = {
            "mean_sharpe": wf["mean_sharpe"],
            "std_sharpe":  wf["std_sharpe"],
            "min_sharpe":  wf["min_sharpe"],
            "sharpe_2022": sharpe_2022,
            "is_robust": int(wf["mean_sharpe"] > 1.0),
            **{f"sharpe_{yr}": wf["per_year"].get(str(yr), 0.0) for yr in range(2019, 2024)},
        }
        config = {
            "strategy": "donchian_adx_robustness",
            "perturbation": label, "adx_period": adx_p, "adx_threshold": adx_t,
            "entry_period": entry, "exit_period": exit_p,
        }
        run_name = f"robust_don_adx_{label}_S{wf['mean_sharpe']:.2f}"
        log_run(tracker, run_name, config, metrics, group="robustness_testing",
                notes=f"P4: robustness test ADX(14,30) Donchian; {label}; {robust}")
        results.append((cfg_name, wf["mean_sharpe"], wf["std_sharpe"], robust))

    # Summary
    robust_count = sum(1 for _, sh, _, r in results if r == "ROBUST")
    print(f"\n  P4 Summary: {robust_count}/{len(results)} configs ROBUST (Sharpe > 1.0)")
    for name, sh, std, robust in sorted(results, key=lambda x: -x[1]):
        print(f"    {name}: Sharpe={sh:.3f}, std={std:.3f} [{robust}]")

    return results


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 80)
    print("CONTRACT #004 — STEP 5 (v2): NOVEL EXPLORATION — DEEP DIVE")
    print("=" * 80)

    # Load data once
    prices, daily_returns, df_daily, target_daily, feat10, feat20 = load_data()

    # Initialize tracker
    tracker = ExperimentTracker(experiment_name="contract_004_novel_v2")
    print(f"\n  W&B project: datadex_ai/sparky-ai")
    print(f"  Tags: {TAGS}")
    print(f"  Target: 15+ runs")
    print()

    all_results = {}

    # Priority 1: Momentum + Regime Sweep (10 configs → 10 runs)
    p1_results = p1_momentum_regime_sweep(prices, daily_returns, tracker)
    all_results["p1_momentum_regime"] = p1_results

    # Priority 2: Target Re-Engineering (3 models × 3 windows → up to 9 runs,
    # but breakout data is sparse so skip invalid combos; expect ~6-9 runs)
    p2_results = p2_target_reengineering(prices, daily_returns, df_daily, tracker, feat10)
    all_results["p2_target_reengineering"] = p2_results

    # Priority 3: Stability Optimization (5 configs → 5 runs)
    p3_results = p3_stability_optimization(prices, daily_returns, tracker)
    all_results["p3_stability"] = p3_results

    # Priority 4: Robustness Testing (7+7 = 14 configs → 14 runs)
    p4_results = p4_robustness_testing(prices, daily_returns, tracker)
    all_results["p4_robustness"] = p4_results

    # ── Final Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — ALL STEP 5 (v2) RESULTS")
    print("=" * 80)

    # Collect all results for JSON
    final_summary = {
        "elapsed_seconds": elapsed,
        "p1_best": sorted(p1_results, key=lambda x: -x[1])[:3] if p1_results else [],
        "p2_best": sorted([(n, sh, std, s22) for n, sh, std, s22 in p2_results], key=lambda x: -x[1])[:3] if p2_results else [],
        "p3_best": sorted([(n, sh, std, s22, sps) for n, sh, std, s22, sps in p3_results], key=lambda x: -x[4])[:3] if p3_results else [],
        "p4_robust_count": sum(1 for _, sh, _, r in p4_results if r == "ROBUST") if p4_results else 0,
    }

    print("\nP1 — Momentum+Regime best:")
    for item in final_summary["p1_best"]:
        print(f"  {item[0]}: Sharpe={item[1]:.3f}")

    print("\nP2 — Target Re-Engineering best:")
    for item in final_summary["p2_best"]:
        print(f"  {item[0]}: Sharpe={item[1]:.3f}, 2022={item[3]:.3f}")

    print("\nP3 — Stability best (by Sharpe/Std):")
    for item in final_summary["p3_best"]:
        print(f"  {item[0]}: Sharpe={item[1]:.3f}, std={item[2]:.3f}, Sharpe/Std={item[4]:.3f}")

    print(f"\nP4 — Robustness: {final_summary['p4_robust_count']}/{len(p4_results)} configs passed Sharpe > 1.0 threshold")

    # Save to JSON
    out_path = RESULTS_DIR / "novel_v2_results.json"
    with open(out_path, "w") as f:
        # Convert tuples to lists for JSON serialization
        json_ready = {
            "elapsed_seconds": elapsed,
            "p1_best": [[str(x) for x in item] for item in final_summary["p1_best"]],
            "p2_best": [[str(x) for x in item] for item in final_summary["p2_best"]],
            "p3_best": [[str(x) for x in item] for item in final_summary["p3_best"]],
            "p4_robust_count": final_summary["p4_robust_count"],
            "p4_total": len(p4_results),
        }
        json.dump(json_ready, f, indent=2)
    print(f"\nSaved summary to {out_path}")
    print(f"Total elapsed: {elapsed:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
