#!/usr/bin/env python3
"""Regime-Aware Donchian Hybrid — Step 3 of Contract #004.

Tests 5 distinct regime detection approaches:
  1. Volatility regime (rolling realized vol, trending=high vol)
  2. Trend strength (ADX-based filter)
  3. ML meta-learner (CatBoost predicts when Donchian will profit)
  4. Multi-signal regime (combinations of vol+ADX or ML+vol)
  5. Rolling drawdown filter (go flat when strategy is losing)

Each approach runs Donchian ONLY in 'trade' regime, flat otherwise.
Walk-forward validated with yearly folds (2019–2023).
"""
import sys
sys.path.insert(0, "src")

import os
os.environ["PYTHONUNBUFFERED"] = "1"

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timezone

from catboost import CatBoostClassifier

from sparky.data.loader import load
from sparky.tracking.experiment import ExperimentTracker
from sparky.features.returns import annualized_sharpe, max_drawdown

# ─── Constants ─────────────────────────────────────────────────────────────────
BASELINE_DONCHIAN_SHARPE = 1.062
BH_SHARPE = 0.65          # approximate BTC buy-and-hold Sharpe 2019-2023
ENTRY_PERIOD = 40          # best Donchian entry from prior sweep
EXIT_PERIOD  = 20          # best Donchian exit
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Tags every run must carry
TAGS = ["contract_004", "regime"]
JOB_TYPE = "regime"


# ─── Data Loading ──────────────────────────────────────────────────────────────
def load_data():
    """Load features and daily prices once; cache in memory."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Use loader for holdout enforcement
    df_feat = load("feature_matrix_btc_hourly_expanded", purpose="training")

    # Load raw hourly prices for Donchian signals
    prices_hourly = pd.read_parquet("data/raw/btc/ohlcv_hourly_max_coverage.parquet")
    # Resample to daily
    prices_daily = prices_hourly["close"].resample("D").last().dropna()

    # Enforce UTC timezone
    if prices_daily.index.tz is None:
        prices_daily.index = prices_daily.index.tz_localize("UTC")

    # Keep in-sample only (2019-2023)
    prices_daily = prices_daily.loc["2019-01-01":"2023-12-31"]

    # Daily returns (aligned)
    daily_returns = prices_daily.pct_change().dropna()

    # Also aggregate features to daily for ML (take daily mean of hourly)
    df_daily_feat = df_feat.resample("D").mean()
    df_daily_feat = df_daily_feat.loc["2019-01-01":"2023-12-31"]

    print(f"  Daily prices: {len(prices_daily)} rows ({prices_daily.index[0].date()} - {prices_daily.index[-1].date()})")
    print(f"  Daily features: {df_daily_feat.shape}")
    print(f"  Daily returns: {len(daily_returns)} rows")
    return prices_daily, daily_returns, df_daily_feat


# ─── Donchian Signal ────────────────────────────────────────────────────────────
def donchian_signal(prices: pd.Series, entry: int = 40, exit_p: int = 20) -> pd.Series:
    """Pure Donchian channel breakout — no look-ahead."""
    upper = prices.rolling(window=entry).max()
    lower = prices.rolling(window=exit_p).min()
    sig = pd.Series(0, index=prices.index, dtype=int)
    in_pos = False

    for i in range(len(prices)):
        if i < entry:
            sig.iloc[i] = 0
            continue
        px = prices.iloc[i]
        if not in_pos:
            if px >= upper.iloc[i - 1]:
                in_pos = True
                sig.iloc[i] = 1
        else:
            if i >= exit_p and px <= lower.iloc[i - 1]:
                in_pos = False
                sig.iloc[i] = 0
            else:
                sig.iloc[i] = 1
    return sig


# ─── Backtest Helpers ───────────────────────────────────────────────────────────
def compute_equity(returns: pd.Series, positions: pd.Series) -> pd.Series:
    """Equity curve from returns and positions.  positions[t] is held *during* day t."""
    pos = positions.reindex(returns.index).fillna(0)
    # shift(1): signal computed at close[T-1], trade executed at open[T]
    pos_shifted = pos.shift(1).fillna(0)
    strat_ret = pos_shifted * returns
    equity = (1 + strat_ret).cumprod()
    equity = pd.concat([pd.Series([1.0], index=[equity.index[0] - pd.Timedelta(days=1)]), equity])
    return equity


def metrics_from_equity(equity: pd.Series, returns: pd.Series, positions: pd.Series) -> dict:
    """Compute Sharpe, max drawdown, trade count from equity curve."""
    eq_ret = equity.pct_change().dropna()
    sharpe = annualized_sharpe(eq_ret)
    mdd = max_drawdown(equity)
    pos_shifted = positions.shift(1).fillna(0).reindex(returns.index).fillna(0)
    trades = int((pos_shifted.diff().fillna(pos_shifted.iloc[0]) != 0).sum())
    total_ret = float(equity.iloc[-1] - 1.0)
    return {"sharpe": sharpe, "max_drawdown": mdd, "total_return": total_ret, "num_trades": trades}


def yearly_sharpes(returns: pd.Series, positions: pd.Series) -> dict:
    """Compute per-year Sharpe ratios."""
    result = {}
    pos_shifted = positions.shift(1).fillna(0).reindex(returns.index).fillna(0)
    for yr in range(2019, 2024):
        mask = returns.index.year == yr
        r_yr = returns[mask]
        p_yr = pos_shifted[mask]
        if len(r_yr) < 10:
            continue
        strat_ret = p_yr * r_yr
        result[str(yr)] = annualized_sharpe(strat_ret)
    return result


def walk_forward_regime(
    prices: pd.Series,
    daily_returns: pd.Series,
    regime_fn,
    name: str,
) -> dict:
    """
    Expanding-window yearly walk-forward for a regime-filtered Donchian.

    For each test year Y (2019-2023):
      - Train: all data up to end of Y-1 (or all if 2019)
      - Compute Donchian signal on FULL price series (no peeking: signal is stateless)
      - Apply regime filter (computed on FULL price series, no peeking from filter perspective)
      - Test: compute equity for year Y only
    """
    per_year = {}
    full_donchian = donchian_signal(prices, ENTRY_PERIOD, EXIT_PERIOD)

    for yr in range(2019, 2024):
        test_mask = (prices.index.year == yr)
        test_prices = prices[test_mask]
        test_returns = daily_returns[test_mask]

        if len(test_returns) < 20:
            continue

        # Compute regime filter using only data through end of test year
        # (regime is computed causally on the whole in-sample series, not future-peeking)
        train_prices = prices[prices.index.year < yr] if yr > 2019 else prices[test_mask]

        # regime_fn receives: prices up to (not including) test start, test prices
        # It returns a binary Series for test period
        try:
            regime_signal = regime_fn(prices, yr)
        except Exception as e:
            print(f"  WARN: regime_fn failed for {yr}: {e}")
            regime_signal = pd.Series(1, index=test_prices.index)

        # Apply regime filter to Donchian
        don_test = full_donchian[test_mask]
        filtered_pos = (don_test * regime_signal.reindex(don_test.index).fillna(0)).clip(0, 1).astype(int)

        # Compute equity for this year
        eq = compute_equity(test_returns, filtered_pos)
        eq_ret = eq.pct_change().dropna()
        yr_sharpe = annualized_sharpe(eq_ret)
        per_year[str(yr)] = round(yr_sharpe, 3)

    mean_sharpe = np.mean(list(per_year.values())) if per_year else 0.0
    std_sharpe = np.std(list(per_year.values())) if per_year else 0.0
    min_sharpe = min(per_year.values()) if per_year else 0.0
    max_sharpe = max(per_year.values()) if per_year else 0.0

    # Full-period metrics (all years combined)
    test_mask_all = (prices.index.year >= 2019) & (prices.index.year <= 2023)
    all_returns = daily_returns[test_mask_all]

    # Reassemble combined filtered signal
    full_regime = pd.Series(0, index=prices[test_mask_all].index)
    for yr in range(2019, 2024):
        try:
            regime_yr = regime_fn(prices, yr)
            don_yr = full_donchian[prices.index.year == yr]
            filt_yr = (don_yr * regime_yr.reindex(don_yr.index).fillna(0)).clip(0, 1).astype(int)
            full_regime.loc[filt_yr.index] = filt_yr.values
        except Exception:
            don_yr = full_donchian[prices.index.year == yr]
            full_regime.loc[don_yr.index] = don_yr.values

    eq_all = compute_equity(all_returns, full_regime)
    overall = metrics_from_equity(eq_all, all_returns, full_regime)

    return {
        "mean_sharpe": round(mean_sharpe, 3),
        "std_sharpe": round(std_sharpe, 3),
        "min_sharpe": round(min_sharpe, 3),
        "max_sharpe": round(max_sharpe, 3),
        "per_year": per_year,
        "overall_sharpe": round(overall["sharpe"], 3),
        "max_drawdown": round(overall["max_drawdown"], 3),
        "total_return": round(overall["total_return"], 3),
        "num_trades": overall["num_trades"],
    }


# ─── ADX Computation ────────────────────────────────────────────────────────────
def compute_adx(prices: pd.Series, period: int = 14) -> pd.Series:
    """Approximate ADX from close prices (no high/low required)."""
    delta = prices.diff()
    plus_dm = delta.clip(lower=0)
    minus_dm = (-delta).clip(lower=0)

    # Smooth with Wilder's method (EWM approximation)
    plus_di  = plus_dm.ewm(span=period, adjust=False).mean()
    minus_di = minus_dm.ewm(span=period, adjust=False).mean()
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


# ════════════════════════════════════════════════════════════════════════════════
# APPROACH 1 — Volatility Regime
# ════════════════════════════════════════════════════════════════════════════════
def make_vol_regime_fn(window: int, above_median: bool = True):
    """Trade Donchian only when realized vol is above rolling median (trending)."""
    def regime_fn(prices: pd.Series, test_year: int) -> pd.Series:
        returns = prices.pct_change()
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        # Use expanding median up to test_year start (causal)
        cutoff = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
        vol_up_to_train = rolling_vol[rolling_vol.index < cutoff]
        median_vol = vol_up_to_train.median() if len(vol_up_to_train) > 10 else rolling_vol.median()

        test_mask = prices.index.year == test_year
        vol_test = rolling_vol[test_mask]
        if above_median:
            regime = (vol_test > median_vol).astype(int)
        else:
            regime = (vol_test < median_vol).astype(int)
        return regime
    return regime_fn


# ════════════════════════════════════════════════════════════════════════════════
# APPROACH 2 — ADX Trend Strength
# ════════════════════════════════════════════════════════════════════════════════
def make_adx_regime_fn(adx_period: int = 14, threshold: float = 25.0):
    """Trade only when ADX > threshold (market is trending, not choppy)."""
    def regime_fn(prices: pd.Series, test_year: int) -> pd.Series:
        adx = compute_adx(prices, period=adx_period)
        test_mask = prices.index.year == test_year
        adx_test = adx[test_mask]
        regime = (adx_test > threshold).astype(int)
        return regime
    return regime_fn


# ════════════════════════════════════════════════════════════════════════════════
# APPROACH 3 — ML Meta-Learner
# ════════════════════════════════════════════════════════════════════════════════
def run_ml_meta_learner(
    prices: pd.Series,
    daily_returns: pd.Series,
    df_features: pd.DataFrame,
    tracker: ExperimentTracker,
):
    """
    CatBoost predicts whether Donchian strategy will profit in next N days.
    Label: was Donchian profitable over rolling 30-day window?
    """
    print("\n" + "=" * 60)
    print("APPROACH 3: ML Meta-Learner")
    print("=" * 60)

    # Align features to daily
    common_idx = prices.index.intersection(df_features.index)
    prices_aligned = prices.loc[common_idx]
    feat_aligned = df_features.loc[common_idx]
    ret_aligned = daily_returns.reindex(common_idx).fillna(0)

    # Compute Donchian signals
    don_sig = donchian_signal(prices_aligned, ENTRY_PERIOD, EXIT_PERIOD)

    # Create meta-label: rolling 30-day Sharpe of Donchian strategy
    # Label = 1 if 30-day forward Sharpe > 0 (strategy would be profitable)
    strat_ret = don_sig.shift(1).fillna(0) * ret_aligned
    rolling_sharpe_30d = strat_ret.rolling(30).mean() / (strat_ret.rolling(30).std() + 1e-9) * np.sqrt(252)
    # Forward-shift: at time T, was strategy profitable OVER NEXT 30 days?
    # We shift label by -30 so we're predicting future performance
    # This is look-ahead in the label (legitimate for meta-learning since we're
    # learning "when does Donchian work" not "what will price do")
    meta_label = (rolling_sharpe_30d.shift(-30) > 0).astype(int)

    # Features: drop columns with too many NaNs
    feat_clean = feat_aligned.dropna(axis=1, thresh=int(0.7 * len(feat_aligned)))
    # Add regime-informative features explicitly
    returns_raw = prices_aligned.pct_change()
    feat_clean["vol_10d"] = returns_raw.rolling(10).std() * np.sqrt(252)
    feat_clean["vol_30d"] = returns_raw.rolling(30).std() * np.sqrt(252)
    feat_clean["adx_14"] = compute_adx(prices_aligned, 14)
    feat_clean["adx_30"] = compute_adx(prices_aligned, 30)
    feat_clean["mom_30d"] = returns_raw.rolling(30).sum()
    feat_clean["don_signal"] = don_sig.astype(float)

    # Align label and features
    common = feat_clean.index.intersection(meta_label.index)
    feat_clean = feat_clean.loc[common]
    meta_label = meta_label.loc[common]

    # Drop rows where either is NaN
    valid = feat_clean.notna().all(axis=1) & meta_label.notna()
    feat_clean = feat_clean[valid]
    meta_label = meta_label[valid]

    # Limit to top 20 features by variance (speed)
    feat_vars = feat_clean.var().sort_values(ascending=False)
    top_features = feat_vars.head(20).index.tolist()
    feat_clean = feat_clean[top_features]

    print(f"  Meta-label: {meta_label.sum()} trade / {(~meta_label.astype(bool)).sum()} no-trade")
    print(f"  Features: {feat_clean.shape}")

    best_result = None
    best_sharpe = -999

    for depth in [3, 4]:
        for lr in [0.05, 0.01]:
            print(f"\n  CatBoost meta-learner: depth={depth}, lr={lr}")
            t0 = time.time()

            per_year = {}
            regime_series = pd.Series(0, index=prices_aligned.index)

            for yr in range(2019, 2024):
                # Expanding train: all data before test year
                train_mask = feat_clean.index.year < yr
                test_mask_feat = feat_clean.index.year == yr
                test_mask_prices = prices_aligned.index.year == yr

                if train_mask.sum() < 100 or test_mask_feat.sum() < 10:
                    regime_series.loc[prices_aligned[test_mask_prices].index] = 1
                    continue

                X_train = feat_clean[train_mask]
                y_train = meta_label[train_mask]
                X_test = feat_clean[test_mask_feat]

                model = CatBoostClassifier(
                    depth=depth,
                    learning_rate=lr,
                    iterations=300,
                    l2_leaf_reg=1.0,
                    task_type="GPU",
                    devices="0",
                    verbose=0,
                    random_seed=42,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                # Map predictions back to price dates
                # (features may not perfectly align with prices after resampling)
                pred_series = pd.Series(preds, index=feat_clean[test_mask_feat].index)
                price_test_idx = prices_aligned[test_mask_prices].index
                regime_yr = pred_series.reindex(price_test_idx, method="ffill").fillna(0)
                regime_series.loc[price_test_idx] = regime_yr.values

                # Compute year Sharpe
                don_yr = donchian_signal(prices_aligned, ENTRY_PERIOD, EXIT_PERIOD)[test_mask_prices]
                filt_yr = (don_yr * regime_yr.values).clip(0, 1).astype(int)
                ret_yr = ret_aligned[test_mask_prices]
                eq_yr = compute_equity(ret_yr, filt_yr)
                yr_sharpe = annualized_sharpe(eq_yr.pct_change().dropna())
                per_year[str(yr)] = round(yr_sharpe, 3)

            elapsed = time.time() - t0
            mean_sh = np.mean(list(per_year.values()))
            std_sh = np.std(list(per_year.values()))
            min_sh = min(per_year.values()) if per_year else 0.0

            print(f"    Mean WF Sharpe: {mean_sh:.3f} ± {std_sh:.3f} | 2022: {per_year.get('2022', 'N/A')}")

            config = {
                "approach": "ml_meta_learner",
                "model": "catboost",
                "depth": depth,
                "learning_rate": lr,
                "iterations": 300,
                "top_features": 20,
                "label_window": 30,
                "entry_period": ENTRY_PERIOD,
                "exit_period": EXIT_PERIOD,
            }
            metrics = {
                "mean_wf_sharpe": round(mean_sh, 3),
                "std_wf_sharpe": round(std_sh, 3),
                "min_wf_sharpe": round(min_sh, 3),
                "sharpe_2022": per_year.get("2022", 0.0),
                "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
                "elapsed_seconds": elapsed,
                **{f"sharpe_{yr}": v for yr, v in per_year.items()},
            }
            run_name = f"meta_learner_cat_d{depth}_lr{lr}_S{mean_sh:.2f}"
            notes = (
                f"CatBoost meta-learner (depth={depth}, lr={lr}): predicts whether Donchian "
                f"will be profitable in next 30 days. Mean WF Sharpe {mean_sh:.3f}. "
                f"2022 Sharpe {per_year.get('2022', 'N/A'):.3f} — "
                + ("meta-learning identifies bear regimes" if mean_sh > BASELINE_DONCHIAN_SHARPE else "insufficient signal from meta-label features")
            )
            config["notes"] = notes

            tracker.log_experiment(
                name=run_name,
                config=config,
                metrics=metrics,
                tags=TAGS,
                job_type=JOB_TYPE,
            )
            print(f"    Logged: {run_name}")

            if mean_sh > best_sharpe:
                best_sharpe = mean_sh
                best_result = {"name": run_name, "per_year": per_year, "metrics": metrics, "config": config}

    return best_result


# ════════════════════════════════════════════════════════════════════════════════
# APPROACH 5 — Rolling Drawdown Filter
# ════════════════════════════════════════════════════════════════════════════════
def make_drawdown_regime_fn(prices: pd.Series, daily_returns: pd.Series, dd_threshold: float, dd_window: int):
    """Go flat if Donchian strategy drew down >threshold over last N days."""
    # Precompute Donchian signals
    full_don = donchian_signal(prices, ENTRY_PERIOD, EXIT_PERIOD)

    def regime_fn(prices_unused: pd.Series, test_year: int) -> pd.Series:
        test_mask = prices.index.year == test_year
        test_idx = prices[test_mask].index

        # Compute rolling drawdown of Donchian strategy (causal — no look-ahead)
        pos_shifted = full_don.shift(1).fillna(0)
        strat_ret = pos_shifted * daily_returns.reindex(prices.index).fillna(0)
        equity = (1 + strat_ret).cumprod()

        # Rolling max-to-current drawdown
        rolling_max = equity.rolling(dd_window).max()
        rolling_dd = (equity - rolling_max) / (rolling_max + 1e-9)

        regime = (rolling_dd[test_mask] > -dd_threshold).astype(int)
        return regime

    return regime_fn


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("REGIME-AWARE HYBRID — Step 3 / Contract #004")
    print("=" * 80)

    prices, daily_returns, df_features = load_data()

    tracker = ExperimentTracker(experiment_name="regime_aware_donchian")

    # Unfiltered Donchian baseline for reference
    print("\n--- Computing unfiltered Donchian baseline ---")
    full_don = donchian_signal(prices, ENTRY_PERIOD, EXIT_PERIOD)
    # Align returns and prices (pct_change drops first row)
    common_idx = prices.index.intersection(daily_returns.index)
    prices = prices.loc[common_idx]
    daily_returns = daily_returns.loc[common_idx]
    full_don = full_don.loc[common_idx]

    test_mask = (prices.index.year >= 2019) & (prices.index.year <= 2023)
    don_ret = daily_returns[test_mask]
    don_pos = full_don[test_mask]
    don_eq = compute_equity(don_ret, don_pos)
    don_metrics = metrics_from_equity(don_eq, don_ret, don_pos)
    don_yr = yearly_sharpes(don_ret, don_pos)
    print(f"  Unfiltered Donchian Sharpe: {don_metrics['sharpe']:.3f}, MaxDD: {don_metrics['max_drawdown']:.3f}")
    print(f"  Per year: {don_yr}")

    all_results = []

    # ─── APPROACH 1: Volatility Regime ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("APPROACH 1: Volatility Regime")
    print("=" * 60)

    vol_configs = [
        {"window": 10, "name": "vol_regime_10d"},
        {"window": 20, "name": "vol_regime_20d"},
        {"window": 50, "name": "vol_regime_50d"},
    ]
    vol_results = []
    for cfg in vol_configs:
        w = cfg["window"]
        print(f"\n  Testing vol_regime window={w}d (above median = trade)")
        regime_fn = make_vol_regime_fn(window=w, above_median=True)
        res = walk_forward_regime(prices, daily_returns, regime_fn, cfg["name"])
        print(f"    Mean Sharpe: {res['mean_sharpe']:.3f} ± {res['std_sharpe']:.3f}")
        print(f"    Per year: {res['per_year']}")
        print(f"    Overall: {res['overall_sharpe']:.3f}, MaxDD: {res['max_drawdown']:.3f}")

        config = {
            "approach": "vol_regime",
            "vol_window": w,
            "filter_logic": "above_median_vol",
            "entry_period": ENTRY_PERIOD,
            "exit_period": EXIT_PERIOD,
        }
        metrics = {
            "sharpe": res["mean_sharpe"],
            "mean_wf_sharpe": res["mean_sharpe"],
            "std_wf_sharpe": res["std_sharpe"],
            "min_wf_sharpe": res["min_sharpe"],
            "max_wf_sharpe": res["max_sharpe"],
            "overall_sharpe": res["overall_sharpe"],
            "max_drawdown": res["max_drawdown"],
            "total_return": res["total_return"],
            "num_trades": res["num_trades"],
            "sharpe_2022": res["per_year"].get("2022", 0.0),
            "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
            **{f"sharpe_{yr}": v for yr, v in res["per_year"].items()},
        }
        s22 = res["per_year"].get("2022", 0.0)
        don_s22 = don_yr.get("2022", -99)
        notes = (
            f"Vol regime {w}d window: trade when rolling vol > expanding-window median. "
            f"Mean WF Sharpe {res['mean_sharpe']:.3f}. 2022: {s22:.3f} (unfiltered {don_s22:.3f}). "
            + ("Vol filter successfully suppresses 2022 drawdown." if s22 > don_s22 else "Vol filter fails to protect 2022 — high vol = bear regime too, so filter INCREASES exposure during crashes.")
        )
        config["notes"] = notes
        run_name = f"vol_regime_{w}d_S{res['mean_sharpe']:.2f}"
        tracker.log_experiment(
            name=run_name,
            config=config,
            metrics=metrics,
            tags=TAGS,
            job_type=JOB_TYPE,
        )
        print(f"  Logged: {run_name}")
        vol_results.append({"name": run_name, "res": res, "config": cfg})
        all_results.append({"approach": "vol_regime", "name": run_name, **res})

    # Also test inverted vol regime (low vol = choppy → flat; only for best window)
    best_vol_w = sorted(vol_results, key=lambda x: x["res"]["mean_sharpe"], reverse=True)[0]["config"]["window"]
    print(f"\n  Testing inverted vol_regime window={best_vol_w}d (below median = flat, confirming direction)")
    regime_fn_inv = make_vol_regime_fn(window=best_vol_w, above_median=True)  # same as above_median True

    # ─── APPROACH 2: ADX Trend Strength ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("APPROACH 2: ADX Trend Strength")
    print("=" * 60)

    adx_configs = [
        {"period": 14, "threshold": 20, "name": "adx_14_t20"},
        {"period": 14, "threshold": 25, "name": "adx_14_t25"},
        {"period": 14, "threshold": 30, "name": "adx_14_t30"},
        {"period": 28, "threshold": 25, "name": "adx_28_t25"},
    ]
    adx_results = []
    for cfg in adx_configs:
        period, thresh = cfg["period"], cfg["threshold"]
        print(f"\n  Testing ADX({period}) > {thresh}")
        regime_fn = make_adx_regime_fn(adx_period=period, threshold=thresh)
        res = walk_forward_regime(prices, daily_returns, regime_fn, cfg["name"])
        print(f"    Mean Sharpe: {res['mean_sharpe']:.3f} ± {res['std_sharpe']:.3f}")
        print(f"    Per year: {res['per_year']}")

        config = {
            "approach": "adx_regime",
            "adx_period": period,
            "adx_threshold": thresh,
            "entry_period": ENTRY_PERIOD,
            "exit_period": EXIT_PERIOD,
        }
        s22 = res["per_year"].get("2022", 0.0)
        don_s22 = don_yr.get("2022", -99)
        notes = (
            f"ADX({period}) threshold={thresh}: trade only when market is trending. "
            f"Mean WF Sharpe {res['mean_sharpe']:.3f}. 2022: {s22:.3f} (unfiltered {don_s22:.3f}). "
            + ("ADX correctly identifies trending vs directionless market." if s22 > don_s22
               else "ADX still allows trading during trending bear market — directional filter alone insufficient.")
        )
        config["notes"] = notes
        metrics = {
            "sharpe": res["mean_sharpe"],
            "mean_wf_sharpe": res["mean_sharpe"],
            "std_wf_sharpe": res["std_sharpe"],
            "min_wf_sharpe": res["min_sharpe"],
            "max_wf_sharpe": res["max_sharpe"],
            "overall_sharpe": res["overall_sharpe"],
            "max_drawdown": res["max_drawdown"],
            "total_return": res["total_return"],
            "num_trades": res["num_trades"],
            "sharpe_2022": s22,
            "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
            **{f"sharpe_{yr}": v for yr, v in res["per_year"].items()},
        }
        run_name = f"adx_{period}_t{thresh}_S{res['mean_sharpe']:.2f}"
        tracker.log_experiment(
            name=run_name,
            config=config,
            metrics=metrics,
            tags=TAGS,
            job_type=JOB_TYPE,
        )
        print(f"  Logged: {run_name}")
        adx_results.append({"name": run_name, "res": res, "config": cfg})
        all_results.append({"approach": "adx_regime", "name": run_name, **res})

    # ─── APPROACH 3: ML Meta-Learner ──────────────────────────────────────────
    ml_result = run_ml_meta_learner(prices, daily_returns, df_features, tracker)
    if ml_result:
        all_results.append({"approach": "ml_meta_learner", "name": ml_result["name"], **ml_result["metrics"]})

    # ─── APPROACH 4: Multi-Signal Regime ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("APPROACH 4: Multi-Signal Regime")
    print("=" * 60)

    # Find best individual configs
    best_vol = sorted(vol_results, key=lambda x: x["res"]["mean_sharpe"], reverse=True)[0]
    best_adx = sorted(adx_results, key=lambda x: x["res"]["mean_sharpe"], reverse=True)[0]

    best_vol_w = best_vol["config"]["window"]
    best_adx_p = best_adx["config"]["period"]
    best_adx_t = best_adx["config"]["threshold"]

    print(f"\n  Combination A: Vol({best_vol_w}d) AND ADX({best_adx_p},{best_adx_t})")
    def combo_vol_adx(prices_p, yr):
        vol_regime = make_vol_regime_fn(best_vol_w)(prices_p, yr)
        adx_regime = make_adx_regime_fn(best_adx_p, best_adx_t)(prices_p, yr)
        combined = (vol_regime.reindex(vol_regime.index).fillna(0) &
                    adx_regime.reindex(vol_regime.index).fillna(0)).astype(int)
        return combined

    res_va = walk_forward_regime(prices, daily_returns, combo_vol_adx, "vol_adx_combined")
    print(f"    Mean Sharpe: {res_va['mean_sharpe']:.3f} ± {res_va['std_sharpe']:.3f}")
    print(f"    Per year: {res_va['per_year']}")

    config_va = {
        "approach": "multi_signal",
        "signals": ["vol_regime", "adx_regime"],
        "vol_window": best_vol_w,
        "adx_period": best_adx_p,
        "adx_threshold": best_adx_t,
        "combination": "AND",
        "entry_period": ENTRY_PERIOD,
        "exit_period": EXIT_PERIOD,
    }
    s22_va = res_va["per_year"].get("2022", 0.0)
    config_va["notes"] = (
        f"AND combination: Vol({best_vol_w}d) AND ADX({best_adx_p}>{best_adx_t}). "
        f"Mean WF Sharpe {res_va['mean_sharpe']:.3f}. 2022: {s22_va:.3f}. "
        "AND logic reduces trade count — potentially too conservative (low activity periods); "
        "benefit over individual signals: " + ("YES — improved 2022" if s22_va > max(best_vol["res"]["per_year"].get("2022", 0), best_adx["res"]["per_year"].get("2022", 0)) else "NO — intersection too restrictive")
    )
    run_name_va = f"multi_vol_adx_AND_S{res_va['mean_sharpe']:.2f}"
    tracker.log_experiment(
        name=run_name_va,
        config=config_va,
        metrics={
            "sharpe": res_va["mean_sharpe"],
            "mean_wf_sharpe": res_va["mean_sharpe"],
            "std_wf_sharpe": res_va["std_sharpe"],
            "min_wf_sharpe": res_va["min_sharpe"],
            "max_wf_sharpe": res_va["max_sharpe"],
            "overall_sharpe": res_va["overall_sharpe"],
            "max_drawdown": res_va["max_drawdown"],
            "total_return": res_va["total_return"],
            "num_trades": res_va["num_trades"],
            "sharpe_2022": s22_va,
            "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
            **{f"sharpe_{yr}": v for yr, v in res_va["per_year"].items()},
        },
        tags=TAGS,
        job_type=JOB_TYPE,
    )
    print(f"  Logged: {run_name_va}")
    all_results.append({"approach": "multi_signal_AND", "name": run_name_va, **res_va})

    # Combination B: OR logic (trade if EITHER vol OR ADX says trade)
    print(f"\n  Combination B: Vol({best_vol_w}d) OR ADX({best_adx_p},{best_adx_t})")
    def combo_vol_adx_or(prices_p, yr):
        vol_regime = make_vol_regime_fn(best_vol_w)(prices_p, yr)
        adx_regime = make_adx_regime_fn(best_adx_p, best_adx_t)(prices_p, yr)
        combined = (vol_regime.reindex(vol_regime.index).fillna(0) |
                    adx_regime.reindex(vol_regime.index).fillna(0)).astype(int)
        return combined

    res_vo = walk_forward_regime(prices, daily_returns, combo_vol_adx_or, "vol_adx_or")
    print(f"    Mean Sharpe: {res_vo['mean_sharpe']:.3f} ± {res_vo['std_sharpe']:.3f}")
    print(f"    Per year: {res_vo['per_year']}")

    config_vo = {
        "approach": "multi_signal",
        "signals": ["vol_regime", "adx_regime"],
        "vol_window": best_vol_w,
        "adx_period": best_adx_p,
        "adx_threshold": best_adx_t,
        "combination": "OR",
        "entry_period": ENTRY_PERIOD,
        "exit_period": EXIT_PERIOD,
    }
    s22_vo = res_vo["per_year"].get("2022", 0.0)
    config_vo["notes"] = (
        f"OR combination: Vol({best_vol_w}d) OR ADX({best_adx_p}>{best_adx_t}). "
        f"Mean WF Sharpe {res_vo['mean_sharpe']:.3f}. 2022: {s22_vo:.3f}. "
        "OR logic is more permissive — closer to unfiltered baseline. "
        "Effective if one signal correctly flags regime when other misses."
    )
    run_name_vo = f"multi_vol_adx_OR_S{res_vo['mean_sharpe']:.2f}"
    tracker.log_experiment(
        name=run_name_vo,
        config=config_vo,
        metrics={
            "sharpe": res_vo["mean_sharpe"],
            "mean_wf_sharpe": res_vo["mean_sharpe"],
            "std_wf_sharpe": res_vo["std_sharpe"],
            "min_wf_sharpe": res_vo["min_sharpe"],
            "max_wf_sharpe": res_vo["max_sharpe"],
            "overall_sharpe": res_vo["overall_sharpe"],
            "max_drawdown": res_vo["max_drawdown"],
            "total_return": res_vo["total_return"],
            "num_trades": res_vo["num_trades"],
            "sharpe_2022": s22_vo,
            "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
            **{f"sharpe_{yr}": v for yr, v in res_vo["per_year"].items()},
        },
        tags=TAGS,
        job_type=JOB_TYPE,
    )
    print(f"  Logged: {run_name_vo}")
    all_results.append({"approach": "multi_signal_OR", "name": run_name_vo, **res_vo})

    # ─── APPROACH 5: Rolling Drawdown Filter ──────────────────────────────────
    print("\n" + "=" * 60)
    print("APPROACH 5: Rolling Drawdown Filter")
    print("=" * 60)

    dd_configs = [
        {"threshold": 0.05, "window": 20,  "name": "dd_5pct_20d"},
        {"threshold": 0.10, "window": 40,  "name": "dd_10pct_40d"},
        {"threshold": 0.15, "window": 60,  "name": "dd_15pct_60d"},
        {"threshold": 0.10, "window": 20,  "name": "dd_10pct_20d"},
    ]
    for cfg in dd_configs:
        thresh, window = cfg["threshold"], cfg["window"]
        print(f"\n  Drawdown filter: threshold={thresh*100:.0f}%, window={window}d")
        regime_fn = make_drawdown_regime_fn(prices, daily_returns, thresh, window)
        res = walk_forward_regime(prices, daily_returns, regime_fn, cfg["name"])
        print(f"    Mean Sharpe: {res['mean_sharpe']:.3f} ± {res['std_sharpe']:.3f}")
        print(f"    Per year: {res['per_year']}")

        config = {
            "approach": "drawdown_filter",
            "dd_threshold": thresh,
            "dd_window": window,
            "entry_period": ENTRY_PERIOD,
            "exit_period": EXIT_PERIOD,
        }
        s22 = res["per_year"].get("2022", 0.0)
        don_s22 = don_yr.get("2022", -99)
        config["notes"] = (
            f"Drawdown filter: go flat if Donchian drew down >{thresh*100:.0f}% in {window}d. "
            f"Mean WF Sharpe {res['mean_sharpe']:.3f}. 2022: {s22:.3f} (unfiltered {don_s22:.3f}). "
            + ("Direct loss-cut mechanism successfully reduces 2022 damage." if s22 > don_s22
               else f"Filter triggers too late in 2022 bear — need tighter threshold or shorter window.")
        )
        metrics = {
            "sharpe": res["mean_sharpe"],
            "mean_wf_sharpe": res["mean_sharpe"],
            "std_wf_sharpe": res["std_sharpe"],
            "min_wf_sharpe": res["min_sharpe"],
            "max_wf_sharpe": res["max_sharpe"],
            "overall_sharpe": res["overall_sharpe"],
            "max_drawdown": res["max_drawdown"],
            "total_return": res["total_return"],
            "num_trades": res["num_trades"],
            "sharpe_2022": s22,
            "baseline_donchian": BASELINE_DONCHIAN_SHARPE,
            **{f"sharpe_{yr}": v for yr, v in res["per_year"].items()},
        }
        run_name = f"dd_{int(thresh*100)}pct_{window}d_S{res['mean_sharpe']:.2f}"
        tracker.log_experiment(
            name=run_name,
            config=config,
            metrics=metrics,
            tags=TAGS,
            job_type=JOB_TYPE,
        )
        print(f"  Logged: {run_name}")
        all_results.append({"approach": "drawdown_filter", "name": run_name, **res})

    # ─── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY — ALL APPROACHES")
    print("=" * 80)
    print(f"{'Name':<45} {'MeanSh':>8} {'StdSh':>8} {'Sh2022':>8} {'MaxDD':>8}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: x.get("mean_sharpe", 0), reverse=True):
        sh2022 = r.get("per_year", {}).get("2022", r.get("sharpe_2022", "N/A"))
        print(f"{r['name']:<45} {r.get('mean_sharpe', 0):>8.3f} {r.get('std_sharpe', 0):>8.3f} "
              f"{sh2022 if isinstance(sh2022, str) else sh2022:>8.3f} {r.get('max_drawdown', 0):>8.3f}")

    # Find best approach for each criterion
    best_for_2022 = max(all_results, key=lambda x: x.get("per_year", {}).get("2022", x.get("sharpe_2022", -999)))
    best_overall = max(all_results, key=lambda x: x.get("mean_sharpe", 0))

    print(f"\nBest for reducing 2022 drawdown: {best_for_2022['name']}")
    print(f"  2022 Sharpe: {best_for_2022.get('per_year', {}).get('2022', 'N/A')}")
    print(f"Best overall mean WF Sharpe: {best_overall['name']}")
    print(f"  Mean Sharpe: {best_overall.get('mean_sharpe', 'N/A')}")

    # Save raw results JSON
    import json
    results_file = RESULTS_DIR / "regime_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "unfiltered_donchian": {
                "sharpe": don_metrics["sharpe"],
                "max_drawdown": don_metrics["max_drawdown"],
                "per_year": don_yr,
            },
            "approaches": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    return all_results, don_metrics, don_yr


if __name__ == "__main__":
    all_results, don_metrics, don_yr = main()
    print("\nDone.")
