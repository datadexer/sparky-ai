#!/usr/bin/env python3
"""Regime-aware Donchian strategy — CONTRACT #004 Step 3.

Tests hypothesis: Filtering Donchian to 'trending' regimes eliminates 2022 catastrophe.

Baseline: Multi-TF Donchian Sharpe 1.062 (corrected, in-sample 2019-2023)
Best ML:  LightGBM top-10 Sharpe 1.365 (but -0.644 in 2022, std=1.701)

5 distinct regime approaches:
  1. Volatility threshold (high vol → trend, low vol → flat)
  2. ADX-based trend strength (ADX > threshold → trade)
  3. ML meta-learner (CatBoost classifies Donchian-profit vs Donchian-loss periods)
  4. Multi-signal combination (vol AND adx, ML + vol/adx)
  5. Rolling drawdown filter (go flat after strategy drawdown > X%)
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparky.data.loader import load
from sparky.tracking.experiment import ExperimentTracker
from sparky.models.simple_baselines import donchian_channel_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASELINE_SHARPE = 1.062
BEST_ML_SHARPE = 1.365
ANNUALIZE = np.sqrt(365)  # Daily data → annualize


# ============================================================
# DATA LOADING
# ============================================================


def load_merged_daily() -> pd.DataFrame:
    """Load OHLCV daily + feature matrix, merge on date, restrict 2019-2023."""
    df_daily = load("btc_daily", purpose="training")
    df_feat = load("feature_matrix_btc_hourly_expanded", purpose="training")

    # Restrict to in-sample window
    df_daily = df_daily[(df_daily.index.year >= 2019) & (df_daily.index.year <= 2023)]
    df_feat = df_feat[(df_feat.index.year >= 2019) & (df_feat.index.year <= 2023)]

    # Inner join on date index
    merged = df_daily.join(df_feat, how="inner")
    logger.info(f"Merged dataset: {merged.shape[0]} daily rows, {merged.shape[1]} cols, "
                f"{merged.index.min().date()} to {merged.index.max().date()}")
    return merged


# ============================================================
# BACKTEST ENGINE (shift(1) enforced — no look-ahead)
# ============================================================


def backtest_daily(signals: pd.Series, close: pd.Series) -> Dict[str, float]:
    """Backtest strategy signals on daily closes. Returns annual Sharpe + stats."""
    returns = close.pct_change()
    # CRITICAL: signal[T] uses close[T]; we fill position NEXT day (T+1)
    strat_returns = signals.shift(1).fillna(0) * returns
    strat_returns = strat_returns.dropna()

    if len(strat_returns) == 0 or strat_returns.std() == 0:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "time_in_market": 0.0}

    cum = (1 + strat_returns).cumprod()
    total_return = float(cum.iloc[-1] - 1)
    sharpe = float(strat_returns.mean() / strat_returns.std() * ANNUALIZE)
    running_max = cum.cummax()
    max_dd = float(((cum - running_max) / running_max).min())
    win_rate = float((strat_returns > 0).sum() / (strat_returns != 0).sum()) if (strat_returns != 0).sum() > 0 else 0.0
    time_in_market = float((signals.shift(1).fillna(0) != 0).mean())

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "time_in_market": time_in_market,
    }


def backtest_per_year(signals: pd.Series, close: pd.Series) -> Dict[int, Dict[str, float]]:
    """Compute per-year metrics — key test is 2022."""
    results = {}
    for yr in [2019, 2020, 2021, 2022, 2023]:
        mask = close.index.year == yr
        if mask.sum() < 20:
            continue
        yr_sig = signals[mask]
        yr_close = close[mask]
        results[yr] = backtest_daily(yr_sig, yr_close)
    return results


# ============================================================
# WALK-FORWARD VALIDATION (expanding window, yearly test folds)
# ============================================================


def walk_forward_regime(
    df: pd.DataFrame,
    regime_fn,  # callable(train_df, test_df) → test_regime Series (0=flat, 1=trade)
    entry_period: int = 20,
    exit_period: int = 10,
) -> Dict:
    """
    Expanding walk-forward validation with yearly test folds.
    Train: 2019...(year-1), Test: year (for years 2020..2023).
    Returns mean/std Sharpe across folds + per-year breakdown.
    """
    test_years = [2020, 2021, 2022, 2023]
    fold_metrics = []
    per_year = {}

    for test_yr in test_years:
        train_df = df[df.index.year < test_yr].copy()
        test_df = df[df.index.year == test_yr].copy()

        if len(train_df) < 100 or len(test_df) < 20:
            logger.warning(f"Skipping {test_yr}: insufficient data")
            continue

        # Get regime for test period (NO look-ahead: regime_fn trains on train_df)
        try:
            regime_mask = regime_fn(train_df, test_df)  # 0/1 Series indexed like test_df
        except Exception as e:
            logger.error(f"Regime fn failed for {test_yr}: {e}")
            regime_mask = pd.Series(1, index=test_df.index)  # fallback: always trade

        # Base Donchian signals on full data up to and including test year
        # (to compute proper channel levels using history)
        full_up_to_test = df[df.index.year <= test_yr].copy()
        all_signals = donchian_channel_strategy(
            full_up_to_test["close"], entry_period=entry_period, exit_period=exit_period
        )
        test_signals = all_signals[all_signals.index.year == test_yr]

        # Apply regime filter
        regime_mask = regime_mask.reindex(test_signals.index).fillna(0)
        filtered_signals = test_signals * regime_mask

        m = backtest_daily(filtered_signals, test_df["close"])
        m["year"] = test_yr
        fold_metrics.append(m)
        per_year[test_yr] = m

    if not fold_metrics:
        return {"mean_sharpe": 0.0, "std_sharpe": 0.0, "per_year": {}, "fold_metrics": []}

    sharpes = [m["sharpe"] for m in fold_metrics]
    returns = [m["total_return"] for m in fold_metrics]
    dds = [m["max_drawdown"] for m in fold_metrics]

    return {
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "min_sharpe": float(np.min(sharpes)),
        "max_sharpe": float(np.max(sharpes)),
        "mean_return": float(np.mean(returns)),
        "mean_max_dd": float(np.mean(dds)),
        "worst_year_sharpe": float(np.min(sharpes)),
        "per_year": per_year,
        "fold_metrics": fold_metrics,
    }


# ============================================================
# APPROACH 1: VOLATILITY THRESHOLD REGIME
# ============================================================


def vol_regime_fn(window: int, threshold_multiplier: float = 1.0):
    """High rolling volatility = trending, go flat in low-vol chop."""
    def fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
        # Compute vol on full train history to get stable median
        returns_train = train_df["close"].pct_change()
        vol_train = returns_train.rolling(window, min_periods=window // 2).std()
        # Use training-period median as threshold (no look-ahead into test)
        vol_median = vol_train.median() * threshold_multiplier

        returns_test = test_df["close"].pct_change()
        vol_test = returns_test.rolling(window, min_periods=window // 2).std()

        # Trade when vol is ABOVE threshold (trending/volatile regime)
        regime = (vol_test > vol_median).astype(int).fillna(0)
        pct = regime.mean() * 100
        logger.info(f"Vol regime (w={window}): {pct:.1f}% in-market")
        return regime
    return fn


# ============================================================
# APPROACH 2: ADX-BASED TREND STRENGTH
# ============================================================


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX). Returns ADX series."""
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    # Wilder smoothing
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pos_di = 100 * pos_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    neg_di = 100 * neg_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-10)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def adx_regime_fn(adx_period: int, adx_threshold: float):
    """ADX > threshold = trending (directional), trade. Below = choppy, flat."""
    def fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
        # Need OHLC: use training + test for ADX continuity
        combined = pd.concat([train_df, test_df])
        adx_all = compute_adx(combined["high"], combined["low"], combined["close"], period=adx_period)
        adx_test = adx_all[adx_all.index.isin(test_df.index)]
        regime = (adx_test > adx_threshold).astype(int).fillna(0)
        pct = regime.mean() * 100
        logger.info(f"ADX regime (period={adx_period}, thresh={adx_threshold}): {pct:.1f}% in-market")
        return regime
    return fn


# ============================================================
# APPROACH 3: ML META-LEARNER (CatBoost, GPU)
# ============================================================


def make_donchian_labels(
    df: pd.DataFrame, entry_period: int = 20, exit_period: int = 10, lookahead: int = 20
) -> pd.Series:
    """
    META-LEARNING labels: Was Donchian PROFITABLE in the next `lookahead` days?
    For each day T, compute Donchian PnL from T to T+lookahead.
    Label 1 = profitable period (trade), 0 = unprofitable (flat).
    All look-ahead here is OK because these labels are ONLY used for training,
    and the model predicts from features (no future data leaked at inference).
    """
    signals = donchian_channel_strategy(df["close"], entry_period=entry_period, exit_period=exit_period)
    returns = df["close"].pct_change()
    strat_returns = signals.shift(1).fillna(0) * returns

    # Rolling sum of future returns over `lookahead` days
    future_pnl = strat_returns.rolling(lookahead, min_periods=lookahead // 2).sum().shift(-lookahead)
    labels = (future_pnl > 0).astype(int)
    return labels


def ml_meta_learner_fn(
    top_n_features: int = 20,
    entry_period: int = 20,
    exit_period: int = 10,
    label_lookahead: int = 20,
    depth: int = 3,
    lr: float = 0.05,
    n_iter: int = 300,
):
    """ML meta-learner: CatBoost trains to predict when Donchian profits."""
    def fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
        import catboost as cb

        feature_cols = [
            c for c in train_df.columns
            if c not in ["open", "high", "low", "close", "volume"]
            and train_df[c].dtype in [np.float64, np.float32, float, int, np.int64, np.int32]
        ]
        feature_cols = feature_cols[:top_n_features]

        if len(feature_cols) < 5:
            logger.warning("Not enough features for ML meta-learner — using all-trade fallback")
            return pd.Series(1, index=test_df.index)

        # Build training labels (meta-labels: profitable Donchian periods)
        y_train = make_donchian_labels(train_df, entry_period=entry_period,
                                        exit_period=exit_period, lookahead=label_lookahead)
        X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        # Align
        valid_mask = y_train.notna()
        X_train_clean = X_train[valid_mask]
        y_train_clean = y_train[valid_mask]

        if len(X_train_clean) < 100:
            logger.warning("Not enough clean training rows — skipping ML, all-trade fallback")
            return pd.Series(1, index=test_df.index)

        model = cb.CatBoostClassifier(
            iterations=n_iter,
            depth=depth,
            learning_rate=lr,
            task_type="GPU",
            devices="0",
            l2_leaf_reg=1.0,
            verbose=0,
            random_seed=42,
        )
        model.fit(X_train_clean, y_train_clean)

        X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        pred = model.predict(X_test)
        regime = pd.Series(pred.astype(int), index=test_df.index)
        pct = regime.mean() * 100
        logger.info(f"ML meta-learner regime: {pct:.1f}% in-market")
        return regime
    return fn


# ============================================================
# APPROACH 4: MULTI-SIGNAL COMBINATION
# ============================================================


def multi_signal_fn(regime_fns: List, mode: str = "all"):
    """
    Combine multiple regime signals.
    mode='all': trade only when ALL regimes say trade (AND logic, most conservative)
    mode='any': trade when ANY regime says trade (OR logic, more permissive)
    """
    def fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
        signals = [rfn(train_df, test_df) for rfn in regime_fns]
        combined = pd.concat(signals, axis=1)
        combined = combined.reindex(test_df.index).fillna(0)

        if mode == "all":
            regime = (combined.sum(axis=1) == len(regime_fns)).astype(int)
        elif mode == "any":
            regime = (combined.sum(axis=1) >= 1).astype(int)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        pct = regime.mean() * 100
        logger.info(f"Multi-signal ({mode}, {len(regime_fns)} signals): {pct:.1f}% in-market")
        return regime
    return fn


# ============================================================
# APPROACH 5: ROLLING DRAWDOWN FILTER
# ============================================================


def rolling_drawdown_filter_fn(
    dd_threshold: float = 0.10,
    lookback_days: int = 40,
    entry_period: int = 20,
    exit_period: int = 10,
    cooldown_days: int = 10,
):
    """
    Go flat if Donchian strategy has drawn down > dd_threshold in last lookback_days.
    Uses training history to compute rolling strategy performance.
    This is forward-pass safe: at test time T, only history up to T-1 is used.
    """
    def fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
        combined = pd.concat([train_df, test_df])
        signals_all = donchian_channel_strategy(
            combined["close"], entry_period=entry_period, exit_period=exit_period
        )
        returns_all = combined["close"].pct_change()
        strat_returns_all = signals_all.shift(1).fillna(0) * returns_all

        # Running cumulative return (rolling peak)
        cum_returns = (1 + strat_returns_all).cumprod()
        rolling_peak = cum_returns.rolling(lookback_days, min_periods=1).max()
        rolling_dd = (cum_returns - rolling_peak) / rolling_peak

        # In-market: go flat when drawdown exceeds threshold
        regime_all = (rolling_dd > -dd_threshold).astype(int)

        # Cooldown: once triggered, stay flat for cooldown_days
        if cooldown_days > 0:
            flat_trigger = (rolling_dd <= -dd_threshold).astype(int)
            # Propagate flat signal forward for cooldown_days
            flat_extended = flat_trigger.rolling(cooldown_days, min_periods=1).max()
            regime_all = (1 - flat_extended).astype(int).clip(0, 1)

        # Return only test period
        regime_test = regime_all[regime_all.index.isin(test_df.index)]
        regime_test = regime_test.reindex(test_df.index).fillna(0)
        pct = regime_test.mean() * 100
        logger.info(f"Drawdown filter (dd={dd_threshold:.0%}, lb={lookback_days}d): {pct:.1f}% in-market")
        return regime_test
    return fn


# ============================================================
# LOGGING HELPERS
# ============================================================


def log_regime_run(
    tracker: ExperimentTracker,
    name: str,
    config: dict,
    results: dict,
    interpretation: str,
    tags: List[str] = None,
) -> str:
    """Log a single regime experiment run to wandb."""
    per_year = results.get("per_year", {})
    metrics = {
        "mean_sharpe": results["mean_sharpe"],
        "std_sharpe": results["std_sharpe"],
        "min_sharpe": results.get("min_sharpe", 0.0),
        "max_sharpe": results.get("max_sharpe", 0.0),
        "mean_return": results["mean_return"],
        "mean_max_dd": results["mean_max_dd"],
        "beats_baseline": float(results["mean_sharpe"] > BASELINE_SHARPE),
        "sharpe_2022": float(per_year.get(2022, {}).get("sharpe", 0.0)),
        "sharpe_2019": float(per_year.get(2019, {}).get("sharpe", 0.0)),
        "sharpe_2020": float(per_year.get(2020, {}).get("sharpe", 0.0)),
        "sharpe_2021": float(per_year.get(2021, {}).get("sharpe", 0.0)),
        "sharpe_2023": float(per_year.get(2023, {}).get("sharpe", 0.0)),
        "baseline_sharpe": BASELINE_SHARPE,
        "best_ml_sharpe": BEST_ML_SHARPE,
    }

    all_tags = ["contract_004", "regime"] + (tags or [])
    run_id = tracker.log_experiment(
        name=name,
        config=config,
        metrics=metrics,
        tags=all_tags,
        job_type="regime",
    )
    logger.info(f"  → wandb run: {name} | Sharpe: {results['mean_sharpe']:.3f} ± {results['std_sharpe']:.3f}")
    logger.info(f"    2022 Sharpe: {metrics['sharpe_2022']:.3f} | Interpretation: {interpretation}")
    return run_id


# ============================================================
# MAIN SWEEP
# ============================================================


def main():
    start = time.time()
    logger.info("=" * 70)
    logger.info("CONTRACT #004 STEP 3 — REGIME-AWARE DONCHIAN")
    logger.info(f"Baseline: Donchian Sharpe {BASELINE_SHARPE} | Best ML: {BEST_ML_SHARPE}")
    logger.info("=" * 70)

    df = load_merged_daily()
    tracker = ExperimentTracker(experiment_name="regime_aware_donchian")

    all_results = []

    # ----------------------------------------------------------
    # APPROACH 1: VOLATILITY THRESHOLD
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("APPROACH 1: VOLATILITY THRESHOLD REGIME")
    logger.info("=" * 60)

    vol_configs = [
        {"window": 10, "multiplier": 1.0, "entry": 20, "exit": 10},
        {"window": 20, "multiplier": 1.0, "entry": 20, "exit": 10},
        {"window": 50, "multiplier": 1.0, "entry": 40, "exit": 20},
    ]

    for vc in vol_configs:
        cfg = {
            "approach": "vol_threshold",
            "vol_window": vc["window"],
            "vol_multiplier": vc["multiplier"],
            "entry_period": vc["entry"],
            "exit_period": vc["exit"],
        }
        name = f"vol_regime_{vc['window']}d_e{vc['entry']}"
        logger.info(f"\nRunning: {name}")

        results = walk_forward_regime(
            df,
            regime_fn=vol_regime_fn(vc["window"], vc["multiplier"]),
            entry_period=vc["entry"],
            exit_period=vc["exit"],
        )

        interpretation = (
            f"Vol({vc['window']}d) filter: {results['mean_sharpe']:.2f} Sharpe. "
            f"2022: {results['per_year'].get(2022, {}).get('sharpe', 0):.2f}. "
            + ("Reduces 2022 damage vs unfiltered." if results['per_year'].get(2022, {}).get('sharpe', -99) > -1.0 else "2022 still bad — vol not regime-discriminative.")
        )

        log_regime_run(tracker, name=name, config=cfg, results=results,
                       interpretation=interpretation, tags=["vol_regime"])
        all_results.append({"approach": "vol_threshold", "name": name,
                             "config": cfg, "results": results})

    # ----------------------------------------------------------
    # APPROACH 2: ADX-BASED TREND STRENGTH
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("APPROACH 2: ADX TREND STRENGTH REGIME")
    logger.info("=" * 60)

    adx_configs = [
        {"period": 14, "threshold": 20, "entry": 20, "exit": 10},
        {"period": 14, "threshold": 25, "entry": 20, "exit": 10},
        {"period": 14, "threshold": 30, "entry": 40, "exit": 20},
    ]

    for ac in adx_configs:
        cfg = {
            "approach": "adx_trend",
            "adx_period": ac["period"],
            "adx_threshold": ac["threshold"],
            "entry_period": ac["entry"],
            "exit_period": ac["exit"],
        }
        name = f"adx_regime_p{ac['period']}_t{ac['threshold']}"
        logger.info(f"\nRunning: {name}")

        results = walk_forward_regime(
            df,
            regime_fn=adx_regime_fn(ac["period"], ac["threshold"]),
            entry_period=ac["entry"],
            exit_period=ac["exit"],
        )

        sharpe_2022 = results['per_year'].get(2022, {}).get('sharpe', 0)
        interpretation = (
            f"ADX({ac['period']}, thresh={ac['threshold']}): {results['mean_sharpe']:.2f} Sharpe. "
            f"2022: {sharpe_2022:.2f}. "
            + ("ADX successfully avoids chop/bear." if sharpe_2022 > -0.5 else "ADX insufficient to identify bear regime — market had directional momentum in 2022 decline.")
        )

        log_regime_run(tracker, name=name, config=cfg, results=results,
                       interpretation=interpretation, tags=["adx_regime"])
        all_results.append({"approach": "adx_trend", "name": name,
                             "config": cfg, "results": results})

    # ----------------------------------------------------------
    # APPROACH 3: ML META-LEARNER (CatBoost GPU)
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("APPROACH 3: ML META-LEARNER (CATBOOST)")
    logger.info("=" * 60)

    ml_configs = [
        {"top_n": 20, "entry": 20, "exit": 10, "lookahead": 20, "depth": 3, "lr": 0.05, "n": 300},
        {"top_n": 20, "entry": 20, "exit": 10, "lookahead": 10, "depth": 4, "lr": 0.01, "n": 500},
    ]

    for mc in ml_configs:
        cfg = {
            "approach": "ml_meta",
            "top_n_features": mc["top_n"],
            "entry_period": mc["entry"],
            "exit_period": mc["exit"],
            "label_lookahead": mc["lookahead"],
            "depth": mc["depth"],
            "lr": mc["lr"],
            "n_iter": mc["n"],
        }
        name = f"ml_meta_lh{mc['lookahead']}_d{mc['depth']}_lr{mc['lr']}"
        logger.info(f"\nRunning: {name}")

        results = walk_forward_regime(
            df,
            regime_fn=ml_meta_learner_fn(
                top_n_features=mc["top_n"],
                entry_period=mc["entry"],
                exit_period=mc["exit"],
                label_lookahead=mc["lookahead"],
                depth=mc["depth"],
                lr=mc["lr"],
                n_iter=mc["n"],
            ),
            entry_period=mc["entry"],
            exit_period=mc["exit"],
        )

        sharpe_2022 = results['per_year'].get(2022, {}).get('sharpe', 0)
        interpretation = (
            f"ML meta-learner (lh={mc['lookahead']}d, d={mc['depth']}): "
            f"{results['mean_sharpe']:.2f} Sharpe. 2022: {sharpe_2022:.2f}. "
            + ("Meta-learner successfully identifies non-profitable Donchian periods." if sharpe_2022 > -0.3
               else "Meta-labels contain future leakage in label construction but NOT at inference — still saw 2022 loss, model misjudged regime.")
        )

        log_regime_run(tracker, name=name, config=cfg, results=results,
                       interpretation=interpretation, tags=["ml_meta"])
        all_results.append({"approach": "ml_meta", "name": name,
                             "config": cfg, "results": results})

    # ----------------------------------------------------------
    # APPROACH 4: MULTI-SIGNAL COMBINATION
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("APPROACH 4: MULTI-SIGNAL COMBINATION")
    logger.info("=" * 60)

    # Pick best vol and adx params from approach 1/2 (use window=20, adx=25 as representative)
    # Combo A: Vol AND ADX (both must agree = AND logic, more conservative)
    combo_a_cfg = {
        "approach": "multi_signal_and",
        "signals": ["vol_20d", "adx_25"],
        "mode": "all",
        "entry_period": 20,
        "exit_period": 10,
    }
    name_a = "multi_vol20_adx25_AND"
    logger.info(f"\nRunning: {name_a}")

    results_a = walk_forward_regime(
        df,
        regime_fn=multi_signal_fn(
            [
                vol_regime_fn(window=20, threshold_multiplier=1.0),
                adx_regime_fn(adx_period=14, adx_threshold=25),
            ],
            mode="all",
        ),
        entry_period=20,
        exit_period=10,
    )

    sharpe_2022_a = results_a['per_year'].get(2022, {}).get('sharpe', 0)
    interpretation_a = (
        f"Vol(20d) AND ADX(25): {results_a['mean_sharpe']:.2f} Sharpe. "
        f"2022: {sharpe_2022_a:.2f}. "
        + ("Strict AND filter severely reduces time-in-market — less capital at risk but also fewer profits." if results_a.get("fold_metrics") else "No fold results.")
    )
    log_regime_run(tracker, name=name_a, config=combo_a_cfg, results=results_a,
                   interpretation=interpretation_a, tags=["multi_signal"])
    all_results.append({"approach": "multi_signal", "name": name_a,
                         "config": combo_a_cfg, "results": results_a})

    # Combo B: ML meta-learner with vol as additional feature (use OR mode for coverage)
    combo_b_cfg = {
        "approach": "multi_signal_ml_vol",
        "signals": ["ml_meta", "vol_20d"],
        "mode": "any",
        "entry_period": 20,
        "exit_period": 10,
    }
    name_b = "multi_ml_meta_OR_vol20"
    logger.info(f"\nRunning: {name_b}")

    results_b = walk_forward_regime(
        df,
        regime_fn=multi_signal_fn(
            [
                ml_meta_learner_fn(top_n_features=20, entry_period=20, exit_period=10,
                                   label_lookahead=20, depth=3, lr=0.05, n_iter=300),
                vol_regime_fn(window=20, threshold_multiplier=1.0),
            ],
            mode="any",
        ),
        entry_period=20,
        exit_period=10,
    )

    sharpe_2022_b = results_b['per_year'].get(2022, {}).get('sharpe', 0)
    interpretation_b = (
        f"ML OR Vol(20d): {results_b['mean_sharpe']:.2f} Sharpe. "
        f"2022: {sharpe_2022_b:.2f}. "
        + ("OR combination gives ML a vol-regime safety net — if either says trade, we trade." if sharpe_2022_b > -0.5
           else "OR logic too permissive — inherits ML/vol failures without reducing them.")
    )
    log_regime_run(tracker, name=name_b, config=combo_b_cfg, results=results_b,
                   interpretation=interpretation_b, tags=["multi_signal"])
    all_results.append({"approach": "multi_signal", "name": name_b,
                         "config": combo_b_cfg, "results": results_b})

    # ----------------------------------------------------------
    # APPROACH 5: ROLLING DRAWDOWN FILTER
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("APPROACH 5: ROLLING DRAWDOWN FILTER")
    logger.info("=" * 60)

    dd_configs = [
        {"dd": 0.05, "lb": 20, "cooldown": 10, "entry": 20, "exit": 10},
        {"dd": 0.10, "lb": 40, "cooldown": 15, "entry": 20, "exit": 10},
        {"dd": 0.15, "lb": 60, "cooldown": 20, "entry": 40, "exit": 20},
    ]

    for dc in dd_configs:
        cfg = {
            "approach": "drawdown_filter",
            "dd_threshold": dc["dd"],
            "lookback_days": dc["lb"],
            "cooldown_days": dc["cooldown"],
            "entry_period": dc["entry"],
            "exit_period": dc["exit"],
        }
        name = f"dd_filter_{int(dc['dd']*100)}pct_{dc['lb']}d"
        logger.info(f"\nRunning: {name}")

        results = walk_forward_regime(
            df,
            regime_fn=rolling_drawdown_filter_fn(
                dd_threshold=dc["dd"],
                lookback_days=dc["lb"],
                entry_period=dc["entry"],
                exit_period=dc["exit"],
                cooldown_days=dc["cooldown"],
            ),
            entry_period=dc["entry"],
            exit_period=dc["exit"],
        )

        sharpe_2022 = results['per_year'].get(2022, {}).get('sharpe', 0)
        interpretation = (
            f"DD filter ({dc['dd']:.0%}, {dc['lb']}d): {results['mean_sharpe']:.2f} Sharpe. "
            f"2022: {sharpe_2022:.2f}. "
            + ("Drawdown filter successfully cuts off losing streaks — 2022 protected." if sharpe_2022 > 0
               else "Drawdown filter exits too late — 2022 loss already locked in before trigger fires." if sharpe_2022 > -0.5
               else "Drawdown filter insufficient — 2022 bear too deep to avoid with this threshold.")
        )

        log_regime_run(tracker, name=name, config=cfg, results=results,
                       interpretation=interpretation, tags=["dd_filter"])
        all_results.append({"approach": "drawdown_filter", "name": name,
                             "config": cfg, "results": results})

    # ----------------------------------------------------------
    # RESULTS SUMMARY
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("REGIME SWEEP COMPLETE — SUMMARY TABLE")
    logger.info("=" * 70)

    valid = [r for r in all_results if r["results"].get("mean_sharpe") is not None]
    valid.sort(key=lambda r: r["results"]["mean_sharpe"], reverse=True)

    logger.info(f"{'Name':<40} {'Sharpe':>7} {'Std':>6} {'2022':>7} {'In-Mkt':>7}")
    logger.info("-" * 70)
    for r in valid:
        res = r["results"]
        per_yr = res.get("per_year", {})
        sharpe_2022 = per_yr.get(2022, {}).get("sharpe", 0)
        # Estimate time in market from fold metrics
        tims = [m.get("time_in_market", 1.0) for m in res.get("fold_metrics", [])]
        tim = np.mean(tims) if tims else 1.0
        logger.info(f"{r['name']:<40} {res['mean_sharpe']:>7.3f} {res['std_sharpe']:>6.3f} {sharpe_2022:>7.3f} {tim*100:>6.1f}%")

    logger.info("-" * 70)
    logger.info(f"{'Unfiltered Donchian baseline':<40} {BASELINE_SHARPE:>7.3f}  (ref)")
    logger.info(f"{'Best ML (LightGBM step 2)':<40} {BEST_ML_SHARPE:>7.3f} {1.701:>6.3f} {-0.644:>7.3f}")

    # Save results JSON
    results_dir = Path("/home/akamath/sparky-ai/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "regime_results.json"

    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return obj

    with open(out_file, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)
    logger.info(f"\nResults saved: {out_file}")

    elapsed = time.time() - start
    logger.info(f"Total time: {elapsed/60:.1f} minutes | {len(all_results)} runs logged")

    return all_results


if __name__ == "__main__":
    main()
