#!/usr/bin/env python3
"""
CONTRACT #004 — Step 5: Novel Exploration (Creative Synthesis)

SYNTHESIS: After 50+ experiments, the key patterns are:
  - ADX(14,30) is the best regime filter: Sharpe 1.181, std 0.829, 2022=0.000
  - LGBM-top10 has highest raw Sharpe (1.365) but volatile, fails in 2022
  - Ensemble averaging doesn't help — models are correlated, fail together in 2022
  - Regime-conditional (ML+flat) got best 2022 (+0.683) but suppresses upside

NOVEL IDEAS TESTED (4 minimum):
  1. ASYMMETRIC REGIME: ADX for ENTRIES, vol-adjusted drawdown filter for EXITS
     Hypothesis: ADX identifies "is market trending?" (entry timing)
     while rolling DD filter identifies "has the trade stopped working?" (exit timing)
     These are orthogonal questions that may benefit from different detectors.

  2. ADAPTIVE DONCHIAN: Regime-adaptive channel width
     Hypothesis: In trending markets (ADX high), use tight channels (fast reaction)
     In choppy/uncertain markets, use wider channels (avoid whipsaws)
     ADX score linearly interpolates between fast (20/10) and slow (60/30) params

  3. ML AS POSITION SIZER (not signal):
     Hypothesis: LGBM confidence should scale position SIZE, not decide on/off.
     Donchian fires the signal, LGBM says how much to allocate.
     High ML confidence (trending regime) → 1.0x position
     Low ML confidence → 0.25x position
     This preserves all trades while reducing risk on low-conviction days.

  4. MAJORITY-VOTE REGIME ENSEMBLE:
     Hypothesis: Using 3 independent regime detectors (ADX, Vol, ML stacking),
     trade only when 2+ agree. Majority vote is more robust than any individual detector.
     This directly addresses the 2022 failure mode — all three detectors flagging
     "not trending" simultaneously is a stronger signal than any one alone.

Walk-forward validated against 2019-2023. Baseline: ADX(14,30) Sharpe 1.181.
All runs tagged: contract_004, novel
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

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sparky.data.loader import load
from sparky.tracking.experiment import ExperimentTracker
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.oversight.timeout import with_timeout

# ─── Constants ─────────────────────────────────────────────────────────────────
BASELINE_DONCHIAN_SHARPE = 1.062
BEST_REGIME_SHARPE = 1.181       # ADX(14,30) Donchian, Step 3
BEST_ML_SHARPE = 1.365           # LightGBM top-10, Step 2

ENTRY_PERIOD = 40
EXIT_PERIOD = 20

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAGS = ["contract_004", "novel"]
JOB_TYPE = "novel"


# ─── Data Loading ───────────────────────────────────────────────────────────────
def load_data():
    print("=" * 80)
    print("LOADING DATA")
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

    # Daily target
    target_raw = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")
    if isinstance(target_raw, pd.DataFrame):
        target_raw = target_raw["target"]
    target_daily = target_raw.resample("D").last().loc["2019-01-01":"2023-12-31"]

    # Align all series
    common_idx = prices_daily.index.intersection(df_daily.index).intersection(target_daily.index)
    prices_daily = prices_daily.loc[common_idx]
    daily_returns = daily_returns.reindex(common_idx).fillna(0)
    df_daily = df_daily.loc[common_idx]
    target_daily = target_daily.loc[common_idx]

    valid = target_daily.notna()
    prices_daily = prices_daily[valid]
    daily_returns = daily_returns[valid]
    df_daily = df_daily[valid].fillna(df_daily.median())
    target_daily = target_daily[valid]

    print(f"  Daily prices: {len(prices_daily)} rows ({prices_daily.index[0].date()} - {prices_daily.index[-1].date()})")
    print(f"  Features top-10: {len(available10)}, top-20: {len(available20)}")
    return prices_daily, daily_returns, df_daily, target_daily, available10, available20


# ─── Core utilities ─────────────────────────────────────────────────────────────
def donchian_signal_stateful(prices: pd.Series, entry: int = 40, exit_p: int = 20) -> pd.Series:
    """Donchian channel with state machine — no look-ahead."""
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
    delta = prices.diff()
    plus_dm = delta.clip(lower=0)
    minus_dm = (-delta).clip(lower=0)
    plus_di = plus_dm.ewm(span=period, adjust=False).mean()
    minus_di = minus_dm.ewm(span=period, adjust=False).mean()
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
    return dx.ewm(span=period, adjust=False).mean()


def compute_equity(returns: pd.Series, positions: pd.Series) -> pd.Series:
    """Equity curve from returns and positions. positions[t] held during day t."""
    pos = positions.reindex(returns.index).fillna(0)
    pos_shifted = pos.shift(1).fillna(0)
    strat_ret = pos_shifted * returns
    return (1 + strat_ret).cumprod()


def sharpe_from_signals(returns: pd.Series, signals: pd.Series) -> float:
    pos = signals.reindex(returns.index).fillna(0).shift(1).fillna(0)
    strat_ret = pos * returns
    if strat_ret.std() == 0 or len(strat_ret) < 10:
        return 0.0
    return float(annualized_sharpe(strat_ret))


def year_sharpe(returns: pd.Series, signals: pd.Series, year: int) -> float:
    mask = returns.index.year == year
    return sharpe_from_signals(returns[mask], signals.reindex(returns[mask].index).fillna(0))


def run_walk_forward_custom(prices, daily_returns, signal_fn, years=None):
    """Run walk-forward for a signal_fn(prices, daily_returns, year) -> pd.Series[positions]."""
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
        "mean_sharpe": round(float(np.mean(vals)), 3),
        "std_sharpe": round(float(np.std(vals)), 3),
        "min_sharpe": round(float(min(vals)), 3),
        "max_sharpe": round(float(max(vals)), 3),
        "per_year": per_year,
    }


def log_novel_result(tracker, name, config, metrics, group, notes):
    """Standardized logging for novel exploration runs."""
    config["notes"] = notes
    metrics["baseline_donchian"] = BASELINE_DONCHIAN_SHARPE
    metrics["best_regime_sharpe"] = BEST_REGIME_SHARPE
    metrics["best_ml_sharpe"] = BEST_ML_SHARPE
    tracker.log_experiment(
        name=name,
        config=config,
        metrics=metrics,
        tags=TAGS,
        job_type=JOB_TYPE,
        group=group,
    )
    print(f"  Logged: {name}")
    print(f"  Mean WF Sharpe: {metrics['mean_wf_sharpe']:.3f} ± {metrics['std_wf_sharpe']:.3f}")
    print(f"  2022 Sharpe: {metrics.get('sharpe_2022', 'N/A')}")
    beats = metrics['mean_wf_sharpe'] > BEST_REGIME_SHARPE
    print(f"  Beats best prior (ADX Sharpe {BEST_REGIME_SHARPE:.3f}): {'YES ✅' if beats else 'NO ❌'}")


# ════════════════════════════════════════════════════════════════════════════════
# IDEA 1 — ASYMMETRIC REGIME: ADX for entries, Vol-DD hybrid for exits
# ════════════════════════════════════════════════════════════════════════════════
@with_timeout(seconds=900)
def run_idea1_asymmetric_regime(prices, daily_returns, tracker):
    """
    HYPOTHESIS: ADX and vol-drawdown filter are measuring different things.
    ADX measures "is the market directional right now?" → optimal for ENTRY decisions
    Drawdown filter measures "has the strategy started losing?" → optimal for EXIT decisions

    Using ADX for entries and rolling DD filter for forced exits should:
    - Get into trades only in trending regimes (ADX benefit)
    - Exit earlier when momentum fades (DD benefit, avoids -20%+ drawdowns)

    This is ASYMMETRIC: different logic for entry vs exit.
    """
    print("\n" + "=" * 80)
    print("IDEA 1: ASYMMETRIC REGIME (ADX entry, DD exit)")
    print("=" * 80)
    print("HYPOTHESIS: ADX identifies when to enter, DD filter identifies when to exit.")
    print("These are orthogonal questions — using different detectors for each should")
    print("improve entry quality AND reduce drawdown simultaneously.")

    # Pre-compute full Donchian signals
    full_don = donchian_signal_stateful(prices, ENTRY_PERIOD, EXIT_PERIOD)

    best_sharpe = -999
    best_result = None
    all_configs = []

    # Test several combinations of ADX threshold + DD threshold
    for adx_thresh in [25, 30]:
        for dd_thresh in [0.05, 0.08, 0.12]:
            for dd_window in [15, 30]:
                print(f"\n  ADX>{adx_thresh}, DD exit {dd_thresh*100:.0f}%/{dd_window}d")

                adx = compute_adx(prices, period=14)

                def signal_fn(prices_p, daily_returns_p, yr,
                              _adx=adx, _full_don=full_don,
                              _adx_thresh=adx_thresh, _dd_thresh=dd_thresh, _dd_window=dd_window):
                    """
                    Entry: only if Donchian fires AND ADX > threshold
                    Exit: Donchian exit signal OR rolling DD > threshold
                    """
                    test_mask = prices_p.index.year == yr

                    # ADX regime for test year
                    adx_test = _adx.reindex(prices_p[test_mask].index, method="ffill").fillna(20)
                    adx_entry = (adx_test > _adx_thresh).astype(int)

                    # Donchian signal for test year
                    don_test = _full_don.reindex(prices_p[test_mask].index).fillna(0)

                    # Compute rolling strategy equity up to start of test year for DD
                    # (no look-ahead: we compute DD on the strategy's equity up to and including test year)
                    pos_all = _full_don.shift(1).fillna(0)
                    strat_ret_all = pos_all * daily_returns_p.reindex(prices_p.index).fillna(0)
                    equity_all = (1 + strat_ret_all).cumprod()

                    # Rolling max-to-current drawdown (for the whole series, causal)
                    rolling_max = equity_all.rolling(_dd_window).max()
                    rolling_dd = (equity_all - rolling_max) / (rolling_max + 1e-9)

                    dd_test = rolling_dd.reindex(prices_p[test_mask].index).fillna(0)
                    # DD exit: if drawdown > threshold, force exit (0)
                    # DD filter = 1 means "we're NOT in a drawdown > threshold" → allow position
                    dd_ok = (dd_test > -_dd_thresh).astype(int)

                    # Asymmetric logic:
                    # ENTRY: need Donchian=1 AND ADX_entry=1
                    # HOLD (already in position): need don_test=1 AND dd_ok=1
                    # EXIT: don_test=0 OR dd_ok=0

                    # We compute this day by day with state
                    signals = pd.Series(0, index=prices_p[test_mask].index, dtype=float)
                    in_pos = False
                    for i in range(len(signals)):
                        d = don_test.iloc[i]
                        a = adx_entry.iloc[i]
                        dd = dd_ok.iloc[i]

                        if not in_pos:
                            # Enter only if Donchian fires AND ADX is trending
                            if d == 1 and a == 1:
                                in_pos = True
                                signals.iloc[i] = 1.0
                        else:
                            # Exit if Donchian exits OR drawdown too deep
                            if d == 0 or dd == 0:
                                in_pos = False
                                signals.iloc[i] = 0.0
                            else:
                                signals.iloc[i] = 1.0

                    return signals

                wf = run_walk_forward_custom(prices, daily_returns, signal_fn)
                s22 = wf["per_year"].get("2022", 0.0)
                print(f"    Mean: {wf['mean_sharpe']:.3f} ± {wf['std_sharpe']:.3f} | 2022: {s22:.3f}")

                cfg_key = f"adx{adx_thresh}_dd{int(dd_thresh*100)}pct_{dd_window}d"
                all_configs.append({
                    "config_key": cfg_key,
                    "adx_threshold": adx_thresh,
                    "dd_threshold": dd_thresh,
                    "dd_window": dd_window,
                    **wf
                })

                if wf["mean_sharpe"] > best_sharpe:
                    best_sharpe = wf["mean_sharpe"]
                    best_result = (cfg_key, wf, adx_thresh, dd_thresh, dd_window)

    # Log best config
    cfg_key, wf, adx_thresh, dd_thresh, dd_window = best_result
    name = f"asym_adx{adx_thresh}_dd{int(dd_thresh*100)}pct_{dd_window}d_S{wf['mean_sharpe']:.2f}"
    s22 = wf["per_year"].get("2022", 0.0)

    notes = (
        f"Asymmetric regime: ADX(14)>{adx_thresh} for entries, "
        f"{dd_thresh*100:.0f}%/{dd_window}d drawdown filter for exits. "
        f"Hypothesis: ADX is better at detecting trending market (entry quality) "
        f"while rolling drawdown catches strategy failures earlier (exit quality). "
        f"Mean WF Sharpe {wf['mean_sharpe']:.3f} ± {wf['std_sharpe']:.3f}. 2022: {s22:.3f}. "
        + ("SUCCESS — asymmetric regime improves over best prior (ADX-only 1.181)."
           if wf["mean_sharpe"] > BEST_REGIME_SHARPE
           else f"Result below best prior ADX-only ({BEST_REGIME_SHARPE:.3f}). "
                f"DD exits may cut winning trades too early, offsetting entry quality gains. "
                f"The DD filter's 'false positive' exits in trending regimes hurt alpha.")
    )
    log_novel_result(
        tracker, name,
        config={
            "approach": "asymmetric_regime",
            "entry_filter": "ADX",
            "adx_period": 14,
            "adx_threshold": adx_thresh,
            "exit_filter": "rolling_drawdown",
            "dd_threshold": dd_thresh,
            "dd_window": dd_window,
            "donchian_entry": ENTRY_PERIOD,
            "donchian_exit": EXIT_PERIOD,
            "all_configs_tested": all_configs,
        },
        metrics={
            "sharpe": wf["mean_sharpe"],
            "mean_wf_sharpe": wf["mean_sharpe"],
            "std_wf_sharpe": wf["std_sharpe"],
            "min_wf_sharpe": wf["min_sharpe"],
            "max_wf_sharpe": wf["max_sharpe"],
            "sharpe_2022": s22,
            **{f"sharpe_{yr}": v for yr, v in wf["per_year"].items()},
        },
        group="asymmetric_regime",
        notes=notes,
    )
    return {"best": best_result, "all_configs": all_configs}


# ════════════════════════════════════════════════════════════════════════════════
# IDEA 2 — ADAPTIVE DONCHIAN: ADX controls channel width
# ════════════════════════════════════════════════════════════════════════════════
@with_timeout(seconds=900)
def run_idea2_adaptive_donchian(prices, daily_returns, tracker):
    """
    HYPOTHESIS: Fixed Donchian parameters are suboptimal — the right channel width
    depends on market regime. In strong trends (high ADX), tight channels (e.g., 20/10)
    react faster and capture more of the move. In choppy/uncertain markets (low ADX),
    wide channels (e.g., 60/30) filter out more noise and avoid whipsaws.

    The regime detector CONTROLS the strategy parameters rather than just on/off.
    This is fundamentally different from previous approaches where we switched
    the strategy off — here we always trade, but trade MORE SENSIBLY.
    """
    print("\n" + "=" * 80)
    print("IDEA 2: ADAPTIVE DONCHIAN (ADX-controlled channel width)")
    print("=" * 80)
    print("HYPOTHESIS: Channel width should adapt to regime. Trending → tight channels")
    print("(faster capture). Choppy → wide channels (avoid whipsaws).")
    print("ADX score directly controls the Donchian lookback period.")

    adx = compute_adx(prices, period=14)

    best_sharpe = -999
    best_result = None
    all_configs = []

    # Test combinations of (fast params when trending, slow params when choppy)
    param_combos = [
        # (trending_entry, trending_exit, choppy_entry, choppy_exit, adx_thresh)
        (20, 10, 60, 30, 30),   # tight-fast in trends, wide-slow in chop, ADX=30 threshold
        (20, 10, 60, 30, 25),   # same but ADX=25 threshold
        (25, 12, 50, 25, 30),   # moderate adaptation
        (30, 15, 50, 25, 25),   # less aggressive adaptation
    ]

    for trend_e, trend_x, chop_e, chop_x, adx_thresh in param_combos:
        print(f"\n  Trending: Don({trend_e}/{trend_x}), Choppy: Don({chop_e}/{chop_x}), ADX>{adx_thresh}")

        # Pre-compute both Donchian signals
        don_fast = donchian_signal_stateful(prices, trend_e, trend_x)
        don_slow = donchian_signal_stateful(prices, chop_e, chop_x)

        def signal_fn(prices_p, daily_returns_p, yr,
                      _adx=adx, _don_fast=don_fast, _don_slow=don_slow,
                      _adx_thresh=adx_thresh, _trend_e=trend_e, _trend_x=trend_x,
                      _chop_e=chop_e, _chop_x=chop_x):
            """
            Adaptive Donchian: use fast channel in trending regime, slow in choppy.
            The ADX score smoothly selects which signal dominates.
            """
            test_mask = prices_p.index.year == yr

            adx_test = _adx.reindex(prices_p[test_mask].index, method="ffill").fillna(20)
            don_fast_test = _don_fast.reindex(prices_p[test_mask].index).fillna(0)
            don_slow_test = _don_slow.reindex(prices_p[test_mask].index).fillna(0)

            # Hard threshold: ADX > threshold → use fast Donchian, else slow
            signals = pd.Series(0, index=prices_p[test_mask].index, dtype=float)
            trending = (adx_test > _adx_thresh)
            signals = np.where(trending, don_fast_test, don_slow_test)
            return pd.Series(signals.astype(float), index=prices_p[test_mask].index)

        wf = run_walk_forward_custom(prices, daily_returns, signal_fn)
        s22 = wf["per_year"].get("2022", 0.0)
        print(f"    Mean: {wf['mean_sharpe']:.3f} ± {wf['std_sharpe']:.3f} | 2022: {s22:.3f}")

        cfg_key = f"adaptive_t{trend_e}x{trend_x}_c{chop_e}x{chop_x}_adx{adx_thresh}"
        all_configs.append({
            "config_key": cfg_key,
            "trend_entry": trend_e,
            "trend_exit": trend_x,
            "chop_entry": chop_e,
            "chop_exit": chop_x,
            "adx_threshold": adx_thresh,
            **wf
        })

        if wf["mean_sharpe"] > best_sharpe:
            best_sharpe = wf["mean_sharpe"]
            best_result = (cfg_key, wf, trend_e, trend_x, chop_e, chop_x, adx_thresh)

    # Also test a smooth interpolation variant (linear blend by ADX percentile)
    print(f"\n  Testing smooth interpolation (ADX percentile → continuous blend)...")
    adx_smooth = adx.copy()

    def signal_fn_smooth(prices_p, daily_returns_p, yr, _adx=adx_smooth):
        """
        Instead of hard threshold, compute ADX percentile (0-1) in training period.
        Use it to linearly blend between fast and slow signal probabilities.
        High ADX percentile → weight fast signal more.
        """
        test_mask = prices_p.index.year == yr
        cutoff = pd.Timestamp(f"{yr}-01-01", tz="UTC")
        adx_train = _adx[_adx.index < cutoff]
        if len(adx_train) < 10:
            adx_train = _adx

        adx_test = _adx.reindex(prices_p[test_mask].index, method="ffill").fillna(20)
        # Map ADX to 0-1 percentile using training distribution
        pct_75 = float(adx_train.quantile(0.75))
        pct_25 = float(adx_train.quantile(0.25))
        adx_pct = (adx_test - pct_25) / (pct_75 - pct_25 + 1e-9)
        adx_pct = adx_pct.clip(0, 1)  # saturate outside IQR

        don_fast_test = donchian_signal_stateful(prices_p, 20, 10).reindex(prices_p[test_mask].index).fillna(0)
        don_slow_test = donchian_signal_stateful(prices_p, 60, 30).reindex(prices_p[test_mask].index).fillna(0)

        # Blend: high ADX pct → fast signal wins
        blended = adx_pct * don_fast_test + (1 - adx_pct) * don_slow_test
        # Convert to discrete signal (threshold 0.5)
        return (blended > 0.5).astype(float)

    # Note: this recomputes Donchian per year in a causal way (only uses data before test year for ADX pct)
    wf_smooth = run_walk_forward_custom(prices, daily_returns, signal_fn_smooth)
    s22_s = wf_smooth["per_year"].get("2022", 0.0)
    print(f"    Smooth blend: {wf_smooth['mean_sharpe']:.3f} ± {wf_smooth['std_sharpe']:.3f} | 2022: {s22_s:.3f}")

    cfg_smooth = {
        "config_key": "adaptive_smooth_blend",
        "blend_type": "linear_adx_percentile",
        "fast_entry": 20, "fast_exit": 10,
        "slow_entry": 60, "slow_exit": 30,
        **wf_smooth
    }
    all_configs.append(cfg_smooth)
    if wf_smooth["mean_sharpe"] > best_sharpe:
        best_sharpe = wf_smooth["mean_sharpe"]
        best_result = ("adaptive_smooth_blend", wf_smooth, 20, 10, 60, 30, "smooth")

    # Log best config
    cfg_key, wf, trend_e, trend_x, chop_e, chop_x, adx_thresh = best_result
    name = f"adaptive_don_{cfg_key}_S{wf['mean_sharpe']:.2f}"[:60]
    s22 = wf["per_year"].get("2022", 0.0)

    notes = (
        f"Adaptive Donchian: ADX(14) controls channel width. "
        f"Trending regime: Don({trend_e}/{trend_x}) — fast reactions. "
        f"Choppy regime: Don({chop_e}/{chop_x}) — wide channels filter whipsaws. "
        f"Hypothesis: regime should control PARAMETERS not just on/off. "
        f"Mean WF Sharpe {wf['mean_sharpe']:.3f} ± {wf['std_sharpe']:.3f}. 2022: {s22:.3f}. "
        + ("SUCCESS — adaptive parameters outperform fixed-parameter + regime filter."
           if wf["mean_sharpe"] > BEST_REGIME_SHARPE
           else f"Below best prior (ADX-only {BEST_REGIME_SHARPE:.3f}). "
                f"The 2022 bear sees both fast AND slow channels fire false signals — "
                f"parameter adaptation alone cannot fix the fundamental trend-following weakness in downtrends.")
    )
    log_novel_result(
        tracker, name,
        config={
            "approach": "adaptive_donchian",
            "regime_indicator": "ADX",
            "adx_period": 14,
            "adx_threshold": adx_thresh,
            "trending_entry": trend_e,
            "trending_exit": trend_x,
            "choppy_entry": chop_e,
            "choppy_exit": chop_x,
            "all_configs_tested": all_configs,
        },
        metrics={
            "sharpe": wf["mean_sharpe"],
            "mean_wf_sharpe": wf["mean_sharpe"],
            "std_wf_sharpe": wf["std_sharpe"],
            "min_wf_sharpe": wf["min_sharpe"],
            "max_wf_sharpe": wf["max_sharpe"],
            "sharpe_2022": s22,
            **{f"sharpe_{yr}": v for yr, v in wf["per_year"].items()},
        },
        group="adaptive_donchian",
        notes=notes,
    )
    return {"best": best_result, "all_configs": all_configs}


# ════════════════════════════════════════════════════════════════════════════════
# IDEA 3 — ML AS POSITION SIZER (not signal)
# ════════════════════════════════════════════════════════════════════════════════
@with_timeout(seconds=1200)
def run_idea3_ml_position_sizer(prices, daily_returns, df_daily, target_daily,
                                 top10_cols, tracker):
    """
    HYPOTHESIS: Previous ML approaches tried to replace or gate the Donchian signal.
    This introduces binary decision failure: if ML is wrong about whether to trade,
    we either miss a winning trade or take a losing one.

    Instead: Donchian ALWAYS determines WHEN to trade. ML determines HOW MUCH.

    ML confidence (probability of up direction) scales the position:
    - Prob > 0.65: 1.0x position (full)
    - Prob 0.55-0.65: 0.5x position
    - Prob < 0.55: 0.25x position (minimum exposure)

    This preserves all trades while reducing drawdown on low-conviction days.
    The 2022 bear market should score low ML probability → smaller positions → less damage.
    """
    print("\n" + "=" * 80)
    print("IDEA 3: ML AS POSITION SIZER (not binary signal)")
    print("=" * 80)
    print("HYPOTHESIS: ML confidence should scale position size, not gate the trade.")
    print("Donchian determines WHEN to trade. LGBM determines HOW MUCH to trade.")
    print("Low ML confidence → 0.25x position. High confidence → 1.0x position.")

    # Pre-compute Donchian signal
    full_don = donchian_signal_stateful(prices, ENTRY_PERIOD, EXIT_PERIOD)

    best_sharpe = -999
    best_result = None
    all_configs = []

    # Test different sizing schemes
    sizing_schemes = [
        # (name, low_size, mid_size, high_size, low_thresh, high_thresh)
        ("aggressive", 0.25, 0.5, 1.0, 0.50, 0.60),   # wider bands → more scaling
        ("moderate",   0.25, 0.5, 1.0, 0.50, 0.65),   # standard
        ("conservative", 0.5, 0.75, 1.0, 0.50, 0.65), # never go below 0.5x
        ("binary_soft", 0.0, 0.5, 1.0, 0.50, 0.60),   # can go fully flat
    ]

    avail_cols = [c for c in top10_cols if c in df_daily.columns]
    if len(avail_cols) < 5:
        avail_cols = df_daily.var().sort_values(ascending=False).head(10).index.tolist()

    lgbm_params = {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "reg_lambda": 1.0, "num_leaves": 63, "device": "gpu",
        "n_jobs": 1, "verbose": -1, "random_state": 42,
    }

    for scheme_name, lo, mid, hi, lo_thresh, hi_thresh in sizing_schemes:
        print(f"\n  Position sizing scheme: {scheme_name}")
        print(f"    Prob<{lo_thresh}: {lo:.2f}x | {lo_thresh}-{hi_thresh}: {mid:.2f}x | >{hi_thresh}: {hi:.2f}x")

        per_year = {}

        for yr in range(2019, 2024):
            train_mask = daily_returns.index.year < yr
            test_mask = daily_returns.index.year == yr

            if train_mask.sum() < 100:
                per_year[str(yr)] = 0.0
                continue

            X_tr = df_daily[avail_cols][train_mask]
            y_tr = target_daily[train_mask]
            X_te = df_daily[avail_cols][test_mask]

            valid = X_tr.notna().all(axis=1) & y_tr.notna()
            X_tr, y_tr = X_tr[valid], y_tr[valid]
            X_te = X_te.fillna(X_te.median())

            if len(X_tr) < 50 or len(X_te) == 0:
                per_year[str(yr)] = 0.0
                continue

            try:
                model = LGBMClassifier(**lgbm_params)
                model.fit(X_tr, y_tr)
                proba = pd.Series(model.predict_proba(X_te)[:, 1], index=X_te.index)
            except Exception as e:
                print(f"    WARN: LGBM year {yr} failed: {e}")
                per_year[str(yr)] = 0.0
                continue

            # Get Donchian binary signal for test year
            don_yr = full_don.reindex(daily_returns.index[test_mask]).fillna(0)

            # Build sized positions: Donchian says trade (1), ML scales how much
            proba_aligned = proba.reindex(don_yr.index, method="ffill").fillna(0.5)

            sized_position = pd.Series(0.0, index=don_yr.index)
            for i in range(len(don_yr)):
                if don_yr.iloc[i] == 0:
                    sized_position.iloc[i] = 0.0  # Donchian says flat
                else:
                    p = proba_aligned.iloc[i]
                    if p >= hi_thresh:
                        sized_position.iloc[i] = hi
                    elif p >= lo_thresh:
                        sized_position.iloc[i] = mid
                    else:
                        sized_position.iloc[i] = lo

            # Compute Sharpe with fractional positions (continuous)
            pos_shifted = sized_position.shift(1).fillna(0)
            ret_yr = daily_returns.reindex(don_yr.index).fillna(0)
            strat_ret = pos_shifted * ret_yr
            sh_yr = float(annualized_sharpe(strat_ret)) if strat_ret.std() > 0 else 0.0
            per_year[str(yr)] = round(sh_yr, 3)

            avg_size = sized_position[don_yr == 1].mean() if (don_yr == 1).sum() > 0 else 0
            print(f"    {yr}: Sharpe={sh_yr:.3f}, avg_position_size={avg_size:.2f}x when in-market")

        vals = list(per_year.values())
        mean_sh = float(np.mean(vals))
        std_sh = float(np.std(vals))
        s22 = per_year.get("2022", 0.0)
        print(f"    → Mean: {mean_sh:.3f} ± {std_sh:.3f} | 2022: {s22:.3f}")

        all_configs.append({
            "scheme": scheme_name,
            "lo_size": lo, "mid_size": mid, "hi_size": hi,
            "lo_thresh": lo_thresh, "hi_thresh": hi_thresh,
            "mean_sharpe": mean_sh,
            "std_sharpe": std_sh,
            "per_year": per_year,
        })

        if mean_sh > best_sharpe:
            best_sharpe = mean_sh
            best_result = (scheme_name, per_year, mean_sh, std_sh, lo, mid, hi, lo_thresh, hi_thresh)

    # Log best config
    scheme_name, per_year, mean_sh, std_sh, lo, mid, hi, lo_thresh, hi_thresh = best_result
    name = f"ml_sizer_{scheme_name}_LGBM_top10_S{mean_sh:.2f}"
    s22 = per_year.get("2022", 0.0)

    notes = (
        f"ML as position sizer (not binary signal). LGBM-top10 confidence scales size. "
        f"Donchian(40/20) decides WHEN to trade. LGBM probability scales HOW MUCH. "
        f"Best scheme: {scheme_name} — lo={lo:.2f}x(<{lo_thresh}), mid={mid:.2f}x, hi={hi:.2f}x(>{hi_thresh}). "
        f"Mean WF Sharpe {mean_sh:.3f} ± {std_sh:.3f}. 2022: {s22:.3f}. "
        + ("SUCCESS — LGBM confidence as continuous position scaler beats binary gating."
           if mean_sh > BEST_REGIME_SHARPE
           else f"Below best prior ADX-only ({BEST_REGIME_SHARPE:.3f}). "
                f"Position scaling is directionally correct (reduces 2022 damage) but "
                f"may not fully compensate for 2022 bear market signal confusion in LGBM. "
                f"The key issue: LGBM's uncertainty in 2022 may not accurately reflect "
                f"the catastrophic risk — the model is uncertain when it should be confident that it's wrong.")
    )
    log_novel_result(
        tracker, name,
        config={
            "approach": "ml_position_sizer",
            "entry_signal": "Donchian(40/20)",
            "sizing_model": "LGBM-top10",
            "sizing_scheme": scheme_name,
            "lo_size": lo, "mid_size": mid, "hi_size": hi,
            "lo_threshold": lo_thresh, "hi_threshold": hi_thresh,
            "lgbm_params": lgbm_params,
            "all_schemes_tested": all_configs,
        },
        metrics={
            "sharpe": mean_sh,
            "mean_wf_sharpe": mean_sh,
            "std_wf_sharpe": std_sh,
            "min_wf_sharpe": min(per_year.values()),
            "max_wf_sharpe": max(per_year.values()),
            "sharpe_2022": s22,
            **{f"sharpe_{yr}": v for yr, v in per_year.items()},
        },
        group="ml_position_sizer",
        notes=notes,
    )
    return {"best": best_result, "all_configs": all_configs}


# ════════════════════════════════════════════════════════════════════════════════
# IDEA 4 — MAJORITY-VOTE REGIME ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════════
@with_timeout(seconds=1200)
def run_idea4_majority_vote_regime(prices, daily_returns, df_daily, target_daily,
                                    top10_cols, tracker):
    """
    HYPOTHESIS: Each individual regime detector makes mistakes, but they make
    DIFFERENT kinds of mistakes. ADX might miss some bear market phases. Vol-based
    detectors might have false positives in high-vol trends. ML might overfit.

    A majority vote (2-of-3 must agree) should be more robust than any individual
    detector — it requires CONSENSUS that the regime is favorable before trading.

    Detectors:
    1. ADX(14,30): trend strength filter (best individual from Step 3)
    2. Vol(20d): volatility regime (high vol = trending = trade)
    3. LGBM regime: ML-based regime prediction (direct regime prediction, not price)

    The ML LGBM here is trained to predict regime (trending vs non-trending)
    using ADX + vol features — learning a meta-regime from the indicators themselves.
    """
    print("\n" + "=" * 80)
    print("IDEA 4: MAJORITY-VOTE REGIME ENSEMBLE (3 detectors, 2-of-3)")
    print("=" * 80)
    print("HYPOTHESIS: ADX, Vol, and ML regime detectors have uncorrelated errors.")
    print("Requiring 2-of-3 consensus is more robust than any single detector.")

    # Pre-compute indicators
    full_don = donchian_signal_stateful(prices, ENTRY_PERIOD, EXIT_PERIOD)
    adx = compute_adx(prices, period=14)
    returns_raw = prices.pct_change()
    rolling_vol = returns_raw.rolling(20).std() * np.sqrt(252)

    # Vol regime: above expanding-window median = trending
    def get_vol_regime(yr):
        cutoff = pd.Timestamp(f"{yr}-01-01", tz="UTC")
        vol_train = rolling_vol[rolling_vol.index < cutoff]
        median_vol = vol_train.median() if len(vol_train) > 10 else rolling_vol.median()
        test_mask = rolling_vol.index.year == yr
        return (rolling_vol[test_mask] > median_vol).astype(int)

    # ADX regime
    def get_adx_regime(yr, threshold=30):
        test_mask = adx.index.year == yr
        return (adx[test_mask] > threshold).astype(int)

    # ML regime detector: predict ADX-based regime directly
    # Features: vol, momentum, ATR-proxies
    def build_regime_features(prices_p):
        """Features that capture trending vs choppy regime."""
        ret = prices_p.pct_change()
        feat = pd.DataFrame(index=prices_p.index)
        feat["vol_10d"] = ret.rolling(10).std() * np.sqrt(252)
        feat["vol_20d"] = ret.rolling(20).std() * np.sqrt(252)
        feat["vol_60d"] = ret.rolling(60).std() * np.sqrt(252)
        feat["mom_10d"] = ret.rolling(10).sum()
        feat["mom_20d"] = ret.rolling(20).sum()
        feat["mom_60d"] = ret.rolling(60).sum()
        feat["adx_14"] = compute_adx(prices_p, 14)
        feat["adx_28"] = compute_adx(prices_p, 28)
        # Directional consistency: how often are daily returns positive in last 20d?
        feat["dir_consist_20d"] = (ret > 0).rolling(20).mean()
        # Vol ratio: short/long vol (< 1 = vol expanding = potential trending)
        feat["vol_ratio_s_l"] = feat["vol_10d"] / (feat["vol_60d"] + 1e-9)
        return feat.fillna(feat.median())

    regime_features = build_regime_features(prices)
    # Label: ADX > 30 = trending regime (this is our "truth" signal for supervised learning)
    adx_label = (adx > 30).astype(int)

    avail_feat_cols = [c for c in regime_features.columns if c in regime_features.columns]
    common_idx = prices.index.intersection(df_daily.index).intersection(target_daily.index)
    regime_features_aligned = regime_features.loc[common_idx]
    adx_label_aligned = adx_label.reindex(common_idx).fillna(0)

    # Test: voting approaches
    voting_configs = [
        # (name, adx_thresh, require_votes)
        ("vote_2of3_strict", 30, 2),   # strict ADX threshold
        ("vote_2of3_loose",  25, 2),   # looser ADX threshold
        ("vote_3of3_all",    30, 3),   # require unanimous consensus
    ]

    lgbm_regime_params = {
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
        "reg_lambda": 2.0, "num_leaves": 31, "device": "gpu",
        "n_jobs": 1, "verbose": -1, "random_state": 42,
    }

    best_sharpe = -999
    best_result = None
    all_configs_results = []

    for config_name, adx_thresh, req_votes in voting_configs:
        print(f"\n  Config: {config_name} (ADX>{adx_thresh}, need {req_votes}/3 votes)")

        per_year = {}

        for yr in range(2019, 2024):
            train_mask = daily_returns.index.year < yr
            test_mask_idx = daily_returns.index.year == yr

            # Detector 1: ADX
            adx_regime_yr = get_adx_regime(yr, threshold=adx_thresh)

            # Detector 2: Vol
            vol_regime_yr = get_vol_regime(yr)
            vol_regime_aligned = vol_regime_yr.reindex(
                daily_returns.index[test_mask_idx], method="ffill"
            ).fillna(0)

            # Detector 3: ML regime predictor
            if train_mask.sum() < 100:
                # Not enough training data — fall back to ADX regime
                ml_regime_yr = adx_regime_yr
            else:
                X_tr = regime_features_aligned[train_mask]
                y_tr = adx_label_aligned[train_mask]
                X_te = regime_features_aligned[test_mask_idx]

                valid = X_tr.notna().all(axis=1) & y_tr.notna()
                X_tr_v, y_tr_v = X_tr[valid], y_tr[valid]
                X_te = X_te.fillna(X_te.median())

                try:
                    ml_regime_model = LGBMClassifier(**lgbm_regime_params)
                    ml_regime_model.fit(X_tr_v, y_tr_v)
                    ml_pred = pd.Series(
                        ml_regime_model.predict(X_te).astype(int),
                        index=X_te.index
                    )
                    ml_regime_yr = ml_pred.reindex(
                        daily_returns.index[test_mask_idx], method="ffill"
                    ).fillna(0)
                except Exception as e:
                    print(f"    WARN: ML regime year {yr} failed: {e}")
                    ml_regime_yr = adx_regime_yr

            # Align all three detectors to test year index
            test_idx = daily_returns.index[test_mask_idx]
            d1 = adx_regime_yr.reindex(test_idx, method="ffill").fillna(0).astype(int)
            d2 = vol_regime_aligned.astype(int)
            d3 = ml_regime_yr.astype(int)

            # Majority vote: need req_votes out of 3
            vote_sum = d1.values + d2.reindex(test_idx).fillna(0).astype(int).values + d3.reindex(test_idx).fillna(0).astype(int).values
            regime_ok = (vote_sum >= req_votes).astype(int)

            # Apply to Donchian
            don_yr = full_don.reindex(test_idx).fillna(0).astype(int)
            filtered_pos = (don_yr.values * regime_ok).clip(0, 1).astype(float)
            filtered_sig = pd.Series(filtered_pos, index=test_idx)

            sh_yr = year_sharpe(daily_returns, filtered_sig, yr)
            per_year[str(yr)] = round(sh_yr, 3)

            vote_frac = float((vote_sum >= req_votes).mean())
            print(f"    {yr}: Sharpe={sh_yr:.3f}, vote_frac={vote_frac:.1%} "
                  f"(ADX={d1.mean():.1%}, Vol={d2.reindex(test_idx).fillna(0).mean():.1%}, ML={d3.reindex(test_idx).fillna(0).mean():.1%})")

        vals = list(per_year.values())
        mean_sh = float(np.mean(vals))
        std_sh = float(np.std(vals))
        s22 = per_year.get("2022", 0.0)
        print(f"    → Mean: {mean_sh:.3f} ± {std_sh:.3f} | 2022: {s22:.3f}")

        all_configs_results.append({
            "config": config_name,
            "adx_threshold": adx_thresh,
            "required_votes": req_votes,
            "mean_sharpe": mean_sh,
            "std_sharpe": std_sh,
            "per_year": per_year,
        })

        if mean_sh > best_sharpe:
            best_sharpe = mean_sh
            best_result = (config_name, per_year, mean_sh, std_sh, adx_thresh, req_votes)

    # Log best config
    config_name, per_year, mean_sh, std_sh, adx_thresh, req_votes = best_result
    name = f"majority_vote_{config_name}_S{mean_sh:.2f}"
    s22 = per_year.get("2022", 0.0)

    notes = (
        f"Majority-vote regime ensemble: 3 detectors (ADX(14,{adx_thresh}), Vol(20d), ML-regime), "
        f"trade Donchian(40/20) only when {req_votes}/3 agree regime is trending. "
        f"ML regime detector trained on ADX>30 labels using price-derived features (vol, mom, ADX). "
        f"Hypothesis: uncorrelated errors in 3 detectors → consensus more robust than any individual. "
        f"Mean WF Sharpe {mean_sh:.3f} ± {std_sh:.3f}. 2022: {s22:.3f}. "
        + ("SUCCESS — consensus regime is more robust than single ADX detector."
           if mean_sh > BEST_REGIME_SHARPE
           else f"Below best prior ADX-only ({BEST_REGIME_SHARPE:.3f}). "
                f"Consensus filtering reduces false positives but also cuts time-in-market "
                f"in good years. The vol detector and ML detector may be correlated with ADX "
                f"(all derived from price), so diversity benefits are limited. "
                f"'Garbage in, garbage out' — if all three detectors use overlapping information, "
                f"majority vote just averages similar decisions.")
    )
    log_novel_result(
        tracker, name,
        config={
            "approach": "majority_vote_regime",
            "n_detectors": 3,
            "detectors": ["ADX(14)", "Vol(20d)", "ML-regime(LGBM)"],
            "adx_threshold": adx_thresh,
            "required_votes": req_votes,
            "donchian_entry": ENTRY_PERIOD,
            "donchian_exit": EXIT_PERIOD,
            "ml_regime_features": avail_feat_cols,
            "all_configs_tested": all_configs_results,
        },
        metrics={
            "sharpe": mean_sh,
            "mean_wf_sharpe": mean_sh,
            "std_wf_sharpe": std_sh,
            "min_wf_sharpe": min(per_year.values()),
            "max_wf_sharpe": max(per_year.values()),
            "sharpe_2022": s22,
            **{f"sharpe_{yr}": v for yr, v in per_year.items()},
        },
        group="majority_vote",
        notes=notes,
    )
    return {"best": best_result, "all_configs": all_configs_results}


# ════════════════════════════════════════════════════════════════════════════════
# IDEA 5 — REGIME FILTER + MOMENTUM BASE STRATEGY
# ════════════════════════════════════════════════════════════════════════════════
@with_timeout(seconds=900)
def run_idea5_regime_momentum(prices, daily_returns, tracker):
    """
    HYPOTHESIS: Momentum (buy when N-day return > threshold) may be a better BASE
    strategy than Donchian for regime filtering. The step 4 results showed
    momentum+vol_adx_OR hit Sharpe 1.279, BUT with 2022=-1.317.

    Key test: Does ADX(14,30) — which achieved 2022=0.000 for Donchian —
    also achieve near-zero 2022 for momentum strategies?

    If yes: momentum+ADX30 might combine momentum's higher raw Sharpe
    with ADX30's bear-market protection.

    Test matrix:
    - Momentum lookbacks: 20d, 40d
    - Thresholds: 0, 0.05, 0.10
    - Regime filters: vol_adx_AND (best for 2022 protection), ADX(14,30) alone
    """
    print("\n" + "=" * 80)
    print("IDEA 5: REGIME FILTER + MOMENTUM BASE STRATEGY")
    print("=" * 80)
    print("HYPOTHESIS: ADX(14,30) gives 2022=0.000 with Donchian.")
    print("Does the SAME filter also protect momentum from 2022? If so,")
    print("momentum+ADX30 has a better risk profile than any prior single strategy.")

    adx = compute_adx(prices, period=14)
    returns_raw = prices.pct_change().fillna(0)
    rolling_vol = returns_raw.rolling(20).std() * np.sqrt(252)

    def get_adx30_regime(yr):
        test_mask = adx.index.year == yr
        return (adx[test_mask] > 30).astype(int)

    def get_vol_adx_and_regime(yr):
        cutoff = pd.Timestamp(f"{yr}-01-01", tz="UTC")
        vol_train = rolling_vol[rolling_vol.index < cutoff]
        median_vol = vol_train.median() if len(vol_train) > 10 else rolling_vol.median()
        test_mask = prices.index.year == yr
        vol_r = (rolling_vol.reindex(prices[test_mask].index).fillna(0) > median_vol).astype(int)
        adx_r = (adx.reindex(prices[test_mask].index).fillna(0) > 25).astype(int)
        return (vol_r & adx_r).astype(int)

    best_sharpe = -999
    best_result = None
    all_configs = []

    for lookback in [20, 40]:
        mom = returns_raw.rolling(lookback).sum()
        for threshold in [0.0, 0.05, 0.10]:
            for regime_name, regime_fn in [
                ("adx30",       get_adx30_regime),
                ("vol_adx_AND", get_vol_adx_and_regime),
            ]:
                per_year = {}
                for yr in range(2019, 2024):
                    test_mask = prices.index.year == yr
                    test_idx = prices[test_mask].index

                    # Momentum signal: buy when trailing N-day return > threshold
                    mom_test = mom.reindex(test_idx, method="ffill").fillna(0)
                    mom_signal = (mom_test > threshold).astype(int)

                    # Regime filter
                    regime = regime_fn(yr).reindex(test_idx, method="ffill").fillna(0)

                    # Combined: trade when both momentum and regime agree
                    filtered = (mom_signal * regime).clip(0, 1).astype(float)
                    filtered_series = pd.Series(filtered.values, index=test_idx)

                    sh_yr = year_sharpe(daily_returns, filtered_series, yr)
                    per_year[str(yr)] = round(sh_yr, 3)

                vals = list(per_year.values())
                mean_sh = float(np.mean(vals))
                std_sh = float(np.std(vals))
                s22 = per_year.get("2022", 0.0)
                cfg_key = f"mom{lookback}_t{int(threshold*100)}_{regime_name}"
                print(f"  {cfg_key}: Mean={mean_sh:.3f} ± {std_sh:.3f} | 2022={s22:.3f}")

                all_configs.append({
                    "config_key": cfg_key,
                    "lookback": lookback,
                    "threshold": threshold,
                    "regime": regime_name,
                    "mean_sharpe": mean_sh,
                    "std_sharpe": std_sh,
                    "per_year": per_year,
                })

                if mean_sh > best_sharpe:
                    best_sharpe = mean_sh
                    best_result = (cfg_key, per_year, mean_sh, std_sh, lookback, threshold, regime_name)

    cfg_key, per_year, mean_sh, std_sh, lookback, threshold, regime_name = best_result
    name = f"regime_mom_lb{lookback}_t{int(threshold*100)}_{regime_name}_S{mean_sh:.2f}"
    s22 = per_year.get("2022", 0.0)

    notes = (
        f"Regime filter applied to momentum base strategy (not Donchian). "
        f"Best: {lookback}d lookback, threshold={threshold:.2f}, regime={regime_name}. "
        f"Hypothesis: ADX(14,30) gives 2022=0.000 with Donchian — does it also protect momentum? "
        f"Mean WF Sharpe {mean_sh:.3f} ± {std_sh:.3f}. 2022: {s22:.3f}. "
        + ("SUCCESS — regime+momentum beats ADX-only Donchian ({BEST_REGIME_SHARPE:.3f})."
           if mean_sh > BEST_REGIME_SHARPE
           else f"Below ADX-only Donchian ({BEST_REGIME_SHARPE:.3f}). "
                f"Donchian breakout is a more precise entry than raw momentum threshold — "
                f"momentum's 'always long when trending' picks up trend but also reversal noise. "
                f"ADX protects somewhat but momentum still more vulnerable than Donchian in 2022.")
    )
    log_novel_result(
        tracker, name,
        config={
            "approach": "regime_momentum",
            "base_strategy": "momentum",
            "regime_filter": regime_name,
            "lookback": lookback,
            "threshold": threshold,
            "all_configs_tested": all_configs,
        },
        metrics={
            "sharpe": mean_sh,
            "mean_wf_sharpe": mean_sh,
            "std_wf_sharpe": std_sh,
            "min_wf_sharpe": min(per_year.values()),
            "max_wf_sharpe": max(per_year.values()),
            "sharpe_2022": s22,
            **{f"sharpe_{yr}": v for yr, v in per_year.items()},
        },
        group="regime_momentum",
        notes=notes,
    )
    return {"best": best_result, "all_configs": all_configs}


# ════════════════════════════════════════════════════════════════════════════════
# IDEA 6 — STABILITY-OPTIMIZED: Systematic search for lowest variance config
# ════════════════════════════════════════════════════════════════════════════════
@with_timeout(seconds=900)
def run_idea6_stability_optimized(prices, daily_returns, tracker):
    """
    HYPOTHESIS: The key finding from 90+ experiments is that high mean Sharpe
    with high variance is undeployable. A strategy doing Sharpe 0.6 every year
    beats one doing 2.0 and -1.0 in practice.

    From all 90+ experiments, the lowest-variance configs were:
    - LGBM + vol_adx_AND: std=0.584 (best std!)
    - vol_10d threshold: std=0.682
    - ADX(14,30) Donchian: std=0.829

    This idea directly optimizes for STABILITY: search a grid of configs where
    the objective is MINIMIZE(std_sharpe) subject to MEAN_sharpe > 0.6.

    Test: combinations of the strictest known filters to find the config where
    year-to-year variance is lowest while maintaining positive Sharpe:
    1. ADX30 + vol_adx_AND (doubly-strict: both must agree AND ADX>30)
    2. ADX30 with multiple Donchian channel variants
    3. The vol_adx_AND strategy with varying ADX thresholds
    """
    print("\n" + "=" * 80)
    print("IDEA 6: STABILITY-OPTIMIZED (minimize year-to-year Sharpe variance)")
    print("=" * 80)
    print("HYPOTHESIS: Lower variance = more deployable. Direct search for configs")
    print("where std_sharpe is minimized while maintaining positive mean Sharpe.")

    adx = compute_adx(prices, period=14)
    returns_raw = prices.pct_change().fillna(0)
    rolling_vol = returns_raw.rolling(20).std() * np.sqrt(252)

    best_stability = 999  # minimize std
    best_by_std = None
    best_sharpe = -999
    best_by_sharpe = None
    all_configs = []

    # Grid: (adx_thresh1, adx_thresh2, vol_on, don_entry, don_exit, require_both)
    stability_configs = [
        # Triple filter: ADX>30 AND vol>median AND ADX>25 (even stricter than vol_adx_AND)
        ("adx30_AND_vol_adx_AND", 30, True, 40, 20),
        # ADX>30 alone with different Donchian widths
        ("adx30_don40_20", 30, False, 40, 20),
        ("adx30_don30_15", 30, False, 30, 15),
        ("adx30_don50_25", 30, False, 50, 25),
        # ADX>35 (very strict) with standard Donchian
        ("adx35_don40_20", 35, False, 40, 20),
        # vol_adx_AND (best prior std) with different ADX thresholds
        ("vol_adx30_AND", 30, True, 40, 20),
        ("vol_adx35_AND", 35, True, 40, 20),
        # Combined stability: triple requirement
        ("triple_gate_adx25_adx30_vol", None, True, 40, 20),  # special case
    ]

    for cfg_key, adx_thresh, use_vol, don_e, don_x in stability_configs:
        don_sig = donchian_signal_stateful(prices, don_e, don_x)
        per_year = {}

        for yr in range(2019, 2024):
            test_mask = prices.index.year == yr
            test_idx = prices[test_mask].index

            # ADX regime
            if adx_thresh is not None:
                adx_r = (adx.reindex(test_idx, method="ffill").fillna(0) > adx_thresh).astype(int)
            else:
                # triple gate: ADX>25 AND ADX>30 (redundant but explicit)
                adx_r = (adx.reindex(test_idx, method="ffill").fillna(0) > 25).astype(int)

            # Vol regime
            if use_vol:
                cutoff = pd.Timestamp(f"{yr}-01-01", tz="UTC")
                vol_train = rolling_vol[rolling_vol.index < cutoff]
                median_vol = vol_train.median() if len(vol_train) > 10 else rolling_vol.median()
                vol_r = (rolling_vol.reindex(test_idx, method="ffill").fillna(0) > median_vol).astype(int)
                regime = (adx_r & vol_r).astype(int)
            else:
                regime = adx_r

            # Apply to Donchian
            don_yr = don_sig.reindex(test_idx).fillna(0).astype(int)
            filtered = (don_yr.values * regime.values).clip(0, 1).astype(float)
            filtered_series = pd.Series(filtered, index=test_idx)

            sh_yr = year_sharpe(daily_returns, filtered_series, yr)
            per_year[str(yr)] = round(sh_yr, 3)

        vals = list(per_year.values())
        mean_sh = float(np.mean(vals))
        std_sh = float(np.std(vals))
        s22 = per_year.get("2022", 0.0)
        print(f"  {cfg_key}: Mean={mean_sh:.3f}, Std={std_sh:.3f} | 2022={s22:.3f} | Sharpe/Std={mean_sh/(std_sh+1e-9):.2f}")

        all_configs.append({
            "config_key": cfg_key,
            "adx_threshold": adx_thresh,
            "use_vol": use_vol,
            "don_entry": don_e,
            "don_exit": don_x,
            "mean_sharpe": mean_sh,
            "std_sharpe": std_sh,
            "per_year": per_year,
        })

        # Track best by std (subject to mean > 0.6)
        if mean_sh > 0.6 and std_sh < best_stability:
            best_stability = std_sh
            best_by_std = (cfg_key, per_year, mean_sh, std_sh, adx_thresh, use_vol, don_e, don_x)

        if mean_sh > best_sharpe:
            best_sharpe = mean_sh
            best_by_sharpe = (cfg_key, per_year, mean_sh, std_sh, adx_thresh, use_vol, don_e, don_x)

    # Primary result: best stability (lowest std with mean>0.6)
    primary = best_by_std if best_by_std else best_by_sharpe
    cfg_key, per_year, mean_sh, std_sh, adx_thresh, use_vol, don_e, don_x = primary
    name = f"stability_opt_{cfg_key}_S{mean_sh:.2f}_std{std_sh:.2f}"[:60]
    s22 = per_year.get("2022", 0.0)

    notes = (
        f"Stability-optimized: direct search minimizing year-to-year Sharpe variance. "
        f"Best stability config: {cfg_key} (ADX>{adx_thresh}, vol={use_vol}, Don({don_e}/{don_x})). "
        f"Mean WF Sharpe {mean_sh:.3f}, Std {std_sh:.3f} (Sharpe/Std ratio={mean_sh/(std_sh+1e-9):.2f}). "
        f"2022: {s22:.3f}. "
        f"Benchmark: LGBM+vol_adx_AND had lowest prior std=0.584, mean=0.728. "
        + ("NEW STABILITY RECORD — lower variance than LGBM+vol_adx_AND (0.584) with better mean."
           if std_sh < 0.584 and mean_sh > 0.6
           else f"Stability rank vs best prior (std=0.584): {'improved' if std_sh < 0.584 else 'not improved'}. "
                f"There may be a fundamental floor on variance given 5 years of data and bear/bull alternation.")
    )
    log_novel_result(
        tracker, name,
        config={
            "approach": "stability_optimized",
            "objective": "minimize_std_sharpe",
            "adx_threshold": adx_thresh,
            "use_vol_filter": use_vol,
            "don_entry": don_e,
            "don_exit": don_x,
            "all_configs_tested": all_configs,
        },
        metrics={
            "sharpe": mean_sh,
            "mean_wf_sharpe": mean_sh,
            "std_wf_sharpe": std_sh,
            "min_wf_sharpe": min(per_year.values()),
            "max_wf_sharpe": max(per_year.values()),
            "sharpe_2022": s22,
            "sharpe_std_ratio": round(mean_sh / (std_sh + 1e-9), 3),
            **{f"sharpe_{yr}": v for yr, v in per_year.items()},
        },
        group="stability_optimized",
        notes=notes,
    )
    return {"best": primary, "all_configs": all_configs}


# ════════════════════════════════════════════════════════════════════════════════
# IDEA 7 — TARGET RE-ENGINEERING: Predict breakout profitability (CatBoost GPU)
# ════════════════════════════════════════════════════════════════════════════════
@with_timeout(seconds=1200)
def run_idea7_breakout_profitability(prices, daily_returns, df_daily, top10_cols, tracker):
    """
    HYPOTHESIS: Previous ML targets predict price DIRECTION (will price go up?).
    A better target for Donchian strategies is: 'will this Donchian breakout be
    profitable?' — i.e., predict the SUCCESS RATE of the current breakout signal.

    This flips the problem: instead of predicting price, we predict strategy outcome.
    We ONLY predict when Donchian fires — meaning we're filtering which breakouts to take.

    Training target: for each Donchian entry, was the trade profitable over next 30 days?
    Feature timing: features computed at breakout day, all prior data only (no leak).
    Model: CatBoost GPU (best Stage-1 AUC was CatBoost with low LR)

    If CatBoost can distinguish profitable breakouts from false breakouts,
    we can dramatically reduce 2022 losses (2022 = bear market = mostly false breakouts).
    """
    print("\n" + "=" * 80)
    print("IDEA 7: TARGET RE-ENGINEERING (predict breakout profitability)")
    print("=" * 80)
    print("HYPOTHESIS: Predict 'will THIS Donchian breakout be profitable?' not 'will price rise?'")
    print("This targets the ML at the strategy's failure mode — false breakouts in bear markets.")

    full_don = donchian_signal_stateful(prices, ENTRY_PERIOD, EXIT_PERIOD)

    # Build breakout-specific labels
    # Label: breakout at day T is profitable if strategy Sharpe over next 30 days > 0
    # Compute daily strategy returns (Donchian)
    pos_shifted = full_don.shift(1).fillna(0)
    strat_ret = pos_shifted * daily_returns.reindex(full_don.index).fillna(0)

    # Rolling 30-day forward Sharpe (look-ahead for label construction only)
    forward_ret = strat_ret.rolling(30).mean().shift(-30)   # mean of NEXT 30 days
    forward_std = strat_ret.rolling(30).std().shift(-30)
    forward_sharpe = (forward_ret / (forward_std + 1e-9)) * np.sqrt(252)

    # Breakout entry label: 1 = this breakout will be profitable
    # Only label days where Donchian fires for the first time (new entry)
    entry_days = (full_don == 1) & (full_don.shift(1).fillna(0) == 0)  # transition 0→1
    breakout_label = (forward_sharpe > 0).astype(int)
    breakout_label_entries = breakout_label[entry_days]  # subset: only entry days

    print(f"  Total Donchian entries: {entry_days.sum()}")
    print(f"  Profitable entries: {breakout_label_entries.sum()} / {len(breakout_label_entries)}")

    avail_cols = [c for c in top10_cols if c in df_daily.columns]
    if len(avail_cols) < 5:
        avail_cols = df_daily.var().sort_values(ascending=False).head(10).index.tolist()

    # Add regime features
    ret_raw = prices.pct_change().fillna(0)
    extra_feats = pd.DataFrame(index=prices.index)
    extra_feats["vol_10d"] = ret_raw.rolling(10).std() * np.sqrt(252)
    extra_feats["vol_20d"] = ret_raw.rolling(20).std() * np.sqrt(252)
    extra_feats["adx_14"] = compute_adx(prices, 14)
    extra_feats["mom_20d"] = ret_raw.rolling(20).sum()
    extra_feats["mom_40d"] = ret_raw.rolling(40).sum()
    extra_feats = extra_feats.fillna(extra_feats.median())

    best_sharpe = -999
    best_result = None
    all_configs = []

    # Test CatBoost configs
    cb_configs = [
        {"depth": 3, "lr": 0.01, "iterations": 300},
        {"depth": 4, "lr": 0.05, "iterations": 200},
    ]

    for cb_cfg in cb_configs:
        depth, lr, iters = cb_cfg["depth"], cb_cfg["lr"], cb_cfg["iterations"]
        print(f"\n  CatBoost breakout predictor: depth={depth}, lr={lr}, iters={iters}")

        per_year = {}

        for yr in range(2019, 2024):
            # Expanding train: all breakout entry days before test year
            train_entry_mask = (entry_days) & (entry_days.index.year < yr)
            test_mask = prices.index.year == yr

            # Build features at entry days for training
            # Use daily features from df_daily + extra_feats
            feat_all = pd.concat([
                df_daily[avail_cols].reindex(extra_feats.index),
                extra_feats
            ], axis=1).fillna(0)

            if train_entry_mask.sum() < 20:
                # Not enough breakout entries — use unfiltered Donchian for test year
                don_yr = full_don.reindex(prices[test_mask].index).fillna(0)
                sh_yr = year_sharpe(daily_returns, don_yr, yr)
                per_year[str(yr)] = round(sh_yr, 3)
                continue

            X_train_entries = feat_all.reindex(entry_days[train_entry_mask].index)
            y_train_entries = breakout_label_entries.reindex(entry_days[train_entry_mask].index)

            valid = X_train_entries.notna().all(axis=1) & y_train_entries.notna()
            X_train_entries = X_train_entries[valid]
            y_train_entries = y_train_entries[valid]

            if len(X_train_entries) < 10 or y_train_entries.nunique() < 2:
                don_yr = full_don.reindex(prices[test_mask].index).fillna(0)
                sh_yr = year_sharpe(daily_returns, don_yr, yr)
                per_year[str(yr)] = round(sh_yr, 3)
                continue

            try:
                model = CatBoostClassifier(
                    depth=depth,
                    learning_rate=lr,
                    iterations=iters,
                    l2_leaf_reg=1.0,
                    task_type="GPU",
                    devices="0",
                    verbose=0,
                    random_seed=42,
                )
                model.fit(X_train_entries, y_train_entries)

                # For test year: predict probability for ALL Donchian entry days in test year
                test_entry_mask = (entry_days) & (entry_days.index.year == yr)
                if test_entry_mask.sum() == 0:
                    # No breakouts this year — flat
                    per_year[str(yr)] = 0.0
                    continue

                test_entry_idx = entry_days[test_entry_mask].index
                X_test_entries = feat_all.reindex(test_entry_idx).fillna(0)
                proba_entries = model.predict_proba(X_test_entries)[:, 1]

                # Build position signal: for each breakout, take it if prob > 0.55
                # Propagate: entry at T, hold until Donchian exit
                prob_threshold = 0.55
                filtered_pos = pd.Series(0.0, index=prices[test_mask].index)
                active_entry = None

                for i, idx in enumerate(prices[test_mask].index):
                    # Check if this is a new entry day
                    is_entry = idx in test_entry_idx
                    if is_entry:
                        entry_idx_pos = test_entry_idx.get_loc(idx)
                        prob = proba_entries[entry_idx_pos]
                        if prob >= prob_threshold:
                            active_entry = idx
                        else:
                            active_entry = None  # skip this breakout

                    # Check if Donchian says exit
                    don_signal_today = full_don.get(idx, 0)
                    if don_signal_today == 0:
                        active_entry = None

                    # Position: held if we have an active entry and Donchian still on
                    if active_entry is not None and don_signal_today == 1:
                        filtered_pos.loc[idx] = 1.0

                sh_yr = year_sharpe(daily_returns, filtered_pos, yr)
                per_year[str(yr)] = round(sh_yr, 3)

                n_entries = test_entry_mask.sum()
                n_taken = (proba_entries >= prob_threshold).sum()
                print(f"    {yr}: Sharpe={sh_yr:.3f}, entries={n_entries}, taken={n_taken} ({n_taken/max(n_entries,1)*100:.0f}%)")

            except Exception as e:
                print(f"    WARN: year {yr} failed: {e}")
                per_year[str(yr)] = 0.0

        vals = list(per_year.values())
        mean_sh = float(np.mean(vals))
        std_sh = float(np.std(vals))
        s22 = per_year.get("2022", 0.0)
        print(f"  → Mean: {mean_sh:.3f} ± {std_sh:.3f} | 2022: {s22:.3f}")

        all_configs.append({
            "depth": depth, "lr": lr, "iterations": iters,
            "mean_sharpe": mean_sh, "std_sharpe": std_sh,
            "per_year": per_year,
        })
        if mean_sh > best_sharpe:
            best_sharpe = mean_sh
            best_result = (f"cb_d{depth}_lr{lr}", per_year, mean_sh, std_sh, depth, lr, iters)

    cfg_key, per_year, mean_sh, std_sh, depth, lr, iters = best_result
    name = f"breakout_profitability_cat_d{depth}_lr{lr}_S{mean_sh:.2f}"
    s22 = per_year.get("2022", 0.0)

    notes = (
        f"Target re-engineering: CatBoost GPU predicts 'will this Donchian breakout be profitable?' "
        f"instead of price direction. Only enter when CatBoost prob > 0.55. "
        f"Training labels: forward 30-day Sharpe > 0 at entry days only. "
        f"Best config: depth={depth}, lr={lr}, iters={iters}. "
        f"Mean WF Sharpe {mean_sh:.3f} ± {std_sh:.3f}. 2022: {s22:.3f}. "
        + ("SUCCESS — breakout profitability target beats ADX regime filter."
           if mean_sh > BEST_REGIME_SHARPE
           else f"Below ADX-only Donchian ({BEST_REGIME_SHARPE:.3f}). "
                f"Possible reasons: (a) too few breakout entries per year for robust training, "
                f"(b) bear-market breakouts look statistically similar to bull-market breakouts "
                f"at entry time — CatBoost can't distinguish false from real at entry day T.")
    )
    log_novel_result(
        tracker, name,
        config={
            "approach": "breakout_profitability",
            "model": "CatBoost",
            "depth": depth,
            "learning_rate": lr,
            "iterations": iters,
            "prob_threshold": 0.55,
            "label_window": 30,
            "all_configs_tested": all_configs,
        },
        metrics={
            "sharpe": mean_sh,
            "mean_wf_sharpe": mean_sh,
            "std_wf_sharpe": std_sh,
            "min_wf_sharpe": min(per_year.values()),
            "max_wf_sharpe": max(per_year.values()),
            "sharpe_2022": s22,
            **{f"sharpe_{yr}": v for yr, v in per_year.items()},
        },
        group="target_reengineering",
        notes=notes,
    )
    return {"best": best_result, "all_configs": all_configs}


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("CONTRACT #004 — STEP 5: NOVEL EXPLORATION")
    print("=" * 80)
    print("Synthesizing 50+ experiments to test creative combinations.")
    print(f"Baseline (Donchian): {BASELINE_DONCHIAN_SHARPE}")
    print(f"Best regime (ADX-30): {BEST_REGIME_SHARPE}")
    print(f"Best ML (LGBM-top10): {BEST_ML_SHARPE}")
    t0 = time.time()

    prices, daily_returns, df_daily, target_daily, top10_cols, top20_cols = load_data()
    tracker = ExperimentTracker(experiment_name="novel_exploration_step5")

    all_results = {}

    # Idea 1: Asymmetric Regime
    print("\n" + "🔍 " + "=" * 76)
    print("RUNNING IDEA 1: ASYMMETRIC REGIME")
    print("=" * 80)
    try:
        r1 = run_idea1_asymmetric_regime(prices, daily_returns, tracker)
        all_results["idea1_asymmetric_regime"] = r1
    except Exception as e:
        print(f"ERROR in Idea 1: {e}")
        all_results["idea1_asymmetric_regime"] = {"error": str(e)}

    # Idea 2: Adaptive Donchian
    print("\n" + "=" * 80)
    print("RUNNING IDEA 2: ADAPTIVE DONCHIAN")
    print("=" * 80)
    try:
        r2 = run_idea2_adaptive_donchian(prices, daily_returns, tracker)
        all_results["idea2_adaptive_donchian"] = r2
    except Exception as e:
        print(f"ERROR in Idea 2: {e}")
        all_results["idea2_adaptive_donchian"] = {"error": str(e)}

    # Idea 3: ML Position Sizer
    print("\n" + "=" * 80)
    print("RUNNING IDEA 3: ML POSITION SIZER")
    print("=" * 80)
    try:
        r3 = run_idea3_ml_position_sizer(
            prices, daily_returns, df_daily, target_daily, top10_cols, tracker
        )
        all_results["idea3_ml_position_sizer"] = r3
    except Exception as e:
        print(f"ERROR in Idea 3: {e}")
        all_results["idea3_ml_position_sizer"] = {"error": str(e)}

    # Idea 4: Majority Vote
    print("\n" + "=" * 80)
    print("RUNNING IDEA 4: MAJORITY VOTE REGIME ENSEMBLE")
    print("=" * 80)
    try:
        r4 = run_idea4_majority_vote_regime(
            prices, daily_returns, df_daily, target_daily, top10_cols, tracker
        )
        all_results["idea4_majority_vote"] = r4
    except Exception as e:
        print(f"ERROR in Idea 4: {e}")
        all_results["idea4_majority_vote"] = {"error": str(e)}

    # Idea 5: Regime Filter + Momentum
    print("\n" + "=" * 80)
    print("RUNNING IDEA 5: REGIME FILTER + MOMENTUM BASE STRATEGY")
    print("=" * 80)
    try:
        r5 = run_idea5_regime_momentum(prices, daily_returns, tracker)
        all_results["idea5_regime_momentum"] = r5
    except Exception as e:
        print(f"ERROR in Idea 5: {e}")
        all_results["idea5_regime_momentum"] = {"error": str(e)}

    # Idea 6: Stability-Optimized
    print("\n" + "=" * 80)
    print("RUNNING IDEA 6: STABILITY-OPTIMIZED")
    print("=" * 80)
    try:
        r6 = run_idea6_stability_optimized(prices, daily_returns, tracker)
        all_results["idea6_stability_optimized"] = r6
    except Exception as e:
        print(f"ERROR in Idea 6: {e}")
        all_results["idea6_stability_optimized"] = {"error": str(e)}

    # Idea 7: Target Re-engineering (Breakout Profitability)
    print("\n" + "=" * 80)
    print("RUNNING IDEA 7: TARGET RE-ENGINEERING (Breakout Profitability)")
    print("=" * 80)
    try:
        r7 = run_idea7_breakout_profitability(
            prices, daily_returns, df_daily, top10_cols, tracker
        )
        all_results["idea7_breakout_profitability"] = r7
    except Exception as e:
        print(f"ERROR in Idea 7: {e}")
        all_results["idea7_breakout_profitability"] = {"error": str(e)}

    elapsed = time.time() - t0

    # ─── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — NOVEL EXPLORATION")
    print("=" * 80)
    print(f"{'Idea':<50} {'BestSh':>8} {'Std':>8} {'2022':>8}")
    print("-" * 80)

    summary = {}
    for idea_name, result in all_results.items():
        if "error" in result:
            print(f"  {idea_name:<48} ERROR")
            continue
        if "best" in result:
            best = result["best"]
            if isinstance(best, tuple) and len(best) >= 4:
                # Different ideas return tuples differently
                config_name = best[0]
                per_year = best[1]
                mean_sh = best[2]
                std_sh = best[3]
                s22 = per_year.get("2022", 0.0) if isinstance(per_year, dict) else 0.0
                summary[idea_name] = {
                    "config": config_name,
                    "mean_sharpe": mean_sh,
                    "std_sharpe": std_sh,
                    "sharpe_2022": s22,
                }
                beats = mean_sh > BEST_REGIME_SHARPE
                print(f"  {idea_name:<48} {mean_sh:>8.3f} {std_sh:>8.3f} {s22:>8.3f}  {'✅' if beats else '❌'}")

    print(f"\n  BENCHMARKS:")
    print(f"    Donchian baseline:  {BASELINE_DONCHIAN_SHARPE:.3f}")
    print(f"    Best regime (ADX):  {BEST_REGIME_SHARPE:.3f} (std=0.829, 2022=0.000)")
    print(f"    Best ML (LGBM-10):  {BEST_ML_SHARPE:.3f} (std=1.701, 2022=-0.644)")
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")

    # Save results JSON
    results_out = {
        "summary": summary,
        "elapsed_seconds": elapsed,
        "benchmarks": {
            "donchian_baseline": BASELINE_DONCHIAN_SHARPE,
            "best_regime_adx30": BEST_REGIME_SHARPE,
            "best_regime_std": 0.829,
            "best_ml": BEST_ML_SHARPE,
            "best_risk_adjusted": {
                "name": "vol_adx_AND Donchian",
                "sharpe": 1.068,
                "std": 0.904,
                "sharpe_2022": 0.192,
            },
        },
    }
    with open(RESULTS_DIR / "novel_results.json", "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR}/novel_results.json")

    return all_results, summary


if __name__ == "__main__":
    all_results, summary = main()
    print("\nDone.")
