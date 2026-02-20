"""Validation and investigation functions for strategy candidates.

PROTECTED FILE — research agents call these, not edit them.

Usage:
    from analysis_runner import run_full_validation_battery, run_full_investigation
    v = run_full_validation_battery(config)
    i = run_full_investigation(config)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from experiment_runner import build_strategy  # noqa: E402
from sparky.backtest.cpcv import cpcv_paths  # noqa: E402
from sparky.tracking.metrics import compute_all_metrics  # noqa: E402
from sweep_utils import net_ret  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _setup(config):
    """Build strategy + compute net returns at 30bps. Returns (prices, positions, returns_30, ppy)."""
    prices, positions, _df, ppy = build_strategy(config)
    returns_30 = net_ret(prices, positions, 30 / 10_000).dropna()
    return prices, positions, returns_30, ppy


def _safe(func, config, **kwargs):
    """Call func(config, **kwargs), catching exceptions."""
    try:
        return func(config, **kwargs)
    except Exception as e:
        logger.exception(f"{func.__name__} failed")
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# 1-12: Validation functions
# ---------------------------------------------------------------------------


def stress_test(config, cost_range_bps=None):
    """Sharpe at multiple cost levels + breakeven cost."""
    if cost_range_bps is None:
        cost_range_bps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
    prices, positions, _, ppy = _setup(config)
    results = {}
    for bps in cost_range_bps:
        ret = net_ret(prices, positions, bps / 10_000).dropna()
        if len(ret) < 30:
            continue
        m = compute_all_metrics(ret, n_trials=1, periods_per_year=ppy)
        results[bps] = {"sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"], "calmar": m["calmar"]}

    sharpes = {bps: v["sharpe"] for bps, v in results.items()}
    breakeven = None
    for bps in sorted(sharpes):
        if sharpes[bps] <= 0:
            breakeven = bps
            break
    if breakeven is None and sharpes:
        breakeven = max(sharpes.keys()) + 10  # beyond tested range

    return {
        "status": "ok",
        "results": results,
        "sharpe_per_cost": sharpes,
        "breakeven_bps": breakeven,
        "plots_data": {"cost_bps": list(sharpes.keys()), "sharpe": list(sharpes.values()), "breakeven": breakeven},
    }


def bootstrap_sharpe(config, n_samples=10000, block_size=20, seed=42):
    """Block-resample returns, compute Sharpe/MaxDD/Calmar distributions.

    NOTE: maxdd/calmar distributions measure sampling uncertainty over these metrics,
    not path-realistic risk — block resampling breaks path continuity. Use
    maxdd_percentiles for hard-fail gating but treat calmar_percentiles as indicative only.
    """
    prices, positions, returns_30, ppy = _setup(config)
    n = len(returns_30)
    arr = returns_30.values
    rng = np.random.RandomState(seed)

    sharpes = np.empty(n_samples)
    maxdd_dist = np.empty(n_samples)
    calmar_dist = np.empty(n_samples)
    n_blocks = int(np.ceil(n / block_size))
    for i in range(n_samples):
        blocks = []
        for _ in range(n_blocks):
            start = rng.randint(0, max(1, n - block_size + 1))
            blocks.append(arr[start : start + block_size])
        sample = np.concatenate(blocks)[:n]
        std = np.std(sample, ddof=1)
        sr = (np.mean(sample) / std * np.sqrt(ppy)) if std > 0 else 0.0
        sharpes[i] = sr
        cum = np.cumprod(1 + sample)
        running_max = np.maximum.accumulate(cum)
        dd = cum / running_max - 1
        mdd = float(dd.min())
        maxdd_dist[i] = mdd
        ann_ret = cum[-1] ** (ppy / len(sample)) - 1 if cum[-1] > 0 else 0.0
        calmar_dist[i] = ann_ret / abs(mdd) if mdd < -1e-8 else 0.0

    pcts = {int(p): float(np.percentile(sharpes, p)) for p in [5, 25, 50, 75, 95]}
    maxdd_pcts = {int(p): float(np.percentile(maxdd_dist, p)) for p in [5, 25, 50, 75, 95]}
    calmar_pcts = {int(p): float(np.percentile(calmar_dist, p)) for p in [5, 25, 50, 75, 95]}
    return {
        "status": "ok",
        "percentiles": pcts,
        "maxdd_percentiles": maxdd_pcts,
        "calmar_percentiles": calmar_pcts,
        "ci_95": (pcts[5], pcts[95]),
        "mean": float(np.mean(sharpes)),
        "std": float(np.std(sharpes)),
        "n_samples": n_samples,
        "plots_data": {
            "sharpe_dist": sharpes.tolist(),
            "maxdd_dist": maxdd_dist.tolist(),
            "calmar_dist": calmar_dist.tolist(),
            "percentiles": pcts,
            "maxdd_percentiles": maxdd_pcts,
            "calmar_percentiles": calmar_pcts,
        },
    }


def walk_forward_multi(config, window_sizes_days=None):
    """Evaluate on non-overlapping windows of various sizes."""
    if window_sizes_days is None:
        window_sizes_days = [90, 180, 365]
    _, _, returns_30, ppy = _setup(config)
    n = len(returns_30)

    results = {}
    for wsize in window_sizes_days:
        n_windows = n // wsize
        if n_windows < 2:
            continue
        window_sharpes = []
        for j in range(n_windows):
            chunk = returns_30.iloc[j * wsize : (j + 1) * wsize]
            if len(chunk) < 20:
                continue
            std = np.std(chunk, ddof=1)
            sr = (np.mean(chunk) / std * np.sqrt(ppy)) if std > 0 else 0.0
            window_sharpes.append(float(sr))
        frac_positive = np.mean(np.array(window_sharpes) > 0) if window_sharpes else 0
        results[wsize] = {
            "n_windows": len(window_sharpes),
            "sharpes": window_sharpes,
            "frac_positive": float(frac_positive),
            "mean_sharpe": float(np.mean(window_sharpes)) if window_sharpes else 0,
        }

    return {
        "status": "ok",
        "results": results,
        "plots_data": {ws: r["sharpes"] for ws, r in results.items()},
    }


def subsample_stability(config, drop_rates=None, repetitions=100, block_size=20, seed=42):
    """Drop entire blocks of returns (preserving temporal order), recompute Sharpe."""
    if drop_rates is None:
        drop_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    _, _, returns_30, ppy = _setup(config)
    arr = returns_30.values
    n = len(arr)
    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(n / block_size))

    results = {}
    for rate in drop_rates:
        drop_count = int(np.ceil(rate * n_blocks))
        sharpes = []
        for _ in range(repetitions):
            drop_idx = set(rng.choice(n_blocks, size=drop_count, replace=False))
            kept = [arr[b * block_size : (b + 1) * block_size] for b in range(n_blocks) if b not in drop_idx]
            if not kept:
                continue
            sample = np.concatenate(kept)
            std = np.std(sample, ddof=1)
            sr = (np.mean(sample) / std * np.sqrt(ppy)) if std > 0 else 0.0
            sharpes.append(sr)
        results[rate] = {"mean": float(np.mean(sharpes)), "std": float(np.std(sharpes))}

    return {
        "status": "ok",
        "results": results,
        "plots_data": {
            "drop_rates": drop_rates,
            "means": [results[r]["mean"] for r in drop_rates],
            "stds": [results[r]["std"] for r in drop_rates],
        },
    }


def cpcv_validate(config, n_groups=6, purge_days=5):
    """Combinatorial Purged Cross-Validation."""
    _, _, returns_30, ppy = _setup(config)
    result = cpcv_paths(returns_30.values, n_groups=n_groups, purge_days=purge_days, ppy=ppy)
    return {
        "status": "ok",
        "pbo": result["pbo"],
        "n_paths": result["n_paths"],
        "median_path_sharpe": result["median_path_sharpe"],
        "mean_path_sharpe": result["mean_path_sharpe"],
        "sharpe_distribution": result["sharpe_distribution"],
        "plots_data": {"sharpe_dist": result["sharpe_distribution"], "pbo": result["pbo"]},
    }


def multi_seed_test(config, seeds=None):
    """For deterministic strategies: trivially std=0. Tests infra consistency."""
    if seeds is None:
        seeds = list(range(5))
    sharpes = []
    for _ in seeds:
        _, _, returns_30, ppy = _setup(config)
        m = compute_all_metrics(returns_30, n_trials=1, periods_per_year=ppy)
        sharpes.append(m["sharpe"])
    return {
        "status": "ok",
        "sharpes": sharpes,
        "mean": float(np.mean(sharpes)),
        "std": float(np.std(sharpes)),
    }


def tail_risk_analysis(config, confidence_levels=None):
    """CVaR at multiple levels, worst drawdowns, max underwater days."""
    if confidence_levels is None:
        confidence_levels = [0.01, 0.025, 0.05, 0.10]
    _, _, returns_30, ppy = _setup(config)
    arr = returns_30.values

    # CVaR at each level
    cvar = {}
    for alpha in confidence_levels:
        cutoff = np.percentile(arr, alpha * 100)
        tail = arr[arr <= cutoff]
        cvar[alpha] = float(np.mean(tail)) if len(tail) > 0 else 0.0

    # Top 5 worst drawdowns
    cum = np.cumprod(1 + arr)
    running_max = np.maximum.accumulate(cum)
    dd_series = cum / running_max - 1
    # Find drawdown troughs
    dd_df = pd.Series(dd_series, index=returns_30.index)
    worst_5 = _top_drawdowns(dd_df, n=5)

    # Max underwater days
    underwater = dd_series < -0.001
    max_underwater = _max_consecutive(underwater)

    mean_daily = float(np.mean(arr))
    return {
        "status": "ok",
        "cvar": cvar,
        "worst_5_drawdowns": worst_5,
        "max_underwater_periods": max_underwater,
        "mean_daily_return": mean_daily,
        "plots_data": {"dd_series": dd_series.tolist(), "index": returns_30.index.strftime("%Y-%m-%d").tolist()},
    }


def _top_drawdowns(dd_series, n=5):
    """Find top N drawdown episodes with start/end/trough dates."""
    results = []
    dd = dd_series.copy()
    for _ in range(n):
        if dd.min() >= -0.001:
            break
        trough_idx = dd.idxmin()
        trough_val = float(dd[trough_idx])
        # Walk backward to find start
        start_idx = trough_idx
        for i in range(dd.index.get_loc(trough_idx) - 1, -1, -1):
            if dd.iloc[i] >= -0.001:
                start_idx = dd.index[i]
                break
        # Walk forward to find recovery
        end_idx = dd.index[-1]
        for i in range(dd.index.get_loc(trough_idx) + 1, len(dd)):
            if dd.iloc[i] >= -0.001:
                end_idx = dd.index[i]
                break
        results.append(
            {
                "start": str(start_idx),
                "trough": str(trough_idx),
                "end": str(end_idx),
                "depth": trough_val,
                "duration": int((pd.Timestamp(end_idx) - pd.Timestamp(start_idx)).days),
            }
        )
        # Mask this episode
        mask = (dd.index >= start_idx) & (dd.index <= end_idx)
        dd[mask] = 0.0
    return results


def _max_consecutive(mask):
    """Max consecutive True values."""
    if not np.any(mask):
        return 0
    groups = np.diff(np.where(np.concatenate(([mask[0]], mask[:-1] != mask[1:], [True])))[0])
    if len(groups) == 0:
        return 0
    return int(max(groups[::2]) if mask[0] else (max(groups[1::2]) if len(groups) > 1 else 0))


def drawdown_analysis(config):
    """Max DD, recovery times for top 5 drawdowns, underwater curve."""
    _, _, returns_30, _ = _setup(config)
    arr = returns_30.values
    cum = np.cumprod(1 + arr)
    running_max = np.maximum.accumulate(cum)
    dd_series = pd.Series(cum / running_max - 1, index=returns_30.index)

    max_dd = float(dd_series.min())
    worst_5 = _top_drawdowns(dd_series, n=5)

    return {
        "status": "ok",
        "max_drawdown": max_dd,
        "worst_5": worst_5,
        "plots_data": {
            "dd_series": dd_series.values.tolist(),
            "index": dd_series.index.strftime("%Y-%m-%d").tolist(),
            "max_dd": max_dd,
        },
    }


def rolling_stability(config, window_days=180):
    """Rolling Sharpe + flag periods where Sharpe < 0 for extended time."""
    _, _, returns_30, ppy = _setup(config)
    n = len(returns_30)
    if n < window_days * 2:
        return {"status": "error", "error": f"Not enough data ({n}) for {window_days}-day rolling window"}

    rolling_sr = []
    dates = []
    for i in range(n - window_days):
        chunk = returns_30.iloc[i : i + window_days]
        std = np.std(chunk, ddof=1)
        sr = (np.mean(chunk) / std * np.sqrt(ppy)) if std > 0 else 0.0
        rolling_sr.append(float(sr))
        dates.append(str(returns_30.index[i + window_days]))

    rolling_sr = np.array(rolling_sr)
    # Flag contiguous periods where rolling Sharpe < 0
    negative = rolling_sr < 0
    flagged = []
    in_neg = False
    start = None
    for i, neg in enumerate(negative):
        if neg and not in_neg:
            in_neg = True
            start = i
        elif not neg and in_neg:
            duration_days = (pd.Timestamp(dates[i - 1]) - pd.Timestamp(dates[start])).days
            if duration_days >= 180:
                flagged.append({"start": dates[start], "end": dates[i - 1], "duration_days": duration_days})
            in_neg = False
    if in_neg:
        duration_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[start])).days
        if duration_days >= 180:
            flagged.append({"start": dates[start], "end": dates[-1], "duration_days": duration_days})

    return {
        "status": "ok",
        "flagged_periods": flagged,
        "mean_rolling_sharpe": float(np.mean(rolling_sr)),
        "std_rolling_sharpe": float(np.std(rolling_sr)),
        "pct_negative": float(np.mean(negative)),
        "plots_data": {"rolling_sharpe": rolling_sr.tolist(), "dates": dates, "flagged": flagged},
    }


def sub_period_analysis(config, periods=None):
    """Sharpe + MaxDD for multiple sub-periods."""
    if periods is None:
        periods = [
            ("full", None, None),
            ("2019-2020", "2019-01-01", "2020-12-31"),
            ("2021", "2021-01-01", "2021-12-31"),
            ("2022", "2022-01-01", "2022-12-31"),
            ("2023", "2023-01-01", "2023-12-31"),
        ]
    _, _, returns_30, ppy = _setup(config)

    results = {}
    for label, start, end in periods:
        r = returns_30
        if start:
            r = r[r.index >= start]
        if end:
            r = r[r.index <= end]
        if len(r) < 20:
            results[label] = {"sharpe": None, "max_drawdown": None, "n_obs": len(r)}
            continue
        m = compute_all_metrics(r, n_trials=1, periods_per_year=ppy)
        results[label] = {
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "calmar": m["calmar"],
            "n_obs": len(r),
            "win_rate": m["win_rate"],
        }

    return {
        "status": "ok",
        "results": results,
        "plots_data": {
            "periods": [p[0] for p in periods],
            "sharpes": [results.get(p[0], {}).get("sharpe") for p in periods],
            "max_dds": [results.get(p[0], {}).get("max_drawdown") for p in periods],
        },
    }


def correlation_stability(config, window_days=90):
    """Rolling correlation with buy-and-hold."""
    prices, positions, returns_30, ppy = _setup(config)
    bh_ret = prices.pct_change().dropna()
    common = returns_30.index.intersection(bh_ret.index)
    strat = returns_30.loc[common]
    bh = bh_ret.loc[common]

    if len(common) < window_days * 2:
        return {"status": "error", "error": "Not enough data for rolling correlation"}

    rolling_corr = strat.rolling(window_days).corr(bh).dropna()
    return {
        "status": "ok",
        "mean_corr": float(rolling_corr.mean()),
        "std_corr": float(rolling_corr.std()),
        "max_corr": float(rolling_corr.max()),
        "min_corr": float(rolling_corr.min()),
        "plots_data": {
            "rolling_corr": rolling_corr.values.tolist(),
            "dates": rolling_corr.index.strftime("%Y-%m-%d").tolist(),
        },
    }


def target_vol_frontier(config, tv_range=None):
    """Rebuild positions at each target_vol, evaluate metrics."""
    if tv_range is None:
        tv_range = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    from experiment_runner import _load_data, _make_signal  # noqa: E402
    from sweep_utils import inv_vol_sizing  # noqa: E402

    cfg = config.copy()
    asset, tf = cfg["asset"], cfg["timeframe"]
    _df, prices, ppy = _load_data(asset, tf)
    raw_signal = _make_signal(prices, cfg["signal_type"], cfg.get("signal_params", {}))
    vw = cfg.get("sizing_params", {}).get("vol_window", 30)

    results = {}
    for tv in tv_range:
        scale = inv_vol_sizing(prices, vw=vw, tv=tv, periods_per_year=ppy)
        pos = (raw_signal * scale).fillna(0)
        ret = net_ret(prices, pos, 30 / 10_000).dropna()
        if len(ret) < 30:
            continue
        m = compute_all_metrics(ret, n_trials=1, periods_per_year=ppy)
        results[tv] = {"sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"], "calmar": m["calmar"]}

    return {
        "status": "ok",
        "results": results,
        "plots_data": {
            "target_vols": list(results.keys()),
            "sharpes": [v["sharpe"] for v in results.values()],
            "max_dds": [v["max_drawdown"] for v in results.values()],
            "calmars": [v["calmar"] for v in results.values()],
        },
    }


# ---------------------------------------------------------------------------
# 13-15: Investigation functions
# ---------------------------------------------------------------------------


def trade_profile(config):
    """Win rate, avg win/loss, profit factor, holding period distribution, top trades."""
    prices, positions, returns_30, ppy = _setup(config)
    pos = positions.reindex(returns_30.index).fillna(0)

    # Identify trade boundaries (position changes)
    trades = []
    in_trade = False
    entry_idx = None
    for i in range(1, len(pos)):
        if pos.iloc[i] > 0.01 and pos.iloc[i - 1] <= 0.01:
            in_trade = True
            entry_idx = i
        elif pos.iloc[i] <= 0.01 and pos.iloc[i - 1] > 0.01 and in_trade:
            in_trade = False
            trade_ret = returns_30.iloc[entry_idx:i].sum()
            trades.append(
                {
                    "entry": str(returns_30.index[entry_idx]),
                    "exit": str(returns_30.index[i]),
                    "return": float(trade_ret),
                    "holding_periods": i - entry_idx,
                }
            )
    if not trades:
        return {"status": "ok", "n_trades": 0, "win_rate": 0, "trades": []}

    rets = np.array([t["return"] for t in trades])
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    holds = [t["holding_periods"] for t in trades]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 0 else float("inf")

    sorted_trades = sorted(trades, key=lambda t: t["return"])
    return {
        "status": "ok",
        "n_trades": len(trades),
        "win_rate": float(np.mean(rets > 0)),
        "avg_win": float(np.mean(wins)) if len(wins) > 0 else 0,
        "avg_loss": float(np.mean(losses)) if len(losses) > 0 else 0,
        "profit_factor": pf,
        "avg_holding_periods": float(np.mean(holds)),
        "median_holding_periods": float(np.median(holds)),
        "top_5_best": sorted_trades[-5:][::-1],
        "top_5_worst": sorted_trades[:5],
        "plots_data": {"holding_periods": holds, "trade_returns": rets.tolist()},
    }


def edge_attribution(config):
    """Ablation: compare full strategy to random entries and flat sizing."""
    prices, positions, returns_30, ppy = _setup(config)
    full_sharpe = float(compute_all_metrics(returns_30, n_trials=1, periods_per_year=ppy)["sharpe"])

    rng = np.random.RandomState(42)
    n = len(positions)

    # (a) Random entries with median holding period
    pos_nonzero = positions[positions > 0.01]
    if len(pos_nonzero) > 0:
        diffs = positions.diff().abs()
        entries = (diffs > 0.01) & (positions > 0.01)
        n_entries = entries.sum()
        if n_entries > 0:
            avg_hold = len(pos_nonzero) / max(n_entries, 1)
        else:
            avg_hold = 20
    else:
        avg_hold = 20

    random_pos = pd.Series(0.0, index=positions.index)
    i = 0
    while i < n:
        if rng.random() < (1.0 / max(avg_hold, 1)):
            hold = int(avg_hold)
            random_pos.iloc[i : min(i + hold, n)] = 1.0
            i += hold
        else:
            i += 1
    ret_random = net_ret(prices, random_pos, 30 / 10_000).dropna()
    random_sharpe = 0.0
    if len(ret_random) > 30:
        random_sharpe = float(compute_all_metrics(ret_random, n_trials=1, periods_per_year=ppy)["sharpe"])

    # (b) Flat sizing (remove inv_vol)
    from experiment_runner import _load_data, _make_signal  # noqa: E402

    _df, p, _ = _load_data(config["asset"], config["timeframe"])
    flat_pos = _make_signal(p, config["signal_type"], config.get("signal_params", {}))
    ret_flat = net_ret(p, flat_pos, 30 / 10_000).dropna()
    flat_sharpe = 0.0
    if len(ret_flat) > 30:
        flat_sharpe = float(compute_all_metrics(ret_flat, n_trials=1, periods_per_year=ppy)["sharpe"])

    return {
        "status": "ok",
        "full_sharpe": full_sharpe,
        "random_entry_sharpe": random_sharpe,
        "flat_sizing_sharpe": flat_sharpe,
        "signal_edge": full_sharpe - random_sharpe,
        "sizing_edge": full_sharpe - flat_sharpe,
    }


def regime_decomposition(config):
    """Classify bull/bear/sideways regimes. Per-regime and crisis-event metrics."""
    prices, positions, returns_30, ppy = _setup(config)

    # Regime classification based on 90-day rolling return (calendar-aware)
    price_series = prices.reindex(returns_30.index).ffill()
    lookback = int(90 * ppy / 365)
    roll_ret = price_series.pct_change(lookback).fillna(0)

    regimes = pd.Series("sideways", index=returns_30.index)
    regimes[roll_ret > 0.20] = "bull"
    regimes[roll_ret < -0.20] = "bear"

    regime_metrics = {}
    for regime in ["bull", "bear", "sideways"]:
        mask = regimes == regime
        r = returns_30[mask]
        if len(r) < 20:
            regime_metrics[regime] = {"sharpe": None, "n_obs": int(mask.sum())}
            continue
        m = compute_all_metrics(r, n_trials=1, periods_per_year=ppy, strict_ppy=False)
        regime_metrics[regime] = {
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "win_rate": m["win_rate"],
            "n_obs": int(mask.sum()),
        }

    # Crisis events
    crisis_events = [
        ("COVID crash", "2020-02-20", "2020-04-15"),
        ("May 2021 crash", "2021-05-10", "2021-06-30"),
        ("Luna collapse", "2022-05-01", "2022-06-30"),
        ("FTX collapse", "2022-11-01", "2022-12-31"),
    ]
    crisis_metrics = {}
    for name, start, end in crisis_events:
        r = returns_30[(returns_30.index >= start) & (returns_30.index <= end)]
        if len(r) < 5:
            continue
        total_ret = float(np.prod(1 + r) - 1)
        crisis_metrics[name] = {"total_return": total_ret, "n_obs": len(r)}

    # Monthly return heatmap data
    monthly = returns_30.resample("ME").apply(lambda x: float(np.prod(1 + x) - 1))
    heatmap = {}
    for dt, val in monthly.items():
        yr = dt.year
        mo = dt.month
        heatmap.setdefault(yr, {})[mo] = round(val * 100, 2)

    return {
        "status": "ok",
        "regime_metrics": regime_metrics,
        "crisis_metrics": crisis_metrics,
        "monthly_heatmap": heatmap,
        "plots_data": {
            "regime_metrics": regime_metrics,
            "crisis_metrics": crisis_metrics,
            "monthly_heatmap": heatmap,
        },
    }


# ---------------------------------------------------------------------------
# 16-17: Convenience functions
# ---------------------------------------------------------------------------

_HARD_FAIL = {
    "stress_test": lambda r: r.get("breakeven_bps", 999) < 70,
    "bootstrap_sharpe": lambda r: (
        r.get("percentiles", {}).get(5, 999) < 0.5 or r.get("maxdd_percentiles", {}).get(5, 0) < -0.40
    ),
    "cpcv_validate": lambda r: r.get("pbo", 1) > 0.50,
    "drawdown_analysis": lambda r: r.get("max_drawdown", -1) < -0.45,
}

_SOFT_FAIL = {
    "stress_test": lambda r: (r.get("results", {}).get(50, {}) or {}).get("sharpe", 999) < 1.0,
    "bootstrap_sharpe": lambda r: r.get("percentiles", {}).get(5, 999) < 0.8,
    "walk_forward_multi": lambda r: any(v.get("frac_positive", 1) < 0.80 for v in r.get("results", {}).values()),
    "subsample_stability": lambda r: _subsample_degrades(r),
    "cpcv_validate": lambda r: r.get("pbo", 0) > 0.30,
    "multi_seed_test": lambda r: r.get("std", 0) > 0.15,
    "tail_risk_analysis": lambda r: _tail_risk_excess(r),
    "rolling_stability": lambda r: len(r.get("flagged_periods", [])) > 0,
    "sub_period_analysis": lambda r: _sub_period_negatives(r) >= 2,
    "correlation_stability": lambda r: r.get("max_corr", 0) > 0.8,
}


def _subsample_degrades(r):
    results = r.get("results", {})
    if not results:
        return False
    rates = sorted(results.keys())
    if len(rates) < 2:
        return False
    base = results[rates[0]]["mean"]
    worst = results[rates[-1]]["mean"]
    return base > 0 and (base - worst) / abs(base) > 0.20


def _sub_period_negatives(r):
    results = r.get("results", {})
    count = 0
    for label, metrics in results.items():
        if label == "full":
            continue
        if metrics.get("sharpe") is not None and metrics["sharpe"] < 0:
            count += 1
    return count


def _tail_risk_excess(r):
    cvar = r.get("cvar", {})
    mean_daily = r.get("mean_daily_return", 0)
    if mean_daily == 0:
        return False
    cvar_05 = cvar.get(0.05, 0)
    return abs(cvar_05) > 3 * abs(mean_daily)


def run_full_validation_battery(config):
    """Run all 12 validation functions. Returns combined results with verdict."""
    tests = {
        "stress_test": lambda: stress_test(config),
        "bootstrap_sharpe": lambda: bootstrap_sharpe(config, n_samples=5000, seed=42),
        "walk_forward_multi": lambda: walk_forward_multi(config),
        "subsample_stability": lambda: subsample_stability(config, repetitions=50, seed=42),
        "cpcv_validate": lambda: cpcv_validate(config),
        "multi_seed_test": lambda: multi_seed_test(config),
        "tail_risk_analysis": lambda: tail_risk_analysis(config),
        "drawdown_analysis": lambda: drawdown_analysis(config),
        "rolling_stability": lambda: rolling_stability(config),
        "sub_period_analysis": lambda: sub_period_analysis(config),
        "correlation_stability": lambda: correlation_stability(config),
        "target_vol_frontier": lambda: (
            target_vol_frontier(config) if config.get("sizing") == "inverse_vol" else {"status": "skipped"}
        ),
    }

    results = {}
    hard_fails = []
    soft_fails = []

    for name, func in tests.items():
        try:
            r = func()
        except Exception as e:
            r = {"status": "error", "error": str(e)}

        verdict = "pass"
        if r.get("status") == "error":
            verdict = "error"
        else:
            if name in _HARD_FAIL and _HARD_FAIL[name](r):
                verdict = "hard_fail"
                hard_fails.append(name)
            elif name in _SOFT_FAIL and _SOFT_FAIL[name](r):
                verdict = "soft_fail"
                soft_fails.append(name)
        r["verdict"] = verdict
        results[name] = r

    overall = "PASS" if not hard_fails else "FAIL"
    if not hard_fails and soft_fails:
        overall = "CONDITIONAL"

    return {
        "status": "ok",
        "overall_verdict": overall,
        "hard_fails": hard_fails,
        "soft_fails": soft_fails,
        "tests": results,
    }


def run_full_investigation(config):
    """Run trade_profile + edge_attribution + regime_decomposition + target_vol_frontier."""
    results = {
        "trade_profile": _safe(trade_profile, config),
        "edge_attribution": _safe(edge_attribution, config),
        "regime_decomposition": _safe(regime_decomposition, config),
    }
    if config.get("sizing") == "inverse_vol":
        results["target_vol_frontier"] = _safe(target_vol_frontier, config)
    return {"status": "ok", "results": results}
