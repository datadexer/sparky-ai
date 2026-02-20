"""Shared utilities for regime Donchian sweep scripts.

PROTECTED FILE — research agents cannot edit this file directly.
To request changes, write a GATE_REQUEST.md in the project root describing
what you need and why. An oversight session will review and apply the change.
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sparky.data.loader import load  # noqa: E402
from sparky.models.simple_baselines import donchian_channel_strategy  # noqa: E402
from sparky.tracking.experiment import ExperimentTracker  # noqa: E402
from sparky.tracking.guardrails import (  # noqa: E402
    check_costs_specified,
    check_holdout_boundary,
    check_no_lookahead,
    has_blocking_failure,
    log_results,
    run_post_checks,
)
from sparky.tracking.metrics import compute_all_metrics  # noqa: E402

PERIODS_PER_YEAR = 365
OUT_DIR = Path("results/regime_donchian")


def load_daily():
    df = load("btc_daily", purpose="training")
    print(f"[DATA] {len(df)} rows: {df.index.min().date()} → {df.index.max().date()}")
    return df


def net_ret(prices, positions, cf):
    pr = prices.pct_change()
    lp = positions.shift(1).fillna(0)
    costs = lp.diff().abs().fillna(0) * cf
    return (lp * pr - costs).dropna()


def run_pre(data, config):
    pre = [
        check_holdout_boundary(data, asset="btc"),
        check_no_lookahead(data, config),
        check_costs_specified(config),
    ]
    if has_blocking_failure(pre):
        for r in pre:
            if not r.passed and r.severity == "block":
                print(f"  [BLOCK] {r.check_name}: {r.message}")
        return False
    return True


def evaluate(prices, positions, config, n_trials, data_df, costs_bps=30):
    cf = costs_bps / 10_000
    config = {**config, "transaction_costs_bps": costs_bps}
    n_trades = int((positions.diff().abs().fillna(0) > 0.01).sum())
    if n_trades < 5:
        return None
    if not run_pre(data_df, config):
        return None
    ret = net_ret(prices, positions, cf)
    m = compute_all_metrics(ret, n_trials=n_trials, periods_per_year=PERIODS_PER_YEAR)
    m["n_trades"] = n_trades
    m["pct_long"] = round(float((positions > 0).mean() * 100), 2)
    m["statistically_significant"] = bool(m["dsr"] >= 0.95)
    post = run_post_checks(ret, m, config)
    log_results(post, run_id=str(config)[:100])
    return m


def yearly_sharpes(prices, positions, cf):
    ret = net_ret(prices, positions, cf)
    out = {}
    for yr in sorted(ret.index.year.unique()):
        r = ret[ret.index.year == yr]
        if len(r) < 30:
            continue
        try:
            out[yr] = round(compute_all_metrics(r, n_trials=1, periods_per_year=PERIODS_PER_YEAR)["sharpe"], 3)
        except Exception:
            out[yr] = float("nan")
    return out


SUB_PERIODS = [
    ("full", None, None),
    ("2017+", "2017-01-01", None),
    ("2020+", "2020-01-01", None),
]


def subperiod_analysis(prices, positions, cf, periods_per_year=PERIODS_PER_YEAR):
    ret = net_ret(prices, positions, cf)
    bh_ret = prices.pct_change().dropna()
    out = {}
    for label, start, end in SUB_PERIODS:
        r = ret if start is None else ret[ret.index >= start]
        b = bh_ret if start is None else bh_ret[bh_ret.index >= start]
        if len(r) < 30:
            continue
        m = compute_all_metrics(r, n_trials=1, periods_per_year=periods_per_year)
        bh_m = compute_all_metrics(b, n_trials=1, periods_per_year=periods_per_year)
        p_slice = positions[positions.index >= start] if start else positions
        out[label] = {
            "sharpe": round(m["sharpe"], 4),
            "max_drawdown": round(m["max_drawdown"], 4),
            "annual_return": round(m["mean_return"] * periods_per_year, 4),
            "n_trades": int((p_slice.diff().abs().fillna(0) > 0.01).sum()),
            "win_rate": round(m["win_rate"], 4),
            "bh_sharpe": round(bh_m["sharpe"], 4),
        }
    return out


def rvol(prices, w=20, periods_per_year=PERIODS_PER_YEAR):
    return np.log(prices / prices.shift(1)).rolling(w).std() * np.sqrt(periods_per_year)


def inv_vol_sizing(prices, vw=20, tv=0.4, periods_per_year=PERIODS_PER_YEAR):
    return (tv / rvol(prices, vw, periods_per_year)).clip(0.1, 1.5).fillna(0.5)


def baseline_donchian(prices, ep=40, xp=20):
    return donchian_channel_strategy(prices, entry_period=ep, exit_period=xp).astype(float)


def sanitize(obj):
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(data, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    with open(path, "w") as f:
        json.dump(sanitize(data), f, indent=2, default=str)
    print(f"[SAVED] {path}")


def wandb_log_sweep(experiment_name, sweep_name, results, summary, tags):
    try:
        tracker = ExperimentTracker(experiment_name=experiment_name)
        tracker.log_sweep(sweep_name, results, summary_metrics=summary, tags=tags)
        print(f"[WANDB] Logged {len(results)} results")
    except Exception as e:
        print(f"[WANDB] Failed: {e}")


def wandb_log_experiment(experiment_name, run_name, config, metrics, tags):
    try:
        tracker = ExperimentTracker(experiment_name=experiment_name)
        tracker.log_experiment(run_name, config=config, metrics=metrics, tags=tags)
    except Exception as e:
        print(f"[WANDB] Failed: {e}")


def print_top(results, baseline_sharpe, n=10):
    def _dsr(r):
        return (r.get("metrics") or r).get("dsr", 0)

    def _sharpe(r):
        return (r.get("metrics") or r).get("sharpe", 0)

    valid = [r for r in results if _sharpe(r) != 0]
    ranked = sorted(valid, key=_dsr, reverse=True)
    print(f"\n{'=' * 80}")
    print(f"TOP {n} (by DSR) — Baseline Sharpe: {baseline_sharpe:.4f}")
    print(f"{'=' * 80}")
    for r in ranked[:n]:
        cfg = r.get("config", {})
        strat = cfg.get("strategy", "?")[:28]
        m = r.get("metrics") or r
        print(
            f"  {strat:<30} S={_sharpe(r):7.4f} DSR={_dsr(r):.4f} "
            f"DD={m.get('max_drawdown', 0):.3f} trades={m.get('n_trades', '?')}"
        )
    print("=" * 80)


def sweep_results_to_wandb_format(results):
    out = []
    for r in results:
        if "metrics" in r and r["metrics"]:
            out.append(r)
        elif "sharpe" in r:
            out.append(
                {
                    "config": r["config"],
                    "metrics": {k: r[k] for k in ("sharpe", "dsr", "max_drawdown", "n_trades") if k in r},
                }
            )
    return out
