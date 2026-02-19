"""Parameterized experiment runner for signal-based strategy evaluation.

PROTECTED FILE — research agents can call this, not edit it.

Usage:
    from experiment_runner import run, sweep, run_custom_signal, portfolio_combine
    result = run({...})
    results = sweep(base_config={...}, grid={...}, ...)
    result = run_custom_signal(signal_func=my_func, asset="btc", timeframe="4h")
    result = portfolio_combine(strategies=[{"config": cfg, "weight": 0.5}, ...])
"""

import json
import logging
import sys
import warnings
from itertools import product as _product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sparky.data.loader import load  # noqa: E402
from sparky.features.advanced import bollinger_bands  # noqa: E402
from sparky.features.technical import rsi  # noqa: E402
from sparky.models.simple_baselines import donchian_channel_strategy  # noqa: E402
from sparky.tracking.guardrails import log_results, run_post_checks  # noqa: E402
from sparky.tracking.metrics import compute_all_metrics  # noqa: E402

from sweep_utils import (  # noqa: E402
    inv_vol_sizing,
    net_ret,
    run_pre,
    sanitize,
    subperiod_analysis,
    wandb_log_sweep,
)

# Dataset resolution: (asset, timeframe) -> loader alias
_DATASET_MAP = {
    ("btc", "1h"): "btc_ohlcv_hourly",
    ("btc", "2h"): "btc_ohlcv_2h",
    ("btc", "4h"): "btc_ohlcv_4h",
    ("btc", "8h"): "btc_ohlcv_8h",
    ("btc", "daily"): "btc_ohlcv_daily",
    ("eth", "1h"): "eth_ohlcv_hourly",
    ("eth", "2h"): "eth_ohlcv_2h",
    ("eth", "4h"): "eth_ohlcv_4h",
    ("eth", "8h"): "eth_ohlcv_8h",
    ("eth", "daily"): "eth_ohlcv_daily",
}

_PPY = {
    "1h": 8760,
    "2h": 4380,
    "4h": 2190,
    "8h": 1095,
    "daily": 365,
}


def _make_signal(prices, signal_type, signal_params):
    if signal_type == "donchian":
        ep = signal_params.get("entry_period", 30)
        xp = signal_params.get("exit_period", 20)
        return donchian_channel_strategy(prices, entry_period=ep, exit_period=xp).astype(float)

    elif signal_type == "bollinger":
        period = signal_params.get("period", 20)
        num_std = signal_params.get("num_std", 2.0)
        hold_periods = signal_params.get("hold_periods", 10)
        _mid, upper, lower = bollinger_bands(prices, period=period, num_std=num_std)
        # Breakout-with-hold: enter on upper break, exit on lower break or hold expiry
        signals = pd.Series(0.0, index=prices.index)
        in_position = False
        bars_held = 0
        for i in range(period, len(prices)):
            if not in_position:
                if prices.iloc[i] >= upper.iloc[i - 1]:
                    in_position = True
                    bars_held = 0
                    signals.iloc[i] = 1.0
            else:
                bars_held += 1
                if prices.iloc[i] <= lower.iloc[i - 1] or bars_held >= hold_periods:
                    in_position = False
                    signals.iloc[i] = 0.0
                else:
                    signals.iloc[i] = 1.0
        return signals

    elif signal_type == "rsi_extreme":
        period = signal_params.get("period", 14)
        entry_level = signal_params.get("entry", 30)
        exit_level = signal_params.get("exit", 50)
        rsi_vals = rsi(prices, period=period)
        signals = pd.Series(0.0, index=prices.index)
        in_position = False
        for i in range(period + 1, len(prices)):
            v = rsi_vals.iloc[i]
            if np.isnan(v):
                continue
            if not in_position:
                if v <= entry_level:
                    in_position = True
                    signals.iloc[i] = 1.0
            else:
                if v >= exit_level:
                    in_position = False
                    signals.iloc[i] = 0.0
                else:
                    signals.iloc[i] = 1.0
        return signals

    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")


def _apply_sizing(positions, prices, sizing, sizing_params, ppy):
    if sizing == "flat":
        return positions
    elif sizing == "inverse_vol":
        vw = sizing_params.get("vol_window", 20)
        tv = sizing_params.get("target_vol", 0.4)
        scale = inv_vol_sizing(prices, vw=vw, tv=tv, periods_per_year=ppy)
        return (positions * scale).fillna(0)
    else:
        raise ValueError(f"Unknown sizing: {sizing}")


def _eval_at_cost(prices, positions, config, n_trials, data_df, cost_bps, ppy):
    cf = cost_bps / 10_000
    n_trades = int((positions.diff().abs().fillna(0) > 0.01).sum())
    if n_trades < 5:
        return None
    cfg = {**config, "transaction_costs_bps": cost_bps}
    if not run_pre(data_df, cfg):
        return None
    ret = net_ret(prices, positions, cf)
    ret = ret.dropna()
    if len(ret) < 30:
        return None
    m = compute_all_metrics(ret, n_trials=n_trials, periods_per_year=ppy)
    m["n_trades"] = n_trades
    m["pct_long"] = round(float((positions > 0).mean() * 100), 2)
    m["statistically_significant"] = bool(m.get("dsr", 0) >= 0.95)
    post = run_post_checks(ret, m, cfg)
    log_results(post, run_id=str(cfg)[:100])
    sp = subperiod_analysis(prices, positions, cf, periods_per_year=ppy)
    m["sub_periods"] = sp
    return m


def _load_data(asset, tf):
    key = (asset, tf)
    if key not in _DATASET_MAP:
        raise ValueError(f"No dataset for {key}. Available: {list(_DATASET_MAP.keys())}")
    df = load(_DATASET_MAP[key], purpose="training")
    return df, df["close"].dropna(), _PPY[tf]


def _build_result(prices, positions, config, n_trials, df, ppy, benchmark_ret=None):
    m30 = _eval_at_cost(prices, positions, config, n_trials, df, 30, ppy)
    m50 = _eval_at_cost(prices, positions, config, n_trials, df, 50, ppy)
    if m30 is None:
        return None
    result = {
        "sharpe_30bps": m30["sharpe"],
        "sharpe_50bps": m50["sharpe"] if m50 else None,
        "dsr_30bps": m30.get("dsr"),
        "dsr_50bps": m50.get("dsr") if m50 else None,
        "max_drawdown": m30["max_drawdown"],
        "n_trades": m30["n_trades"],
        "win_rate": m30.get("win_rate"),
        "sub_periods": m30.get("sub_periods", {}),
        "periods_per_year": ppy,
        "statistically_significant": m30.get("statistically_significant", False),
        "metrics_30bps": m30,
        "metrics_50bps": m50,
    }
    if benchmark_ret is not None:
        strat_ret = net_ret(prices, positions, 30 / 10_000).dropna()
        common = strat_ret.index.intersection(benchmark_ret.index)
        if len(common) > 30:
            result["corr_with_benchmark"] = float(strat_ret.loc[common].corr(benchmark_ret.loc[common]))
    return result


def run(config: dict) -> dict | None:
    """Run one strategy config through full eval pipeline.

    Config keys: asset, timeframe, signal_type, signal_params, sizing,
    sizing_params, n_trials, benchmark_returns.
    """
    df, prices, ppy = _load_data(config["asset"], config["timeframe"])
    positions = _make_signal(prices, config["signal_type"], config.get("signal_params", {}))
    positions = _apply_sizing(positions, prices, config.get("sizing", "flat"), config.get("sizing_params", {}), ppy)
    eval_config = {
        "asset": config["asset"],
        "timeframe": config["timeframe"],
        "signal_type": config["signal_type"],
        "periods_per_year": ppy,
        **config.get("signal_params", {}),
        "sizing": config.get("sizing", "flat"),
        **config.get("sizing_params", {}),
    }
    return _build_result(
        prices,
        positions,
        eval_config,
        config.get("n_trials", 1),
        df,
        ppy,
        config.get("benchmark_returns"),
    )


def sweep(
    base_config,
    grid,
    n_trials_start=0,
    benchmark_returns=None,
    wandb_name=None,
    wandb_tags=None,
    save_path=None,
    top_k=10,
):
    """Run parameter grid sweep. Grid keys route to signal_params by default,
    vol_window/target_vol to sizing_params, asset/timeframe/signal_type/sizing to top-level.
    """
    _SIZ = {"vol_window", "target_vol"}
    _TOP = {"asset", "timeframe", "signal_type", "sizing"}
    param_names = list(grid.keys())
    combos = list(_product(*(list(v) for v in grid.values())))
    results, n_trials = [], n_trials_start

    for combo in combos:
        n_trials += 1
        overrides = dict(zip(param_names, combo))
        cfg = {**base_config}
        sig_p = dict(cfg.pop("signal_params", {}))
        siz_p = dict(cfg.pop("sizing_params", {}))
        for k, v in overrides.items():
            if k in _TOP:
                cfg[k] = v
            elif k in _SIZ:
                siz_p[k] = v
            else:
                sig_p[k] = v
        cfg.update(signal_params=sig_p, sizing_params=siz_p, n_trials=n_trials)
        if benchmark_returns is not None:
            cfg["benchmark_returns"] = benchmark_returns
        try:
            r = run(cfg)
            if r:
                r["config"] = overrides
                results.append(r)
                print(f"  {overrides}: S@30={r['sharpe_30bps']:.3f} DSR={r.get('dsr_30bps') or 0:.3f}")
        except Exception as e:
            print(f"  {overrides}: ERROR {e}")

    results.sort(key=lambda r: r.get("dsr_30bps") or 0, reverse=True)

    if results:
        print(f"\n{'=' * 70}")
        print(f"TOP {min(top_k, len(results))} by DSR ({len(results)}/{len(combos)} valid)")
        print(f"{'=' * 70}")
        for r in results[:top_k]:
            sp20 = (r.get("sub_periods") or {}).get("2020+", {}).get("sharpe", "?")
            print(
                f"  {r['config']}: S@30={r['sharpe_30bps']:.4f} "
                f"DSR={r.get('dsr_30bps') or 0:.4f} DD={r['max_drawdown']:.3f} 2020+={sp20}"
            )

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(
                sanitize({"n_trials": n_trials, "results": results}),
                f,
                indent=2,
                default=str,
            )
        print(f"[SAVED] {p}")

    if wandb_name and results:
        wb = [{"config": r.get("config", {}), "metrics": r.get("metrics_30bps", {})} for r in results]
        summary = {
            "best_sharpe": results[0]["sharpe_30bps"],
            "best_dsr": results[0].get("dsr_30bps"),
            "n_valid": len(results),
            "n_total": len(combos),
            "n_trials": n_trials,
        }
        wandb_log_sweep(wandb_name, wandb_name, wb, summary, wandb_tags or [])

    return {
        "results": results,
        "n_trials": n_trials,
        "n_valid": len(results),
        "n_total": len(combos),
    }


def run_custom_signal(
    signal_func,
    signal_kwargs=None,
    asset="btc",
    timeframe="4h",
    sizing="flat",
    sizing_params=None,
    n_trials=1,
    benchmark_returns=None,
):
    """Run a custom signal function: signal_func(prices, **kwargs) -> pd.Series of positions."""
    df, prices, ppy = _load_data(asset, timeframe)
    positions = signal_func(prices, **(signal_kwargs or {}))
    positions = _apply_sizing(positions, prices, sizing, sizing_params or {}, ppy)
    config = {
        "asset": asset,
        "timeframe": timeframe,
        "signal_type": "custom",
        "periods_per_year": ppy,
        "sizing": sizing,
        **(sizing_params or {}),
    }
    return _build_result(prices, positions, config, n_trials, df, ppy, benchmark_returns)


def portfolio_combine(
    strategies,
    weighting="equal",
    n_trials=1,
    benchmark_returns=None,
):
    """Combine multiple strategies into a portfolio. All must share same asset/timeframe.

    strategies: list of {"config": run()-compatible config, "weight": float (for "specified")}
    weighting: "equal" | "specified" | "inverse_vol"
    """
    cfg0 = strategies[0]["config"]
    asset_tf = (cfg0["asset"], cfg0["timeframe"])
    for s in strategies[1:]:
        key = (s["config"]["asset"], s["config"]["timeframe"])
        if key != asset_tf:
            raise ValueError(f"All strategies must share asset/timeframe. Got {asset_tf} and {key}")

    df, prices, ppy = _load_data(*asset_tf)
    all_pos = []
    for s in strategies:
        cfg = s["config"]
        sig = _make_signal(prices, cfg["signal_type"], cfg.get("signal_params", {}))
        sig = _apply_sizing(sig, prices, cfg.get("sizing", "flat"), cfg.get("sizing_params", {}), ppy)
        all_pos.append(sig)

    # Align on common index (different signal warm-up periods)
    common = all_pos[0].index
    for p in all_pos[1:]:
        common = common.intersection(p.index)
    aligned = [p.reindex(common).fillna(0) for p in all_pos]
    prices_c = prices.reindex(common)

    n = len(aligned)
    if weighting == "specified":
        weights = [s.get("weight", 1.0 / n) for s in strategies]
    elif weighting == "inverse_vol":
        vols = [net_ret(prices_c, p, 0).std() for p in aligned]
        inv = [1.0 / (v + 1e-10) for v in vols]
        total = sum(inv)
        weights = [w / total for w in inv]
    else:
        weights = [1.0 / n] * n

    combined = sum(w * p for w, p in zip(weights, aligned))
    pc = {
        "strategy": "portfolio",
        "weighting": weighting,
        "n_strategies": n,
        "periods_per_year": ppy,
        "transaction_costs_bps": 30,
    }
    result = _build_result(prices_c, combined, pc, n_trials, df, ppy, benchmark_returns)
    if result:
        rets = pd.DataFrame({f"s{i}": net_ret(prices_c, p, 30 / 10_000) for i, p in enumerate(aligned)})
        result["correlation_matrix"] = rets.corr().to_dict()
        result["weights"] = weights
    return result
