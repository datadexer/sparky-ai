"""Parameterized experiment runner for signal-based strategy evaluation.

PROTECTED FILE — research agents can call this, not edit it.

Usage:
    from experiment_runner import run
    result = run({
        "asset": "btc", "timeframe": "4h",
        "signal_type": "donchian",
        "signal_params": {"entry_period": 30, "exit_period": 20},
        "sizing": "flat", "n_trials": 1,
    })
"""

import logging
import sys
import warnings
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

from sweep_utils import inv_vol_sizing, net_ret, run_pre, subperiod_analysis  # noqa: E402

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


def run(config: dict) -> dict | None:
    """Run one strategy config through full eval pipeline.

    Config keys:
      asset: "btc" | "eth"
      timeframe: "2h" | "4h" | "8h" | "daily"
      signal_type: "donchian" | "bollinger" | "rsi_extreme"
      signal_params: dict
      sizing: "flat" | "inverse_vol"
      sizing_params: dict (optional)
      n_trials: int (for DSR)
      benchmark_returns: pd.Series | None (for correlation)

    Returns combined results dict or None on failure.
    """
    asset = config["asset"]
    tf = config["timeframe"]
    key = (asset, tf)
    if key not in _DATASET_MAP:
        raise ValueError(f"No dataset for {key}. Available: {list(_DATASET_MAP.keys())}")

    ppy = _PPY[tf]
    alias = _DATASET_MAP[key]
    df = load(alias, purpose="training")
    prices = df["close"].dropna()

    signal_type = config["signal_type"]
    signal_params = config.get("signal_params", {})
    sizing = config.get("sizing", "flat")
    sizing_params = config.get("sizing_params", {})
    n_trials = config.get("n_trials", 1)
    benchmark_ret = config.get("benchmark_returns")

    positions = _make_signal(prices, signal_type, signal_params)
    positions = _apply_sizing(positions, prices, sizing, sizing_params, ppy)

    base_config = {
        "asset": asset,
        "timeframe": tf,
        "signal_type": signal_type,
        "periods_per_year": ppy,
        **signal_params,
        "sizing": sizing,
        **sizing_params,
    }

    m30 = _eval_at_cost(prices, positions, base_config, n_trials, df, 30, ppy)
    m50 = _eval_at_cost(prices, positions, base_config, n_trials, df, 50, ppy)

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
        cf30 = 30 / 10_000
        strat_ret = net_ret(prices, positions, cf30).dropna()
        common = strat_ret.index.intersection(benchmark_ret.index)
        if len(common) > 30:
            result["corr_with_benchmark"] = float(strat_ret.loc[common].corr(benchmark_ret.loc[common]))

    return result
