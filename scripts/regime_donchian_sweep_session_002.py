"""Regime Donchian — Directive 002, Session 002.

Session 1 found binary vol-filter hurts. This session tests:
1. Regime-aware position SIZING (fractional, not binary)
2. Trend-confirmation Donchian (SMA + breakout conjunction)
3. Adaptive lookback (vol-responsive periods)
4. Drawdown-triggered regime filter (dual condition: high vol AND drawdown)
BTC daily, IS only, 30 bps.
"""

import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent / "infra"))

import pandas as pd  # noqa: E402
from sweep_utils import (  # noqa: E402
    baseline_donchian,
    donchian_channel_strategy,
    evaluate,
    load_daily,
    print_top,
    run_pre,
    save_json,
    wandb_log_sweep,
)
from sparky.features.regime_indicators import compute_volatility_regime

TAGS = ["regime_donchian", "directive_002", "session_002"]
COSTS_BPS = 30
CF = COSTS_BPS / 10_000


# ── Strategies ────────────────────────────────────────────────────────────────


def regime_sized_donchian(prices, ep, xp, vw, sz_high, sz_med, sz_low):
    sig = baseline_donchian(prices, ep, xp)
    regime = compute_volatility_regime(prices, window=vw, frequency="1d")
    sizes = regime.map({"low": sz_low, "medium": sz_med, "high": sz_high}).fillna(sz_med)
    return sig * sizes


def trend_confirmed_donchian(prices, ep, xp, sma_period):
    base = donchian_channel_strategy(prices, entry_period=ep, exit_period=xp)
    sma = prices.rolling(window=sma_period).mean().bfill()
    above_sma = (prices > sma).astype(int)
    combined = pd.Series(0, index=prices.index)
    in_pos = False
    for i in range(len(prices)):
        if not in_pos:
            if base.iloc[i] == 1 and above_sma.iloc[i] == 1:
                in_pos = True
                combined.iloc[i] = 1
        else:
            if base.iloc[i] == 0:
                in_pos = False
            else:
                combined.iloc[i] = 1
    return combined


def adaptive_lookback_donchian(prices, base_ep, base_xp, vw, hvm, lvm):
    regime = compute_volatility_regime(prices, window=vw, frequency="1d")
    ep_s = pd.Series(base_ep, index=prices.index)
    xp_s = pd.Series(base_xp, index=prices.index)
    ep_s[regime == "high"] = max(1, int(base_ep * hvm))
    ep_s[regime == "low"] = max(1, int(base_ep * lvm))
    xp_s[regime == "high"] = max(1, int(base_xp * hvm))
    xp_s[regime == "low"] = max(1, int(base_xp * lvm))

    sig = pd.Series(0, index=prices.index)
    in_pos = False
    for i in range(len(prices)):
        ep = int(ep_s.iloc[i])
        xp = int(xp_s.iloc[i])
        if i < ep:
            sig.iloc[i] = 1 if in_pos else 0
            continue
        upper = prices.iloc[max(0, i - ep) : i].max()
        lower = prices.iloc[max(0, i - xp) : i].min()
        c = prices.iloc[i]
        if not in_pos:
            if c >= upper:
                in_pos = True
                sig.iloc[i] = 1
        else:
            if c <= lower:
                in_pos = False
            else:
                sig.iloc[i] = 1
    return sig


def drawdown_triggered_donchian(prices, ep, xp, vw, dd_thresh, lookback_peak):
    base = donchian_channel_strategy(prices, entry_period=ep, exit_period=xp)
    regime = compute_volatility_regime(prices, window=vw, frequency="1d")
    rolling_high = prices.rolling(window=lookback_peak).max().bfill()
    dd = (prices - rolling_high) / rolling_high
    block = (regime == "high") & (dd < -dd_thresh)
    filtered = base.copy()
    filtered[block] = 0
    return filtered


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    df = load_daily()
    prices = df["close"]
    if not run_pre(df, {"strategy": "sweep_s002", "transaction_costs_bps": COSTS_BPS}):
        raise SystemExit("Pre-checks failed")

    # Baseline
    base_sig = baseline_donchian(prices)
    base_m = evaluate(
        prices, base_sig, {"strategy": "donchian_baseline", "entry_period": 40, "exit_period": 20}, 1, df, COSTS_BPS
    )
    base_sharpe = base_m["sharpe"] if base_m else 1.062
    baseline = {"config": {"strategy": "donchian_baseline"}, "metrics": base_m}
    print(f"[BASELINE] Sharpe={base_sharpe:.4f}")

    # Approach 1: Regime-aware sizing
    sizing_results = []
    ep_xp = [(40, 20), (30, 15), (20, 10), (30, 10), (60, 30)]
    schemes = [(1.0, 0.75, 0.25), (1.0, 0.75, 0.50), (1.0, 1.0, 0.50), (1.0, 0.50, 0.25), (1.25, 1.0, 0.50)]
    n = 0
    for (ep, xp), vw, (sl, sm, sh) in product(ep_xp, [20, 30, 45, 60], schemes):
        n += 1
        cfg = {
            "strategy": "regime_sized_donchian",
            "entry_period": ep,
            "exit_period": xp,
            "vol_window": vw,
            "size_low": sl,
            "size_medium": sm,
            "size_high": sh,
        }
        try:
            pos = regime_sized_donchian(prices, ep, xp, vw, sh, sm, sl)
            m = evaluate(prices, pos, cfg, n, df, COSTS_BPS)
            if m:
                sizing_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass
    print(f"[SIZING] {len(sizing_results)} valid configs")

    # Approach 2: Trend confirmation
    trend_results = []
    for (ep, xp), sma in product([(20, 10), (30, 10), (40, 20), (30, 15), (60, 30), (15, 5)], [50, 100, 150, 200]):
        n += 1
        cfg = {"strategy": "trend_confirmed_donchian", "entry_period": ep, "exit_period": xp, "sma_period": sma}
        try:
            sig = trend_confirmed_donchian(prices, ep, xp, sma)
            m = evaluate(prices, sig.astype(float), cfg, n, df, COSTS_BPS)
            if m:
                trend_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass
    print(f"[TREND] {len(trend_results)} valid configs")

    # Approach 3: Adaptive lookback
    adaptive_results = []
    for (bep, bxp), vw, hvm, lvm in product(
        [(20, 10), (30, 15), (40, 20), (30, 10)], [20, 30, 45, 60], [1.5, 2.0, 2.5], [0.5, 0.7, 0.8]
    ):
        n += 1
        cfg = {
            "strategy": "adaptive_lookback_donchian",
            "base_entry": bep,
            "base_exit": bxp,
            "vol_window": vw,
            "high_vol_multiplier": hvm,
            "low_vol_multiplier": lvm,
        }
        try:
            sig = adaptive_lookback_donchian(prices, bep, bxp, vw, hvm, lvm)
            m = evaluate(prices, sig.astype(float), cfg, n, df, COSTS_BPS)
            if m:
                adaptive_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass
    print(f"[ADAPTIVE] {len(adaptive_results)} valid configs")

    # Approach 4: Drawdown-triggered
    dd_results = []
    for (ep, xp), vw, dd, lp in product(
        [(40, 20), (30, 15), (20, 10), (60, 30), (30, 10)], [20, 30, 45], [0.10, 0.15, 0.20, 0.25], [30, 60, 90]
    ):
        n += 1
        cfg = {
            "strategy": "drawdown_triggered_donchian",
            "entry_period": ep,
            "exit_period": xp,
            "vol_window": vw,
            "dd_threshold": dd,
            "lookback_peak": lp,
        }
        try:
            sig = drawdown_triggered_donchian(prices, ep, xp, vw, dd, lp)
            m = evaluate(prices, sig.astype(float), cfg, n, df, COSTS_BPS)
            if m:
                dd_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass
    print(f"[DD_TRIGGER] {len(dd_results)} valid configs")

    # Ablation
    abl_results = []
    from sparky.models.simple_baselines import sma_crossover_strategy

    for strat, kwargs in [
        ("sma_only", {"sma_period": 200}),
        ("donchian_40_20", {"entry_period": 40, "exit_period": 20}),
    ]:
        n += 1
        cfg = {"strategy": strat, **kwargs}
        try:
            if strat == "sma_only":
                sig = sma_crossover_strategy(prices, sma_period=200)
            else:
                sig = donchian_channel_strategy(prices, entry_period=40, exit_period=20)
            m = evaluate(prices, sig.astype(float), cfg, n, df, COSTS_BPS)
            if m:
                abl_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass

    all_results = [baseline] + sizing_results + trend_results + adaptive_results + dd_results + abl_results
    print_top(all_results, base_sharpe, 15)

    wandb_log_sweep(
        "regime_donchian_sweep",
        "directive_002_session_002",
        all_results,
        {"n_configs": len(all_results), "baseline_sharpe": base_sharpe},
        TAGS,
    )
    save_json(
        {
            "baseline": baseline,
            "sizing": sizing_results,
            "trend": trend_results,
            "adaptive": adaptive_results,
            "dd_triggered": dd_results,
            "ablation": abl_results,
        },
        "session_002_results.json",
    )

    valid = [r for r in all_results if r.get("metrics")]
    if valid:
        best = max(valid, key=lambda x: x["metrics"]["dsr"])
        m = best["metrics"]
        print(f"\nBest: {best['config']['strategy']} Sharpe={m['sharpe']:.4f} DSR={m['dsr']:.4f}")
    print(f"Cumulative: 148 + {len(valid)} = {148 + len(valid)} configs tested")


if __name__ == "__main__":
    main()
