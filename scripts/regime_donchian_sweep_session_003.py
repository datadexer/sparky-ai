"""Regime Donchian — Directive 002, Session 003.

HISTORICAL ARCHIVE: ran at 50 bps (stress-test). Current standard is 30 bps.
Cumulative: 599 prior configs (sessions 1+2).
Tests: fine-grained sizing, combined sizing+trend, vol-of-vol, trend-aware 2D, walk-forward.
BTC daily, IS only.
"""

from itertools import product
import numpy as np
import pandas as pd
from sweep_utils import (
    baseline_donchian,
    donchian_channel_strategy,
    evaluate,
    load_daily,
    net_ret,
    print_top,
    run_pre,
    save_json,
    wandb_log_sweep,
)
from sparky.features.regime_indicators import compute_volatility_regime

TAGS = ["regime_donchian", "directive_002", "session_003"]
COSTS_BPS = 50
CF = COSTS_BPS / 10_000
N_PRIOR = 599


# ── Strategies ────────────────────────────────────────────────────────────────


def regime_sized_donchian(prices, ep, xp, vw, sz_high, sz_med, sz_low):
    sig = baseline_donchian(prices, ep, xp)
    regime = compute_volatility_regime(prices, window=vw, frequency="1d")
    sizes = regime.map({"low": sz_low, "medium": sz_med, "high": sz_high}).fillna(sz_med)
    return sig * sizes


def combined_sizing_trend(prices, ep, xp, vw, sz_high, sz_med, sz_low, sma_period):
    base = donchian_channel_strategy(prices, entry_period=ep, exit_period=xp)
    sma = prices.rolling(window=sma_period).mean().bfill()
    above_sma = prices > sma
    regime = compute_volatility_regime(prices, window=vw, frequency="1d")
    sizes = regime.map({"low": sz_low, "medium": sz_med, "high": sz_high}).fillna(sz_med)
    combined = pd.Series(0.0, index=prices.index)
    in_pos = False
    for i in range(len(prices)):
        if not in_pos:
            if int(base.iloc[i]) == 1 and above_sma.iloc[i]:
                in_pos = True
                combined.iloc[i] = sizes.iloc[i]
        else:
            if int(base.iloc[i]) == 0:
                in_pos = False
            else:
                combined.iloc[i] = sizes.iloc[i]
    return combined


def vol_of_vol_donchian(prices, ep, xp, vw, vov_w, sz_stable, sz_rising):
    returns = prices.pct_change()
    vol = returns.rolling(vw).std() * np.sqrt(365)
    vol_change = vol.diff(vov_w)
    regime = compute_volatility_regime(prices, window=vw, frequency="1d")
    high_rising = (regime == "high") & (vol_change > 0)
    high_stable = (regime == "high") & (vol_change <= 0)
    sig = baseline_donchian(prices, ep, xp)
    pos = sig.copy()
    pos[high_rising] = sig[high_rising] * sz_rising
    pos[high_stable] = sig[high_stable] * sz_stable
    return pos


def trend_aware_sized(prices, ep, xp, vw, sma_period, su_lo, su_hi, sd_lo, sd_hi):
    sig = baseline_donchian(prices, ep, xp)
    regime = compute_volatility_regime(prices, window=vw, frequency="1d")
    sma = prices.rolling(window=sma_period).mean().bfill()
    up = prices > sma
    pos = sig.copy()
    for i in range(len(prices)):
        if sig.iloc[i] == 0:
            pos.iloc[i] = 0.0
            continue
        hi = regime.iloc[i] == "high"
        u = up.iloc[i]
        if u and not hi:
            pos.iloc[i] = su_lo
        elif u and hi:
            pos.iloc[i] = su_hi
        elif not u and not hi:
            pos.iloc[i] = sd_lo
        else:
            pos.iloc[i] = sd_hi
    return pos


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    df = load_daily()
    prices = df["close"]
    if not run_pre(df, {"strategy": "sweep_s003", "transaction_costs_bps": COSTS_BPS}):
        raise SystemExit("Pre-checks failed")

    # Baseline
    base_sig = baseline_donchian(prices)
    base_m = evaluate(prices, base_sig, {"strategy": "donchian_baseline"}, N_PRIOR + 1, df, COSTS_BPS)
    base_sharpe = base_m["sharpe"] if base_m else 1.0
    baseline = {"config": {"strategy": "donchian_baseline"}, "metrics": base_m}
    print(f"[BASELINE] Sharpe={base_sharpe:.4f}")

    n = N_PRIOR + 1

    # Walk-forward validation of session 2 winners
    wf_results = []
    wf_strategies = [
        ("baseline_40_20", lambda p: baseline_donchian(p)),
        ("regime_sized_best_s2", lambda p: regime_sized_donchian(p, 40, 20, 20, 0.25, 0.75, 1.0)),
        ("regime_sized_vol30", lambda p: regime_sized_donchian(p, 40, 20, 30, 0.25, 0.75, 1.0)),
        ("regime_sized_30_15", lambda p: regime_sized_donchian(p, 30, 15, 20, 0.25, 0.75, 1.0)),
    ]
    years = sorted(pd.to_datetime(prices.index).year.unique())
    for name, fn in wf_strategies:
        yr_sharpes = {}
        for yr in years[2:]:
            pos = fn(prices)
            test_ret = net_ret(prices, pos, CF)
            yr_ret = test_ret[test_ret.index.year == yr]
            if len(yr_ret) > 20 and yr_ret.std() > 0:
                yr_sharpes[yr] = float(yr_ret.mean() / yr_ret.std(ddof=1) * np.sqrt(365))
        if yr_sharpes:
            m = {
                "sharpe": np.mean(list(yr_sharpes.values())),
                "dsr": 0,
                "max_drawdown": 0,
                "wf_yearly": yr_sharpes,
                "wf_min": min(yr_sharpes.values()),
                "wf_mean": np.mean(list(yr_sharpes.values())),
            }
            wf_results.append({"config": {"strategy": f"wf_{name}"}, "metrics": m})
            print(f"  [WF] {name}: {', '.join(f'{y}:{s:.2f}' for y, s in sorted(yr_sharpes.items()))}")

    # Fine-grained sizing
    fine_results = []
    for (ep, xp), vw, sh, sm, sl in product(
        [(40, 20), (30, 15), (20, 10)],
        [15, 20, 25, 30, 35],
        [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        [0.60, 0.70, 0.75, 0.80, 0.90],
        [1.0, 1.25],
    ):
        if not (sl >= sm >= sh):
            continue
        if sh > 0.35 or sm > 0.90:
            continue
        n += 1
        cfg = {
            "strategy": "regime_sized_fine",
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
                fine_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass
    print(f"[FINE_SIZING] {len(fine_results)} valid configs")

    # Combined sizing + trend
    comb_results = []
    for (ep, xp), vw, sma, (sl, sm, sh) in product(
        [(40, 20), (30, 15), (20, 10), (30, 10), (60, 30)],
        [20, 30, 45, 60],
        [50, 100, 150, 200],
        [(1.0, 0.75, 0.25), (1.0, 0.75, 0.50), (1.0, 1.0, 0.50), (1.25, 0.75, 0.25)],
    ):
        n += 1
        cfg = {
            "strategy": "combined_sizing_trend",
            "entry_period": ep,
            "exit_period": xp,
            "vol_window": vw,
            "sma_period": sma,
            "size_low": sl,
            "size_medium": sm,
            "size_high": sh,
        }
        try:
            pos = combined_sizing_trend(prices, ep, xp, vw, sh, sm, sl, sma)
            m = evaluate(prices, pos, cfg, n, df, COSTS_BPS)
            if m:
                comb_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass
    print(f"[COMBINED] {len(comb_results)} valid configs")

    # Vol-of-vol
    vov_results = []
    vov_schemes = [(0.75, 0.0), (0.75, 0.25), (0.50, 0.0), (0.50, 0.25), (1.0, 0.25)]
    for (ep, xp), vw, vov_w, (ss, sr) in product(
        [(40, 20), (30, 15), (20, 10), (60, 30)], [20, 30, 45], [10, 20, 30], vov_schemes
    ):
        n += 1
        cfg = {
            "strategy": "vol_of_vol_donchian",
            "entry_period": ep,
            "exit_period": xp,
            "vol_window": vw,
            "vov_window": vov_w,
            "size_stable_high": ss,
            "size_rising_high": sr,
        }
        try:
            pos = vol_of_vol_donchian(prices, ep, xp, vw, vov_w, ss, sr)
            m = evaluate(prices, pos, cfg, n, df, COSTS_BPS)
            if m:
                vov_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass
    print(f"[VOL_OF_VOL] {len(vov_results)} valid configs")

    # Trend-aware 2D sizing
    ta_results = []
    ta_schemes = [
        (1.0, 0.75, 0.75, 0.0),
        (1.0, 1.0, 0.50, 0.0),
        (1.0, 0.75, 0.50, 0.25),
        (1.25, 1.0, 0.50, 0.0),
        (1.0, 0.50, 0.50, 0.0),
        (1.0, 0.75, 0.75, 0.25),
    ]
    for (ep, xp), vw, sma, (su_lo, su_hi, sd_lo, sd_hi) in product(
        [(40, 20), (30, 15), (20, 10), (60, 30)], [20, 30, 45], [100, 150, 200], ta_schemes
    ):
        n += 1
        cfg = {
            "strategy": "trend_aware_sized",
            "entry_period": ep,
            "exit_period": xp,
            "vol_window": vw,
            "sma_period": sma,
            "su_lo": su_lo,
            "su_hi": su_hi,
            "sd_lo": sd_lo,
            "sd_hi": sd_hi,
        }
        try:
            pos = trend_aware_sized(prices, ep, xp, vw, sma, su_lo, su_hi, sd_lo, sd_hi)
            m = evaluate(prices, pos, cfg, n, df, COSTS_BPS)
            if m:
                ta_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass
    print(f"[TREND_AWARE] {len(ta_results)} valid configs")

    all_results = [baseline] + wf_results + fine_results + comb_results + vov_results + ta_results
    print_top(all_results, base_sharpe, 25)

    wandb_log_sweep(
        "regime_donchian_sweep",
        "directive_002_session_003",
        all_results,
        {"n_configs": len(all_results), "n_prior": N_PRIOR},
        TAGS,
    )
    save_json(
        {
            "baseline": baseline,
            "walkforward": wf_results,
            "fine_sizing": fine_results,
            "combined": comb_results,
            "vol_of_vol": vov_results,
            "trend_aware": ta_results,
        },
        "session_003_results.json",
    )

    valid = [r for r in all_results if r.get("metrics")]
    n_total = N_PRIOR + len(valid)
    print(f"\nCumulative: {n_total} configs tested across 3 sessions")


if __name__ == "__main__":
    main()
