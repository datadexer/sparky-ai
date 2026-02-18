#!/usr/bin/env python3
"""Regime Donchian V3 — Session 002.

Cumulative N_TRIALS starts at 1507 (from session 001).
Approaches: A) Long/Short Donchian, B) Adaptive Exit, C) HMM Soft Prob, D) Validate S001 best.
BTC daily, IS only, 30 bps standard + 50 bps stress.
"""

import numpy as np
import pandas as pd
from sparky.tracking.metrics import compute_all_metrics
from sweep_utils import (
    ExperimentTracker,
    PERIODS_PER_YEAR,
    baseline_donchian,
    evaluate,
    inv_vol_sizing,
    load_daily,
    net_ret,
    run_pre,
    rvol,
    save_json,
    sweep_results_to_wandb_format,
    wandb_log_sweep,
    yearly_sharpes,
)

TAGS = ["regime_donchian_v3", "session_002"]
COSTS_BPS = 30
CF = COSTS_BPS / 10_000
CF_STRESS = 50 / 10_000
N_TRIALS = 1507

df = load_daily()
prices = df["close"]
if not run_pre(df, {"transaction_costs_bps": COSTS_BPS}):
    raise SystemExit("Pre-checks failed")

baseline_pos = baseline_donchian(prices)
base_m = compute_all_metrics(net_ret(prices, baseline_pos, CF), n_trials=N_TRIALS, periods_per_year=PERIODS_PER_YEAR)
BASELINE_SHARPE = base_m["sharpe"]
print(f"[BASELINE] Sharpe={BASELINE_SHARPE:.3f}, Yearly: {yearly_sharpes(prices, baseline_pos, CF)}")

ALL_RESULTS = []

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH A: Long/Short Donchian
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nAPPROACH A: Long/Short Donchian\n" + "=" * 70)


def ls_donchian(prices, ep, xp, tw=200):
    upper = prices.rolling(ep).max().shift(1)
    lower = prices.rolling(ep).min().shift(1)
    xu = prices.rolling(xp).max().shift(1)
    xl = prices.rolling(xp).min().shift(1)
    sma = prices.rolling(tw).mean() if tw > 0 else None
    sig = pd.Series(0.0, index=prices.index)
    pos = 0.0
    for i in range(len(prices)):
        if pd.isna(upper.iloc[i]):
            continue
        c = prices.iloc[i]
        if pos == 0:
            if c > upper.iloc[i]:
                pos = 1.0
            elif c < lower.iloc[i] and (sma is None or (not pd.isna(sma.iloc[i]) and c < sma.iloc[i])):
                pos = -1.0
        elif pos == 1:
            if not pd.isna(xl.iloc[i]) and c < xl.iloc[i]:
                pos = 0.0
        elif pos == -1:
            if not pd.isna(xu.iloc[i]) and c > xu.iloc[i]:
                pos = 0.0
        sig.iloc[i] = pos
    return sig


results_a = []
for ep, xp, tw in [
    (40, 20, 200),
    (40, 20, 100),
    (40, 20, 0),
    (30, 15, 200),
    (30, 15, 100),
    (30, 15, 0),
    (20, 10, 200),
    (20, 10, 0),
    (60, 20, 200),
    (60, 30, 200),
    (60, 30, 0),
    (50, 25, 200),
    (50, 25, 100),
    (40, 10, 200),
    (40, 15, 200),
]:
    N_TRIALS += 1
    cfg = {"strategy": "long_short_donchian", "entry_period": ep, "exit_period": xp, "trend_window": tw}
    try:
        pos = ls_donchian(prices, ep, xp, tw)
        m = evaluate(prices, pos, cfg, N_TRIALS, df, COSTS_BPS)
        if m is None:
            continue
        m50 = compute_all_metrics(net_ret(prices, pos, CF_STRESS), n_trials=N_TRIALS, periods_per_year=PERIODS_PER_YEAR)
        yr = yearly_sharpes(prices, pos, CF)
        res = {
            "config": cfg,
            "sharpe": m["sharpe"],
            "dsr": m["dsr"],
            "sharpe_50bps": m50["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "n_trades": m["n_trades"],
            "yearly": yr,
            "beats_baseline": m["sharpe"] > BASELINE_SHARPE,
        }
        results_a.append(res)
        ALL_RESULTS.append(res)
        print(f"  LS({ep}/{xp},tw={tw}): S={m['sharpe']:.3f}, 50bps={m50['sharpe']:.3f}, 2022={yr.get(2022, '?')}")
    except Exception:
        pass

# L/S with inverse vol sizing
for ep, xp, tw, vw, tv in [
    (40, 20, 200, 45, 0.4),
    (40, 20, 200, 20, 0.4),
    (60, 30, 200, 45, 0.4),
    (30, 15, 200, 45, 0.4),
]:
    N_TRIALS += 1
    cfg = {
        "strategy": "long_short_inv_vol",
        "entry_period": ep,
        "exit_period": xp,
        "trend_window": tw,
        "vol_window": vw,
        "target_vol": tv,
    }
    try:
        pos = (ls_donchian(prices, ep, xp, tw) * inv_vol_sizing(prices, vw, tv)).clip(-1.5, 1.5)
        m = evaluate(prices, pos, cfg, N_TRIALS, df, COSTS_BPS)
        if m:
            yr = yearly_sharpes(prices, pos, CF)
            res = {
                "config": cfg,
                "sharpe": m["sharpe"],
                "dsr": m["dsr"],
                "max_drawdown": m["max_drawdown"],
                "n_trades": m["n_trades"],
                "yearly": yr,
                "beats_baseline": m["sharpe"] > BASELINE_SHARPE,
            }
            results_a.append(res)
            ALL_RESULTS.append(res)
            print(f"  LS_InvVol({ep}/{xp},vw={vw}): S={m['sharpe']:.3f}, 2022={yr.get(2022, '?')}")
    except Exception:
        pass

print(f"Approach A: {len(results_a)} configs, N={N_TRIALS}")

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH B: Adaptive Exit Speed
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nAPPROACH B: Adaptive Exit\n" + "=" * 70)


def adaptive_exit(prices, ep=40, bull_xp=20, bear_xp=5, vw=20, tw=200):
    sma = prices.rolling(tw).mean()
    vol = rvol(prices, vw)
    med = vol.rolling(252, min_periods=60).median().fillna(vol.median())
    is_bear = (prices < sma) | (vol >= med)
    upper = prices.rolling(ep).max().shift(1)
    sig = pd.Series(0.0, index=prices.index)
    pos = 0.0
    for i in range(len(prices)):
        if pd.isna(upper.iloc[i]):
            continue
        c = prices.iloc[i]
        xp = bear_xp if is_bear.iloc[i] else bull_xp
        xl = prices.rolling(xp).min().shift(1).iloc[i]
        if pos == 0:
            if c > upper.iloc[i]:
                pos = 1.0
        elif pos == 1:
            if not pd.isna(xl) and c < xl:
                pos = 0.0
        sig.iloc[i] = pos
    return sig


results_b = []
for ep, bxp, bearxp, vw, tw in [
    (40, 20, 5, 20, 200),
    (40, 20, 7, 20, 200),
    (40, 20, 10, 20, 200),
    (40, 20, 5, 15, 200),
    (40, 20, 5, 30, 200),
    (40, 20, 5, 20, 100),
    (40, 20, 5, 20, 150),
    (40, 30, 5, 20, 200),
    (40, 30, 10, 20, 200),
    (30, 15, 5, 20, 200),
    (30, 15, 3, 20, 200),
    (60, 30, 5, 20, 200),
    (60, 20, 5, 20, 200),
    (50, 25, 5, 20, 200),
    (40, 20, 3, 20, 200),
]:
    N_TRIALS += 1
    cfg = {
        "strategy": "adaptive_exit",
        "entry_period": ep,
        "bull_exit": bxp,
        "bear_exit": bearxp,
        "vol_window": vw,
        "trend_window": tw,
    }
    try:
        pos = adaptive_exit(prices, ep, bxp, bearxp, vw, tw)
        m = evaluate(prices, pos, cfg, N_TRIALS, df, COSTS_BPS)
        if m:
            yr = yearly_sharpes(prices, pos, CF)
            res = {
                "config": cfg,
                "sharpe": m["sharpe"],
                "dsr": m["dsr"],
                "max_drawdown": m["max_drawdown"],
                "n_trades": m["n_trades"],
                "yearly": yr,
                "beats_baseline": m["sharpe"] > BASELINE_SHARPE,
            }
            results_b.append(res)
            ALL_RESULTS.append(res)
            print(f"  AdaptExit({ep},{bxp}/{bearxp}): S={m['sharpe']:.3f}, 2022={yr.get(2022, '?')}")
    except Exception:
        pass

print(f"Approach B: {len(results_b)} configs, N={N_TRIALS}")

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH C: HMM Soft Probability Sizing
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nAPPROACH C: HMM Soft Prob\n" + "=" * 70)
results_c = []

try:
    from hmmlearn import hmm as hmmlearn_mod

    HMM_OK = True
except ImportError:
    HMM_OK = False
    print("hmmlearn unavailable — using vol fallback")


def get_hmm_p_bull(prices, n_states=2, feats="returns_vol", tw=252, step=20):
    if not HMM_OK:
        return None
    log_r = np.log(prices / prices.shift(1)).fillna(0)
    v = log_r.rolling(20).std().fillna(log_r.std())
    feat_map = {
        "returns_only": pd.DataFrame({"r": log_r}),
        "returns_vol": pd.DataFrame({"r": log_r, "v": v}),
        "returns_vol_trend": pd.DataFrame({"r": log_r, "v": v, "m": log_r.rolling(20).mean().fillna(0)}),
    }
    X = feat_map.get(feats, feat_map["returns_vol"]).fillna(0)
    probs = pd.DataFrame(0.0, index=prices.index, columns=range(n_states))
    labels = pd.Series(0, index=prices.index, dtype=int)
    last_m = None
    for i in range(tw, len(prices), step):
        try:
            mdl = hmmlearn_mod.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=50, random_state=42)
            mdl.fit(X.iloc[max(0, i - tw) : i].values)
            last_m = mdl
        except Exception:
            if last_m is None:
                continue
            mdl = last_m
        pe = min(i + step, len(prices))
        Xp = X.iloc[i:pe].values
        if len(Xp) == 0:
            continue
        try:
            lbs = mdl.predict(Xp)
            pbs = mdl.predict_proba(Xp)
            for j in range(len(lbs)):
                labels.iloc[i + j] = lbs[j]
                for s in range(min(n_states, pbs.shape[1])):
                    probs.iloc[i + j, s] = pbs[j, s]
        except Exception:
            pass
    lr = np.log(prices / prices.shift(1)).fillna(0).values
    sv = {s: lr[(labels == s).values].std() for s in range(n_states) if (labels == s).sum() > 20}
    bull = min(sv, key=sv.get) if sv else 0
    return probs[bull]


def soft_prob_pos(prices, ep, xp, p_bull, mode="linear"):
    base = baseline_donchian(prices, ep, xp)
    pb = p_bull.reindex(prices.index).fillna(0.5)
    if mode == "sigmoid":
        sz = 1 / (1 + np.exp(-(pb - 0.5) * 6))
    elif mode == "threshold_soft":
        sz = 0.1 + 0.9 * pb
    else:
        sz = pb
    return base * sz


if HMM_OK:
    hmm_cache = {}
    hmm_cfgs = [
        (2, "returns_vol", 252, 40, 20, "linear"),
        (2, "returns_vol", 252, 40, 20, "sigmoid"),
        (2, "returns_vol", 252, 40, 20, "threshold_soft"),
        (2, "returns_vol", 252, 30, 15, "linear"),
        (2, "returns_vol", 252, 60, 20, "linear"),
        (2, "returns_vol", 504, 40, 20, "linear"),
        (2, "returns_vol", 504, 40, 20, "sigmoid"),
        (3, "returns_vol", 252, 40, 20, "linear"),
        (2, "returns_only", 252, 40, 20, "linear"),
        (2, "returns_vol_trend", 252, 40, 20, "linear"),
        (2, "returns_vol_trend", 252, 40, 20, "sigmoid"),
    ]
    for ns, ft, tw, ep, xp, mode in hmm_cfgs:
        ck = (ns, ft, tw)
        if ck not in hmm_cache:
            print(f"  Fitting HMM({ns},{ft},{tw})...")
            hmm_cache[ck] = get_hmm_p_bull(prices, ns, ft, tw)
        p = hmm_cache[ck]
        if p is None:
            continue
        N_TRIALS += 1
        cfg = {
            "strategy": "hmm_soft_prob",
            "n_states": ns,
            "hmm_features": ft,
            "hmm_tw": tw,
            "entry_period": ep,
            "exit_period": xp,
            "sizing_mode": mode,
        }
        try:
            pos = soft_prob_pos(prices, ep, xp, p, mode)
            m = evaluate(prices, pos, cfg, N_TRIALS, df, COSTS_BPS)
            if m:
                yr = yearly_sharpes(prices, pos, CF)
                res = {
                    "config": cfg,
                    "sharpe": m["sharpe"],
                    "dsr": m["dsr"],
                    "max_drawdown": m["max_drawdown"],
                    "n_trades": m["n_trades"],
                    "yearly": yr,
                    "beats_baseline": m["sharpe"] > BASELINE_SHARPE,
                }
                results_c.append(res)
                ALL_RESULTS.append(res)
                print(f"  HMM({ns},{ft},{mode}): S={m['sharpe']:.3f}, DSR={m['dsr']:.3f}")
        except Exception:
            pass
else:
    vol = rvol(prices, 20)
    med = vol.rolling(252, min_periods=60).median().fillna(vol.median())
    p_fb = (med / vol).clip(upper=1.0).fillna(0.5)
    for ep, xp, mode in [
        (40, 20, "linear"),
        (40, 20, "sigmoid"),
        (30, 15, "linear"),
        (60, 20, "linear"),
        (40, 20, "threshold_soft"),
    ]:
        N_TRIALS += 1
        cfg = {"strategy": "vol_soft_prob", "entry_period": ep, "exit_period": xp, "sizing_mode": mode}
        try:
            pos = soft_prob_pos(prices, ep, xp, p_fb, mode)
            m = evaluate(prices, pos, cfg, N_TRIALS, df, COSTS_BPS)
            if m:
                res = {
                    "config": cfg,
                    "sharpe": m["sharpe"],
                    "dsr": m["dsr"],
                    "max_drawdown": m["max_drawdown"],
                    "beats_baseline": m["sharpe"] > BASELINE_SHARPE,
                }
                results_c.append(res)
                ALL_RESULTS.append(res)
        except Exception:
            pass

print(f"Approach C: {len(results_c)} configs, N={N_TRIALS}")

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH D: Validate Session 001 Best
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nAPPROACH D: Session 001 Best Validation\n" + "=" * 70)
pos_best = (baseline_donchian(prices, 30, 20) * inv_vol_sizing(prices, 45, 0.4)).clip(0, 1.5)
m30 = compute_all_metrics(net_ret(prices, pos_best, CF), n_trials=N_TRIALS, periods_per_year=PERIODS_PER_YEAR)
m50 = compute_all_metrics(net_ret(prices, pos_best, CF_STRESS), n_trials=N_TRIALS, periods_per_year=PERIODS_PER_YEAR)
yr30 = yearly_sharpes(prices, pos_best, CF)
yr50 = yearly_sharpes(prices, pos_best, CF_STRESS)
yr_base = yearly_sharpes(prices, baseline_pos, CF)

print(f"S001 Best @30bps: Sharpe={m30['sharpe']:.3f}, DSR={m30['dsr']:.3f}")
print(f"S001 Best @50bps: Sharpe={m50['sharpe']:.3f}, DSR={m50['dsr']:.3f}")
print(f"{'Year':5} | {'Best@30':>8} | {'Best@50':>8} | {'Base':>8} | {'Edge':>8}")
for yr in sorted(yr30.keys()):
    print(
        f"{yr:5} | {yr30.get(yr, 0):8.3f} | {yr50.get(yr, 0):8.3f} | {yr_base.get(yr, 0):8.3f} | {yr30.get(yr, 0) - yr_base.get(yr, 0):+8.3f}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nSESSION 002 SUMMARY\n" + "=" * 70)
print(f"N_TRIALS: {N_TRIALS}, Session configs: {len(ALL_RESULTS)}")
print(f"Baseline: S={BASELINE_SHARPE:.3f}")
n_beats = sum(r.get("beats_baseline", False) for r in ALL_RESULTS)
winners = [r for r in ALL_RESULTS if r.get("dsr", 0) > 0.95 and r.get("sharpe", 0) >= 1.0]
print(f"Beats baseline: {n_beats}, DSR>0.95: {len(winners)}")

if ALL_RESULTS:
    ob = max(ALL_RESULTS, key=lambda x: x.get("sharpe", 0))
    print(f"Best Sharpe: {ob['config']['strategy']} S={ob['sharpe']:.3f}")
    for nm, res in [("A L/S", results_a), ("B AdaptExit", results_b), ("C HMM", results_c)]:
        if res:
            best = max(res, key=lambda x: x.get("sharpe", 0))
            print(f"  {nm}: N={len(res)}, Best S={best['sharpe']:.3f}")

save_json(
    {
        "cumulative_n_trials": N_TRIALS,
        "session_n": len(ALL_RESULTS),
        "baseline_sharpe": BASELINE_SHARPE,
        "n_beats": n_beats,
        "s001_validation": {
            "sharpe_30": m30["sharpe"],
            "sharpe_50": m50["sharpe"],
            "yearly_30": yr30,
            "yearly_50": yr50,
        },
        "all_results": ALL_RESULTS,
    },
    "session_002_v3_summary.json",
)

tracker = ExperimentTracker(experiment_name="regime_donchian_v3")
if ALL_RESULTS:
    ob = max(ALL_RESULTS, key=lambda x: x.get("sharpe", 0))
    wandb_log_sweep(
        "regime_donchian_v3",
        "session_002_all",
        sweep_results_to_wandb_format(ALL_RESULTS),
        {"sharpe": ob["sharpe"], "n_configs": len(ALL_RESULTS), "n_trials": N_TRIALS},
        TAGS + ["session_002"],
    )

print(f"\nDone. {'DSR>0.95 found!' if winners else 'No DSR>0.95.'}")
