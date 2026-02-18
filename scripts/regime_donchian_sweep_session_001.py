#!/usr/bin/env python3
"""Regime Donchian V3 — Session 001.

4 strategy families + refinement + stress test. BTC daily, IS only, 30 bps.
Prior work (directive_002, 2370 configs): best DSR 0.710 at 50 bps.
V3 uses 30 bps standard, continuous sizing, soft probability, parameter switching.
"""

import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent / "infra"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sparky.tracking.metrics import compute_all_metrics  # noqa: E402
from sweep_utils import (  # noqa: E402
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
    wandb_log_experiment,
    wandb_log_sweep,
    yearly_sharpes,
)

TAGS = ["regime_donchian_v3", "session_001"]
COSTS_BPS = 30
CF = COSTS_BPS / 10_000
CF_STRESS = 50 / 10_000
N_TRIALS = 0

# ── Data ──────────────────────────────────────────────────────────────────────
df = load_daily()
prices = df["close"]
if not run_pre(df, {"transaction_costs_bps": COSTS_BPS}):
    raise SystemExit("Pre-checks failed")

# ── Regime helpers ────────────────────────────────────────────────────────────


def vol_momentum_regime(prices, vw=20, mw=90):
    vol = rvol(prices, vw)
    med = vol.rolling(252, min_periods=60).median().fillna(vol.median())
    high_vol = (vol >= med).astype(int)
    up_trend = (prices.pct_change(mw).fillna(0) > 0).astype(int)
    return high_vol * 2 + up_trend


# ── Baseline ──────────────────────────────────────────────────────────────────
baseline_pos = baseline_donchian(prices)
base_m = compute_all_metrics(net_ret(prices, baseline_pos, CF), n_trials=1, periods_per_year=PERIODS_PER_YEAR)
BASELINE_SHARPE = base_m["sharpe"]
print(f"[BASELINE] Sharpe={BASELINE_SHARPE:.3f}, Yearly: {yearly_sharpes(prices, baseline_pos, CF)}")

ALL_RESULTS = []

# ══════════════════════════════════════════════════════════════════════════════
# ROUND 1: Inverse-Vol Continuous Sizing
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nROUND 1: Inverse-Vol Sizing\n" + "=" * 70)
r1 = []
for ep, xp, vw, tv in product(
    [20, 30, 40, 60], [5, 10, 15, 20, 30], [10, 15, 20, 30, 45], [0.20, 0.30, 0.40, 0.60, 0.80]
):
    if xp >= ep:
        continue
    N_TRIALS += 1
    cfg = {"strategy": "inverse_vol_sizing", "entry_period": ep, "exit_period": xp, "vol_window": vw, "target_vol": tv}
    try:
        pos = (baseline_donchian(prices, ep, xp) * inv_vol_sizing(prices, vw, tv)).clip(0, 1.5)
        m = evaluate(prices, pos, cfg, N_TRIALS, df, COSTS_BPS)
        if m:
            r1.append(
                {
                    "config": cfg,
                    "sharpe": m["sharpe"],
                    "dsr": m["dsr"],
                    "max_drawdown": m["max_drawdown"],
                    "n_trades": m["n_trades"],
                    "n_trials": N_TRIALS,
                    "beats_baseline": m["sharpe"] > BASELINE_SHARPE,
                }
            )
            ALL_RESULTS.append(r1[-1])
    except Exception:
        pass

print(f"Round 1: {len(r1)} configs, N_TRIALS={N_TRIALS}")
if r1:
    best = max(r1, key=lambda x: x["dsr"])
    print(f"  Best DSR: S={best['sharpe']:.3f}, DSR={best['dsr']:.3f}")
save_json(r1, "session_001_round1_results.json")
wandb_log_sweep(
    "regime_donchian_v3",
    "session_001_round1",
    sweep_results_to_wandb_format(r1),
    {"n": len(r1), "best_sharpe": max((x["sharpe"] for x in r1), default=0)},
    TAGS + ["round1"],
)

# ══════════════════════════════════════════════════════════════════════════════
# ROUND 2: Regime-Conditioned Parameter Switching
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nROUND 2: Regime Parameter Switching\n" + "=" * 70)
r2 = []

param_combos = [
    (20, 10, 40, 20),
    (20, 10, 60, 30),
    (15, 5, 40, 20),
    (15, 5, 60, 30),
    (30, 15, 60, 30),
    (20, 10, 30, 15),
    (15, 5, 30, 15),
    (10, 5, 40, 20),
    (10, 5, 60, 30),
    (10, 5, 20, 10),
    (20, 10, 40, 20),
    (30, 15, 40, 20),
]


def regime_switching_donchian(prices, le, lx, he, hx, vw=20):
    vol = rvol(prices, vw)
    med = vol.rolling(252, min_periods=60).median().fillna(vol.median())
    is_high = vol >= med
    sig_low = baseline_donchian(prices, le, lx)
    sig_high = baseline_donchian(prices, he, hx)
    combined = pd.Series(0.0, index=prices.index)
    combined[~is_high] = sig_low[~is_high]
    combined[is_high] = sig_high[is_high]
    return combined


for (le, lx, he, hx), vw in product(param_combos, [10, 15, 20, 30]):
    N_TRIALS += 1
    cfg = {
        "strategy": "regime_param_switching",
        "low_vol_entry": le,
        "low_vol_exit": lx,
        "high_vol_entry": he,
        "high_vol_exit": hx,
        "vol_window": vw,
    }
    try:
        pos = regime_switching_donchian(prices, le, lx, he, hx, vw)
        m = evaluate(prices, pos, cfg, N_TRIALS, df, COSTS_BPS)
        if m:
            r2.append(
                {
                    "config": cfg,
                    "sharpe": m["sharpe"],
                    "dsr": m["dsr"],
                    "max_drawdown": m["max_drawdown"],
                    "n_trades": m["n_trades"],
                    "n_trials": N_TRIALS,
                    "beats_baseline": m["sharpe"] > BASELINE_SHARPE,
                }
            )
            ALL_RESULTS.append(r2[-1])
    except Exception:
        pass

print(f"Round 2: {len(r2)} configs, N_TRIALS={N_TRIALS}")
save_json(r2, "session_001_round2_results.json")
wandb_log_sweep(
    "regime_donchian_v3", "session_001_round2", sweep_results_to_wandb_format(r2), {"n": len(r2)}, TAGS + ["round2"]
)

# ══════════════════════════════════════════════════════════════════════════════
# ROUND 3: Vol × Momentum 4-State Regime Sizing
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nROUND 3: Vol × Momentum 4-State\n" + "=" * 70)
r3 = []

sizing_grids = [
    (1.0, 0.5, 0.75, 0.25),
    (1.0, 0.0, 0.5, 0.0),
    (1.0, 0.3, 0.6, 0.2),
    (1.2, 0.5, 0.8, 0.2),
    (1.0, 0.5, 1.0, 0.3),
    (1.0, 0.0, 0.75, 0.0),
    (1.0, 0.5, 0.5, 0.0),
    (1.25, 0.5, 0.5, 0.25),
    (1.0, 0.25, 0.75, 0.1),
    (0.75, 0.25, 1.0, 0.5),
]


def vm4_sizing(prices, base_sig, sizes, vw=20, mw=90):
    regime = vol_momentum_regime(prices, vw, mw)
    sz = regime.map({0: sizes[0], 1: sizes[1], 2: sizes[2], 3: sizes[3]}).fillna(0.5)
    return base_sig * sz


for ep, xp in product([20, 30, 40], [10, 15, 20]):
    if xp >= ep:
        continue
    base_sig = baseline_donchian(prices, ep, xp)
    for sizes, vw, mw in product(sizing_grids, [15, 20, 30], [60, 90, 120]):
        N_TRIALS += 1
        cfg = {
            "strategy": "vol_momentum_4state",
            "entry_period": ep,
            "exit_period": xp,
            "vol_window": vw,
            "mom_window": mw,
            "size_low_up": sizes[0],
            "size_low_down": sizes[1],
            "size_high_up": sizes[2],
            "size_high_down": sizes[3],
        }
        try:
            pos = vm4_sizing(prices, base_sig, sizes, vw, mw)
            m = evaluate(prices, pos, cfg, N_TRIALS, df, COSTS_BPS)
            if m:
                r3.append(
                    {
                        "config": cfg,
                        "sharpe": m["sharpe"],
                        "dsr": m["dsr"],
                        "max_drawdown": m["max_drawdown"],
                        "n_trades": m["n_trades"],
                        "n_trials": N_TRIALS,
                        "beats_baseline": m["sharpe"] > BASELINE_SHARPE,
                    }
                )
                ALL_RESULTS.append(r3[-1])
        except Exception:
            pass

print(f"Round 3: {len(r3)} configs, N_TRIALS={N_TRIALS}")
save_json(r3, "session_001_round3_results.json")
wandb_log_sweep(
    "regime_donchian_v3", "session_001_round3", sweep_results_to_wandb_format(r3), {"n": len(r3)}, TAGS + ["round3"]
)

# ══════════════════════════════════════════════════════════════════════════════
# ROUND 4: Adaptive Lookback — Vol-Weighted Signal Blending
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nROUND 4: Adaptive Lookback\n" + "=" * 70)
r4 = []


def adaptive_lookback(prices, fe, fx, se, sx, vw=20, blend="linear"):
    vol = rvol(prices, vw)
    med = vol.rolling(252, min_periods=60).median().fillna(vol.median())
    sig_fast = baseline_donchian(prices, fe, fx)
    sig_slow = baseline_donchian(prices, se, sx)
    vr = (vol / med).fillna(1.0)
    if blend == "linear":
        w = ((vr - 0.5) / 1.5).clip(0, 1)
    elif blend == "sigmoid":
        x = (vr - 1.0) * 3
        w = 1 / (1 + np.exp(-x))
    else:  # threshold
        w = (vr >= 1.0).astype(float)
    w = pd.Series(w, index=prices.index) if not isinstance(w, pd.Series) else w
    return sig_fast * (1 - w) + sig_slow * w


fs_combos = [
    (10, 5, 40, 20),
    (15, 5, 40, 20),
    (20, 10, 40, 20),
    (20, 10, 60, 30),
    (15, 5, 60, 30),
    (10, 5, 60, 30),
    (20, 10, 30, 15),
    (15, 5, 30, 15),
]

for (fe, fx, se, sx), bm, vw in product(fs_combos, ["linear", "sigmoid", "threshold"], [10, 15, 20, 30]):
    N_TRIALS += 1
    cfg = {
        "strategy": "adaptive_lookback",
        "fast_entry": fe,
        "fast_exit": fx,
        "slow_entry": se,
        "slow_exit": sx,
        "blend_mode": bm,
        "vol_window": vw,
    }
    try:
        pos = adaptive_lookback(prices, fe, fx, se, sx, vw, bm)
        m = evaluate(prices, pos, cfg, N_TRIALS, df, COSTS_BPS)
        if m:
            r4.append(
                {
                    "config": cfg,
                    "sharpe": m["sharpe"],
                    "dsr": m["dsr"],
                    "max_drawdown": m["max_drawdown"],
                    "n_trades": m["n_trades"],
                    "n_trials": N_TRIALS,
                    "beats_baseline": m["sharpe"] > BASELINE_SHARPE,
                }
            )
            ALL_RESULTS.append(r4[-1])
    except Exception:
        pass

print(f"Round 4: {len(r4)} configs, N_TRIALS={N_TRIALS}")
save_json(r4, "session_001_round4_results.json")
wandb_log_sweep(
    "regime_donchian_v3", "session_001_round4", sweep_results_to_wandb_format(r4), {"n": len(r4)}, TAGS + ["round4"]
)

# ══════════════════════════════════════════════════════════════════════════════
# ROUND 5: Deep grid around best configs
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nROUND 5: Refined Grid\n" + "=" * 70)
r5 = []

if r1:
    bc = max(r1, key=lambda x: x["dsr"])["config"]
    ep_c, xp_c, vw_c, tv_c = bc["entry_period"], bc["exit_period"], bc["vol_window"], bc["target_vol"]
    for ep, xp, vw, tv in product(
        [max(10, ep_c - 10), ep_c, ep_c + 10],
        [max(5, xp_c - 5), xp_c, xp_c + 5],
        [max(5, vw_c - 5), vw_c, vw_c + 5, vw_c + 10],
        [max(0.1, tv_c - 0.1), tv_c, tv_c + 0.1, tv_c + 0.2],
    ):
        if xp >= ep:
            continue
        N_TRIALS += 1
        cfg = {
            "strategy": "inverse_vol_sizing_refined",
            "entry_period": ep,
            "exit_period": xp,
            "vol_window": vw,
            "target_vol": tv,
        }
        try:
            pos = (baseline_donchian(prices, ep, xp) * inv_vol_sizing(prices, vw, tv)).clip(0, 1.5)
            m = evaluate(prices, pos, cfg, N_TRIALS, df, COSTS_BPS)
            if m:
                r5.append(
                    {
                        "config": cfg,
                        "sharpe": m["sharpe"],
                        "dsr": m["dsr"],
                        "max_drawdown": m["max_drawdown"],
                        "n_trades": m["n_trades"],
                        "n_trials": N_TRIALS,
                        "beats_baseline": m["sharpe"] > BASELINE_SHARPE,
                    }
                )
                ALL_RESULTS.append(r5[-1])
        except Exception:
            pass

if r3:
    bc = max(r3, key=lambda x: x["dsr"])["config"]
    ep, xp = bc["entry_period"], bc["exit_period"]
    vw_c, mw_c = bc["vol_window"], bc["mom_window"]
    s0, s1, s2, s3 = bc["size_low_up"], bc["size_low_down"], bc["size_high_up"], bc["size_high_down"]
    base_sig = baseline_donchian(prices, ep, xp)
    size_variants = [
        (s0, s1, s2, s3),
        (s0 + 0.1, s1, s2, s3),
        (s0, s1 + 0.1, s2, s3),
        (s0, s1, s2 + 0.1, s3),
        (s0, s1, s2, s3 + 0.1),
        (s0 + 0.15, s1, s2 - 0.1, s3),
        (s0 + 0.25, s1 * 0.5, s2 - 0.25, s3 * 0.5),
        (1.25, 0.25, 0.75, 0.0),
        (1.25, 0.0, 0.5, 0.0),
        (1.5, 0.5, 0.75, 0.25),
    ]
    for sv in size_variants:
        for vw, mw in product([vw_c, max(10, vw_c - 5), vw_c + 5], [mw_c, max(30, mw_c - 30), mw_c + 30]):
            N_TRIALS += 1
            cfg = {
                "strategy": "vol_momentum_4state_refined",
                "entry_period": ep,
                "exit_period": xp,
                "vol_window": vw,
                "mom_window": mw,
                "size_low_up": sv[0],
                "size_low_down": sv[1],
                "size_high_up": sv[2],
                "size_high_down": sv[3],
            }
            try:
                pos = vm4_sizing(prices, base_sig, sv, vw, mw)
                m = evaluate(prices, pos, cfg, N_TRIALS, df, COSTS_BPS)
                if m:
                    r5.append(
                        {
                            "config": cfg,
                            "sharpe": m["sharpe"],
                            "dsr": m["dsr"],
                            "max_drawdown": m["max_drawdown"],
                            "n_trades": m["n_trades"],
                            "n_trials": N_TRIALS,
                            "beats_baseline": m["sharpe"] > BASELINE_SHARPE,
                        }
                    )
                    ALL_RESULTS.append(r5[-1])
            except Exception:
                pass

print(f"Round 5: {len(r5)} configs, N_TRIALS={N_TRIALS}")
save_json(r5, "session_001_round5_results.json")

# ══════════════════════════════════════════════════════════════════════════════
# STRESS TEST: Top 10 at 50 bps
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nSTRESS TEST: 50 bps\n" + "=" * 70)
stress = []


def reconstruct_positions(cfg):
    s = cfg["strategy"]
    if s in ("inverse_vol_sizing", "inverse_vol_sizing_refined"):
        return (
            baseline_donchian(prices, cfg["entry_period"], cfg["exit_period"])
            * inv_vol_sizing(prices, cfg["vol_window"], cfg["target_vol"])
        ).clip(0, 1.5)
    elif s == "regime_param_switching":
        return regime_switching_donchian(
            prices,
            cfg["low_vol_entry"],
            cfg["low_vol_exit"],
            cfg["high_vol_entry"],
            cfg["high_vol_exit"],
            cfg["vol_window"],
        )
    elif s in ("vol_momentum_4state", "vol_momentum_4state_refined"):
        base = baseline_donchian(prices, cfg["entry_period"], cfg["exit_period"])
        return vm4_sizing(
            prices,
            base,
            (cfg["size_low_up"], cfg["size_low_down"], cfg["size_high_up"], cfg["size_high_down"]),
            cfg["vol_window"],
            cfg["mom_window"],
        )
    elif s == "adaptive_lookback":
        return adaptive_lookback(
            prices,
            cfg["fast_entry"],
            cfg["fast_exit"],
            cfg["slow_entry"],
            cfg["slow_exit"],
            cfg["vol_window"],
            cfg["blend_mode"],
        )
    return None


for r in sorted(ALL_RESULTS, key=lambda x: x["dsr"], reverse=True)[:10]:
    try:
        pos = reconstruct_positions(r["config"])
        if pos is None:
            continue
        m50 = compute_all_metrics(net_ret(prices, pos, CF_STRESS), n_trials=N_TRIALS, periods_per_year=PERIODS_PER_YEAR)
        stress.append({**r, "sharpe_50bps": m50["sharpe"], "dsr_50bps": m50["dsr"]})
        print(f"  {r['config']['strategy']}: 30bps={r['sharpe']:.3f}, 50bps={m50['sharpe']:.3f}")
    except Exception:
        pass

b50 = compute_all_metrics(
    net_ret(prices, baseline_pos, CF_STRESS), n_trials=N_TRIALS, periods_per_year=PERIODS_PER_YEAR
)
print(f"Baseline: 30bps={BASELINE_SHARPE:.3f}, 50bps={b50['sharpe']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nSESSION 001 SUMMARY\n" + "=" * 70)
n_beats = sum(r["beats_baseline"] for r in ALL_RESULTS)
winners = [r for r in ALL_RESULTS if r["dsr"] > 0.95 and r["sharpe"] >= 1.0]
print(f"Total: {N_TRIALS}, Beats baseline: {n_beats}, DSR>0.95: {len(winners)}")

if ALL_RESULTS:
    best_d = max(ALL_RESULTS, key=lambda x: x["dsr"])
    best_s = max(ALL_RESULTS, key=lambda x: x["sharpe"])
    print(f"Best DSR: {best_d['config']['strategy']} S={best_d['sharpe']:.3f} DSR={best_d['dsr']:.3f}")
    print(f"Best Sharpe: {best_s['config']['strategy']} S={best_s['sharpe']:.3f} DSR={best_s['dsr']:.3f}")
    for nm, rr in [
        ("R1 inv_vol", r1),
        ("R2 param_switch", r2),
        ("R3 4state", r3),
        ("R4 adaptive", r4),
        ("R5 refined", r5),
    ]:
        if rr:
            print(
                f"  {nm}: N={len(rr)}, best_S={max(x['sharpe'] for x in rr):.3f}, best_DSR={max(x['dsr'] for x in rr):.3f}"
            )

save_json(
    {
        "baseline_sharpe": BASELINE_SHARPE,
        "total_configs": N_TRIALS,
        "n_beats_baseline": n_beats,
        "n_dsr095": len(winners),
        "best_dsr_config": max(ALL_RESULTS, key=lambda x: x["dsr"]) if ALL_RESULTS else None,
        "best_sharpe_config": max(ALL_RESULTS, key=lambda x: x["sharpe"]) if ALL_RESULTS else None,
        "stress_test": stress,
    },
    "session_001_v3_summary.json",
)

if ALL_RESULTS:
    wandb_log_experiment(
        "regime_donchian_v3",
        "session_001_v3_consolidated",
        config={"n_rounds": 5, "costs_bps": COSTS_BPS},
        metrics={
            "sharpe": best_d["sharpe"],
            "dsr": best_d["dsr"],
            "n_configs": N_TRIALS,
            "baseline_sharpe": BASELINE_SHARPE,
        },
        tags=TAGS + ["consolidated"],
    )

print(f"\nDone. {N_TRIALS} configs. {'DSR>0.95 found!' if winners else 'No DSR>0.95.'}")
