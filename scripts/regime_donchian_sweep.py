"""Regime Donchian Sweep — Directive 002, Session 001.

Tests vol_filtered, HMM regime, and regime_weighted_ensemble against Donchian baseline.
BTC daily, IS only, 30 bps.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "infra"))

from sweep_utils import (  # noqa: E402
    baseline_donchian,
    evaluate,
    load_daily,
    print_top,
    run_pre,
    save_json,
    wandb_log_sweep,
)
from sparky.models.regime_filtered_donchian import regime_filtered_donchian
from sparky.models.regime_hmm import hmm_regime_donchian, hmm_probabilistic_ensemble
from sparky.models.regime_weighted_ensemble import regime_weighted_ensemble

TAGS = ["regime_donchian", "directive_002", "session_001"]
COSTS_BPS = 30


def main():
    df = load_daily()
    prices = df["close"]
    if not run_pre(df, {"strategy": "sweep", "transaction_costs_bps": COSTS_BPS}):
        raise SystemExit("Pre-checks failed")

    # Baseline
    base_sig = baseline_donchian(prices)
    base_m = evaluate(
        prices, base_sig, {"strategy": "donchian_baseline", "entry_period": 40, "exit_period": 20}, 1, df, COSTS_BPS
    )
    base_sharpe = base_m["sharpe"] if base_m else 1.062
    baseline = {"config": {"strategy": "donchian_baseline"}, "metrics": base_m}
    print(f"[BASELINE] Sharpe={base_sharpe:.4f}")

    # Priority 1: vol_filtered_donchian
    vol_results = []
    configs = [
        (ep, xp, vw) for ep in [15, 20, 30, 40, 60] for xp in [5, 10, 15, 20, 30] for vw in [20, 30, 45, 60] if xp < ep
    ]
    print(f"\n[VOL_FILTER] {len(configs)} configs...")
    for i, (ep, xp, vw) in enumerate(configs):
        try:
            sig = regime_filtered_donchian(prices, entry_period=ep, exit_period=xp, vol_window=vw, filter_high_vol=True)
            cfg = {"strategy": "vol_filtered_donchian", "entry_period": ep, "exit_period": xp, "vol_window": vw}
            m = evaluate(prices, sig, cfg, i + 1, df, COSTS_BPS)
            if m:
                vol_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass

    # Priority 2: HMM regime donchian
    hmm_results = []
    n_prior = len(vol_results) + 1
    hmm_configs = [
        (ns, epl, xpl, eph, xph)
        for ns in [2, 3]
        for epl in [15, 20, 30]
        for xpl in [5, 10]
        for eph in [40, 60]
        for xph in [20, 30]
        if xpl < epl and xph < eph
    ]
    print(f"\n[HMM] {len(hmm_configs) + 2} configs...")
    for i, (ns, epl, xpl, eph, xph) in enumerate(hmm_configs):
        try:
            sig = hmm_regime_donchian(prices, n_states=ns, aggressive_params=(epl, xpl), conservative_params=(eph, xph))
            cfg = {
                "strategy": "hmm_regime_donchian",
                "n_states": ns,
                "entry_period_low_vol": epl,
                "exit_period_low_vol": xpl,
                "entry_period_high_vol": eph,
                "exit_period_high_vol": xph,
            }
            m = evaluate(prices, sig, cfg, n_prior + i, df, COSTS_BPS)
            if m:
                hmm_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass

    for ns in [2, 3]:
        try:
            sig = hmm_probabilistic_ensemble(prices, n_states=ns)
            cfg = {"strategy": "hmm_probabilistic_ensemble", "n_states": ns}
            m = evaluate(prices, sig, cfg, n_prior + len(hmm_configs) + ns, df, COSTS_BPS)
            if m:
                hmm_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass

    # Priority 3: regime_weighted_ensemble
    ens_results = []
    n_prior2 = n_prior + len(hmm_results)
    det_map = {"vol_threshold": "combined", "hmm": "volatility"}
    ens_configs = [
        (rm, det_map[rm], w, vw)
        for rm in ["vol_threshold", "hmm"]
        for w in ["aggressive", "balanced", "defensive"]
        for vw in [20, 30, 45]
    ]
    print(f"\n[ENSEMBLE] {len(ens_configs)} configs...")
    for i, (rm, det, w, vw) in enumerate(ens_configs):
        try:
            sig = regime_weighted_ensemble(prices, regime_detection=det, weighting_scheme=w, vol_window=vw)
            cfg = {"strategy": "regime_weighted_ensemble", "regime_method": rm, "weighting_scheme": w, "vol_window": vw}
            m = evaluate(prices, sig, cfg, n_prior2 + i, df, COSTS_BPS)
            if m:
                ens_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass

    # Ablation: filter OFF
    abl_results = []
    n_prior3 = n_prior2 + len(ens_results)
    for i, (ep, xp) in enumerate([(40, 20), (20, 10), (30, 10)]):
        try:
            sig = regime_filtered_donchian(
                prices, entry_period=ep, exit_period=xp, vol_window=30, filter_high_vol=False
            )
            cfg = {"strategy": "vol_filtered_ablation", "entry_period": ep, "exit_period": xp, "filter_high_vol": False}
            m = evaluate(prices, sig, cfg, n_prior3 + i, df, COSTS_BPS)
            if m:
                abl_results.append({"config": cfg, "metrics": m})
        except Exception:
            pass

    all_results = [baseline] + vol_results + hmm_results + ens_results + abl_results
    print_top(all_results, base_sharpe)

    wandb_log_sweep(
        "regime_donchian_sweep",
        "directive_002_session_001",
        all_results,
        {"n_configs": len(all_results), "baseline_sharpe": base_sharpe},
        TAGS,
    )
    save_json(
        {
            "baseline": baseline,
            "vol": vol_results,
            "hmm": hmm_results,
            "ensemble": ens_results,
            "ablation": abl_results,
        },
        "session_001_results.json",
    )


if __name__ == "__main__":
    main()
