"""Revalidate key results after inv_vol_sizing fix.

The old inv_vol_sizing hardcoded periods_per_year=365, which is wrong for
intraday data. This script re-runs key configs with corrected sizing and
reports old vs new metrics.

Run: .venv/bin/python scripts/infra/revalidate_inv_vol.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "infra"))

from experiment_runner import run  # noqa: E402

CONFIGS = [
    {
        "name": "BTC_Don4h(30,25)_invvol — Layer 4 benchmark",
        "config": {
            "asset": "btc",
            "timeframe": "4h",
            "signal_type": "donchian",
            "signal_params": {"entry_period": 30, "exit_period": 25},
            "sizing": "inverse_vol",
            "sizing_params": {"vol_window": 20, "target_vol": 0.4},
            "n_trials": 627,
        },
    },
    {
        "name": "BTC_Don2h(80,40)_invvol_vw40tv0.3",
        "config": {
            "asset": "btc",
            "timeframe": "2h",
            "signal_type": "donchian",
            "signal_params": {"entry_period": 80, "exit_period": 40},
            "sizing": "inverse_vol",
            "sizing_params": {"vol_window": 40, "target_vol": 0.3},
            "n_trials": 627,
        },
    },
    {
        "name": "BTC_Don4h(30,25)_flat — no sizing baseline",
        "config": {
            "asset": "btc",
            "timeframe": "4h",
            "signal_type": "donchian",
            "signal_params": {"entry_period": 30, "exit_period": 25},
            "sizing": "flat",
            "n_trials": 627,
        },
    },
    {
        "name": "ETH_Don8h(30,20)_invvol",
        "config": {
            "asset": "eth",
            "timeframe": "8h",
            "signal_type": "donchian",
            "signal_params": {"entry_period": 30, "exit_period": 20},
            "sizing": "inverse_vol",
            "sizing_params": {"vol_window": 20, "target_vol": 0.4},
            "n_trials": 627,
        },
    },
    {
        "name": "ETH_BB8h(20,2.0)_invvol",
        "config": {
            "asset": "eth",
            "timeframe": "8h",
            "signal_type": "bollinger",
            "signal_params": {"period": 20, "num_std": 2.0, "hold_periods": 10},
            "sizing": "inverse_vol",
            "sizing_params": {"vol_window": 20, "target_vol": 0.4},
            "n_trials": 627,
        },
    },
]


def main():
    print("=" * 90)
    print("inv_vol_sizing REVALIDATION — correct periods_per_year for intraday data")
    print("=" * 90)
    print(f"\n{'Name':<45} {'Sharpe@30':<12} {'Sharpe@50':<12} {'MaxDD':<10} {'DSR':<8} {'Trades':<8}")
    print("-" * 90)

    for entry in CONFIGS:
        name = entry["name"]
        cfg = entry["config"]
        r = run(cfg)
        if r is None:
            print(f"{name:<45} {'FAILED':<12}")
            continue
        s30 = f"{r['sharpe_30bps']:.3f}"
        s50 = f"{r['sharpe_50bps']:.3f}" if r["sharpe_50bps"] else "N/A"
        dd = f"{r['max_drawdown']:.3f}"
        dsr = f"{r['dsr_30bps']:.3f}" if r["dsr_30bps"] else "N/A"
        nt = str(r["n_trades"])
        print(f"{name:<45} {s30:<12} {s50:<12} {dd:<10} {dsr:<8} {nt:<8}")

        # Sub-period detail
        sp = r.get("sub_periods", {})
        for period_name, pm in sp.items():
            if period_name == "full":
                continue
            ps = pm.get("sharpe", "?")
            print(f"  {period_name}: Sharpe={ps}")

    print("=" * 90)
    print("Compare against previous (broken) results to assess impact.")
    print("Flat-sizing configs should be unchanged (no inv_vol_sizing used).")


if __name__ == "__main__":
    main()
