#!/usr/bin/env python3
"""OOS validation sweep — bug checks, parameter exploration, ETH-only evaluation."""

import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")  # noqa: S108

ROOT = Path(__file__).parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bin/infra"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sparky.data.loader import load
from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.tracking.metrics import compute_all_metrics
from sweep_utils import inv_vol_sizing

OUT = Path("/tmp/sparky_oos_reports/validation_sweep")  # noqa: S108
OOS_START = pd.Timestamp("2024-01-01", tz="UTC")
PPY = 1095
COST_BPS = 30


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_full_data(dataset):
    """Load IS + OOS data merged."""
    is_df = load(dataset, purpose="analysis")
    oos_df = load(dataset, purpose="evaluation")
    df = pd.concat([is_df, oos_df])
    return df[~df.index.duplicated(keep="last")].sort_index()


def build_leg(dataset, entry, exit_, vw, tv, cost_bps):
    """Build net returns for one leg using correct IV timing."""
    df = load_full_data(dataset)
    prices = df["close"]
    signals = donchian_channel_strategy(prices, entry, exit_)
    scale = inv_vol_sizing(prices, vw, tv, PPY)
    positions = (signals * scale).shift(1).fillna(0)
    price_returns = prices.pct_change().fillna(0)
    gross = positions * price_returns
    cost_frac = cost_bps / 10_000
    costs = positions.diff().abs().fillna(0) * cost_frac
    net = gross - costs
    return {"returns": net, "positions": positions, "prices": prices, "signals": signals, "scale": scale}


def build_portfolio(legs_data, weights, cost_bps):
    """Combine legs into portfolio returns."""
    all_returns = [d["returns"] for d in legs_data]
    common_idx = all_returns[0].index
    for s in all_returns[1:]:
        common_idx = common_idx.intersection(s.index)
    portfolio = pd.Series(0.0, index=common_idx)
    for ret, w in zip(all_returns, weights):
        portfolio += ret.reindex(common_idx).fillna(0) * w
    return portfolio


# ── Part 1: Bug Validation ───────────────────────────────────────────────────


def test_is_reproduction(btc, eth, port_returns):
    is_port = port_returns[port_returns.index < OOS_START]
    is_btc = btc["returns"][btc["returns"].index < OOS_START]
    is_eth = eth["returns"][eth["returns"].index < OOS_START]

    pm = compute_all_metrics(is_port, n_trials=1, periods_per_year=PPY)
    bm = compute_all_metrics(is_btc, n_trials=1, periods_per_year=PPY)
    em = compute_all_metrics(is_eth, n_trials=1, periods_per_year=PPY)

    checks = [
        ("Port Sharpe", pm["sharpe"], 2.217, 0.1),
        ("Port MaxDD", pm["max_drawdown"], -0.070, 0.01),
        ("BTC Sharpe", bm["sharpe"], 2.220, 0.1),
        ("ETH Sharpe", em["sharpe"], 2.05, 0.2),
        ("BTC MaxDD", bm["max_drawdown"], -0.197, 0.01),
    ]

    results = []
    for name, actual, expected, tol in checks:
        passed = abs(actual - expected) <= tol
        results.append({"name": name, "actual": actual, "expected": expected, "tol": tol, "passed": passed})
    return results


def test_signal_verification(btc, eth):
    btc_prices = load_full_data("btc_ohlcv_8h")["close"]
    btc_signals_check = donchian_channel_strategy(btc_prices, 82, 20)
    eth_prices = load_full_data("eth_ohlcv_8h")["close"]
    eth_signals_check = donchian_channel_strategy(eth_prices, 83, 33)

    btc_match = btc["signals"].equals(btc_signals_check)
    eth_match = eth["signals"].equals(eth_signals_check)

    return [
        {"name": "BTC signals match", "passed": btc_match, "actual": btc_match, "expected": True, "tol": 0},
        {"name": "ETH signals match", "passed": eth_match, "actual": eth_match, "expected": True, "tol": 0},
    ]


def test_iv_scale_match(btc, eth):
    btc_prices = load_full_data("btc_ohlcv_8h")["close"]
    btc_scale_check = inv_vol_sizing(btc_prices, 30, 0.20, PPY)
    eth_prices = load_full_data("eth_ohlcv_8h")["close"]
    eth_scale_check = inv_vol_sizing(eth_prices, 30, 0.15, PPY)

    btc_is = btc["scale"][btc["scale"].index < OOS_START]
    btc_check_is = btc_scale_check.reindex(btc_is.index)
    btc_ok = np.allclose(btc_is.dropna(), btc_check_is.dropna(), atol=1e-6)

    eth_is = eth["scale"][eth["scale"].index < OOS_START]
    eth_check_is = eth_scale_check.reindex(eth_is.index)
    eth_ok = np.allclose(eth_is.dropna(), eth_check_is.dropna(), atol=1e-6)

    return [
        {"name": "BTC IV scale match", "passed": btc_ok, "actual": btc_ok, "expected": True, "tol": 0},
        {"name": "ETH IV scale match", "passed": eth_ok, "actual": eth_ok, "expected": True, "tol": 0},
    ]


def test_return_spot_check(btc):
    is_returns = btc["returns"][btc["returns"].index < OOS_START]
    positions = btc["positions"]
    prices = btc["prices"]
    price_returns = prices.pct_change().fillna(0)

    np.random.seed(42)
    valid_idx = is_returns.index[10:]
    sample_idx = np.random.choice(len(valid_idx), size=min(10, len(valid_idx)), replace=False)
    sample_dates = valid_idx[sample_idx]

    mismatches = []
    for dt in sample_dates:
        pos_t = positions.loc[dt]
        pr_t = price_returns.loc[dt]
        dt_loc = positions.index.get_loc(dt)
        pos_prev = positions.iloc[dt_loc - 1] if dt_loc > 0 else 0.0
        cost_frac = COST_BPS / 10_000
        expected = pos_t * pr_t - abs(pos_t - pos_prev) * cost_frac
        actual = is_returns.loc[dt]
        if abs(expected - actual) > 1e-10:
            mismatches.append({"date": dt, "expected": expected, "actual": actual})

    passed = len(mismatches) == 0
    detail = f"{len(mismatches)} mismatches" if mismatches else "all 10 match"
    return [{"name": "Return spot check", "passed": passed, "actual": detail, "expected": "all match", "tol": 0}]


def test_data_continuity():
    results = []
    for dataset in ["btc_ohlcv_8h", "eth_ohlcv_8h"]:
        df = load_full_data(dataset)
        prices = df["close"]

        no_dupes = not df.index.duplicated().any()
        monotonic = df.index.is_monotonic_increasing
        diffs = df.index.to_series().diff().dropna()
        max_gap = diffs.max()
        no_big_gaps = max_gap <= pd.Timedelta(hours=24)

        boundary = pd.Timestamp("2024-01-01", tz="UTC")
        near_boundary = prices[
            (prices.index >= boundary - pd.Timedelta(days=1)) & (prices.index <= boundary + pd.Timedelta(days=1))
        ]
        if len(near_boundary) >= 2:
            pct_changes = near_boundary.pct_change().dropna().abs()
            no_jumps = (pct_changes < 0.10).all()
        else:
            no_jumps = True

        asset = dataset.split("_")[0].upper()
        all_ok = no_dupes and monotonic and no_big_gaps and no_jumps
        results.append(
            {
                "name": f"{asset} data continuity",
                "passed": all_ok,
                "actual": f"dupes={not no_dupes}, mono={monotonic}, gap={max_gap}, jumps={not no_jumps}",
                "expected": "all clean",
                "tol": 0,
            }
        )
    return results


def write_validation_failure_report(all_results):
    lines = ["# Validation Failure Report", f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    for r in all_results:
        status = "PASS" if r["passed"] else "**FAIL**"
        actual_str = f"{r['actual']:.4f}" if isinstance(r["actual"], float) else str(r["actual"])
        expected_str = f"{r['expected']:.4f}" if isinstance(r["expected"], float) else str(r["expected"])
        tol_str = f" (+-{r['tol']})" if r["tol"] else ""
        lines.append(f"- {status}: {r['name']} — actual={actual_str}, expected={expected_str}{tol_str}")
    (OUT / "validation_failure.md").write_text("\n".join(lines))
    print(f"  Failure report: {OUT / 'validation_failure.md'}")


# ── Part 2: Parameter Exploration ────────────────────────────────────────────


WEIGHT_CONFIGS = [
    ("ETH-only", 0.00, 1.00),
    ("ETH-heavy", 0.10, 0.90),
    ("ETH-tilt", 0.20, 0.80),
    ("Champion", 0.30, 0.70),
    ("Balanced", 0.40, 0.60),
    ("Even", 0.50, 0.50),
]

ETH_TV_CONFIGS = [
    ("Aggressive", 0.25),
    ("Original", 0.15),
    ("Reduced", 0.10),
    ("Conservative", 0.07),
    ("Minimal", 0.05),
]


def run_weight_sweep(btc, eth):
    print("\n  Sweep 1: Weight sensitivity")
    results = []
    for label, w_btc, w_eth in WEIGHT_CONFIGS:
        port = build_portfolio([btc, eth], [w_btc, w_eth], COST_BPS)
        oos = port[port.index >= OOS_START]
        m = compute_all_metrics(oos, n_trials=1, periods_per_year=PPY)
        results.append({"label": label, "w_btc": w_btc, "w_eth": w_eth, **m})

    print(f"\n  {'Config':<12} {'wBTC':>5} {'wETH':>5} {'Sharpe':>8} {'MaxDD':>8} {'Return':>8} {'Calmar':>8}")
    print("  " + "-" * 56)
    for r in results:
        print(
            f"  {r['label']:<12} {r['w_btc']:5.2f} {r['w_eth']:5.2f} "
            f"{r['sharpe']:8.3f} {r['max_drawdown']:8.3f} {r['total_return']:8.1%} {r['calmar']:8.3f}"
        )
    return results


def run_eth_tv_sweep():
    print("\n  Sweep 2: ETH target vol (ETH-only)")
    results = []
    for label, tv in ETH_TV_CONFIGS:
        leg = build_leg("eth_ohlcv_8h", 83, 33, 30, tv, COST_BPS)
        oos = leg["returns"][leg["returns"].index >= OOS_START]
        m = compute_all_metrics(oos, n_trials=1, periods_per_year=PPY)
        results.append({"label": label, "tv": tv, **m})

    print(f"\n  {'Config':<14} {'TV':>5} {'Sharpe':>8} {'MaxDD':>8} {'Return':>8} {'Calmar':>8}")
    print("  " + "-" * 51)
    for r in results:
        print(
            f"  {r['label']:<14} {r['tv']:5.2f} "
            f"{r['sharpe']:8.3f} {r['max_drawdown']:8.3f} {r['total_return']:8.1%} {r['calmar']:8.3f}"
        )
    return results


def plot_sweeps(weight_results, tv_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: weight sweep
    sharpes = [r["sharpe"] for r in weight_results]
    max_dds = [abs(r["max_drawdown"]) for r in weight_results]
    labels = [r["label"] for r in weight_results]
    ax1.scatter(max_dds, sharpes, color="#2196F3", s=100, zorder=5)
    for i, lbl in enumerate(labels):
        ax1.annotate(lbl, (max_dds[i], sharpes[i]), textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax1.set_xlabel("Max Drawdown (absolute)")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Weight Sensitivity — OOS")
    ax1.grid(alpha=0.3)

    # Right: ETH tv sweep
    sharpes = [r["sharpe"] for r in tv_results]
    max_dds = [abs(r["max_drawdown"]) for r in tv_results]
    labels = [r["label"] for r in tv_results]
    ax2.scatter(max_dds, sharpes, color="#FF9800", s=100, zorder=5)
    for i, lbl in enumerate(labels):
        ax2.annotate(lbl, (max_dds[i], sharpes[i]), textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax2.set_xlabel("Max Drawdown (absolute)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("ETH Target Vol Sensitivity — OOS")
    ax2.grid(alpha=0.3)

    fig.suptitle("OOS Parameter Exploration")
    fig.tight_layout()
    fig.savefig(OUT / "parameter_sweeps.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved {OUT / 'parameter_sweeps.png'}")


# ── Part 3: ETH-Only Formal Evaluation ──────────────────────────────────────


def eth_only_evaluation():
    print("\nBuilding ETH-only leg...")
    eth = build_leg("eth_ohlcv_8h", 83, 33, 30, 0.15, COST_BPS)
    ret = eth["returns"]

    is_ret = ret[ret.index < OOS_START]
    oos_ret = ret[ret.index >= OOS_START]

    is_m = compute_all_metrics(is_ret, n_trials=1, periods_per_year=PPY)
    oos_m = compute_all_metrics(oos_ret, n_trials=1, periods_per_year=PPY)
    full_m = compute_all_metrics(ret, n_trials=1, periods_per_year=PPY)

    sharpe_pass = oos_m["sharpe"] >= 0.5
    maxdd_pass = oos_m["max_drawdown"] >= -0.30
    overall = sharpe_pass and maxdd_pass
    verdict = "PASS" if overall else "FAIL"

    report = f"""# ETH-Only OOS Evaluation Report — {verdict}
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Strategy
- ETH Don8h(83,33) inverse_vol(vw=30, tv=0.15)
- Weight: 100%
- Cost: 30 bps/side
- PPY: 1095

## OOS Results (2024-01-01 to present)
- Sharpe: {oos_m["sharpe"]:.3f}
- Sortino: {oos_m["sortino"]:.3f}
- MaxDD: {oos_m["max_drawdown"]:.3f}
- Calmar: {oos_m["calmar"]:.3f}
- Total Return: {oos_m["total_return"]:.1%}
- Win Rate: {oos_m["win_rate"]:.1%}
- Profit Factor: {oos_m["profit_factor"]:.3f}
- N Observations: {oos_m["n_observations"]}
- DSR (n=1): {oos_m["dsr"]:.3f}

## In-Sample Results (to 2024-01-01)
- Sharpe: {is_m["sharpe"]:.3f}
- Sortino: {is_m["sortino"]:.3f}
- MaxDD: {is_m["max_drawdown"]:.3f}
- Total Return: {is_m["total_return"]:.1%}
- N Observations: {is_m["n_observations"]}

## Full-Period Results
- Sharpe: {full_m["sharpe"]:.3f}
- Sortino: {full_m["sortino"]:.3f}
- MaxDD: {full_m["max_drawdown"]:.3f}
- Calmar: {full_m["calmar"]:.3f}
- Total Return: {full_m["total_return"]:.1%}
- Win Rate: {full_m["win_rate"]:.1%}
- Profit Factor: {full_m["profit_factor"]:.3f}
- N Observations: {full_m["n_observations"]}
- DSR (n=1): {full_m["dsr"]:.3f}

## Verdict
- {"PASS" if sharpe_pass else "FAIL"}: Sharpe {oos_m["sharpe"]:.3f} {"≥" if sharpe_pass else "<"} 0.500
- {"PASS" if maxdd_pass else "FAIL"}: MaxDD {oos_m["max_drawdown"]:.3f} {"≥" if maxdd_pass else "<"} -0.300

**Overall: {verdict}**
"""
    path = OUT / "eth_only_evaluation.md"
    path.write_text(report)
    print(f"  Saved {path}")

    print(
        f"\n  ETH-Only OOS: Sharpe={oos_m['sharpe']:.3f}  MaxDD={oos_m['max_drawdown']:.3f}  "
        f"Return={oos_m['total_return']:.1%}  Verdict={verdict}"
    )
    return oos_m, verdict


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OOS Validation Sweep")
    print("=" * 60)

    # Build legs
    print("\nBuilding legs...")
    btc = build_leg("btc_ohlcv_8h", 82, 20, 30, 0.20, COST_BPS)
    eth = build_leg("eth_ohlcv_8h", 83, 33, 30, 0.15, COST_BPS)
    port_returns = build_portfolio([btc, eth], [0.30, 0.70], COST_BPS)

    # Part 1: Validation
    print("\n" + "=" * 60)
    print("PART 1: Bug Validation")
    print("=" * 60)

    all_results = []
    all_results.extend(test_is_reproduction(btc, eth, port_returns))
    all_results.extend(test_signal_verification(btc, eth))
    all_results.extend(test_iv_scale_match(btc, eth))
    all_results.extend(test_return_spot_check(btc))
    all_results.extend(test_data_continuity())

    print(f"\n{'Test':<30} {'Result':<8} {'Actual':<20} {'Expected':<20}")
    print("-" * 78)
    all_passed = True
    for r in all_results:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        actual_str = f"{r['actual']:.4f}" if isinstance(r["actual"], float) else str(r["actual"])
        expected_str = f"{r['expected']:.4f}" if isinstance(r["expected"], float) else str(r["expected"])
        tol_str = f" (+-{r['tol']})" if r["tol"] else ""
        print(f"{r['name']:<30} {status:<8} {actual_str:<20} {expected_str}{tol_str}")

    if not all_passed:
        print("\n*** VALIDATION FAILED — skipping Parts 2 and 3 ***")
        write_validation_failure_report(all_results)
        sys.exit(1)

    print("\nAll validation tests PASSED")

    # Part 2: Parameter exploration
    print("\n" + "=" * 60)
    print("PART 2: Parameter Exploration")
    print("=" * 60)

    weight_results = run_weight_sweep(btc, eth)
    tv_results = run_eth_tv_sweep()
    plot_sweeps(weight_results, tv_results)

    # Part 3: ETH-only formal evaluation
    print("\n" + "=" * 60)
    print("PART 3: ETH-Only Formal Evaluation")
    print("=" * 60)

    eth_only_evaluation()

    print("\nDone. Output in:", OUT)


if __name__ == "__main__":
    main()
