#!/usr/bin/env python3
"""OOS diagnostics — drawdown analysis for champion portfolio."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bin/infra"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml

from sparky.data.loader import load
from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.tracking.metrics import compute_all_metrics

OUT = Path("/tmp/sparky_oos_reports/diagnostics")  # noqa: S108
OUT.mkdir(parents=True, exist_ok=True, mode=0o700)

with open(ROOT / "configs/holdout_policy.yaml") as _f:
    _holdout = yaml.safe_load(_f)
OOS_START = pd.Timestamp(_holdout["holdout_periods"]["cross_asset"]["oos_start"], tz="UTC")
PPY = 1095
COST_BPS = 30

# Colors
BLUE, GREEN, RED, ORANGE, PURPLE = "#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"


def inv_vol_sizing(prices, vw, tv, ppy=1095):
    rvol = np.log(prices / prices.shift(1)).rolling(vw).std() * np.sqrt(ppy)
    return (tv / rvol).clip(0.1, 1.5).fillna(0.5)


def build_leg_returns(dataset, entry, exit_, vw, tv):
    is_df = load(dataset, purpose="training")
    oos_df = load(dataset, purpose="evaluation")
    oos_df = oos_df[oos_df.index >= OOS_START]
    df = pd.concat([is_df, oos_df]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    gap = oos_df.index[0] - is_df.index[-1]
    assert gap <= pd.Timedelta(hours=16), f"Gap at IS/OOS boundary: {gap}"
    prices = df["close"]

    signals = donchian_channel_strategy(prices, entry, exit_)
    sizing = inv_vol_sizing(prices, vw, tv, PPY)
    positions = (signals * sizing).shift(1).fillna(0)

    price_returns = prices.pct_change().fillna(0)
    if prices.iloc[1:].isna().any():
        print(f"  WARNING: interior NaN prices in {dataset}")
    gross = positions * price_returns
    cost_frac = COST_BPS / 10_000
    costs = positions.diff().abs().fillna(0) * cost_frac
    net = gross - costs
    return {"returns": net, "positions": positions, "prices": prices}


def build_portfolio(btc_leg, eth_leg, w_btc=0.30, w_eth=0.70):
    common = btc_leg["returns"].index.intersection(eth_leg["returns"].index)
    return w_btc * btc_leg["returns"].loc[common] + w_eth * eth_leg["returns"].loc[common]


def save_fig(fig, name):
    fig.savefig(OUT / name, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


def main():
    # ── Build portfolio ──────────────────────────────────────────────────────────
    print("Building champion portfolio...")
    btc = build_leg_returns("btc_ohlcv_8h", 82, 20, 30, 0.20)
    eth = build_leg_returns("eth_ohlcv_8h", 83, 33, 30, 0.15)
    port_returns = build_portfolio(btc, eth)
    oos = port_returns[port_returns.index >= OOS_START]
    print(f"  OOS period: {oos.index[0].date()} to {oos.index[-1].date()} ({len(oos)} bars)")

    # Full metrics
    metrics = compute_all_metrics(oos, n_trials=1, periods_per_year=PPY)
    print(
        f"  Sharpe={metrics['sharpe']:.3f}  MaxDD={metrics['max_drawdown']:.3f}  "
        f"Return={metrics['total_return']:.3f}  DSR={metrics['dsr']:.3f}"
    )

    # ── Diagnostic 1: Equity Curve + Drawdown ────────────────────────────────────
    print("\nDiagnostic 1: Equity curve + drawdown...")
    equity = (1 + oos).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    # Find max drawdown details
    dd_trough_idx = drawdown.idxmin()
    dd_trough_val = drawdown.min()
    mask_before = drawdown.loc[:dd_trough_idx]
    dd_start_idx = mask_before[mask_before == 0].index[-1] if (mask_before == 0).any() else oos.index[0]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity.index, equity.values, color=BLUE, linewidth=1.2)
    ax.axvline(dd_trough_idx, color=RED, linestyle="--", alpha=0.6, label=f"Max DD trough: {dd_trough_val:.1%}")
    ax.set_title("OOS Equity Curve — Champion Portfolio (30/70 BTC/ETH)")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    save_fig(fig, "equity_curve.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(drawdown.index, drawdown.values, 0, color=RED, alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color=RED, linewidth=0.8)
    ax.axhline(-0.15, color=ORANGE, linestyle="--", alpha=0.7, label="-15% threshold")
    ax.annotate(
        f"Max DD: {dd_trough_val:.1%}\n{dd_trough_idx.date()}",
        xy=(dd_trough_idx, dd_trough_val),
        xytext=(dd_trough_idx, dd_trough_val - 0.03),
        fontsize=9,
        color=RED,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED),
    )
    # Mark drawdowns exceeding -15%
    severe = drawdown[drawdown < -0.15]
    if len(severe) > 0:
        ax.scatter(severe.index, severe.values, color=RED, s=10, zorder=5, label="Below -15%")
    ax.set_title("Underwater Chart — Drawdown from Peak")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    save_fig(fig, "underwater.png")

    # ── Diagnostic 2: Per-Leg Decomposition ──────────────────────────────────────
    print("\nDiagnostic 2: Per-leg decomposition...")
    btc_oos = btc["returns"][btc["returns"].index >= OOS_START]
    eth_oos = eth["returns"][eth["returns"].index >= OOS_START]

    btc_m = compute_all_metrics(btc_oos, n_trials=1, periods_per_year=PPY)
    eth_m = compute_all_metrics(eth_oos, n_trials=1, periods_per_year=PPY)

    leg_table = {
        "BTC Don8h(82,20)": btc_m,
        "ETH Don8h(83,33)": eth_m,
        "Portfolio 30/70": metrics,
    }

    for name, m in leg_table.items():
        print(
            f"  {name}: S={m['sharpe']:.3f} DD={m['max_drawdown']:.3f} Ret={m['total_return']:.3f} Calmar={m['calmar']:.3f}"
        )

    # Rolling correlation
    common = btc_oos.index.intersection(eth_oos.index)
    btc_common = btc_oos.loc[common]
    eth_common = eth_oos.loc[common]
    rolling_corr = btc_common.rolling(60).corr(eth_common)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top: cumulative returns
    btc_eq = (1 + btc_oos).cumprod()
    eth_eq = (1 + eth_oos).cumprod()
    port_eq = (1 + oos).cumprod()
    ax1.plot(btc_eq.index, btc_eq.values, color=ORANGE, label="BTC leg", linewidth=1)
    ax1.plot(eth_eq.index, eth_eq.values, color=PURPLE, label="ETH leg", linewidth=1)
    ax1.plot(port_eq.index, port_eq.values, color=BLUE, label="Portfolio", linewidth=1.5)
    ax1.set_title("Per-Leg Decomposition — OOS Cumulative Returns")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Bottom: rolling correlation with drawdown shading
    ax2.plot(rolling_corr.index, rolling_corr.values, color=GREEN, linewidth=1)
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax2.axhline(0.5, color=ORANGE, linestyle="--", alpha=0.5, label="ρ=0.5")
    # Shade drawdown periods
    dd_mask = drawdown < -0.05
    if dd_mask.any():
        dd_periods = dd_mask.astype(int).diff().fillna(0)
        starts = dd_periods[dd_periods == 1].index
        ends = dd_periods[dd_periods == -1].index
        for s in starts:
            e_candidates = ends[ends > s]
            e = e_candidates[0] if len(e_candidates) > 0 else drawdown.index[-1]
            ax2.axvspan(s, e, color=RED, alpha=0.1)
    ax2.set_title("Rolling 60-bar BTC-ETH Correlation")
    ax2.set_ylabel("Correlation")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    save_fig(fig, "leg_decomposition.png")

    # ── Diagnostic 3: Target Vol Sensitivity ─────────────────────────────────────
    print("\nDiagnostic 3: Target vol sensitivity...")
    tv_configs = [
        ("Original", 0.20, 0.15),
        ("Reduced", 0.15, 0.10),
        ("Conservative", 0.10, 0.07),
        ("Minimal", 0.05, 0.035),
    ]

    tv_results = []
    for name, btc_tv, eth_tv in tv_configs:
        b = build_leg_returns("btc_ohlcv_8h", 82, 20, 30, btc_tv)
        e = build_leg_returns("eth_ohlcv_8h", 83, 33, 30, eth_tv)
        p = build_portfolio(b, e)
        p_oos = p[p.index >= OOS_START]
        m = compute_all_metrics(p_oos, n_trials=1, periods_per_year=PPY)
        tv_results.append({"name": name, "btc_tv": btc_tv, "eth_tv": eth_tv, **m})
        print(
            f"  {name} (BTC={btc_tv}, ETH={eth_tv}): "
            f"S={m['sharpe']:.3f} DD={m['max_drawdown']:.3f} Ret={m['total_return']:.3f}"
        )

    fig, ax = plt.subplots(figsize=(14, 6))
    sharpes = [r["sharpe"] for r in tv_results]
    max_dds = [abs(r["max_drawdown"]) for r in tv_results]
    names = [r["name"] for r in tv_results]
    ax.scatter(max_dds, sharpes, color=BLUE, s=100, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, (max_dds[i], sharpes[i]), textcoords="offset points", xytext=(10, 5), fontsize=9)
    ax.set_xlabel("Max Drawdown (absolute)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Target Vol Sensitivity — Sharpe vs Max Drawdown")
    ax.grid(alpha=0.3)
    save_fig(fig, "tv_sensitivity.png")

    # ── Diagnostic 4: Monthly Return Heatmap ─────────────────────────────────────
    print("\nDiagnostic 4: Monthly returns heatmap...")
    oos_daily = (1 + oos).resample("D").prod() - 1
    monthly = (1 + oos_daily).resample("ME").prod() - 1
    monthly_df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
    pivot = monthly_df.pivot(index="year", columns="month", values="ret")

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            weight = "bold" if val < -0.05 else "normal"
            color = "white" if abs(val) > 0.10 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8, fontweight=weight, color=color)
    ax.set_title("Monthly Returns Heatmap — OOS Period")
    fig.colorbar(im, ax=ax, label="Return", shrink=0.8)
    save_fig(fig, "monthly_returns.png")

    # ── Diagnostic 5: Regime Overlay ─────────────────────────────────────────────
    print("\nDiagnostic 5: Regime overlay...")
    try:
        onchain = load("btc_onchain_bgeometrics", purpose="analysis")
        available_cols = [c for c in onchain.columns if c in ["mvrv_zscore", "sopr", "nupl", "puell_multiple"]]
        if len(available_cols) < 2:
            raise ValueError(f"Only found columns: {list(onchain.columns)}")

        # Resample portfolio to daily for alignment
        port_daily = (1 + oos).resample("D").prod() - 1
        port_daily_eq = (1 + port_daily).cumprod()
        port_daily_dd = port_daily_eq / port_daily_eq.cummax() - 1

        # Align dates
        common_idx = port_daily.index.intersection(onchain.index)
        if len(common_idx) < 30:
            # Try tz-naive alignment
            onchain.index = onchain.index.tz_localize("UTC") if onchain.index.tz is None else onchain.index
            common_idx = port_daily.index.intersection(onchain.index)

        has_mvrv = "mvrv_zscore" in available_cols
        has_nupl = "nupl" in available_cols
        has_sopr = "sopr" in available_cols

        n_panels = 1 + (1 if has_mvrv or has_nupl else 0) + (1 if has_sopr else 0)
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        panel = 0
        # Top: portfolio equity
        axes[panel].plot(
            port_daily_eq.loc[common_idx].index, port_daily_eq.loc[common_idx].values, color=BLUE, linewidth=1.2
        )
        axes[panel].set_title("OOS Portfolio Cumulative Returns + Regime Signals")
        axes[panel].set_ylabel("Equity")
        axes[panel].grid(alpha=0.3)

        # Shade drawdown > 5%
        dd_5 = port_daily_dd.loc[common_idx] < -0.05
        for ax_i in axes:
            if dd_5.any():
                dd_int = dd_5.astype(int).diff().fillna(0)
                dd_starts = dd_int[dd_int == 1].index
                dd_ends = dd_int[dd_int == -1].index
                for s in dd_starts:
                    ec = dd_ends[dd_ends > s]
                    e = ec[0] if len(ec) > 0 else common_idx[-1]
                    ax_i.axvspan(s, e, color=RED, alpha=0.08)

        # Middle: MVRV + NUPL
        if has_mvrv or has_nupl:
            panel += 1
            if has_mvrv:
                axes[panel].plot(
                    onchain.loc[common_idx, "mvrv_zscore"], color=ORANGE, label="MVRV Z-Score", linewidth=1
                )
            if has_nupl:
                ax2 = axes[panel].twinx() if has_mvrv else axes[panel]
                ax2.plot(onchain.loc[common_idx, "nupl"], color=PURPLE, label="NUPL", linewidth=1)
                if has_mvrv:
                    ax2.set_ylabel("NUPL")
                    ax2.legend(loc="upper left")
            axes[panel].set_ylabel("MVRV Z-Score" if has_mvrv else "NUPL")
            axes[panel].legend(loc="upper right")
            axes[panel].grid(alpha=0.3)

        # Bottom: SOPR
        if has_sopr:
            panel += 1
            axes[panel].plot(onchain.loc[common_idx, "sopr"], color=GREEN, label="SOPR", linewidth=1)
            axes[panel].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
            axes[panel].set_ylabel("SOPR")
            axes[panel].legend()
            axes[panel].grid(alpha=0.3)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        save_fig(fig, "regime_overlay.png")
        regime_ok = True
    except Exception as exc:
        print(f"  WARNING: Skipping regime overlay — {exc}")
        regime_ok = False

    # ── Write Report ─────────────────────────────────────────────────────────────
    print("\nWriting diagnostic report...")

    dd_duration = (dd_trough_idx - dd_start_idx).days
    # Recovery: find when equity recovers to pre-drawdown peak after trough
    post_trough = equity.loc[dd_trough_idx:]
    recovery_mask = post_trough >= running_max.loc[dd_start_idx]
    dd_recovery_idx = recovery_mask[recovery_mask].index[0] if recovery_mask.any() else None
    dd_recovery_str = (
        f"{dd_recovery_idx.date()} ({(dd_recovery_idx - dd_trough_idx).days}d)"
        if dd_recovery_idx
        else "Not yet recovered"
    )

    # Correlation summary
    mean_corr = rolling_corr.mean()
    corr_during_dd = rolling_corr[drawdown < -0.05].mean() if (drawdown < -0.05).any() else float("nan")

    # Verdict logic
    max_dd_abs = abs(metrics["max_drawdown"])
    if max_dd_abs > 0.30:
        verdict = "OVER-LEVERED"
        verdict_detail = "Max drawdown exceeds 30%, indicating excessive leverage from IV sizing."
    elif not np.isnan(corr_during_dd) and corr_during_dd > 0.7:
        verdict = "CORRELATION BREAK"
        verdict_detail = "BTC-ETH correlation spiked during drawdowns, eliminating diversification benefit."
    elif metrics["sharpe"] < 0.5:
        verdict = "STRUCTURAL FAIL"
        verdict_detail = "OOS Sharpe below 0.5 indicates the trend-following edge has degraded structurally."
    else:
        verdict = "MIXED"
        verdict_detail = "Performance is mixed — some degradation but not a clear single-cause failure."

    # TV sensitivity table
    tv_table_rows = []
    for r in tv_results:
        tv_table_rows.append(
            f"| {r['name']} | {r['btc_tv']:.2f} | {r['eth_tv']:.3f} | "
            f"{r['sharpe']:.3f} | {r['max_drawdown']:.3f} | {r['total_return']:.3f} | {r['calmar']:.3f} |"
        )

    # Leg table
    leg_table_rows = []
    for name, m in leg_table.items():
        leg_table_rows.append(
            f"| {name} | {m['sharpe']:.3f} | {m['max_drawdown']:.3f} | "
            f"{m['total_return']:.3f} | {m['calmar']:.3f} | {m['sortino']:.3f} |"
        )

    report = f"""# OOS Diagnostic Report — Champion Portfolio

Generated: {pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")} UTC
OOS Period: {oos.index[0].date()} to {oos.index[-1].date()} ({len(oos)} bars, {(oos.index[-1] - oos.index[0]).days} days)

## Section 1: Corrected OOS Evaluation

| Metric | Value |
|--------|-------|
| Sharpe | {metrics["sharpe"]:.3f} |
| Sortino | {metrics["sortino"]:.3f} |
| Max Drawdown | {metrics["max_drawdown"]:.3f} |
| Total Return | {metrics["total_return"]:.3f} ({metrics["total_return"]:.1%}) |
| Calmar | {metrics["calmar"]:.3f} |
| DSR (n=1) | {metrics["dsr"]:.3f} |
| Win Rate | {metrics["win_rate"]:.3f} |
| Profit Factor | {metrics["profit_factor"]:.3f} |
| Observations | {metrics["n_observations"]} |

## Section 2: Drawdown Anatomy

| Detail | Value |
|--------|-------|
| Max Drawdown | {dd_trough_val:.3f} ({dd_trough_val:.1%}) |
| DD Start | {dd_start_idx.date()} |
| DD Trough | {dd_trough_idx.date()} |
| DD Duration to Trough | {dd_duration} days |
| Recovery | {dd_recovery_str} |
| Severe DDs (>15%) | {len(severe)} bars below -15% |

![Equity Curve](equity_curve.png)
![Underwater Chart](underwater.png)

## Section 3: Per-Leg Decomposition

| Leg | Sharpe | MaxDD | Return | Calmar | Sortino |
|-----|--------|-------|--------|--------|---------|
{chr(10).join(leg_table_rows)}

**Correlation Analysis:**
- Mean rolling 60-bar correlation: {mean_corr:.3f}
- Correlation during drawdowns (>5%): {corr_during_dd:.3f}

![Leg Decomposition](leg_decomposition.png)

## Section 4: Target Vol Sensitivity

| Config | BTC TV | ETH TV | Sharpe | MaxDD | Return | Calmar |
|--------|--------|--------|--------|-------|--------|--------|
{chr(10).join(tv_table_rows)}

![TV Sensitivity](tv_sensitivity.png)

## Section 5: Monthly Returns

![Monthly Returns](monthly_returns.png)

## Section 6: Regime Signals

{"Regime overlay generated — see regime_overlay.png for visual analysis." if regime_ok else "On-chain data unavailable — regime analysis skipped."}

{"![Regime Overlay](regime_overlay.png)" if regime_ok else ""}

## Verdict: {verdict}

{verdict_detail}

## Recommendation

{"Reduce target volatility parameters and re-evaluate. The Original IV sizing may be too aggressive for OOS conditions." if verdict == "OVER-LEVERED" else ""}{"Consider reducing ETH weight or adding a correlation regime filter to reduce exposure when BTC-ETH correlation spikes." if verdict == "CORRELATION BREAK" else ""}{"The trend-following approach may need fundamental rethinking — consider regime-conditional entry or alternative signal generation." if verdict == "STRUCTURAL FAIL" else ""}{"Investigate sub-period performance to identify whether degradation is continuous or concentrated in specific regimes. Consider reduced sizing as a first mitigation." if verdict == "MIXED" else ""}

---
*Report generated by run_diagnostics.py under OOS evaluation protocol.*
"""

    (OUT / "oos_diagnostic_report.md").write_text(report)
    print("  saved oos_diagnostic_report.md")

    # ── Summary ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("OOS Diagnostics Complete")
    print(f"{'=' * 60}")
    print(f"  Output dir: {OUT}")
    print(f"  Report:     {OUT / 'oos_diagnostic_report.md'}")
    print(f"  Sharpe:     {metrics['sharpe']:.3f}")
    print(f"  MaxDD:      {metrics['max_drawdown']:.3f}")
    print(f"  Return:     {metrics['total_return']:.1%}")
    print(f"  DSR:        {metrics['dsr']:.3f}")
    print(f"  Verdict:    {verdict}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if os.environ.get("SPARKY_OOS_ENABLED") != "1":
        print("ERROR: Set SPARKY_OOS_ENABLED=1 to run diagnostics")
        sys.exit(2)
    main()
