"""Report generation: plots + HTML reports for validation/investigation results.

PROTECTED FILE — research agents call these, not edit them.

Usage:
    from report_generator import generate_candidate_report, generate_project_summary
    generate_candidate_report("btc_don4h_160_25_iv", inv_results, val_results, "reports/project_001")
    generate_project_summary("project_001", all_results, "reports/project_001")
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Plot functions — each returns saved PNG path
# ---------------------------------------------------------------------------


def _savefig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def plot_cost_curve(plots_data, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bps = plots_data["cost_bps"]
    sharpes = plots_data["sharpe"]
    ax.plot(bps, sharpes, "o-", color="#2196F3", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    be = plots_data.get("breakeven")
    if be and be <= max(bps):
        ax.axvline(x=be, color="red", linestyle="--", alpha=0.7, label=f"Breakeven: {be} bps")
        ax.legend()
    ax.set_xlabel("Transaction Cost (bps per side)")
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title("Cost Sensitivity")
    ax.grid(True, alpha=0.3)
    return _savefig(fig, output_path)


def plot_bootstrap_distribution(plots_data, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    dist = plots_data["sharpe_dist"]
    ax.hist(dist, bins=80, color="#4CAF50", alpha=0.7, edgecolor="white")
    pcts = plots_data.get("percentiles", {})
    colors = {5: "red", 25: "orange", 50: "blue", 75: "orange", 95: "red"}
    for p, v in pcts.items():
        ax.axvline(x=v, color=colors.get(int(p), "gray"), linestyle="--", alpha=0.7, label=f"{p}th: {v:.2f}")
    ax.set_xlabel("Annualized Sharpe")
    ax.set_ylabel("Count")
    ax.set_title("Bootstrap Sharpe Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _savefig(fig, output_path)


def plot_walk_forward(plots_data, output_path):
    fig, axes = plt.subplots(1, len(plots_data), figsize=(5 * len(plots_data), 5), squeeze=False)
    for idx, (ws, sharpes) in enumerate(sorted(plots_data.items())):
        ax = axes[0][idx]
        colors = ["#4CAF50" if s > 0 else "#F44336" for s in sharpes]
        ax.bar(range(len(sharpes)), sharpes, color=colors)
        ax.axhline(y=0, color="gray", linestyle="--")
        ax.set_title(f"{ws}-day windows")
        ax.set_xlabel("Window #")
        ax.set_ylabel("Sharpe")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Walk-Forward Analysis", fontsize=14)
    fig.tight_layout()
    return _savefig(fig, output_path)


def plot_subsample_degradation(plots_data, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    rates = plots_data["drop_rates"]
    means = plots_data["means"]
    stds = plots_data["stds"]
    ax.errorbar(rates, means, yerr=stds, fmt="o-", color="#9C27B0", capsize=5, linewidth=2)
    ax.set_xlabel("Drop Rate")
    ax.set_ylabel("Mean Sharpe")
    ax.set_title("Subsample Stability")
    ax.grid(True, alpha=0.3)
    return _savefig(fig, output_path)


def plot_cpcv_distribution(plots_data, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    dist = plots_data["sharpe_dist"]
    ax.hist(dist, bins=max(5, len(dist) // 2), color="#FF9800", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    pbo = plots_data.get("pbo", 0)
    ax.set_title(f"CPCV Path Sharpe Distribution (PBO={pbo:.2f})")
    ax.set_xlabel("Path Sharpe")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    return _savefig(fig, output_path)


def plot_rolling_sharpe(plots_data, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    rs = plots_data["rolling_sharpe"]
    dates = plots_data["dates"]
    n = len(rs)
    # Subsample for display if too many points
    step = max(1, n // 500)
    x = range(0, n, step)
    y = [rs[i] for i in x]
    ax.plot(x, y, color="#2196F3", linewidth=0.8)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.fill_between(x, y, 0, where=[v < 0 for v in y], alpha=0.2, color="red")
    ax.set_title("Rolling Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    # Set a few date labels
    if len(dates) > 4:
        ticks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([dates[t] if t < len(dates) else "" for t in ticks], rotation=30, fontsize=8)
    ax.grid(True, alpha=0.3)
    return _savefig(fig, output_path)


def plot_sub_period(plots_data, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    periods = plots_data["periods"]
    sharpes = [s if s is not None else 0 for s in plots_data["sharpes"]]
    colors = ["#4CAF50" if s > 0 else "#F44336" for s in sharpes]
    ax.bar(periods, sharpes, color=colors)
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_title("Sub-Period Sharpe Ratios")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    return _savefig(fig, output_path)


def plot_tv_frontier(plots_data, output_path):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    tvs = plots_data["target_vols"]
    sharpes = plots_data["sharpes"]
    dds = [abs(d) for d in plots_data["max_dds"]]
    ax1.plot(tvs, sharpes, "o-", color="#2196F3", linewidth=2, label="Sharpe")
    ax1.set_xlabel("Target Volatility")
    ax1.set_ylabel("Sharpe", color="#2196F3")
    ax2 = ax1.twinx()
    ax2.plot(tvs, dds, "s--", color="#F44336", linewidth=2, label="|MaxDD|")
    ax2.set_ylabel("|MaxDD|", color="#F44336")
    ax1.set_title("Target Vol Frontier")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.grid(True, alpha=0.3)
    return _savefig(fig, output_path)


def plot_trade_profile(plots_data, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    holds = plots_data["holding_periods"]
    rets = plots_data["trade_returns"]
    ax1.hist(holds, bins=30, color="#607D8B", alpha=0.7, edgecolor="white")
    ax1.set_title("Holding Period Distribution")
    ax1.set_xlabel("Periods")
    ax1.set_ylabel("Count")
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in rets]
    ax2.bar(range(len(rets)), sorted(rets), color=colors, width=1.0)
    ax2.set_title("Trade Returns (sorted)")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Return")
    ax2.axhline(y=0, color="gray", linestyle="--")
    fig.tight_layout()
    return _savefig(fig, output_path)


def plot_regime_performance(plots_data, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    regimes = plots_data["regime_metrics"]
    names = list(regimes.keys())
    sharpes = [regimes[n].get("sharpe") or 0 for n in names]
    colors = {"bull": "#4CAF50", "bear": "#F44336", "sideways": "#FF9800"}
    ax.bar(names, sharpes, color=[colors.get(n, "#607D8B") for n in names])
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_title("Performance by Market Regime")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    return _savefig(fig, output_path)


def plot_monthly_heatmap(plots_data, output_path):
    heatmap = plots_data["monthly_heatmap"]
    years = sorted(heatmap.keys())
    months = list(range(1, 13))
    data = np.full((len(years), 12), np.nan)
    for i, yr in enumerate(years):
        for mo in months:
            if mo in heatmap.get(yr, {}):
                data[i, mo - 1] = heatmap[yr][mo]

    fig, ax = plt.subplots(figsize=(12, max(3, len(years) * 0.6)))
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    ax.set_title("Monthly Returns (%)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    # Add text annotations
    for i in range(len(years)):
        for j in range(12):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=7)
    return _savefig(fig, output_path)


def plot_drawdown_series(plots_data, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    dd = plots_data["dd_series"]
    dates = plots_data["index"]
    n = len(dd)
    step = max(1, n // 500)
    x = range(0, n, step)
    y = [dd[i] * 100 for i in x]
    ax.fill_between(x, y, 0, color="#F44336", alpha=0.4)
    ax.plot(x, y, color="#F44336", linewidth=0.5)
    max_dd = plots_data.get("max_dd", min(dd))
    ax.set_title(f"Drawdown (Max: {max_dd * 100:.1f}%)")
    ax.set_ylabel("Drawdown (%)")
    if len(dates) > 4:
        ticks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([dates[t] if t < len(dates) else "" for t in ticks], rotation=30, fontsize=8)
    ax.grid(True, alpha=0.3)
    return _savefig(fig, output_path)


def plot_correlation_rolling(plots_data, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    corr = plots_data["rolling_corr"]
    dates = plots_data["dates"]
    n = len(corr)
    step = max(1, n // 500)
    x = range(0, n, step)
    y = [corr[i] for i in x]
    ax.plot(x, y, color="#9C27B0", linewidth=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.3, label="High corr threshold")
    ax.set_title("Rolling Correlation with Buy-and-Hold")
    ax.set_ylabel("Correlation")
    ax.legend()
    if len(dates) > 4:
        ticks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([dates[t] if t < len(dates) else "" for t in ticks], rotation=30, fontsize=8)
    ax.grid(True, alpha=0.3)
    return _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Plot dispatcher
# ---------------------------------------------------------------------------

_PLOT_MAP = {
    "stress_test": plot_cost_curve,
    "bootstrap_sharpe": plot_bootstrap_distribution,
    "walk_forward_multi": plot_walk_forward,
    "subsample_stability": plot_subsample_degradation,
    "cpcv_validate": plot_cpcv_distribution,
    "rolling_stability": plot_rolling_sharpe,
    "sub_period_analysis": plot_sub_period,
    "target_vol_frontier": plot_tv_frontier,
    "trade_profile": plot_trade_profile,
    "regime_decomposition": plot_regime_performance,
    "drawdown_analysis": plot_drawdown_series,
    "correlation_stability": plot_correlation_rolling,
}


# ---------------------------------------------------------------------------
# HTML Report Generation
# ---------------------------------------------------------------------------

_CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }
h1, h2, h3 { color: #1a1a2e; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; color: white; }
.badge-pass { background: #4CAF50; }
.badge-fail { background: #F44336; }
.badge-conditional { background: #FF9800; }
.badge-error { background: #9E9E9E; }
.badge-soft { background: #FF9800; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
th { background: #f5f5f5; }
tr:nth-child(even) { background: #fafafa; }
.metric { font-size: 1.1em; font-weight: bold; }
img { max-width: 100%; margin: 10px 0; border: 1px solid #eee; border-radius: 4px; }
.section { background: white; padding: 20px; margin: 16px 0; border-radius: 8px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.green { color: #4CAF50; } .red { color: #F44336; } .orange { color: #FF9800; }
</style>
"""

_VERDICT_BADGE = {
    "pass": '<span class="badge badge-pass">PASS</span>',
    "hard_fail": '<span class="badge badge-fail">HARD FAIL</span>',
    "soft_fail": '<span class="badge badge-soft">SOFT FAIL</span>',
    "error": '<span class="badge badge-error">ERROR</span>',
    "skipped": '<span class="badge badge-error">SKIPPED</span>',
}


def generate_candidate_report(candidate_id, investigation_results, validation_results, output_dir):
    """Generate full HTML report for a single candidate."""
    output_dir = Path(output_dir) / "candidates" / candidate_id
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_paths = {}
    if validation_results and validation_results.get("tests"):
        for test_name, test_result in validation_results["tests"].items():
            pd_data = test_result.get("plots_data")
            if pd_data and test_name in _PLOT_MAP:
                try:
                    p = _PLOT_MAP[test_name](pd_data, plots_dir / f"{test_name}.png")
                    plot_paths[test_name] = p
                except Exception:
                    pass

    if investigation_results and investigation_results.get("results"):
        for test_name, test_result in investigation_results["results"].items():
            pd_data = test_result.get("plots_data") if isinstance(test_result, dict) else None
            if pd_data and test_name in _PLOT_MAP:
                try:
                    p = _PLOT_MAP[test_name](pd_data, plots_dir / f"{test_name}.png")
                    plot_paths[test_name] = p
                except Exception:
                    pass
            # Monthly heatmap from regime_decomposition
            if test_name == "regime_decomposition" and pd_data and "monthly_heatmap" in pd_data:
                try:
                    p = plot_monthly_heatmap(pd_data, plots_dir / "monthly_heatmap.png")
                    plot_paths["monthly_heatmap"] = p
                except Exception:
                    pass

    # Build HTML
    overall = validation_results.get("overall_verdict", "UNKNOWN") if validation_results else "N/A"
    badge_class = {"PASS": "pass", "FAIL": "fail", "CONDITIONAL": "conditional"}.get(overall, "error")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{candidate_id} — Validation Report</title>{_CSS}</head>
<body>
<h1>{candidate_id}</h1>
<p>Overall: <span class="badge badge-{badge_class}">{overall}</span></p>
"""

    # Investigation section
    if investigation_results and investigation_results.get("results"):
        html += '<div class="section"><h2>Investigation</h2>\n'
        inv = investigation_results["results"]

        if "trade_profile" in inv and inv["trade_profile"].get("status") == "ok":
            tp = inv["trade_profile"]
            html += f"""<h3>Trade Profile</h3>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>N Trades</td><td>{tp.get("n_trades", "N/A")}</td></tr>
<tr><td>Win Rate</td><td>{tp.get("win_rate", 0):.1%}</td></tr>
<tr><td>Avg Win</td><td>{tp.get("avg_win", 0):.4f}</td></tr>
<tr><td>Avg Loss</td><td>{tp.get("avg_loss", 0):.4f}</td></tr>
<tr><td>Profit Factor</td><td>{tp.get("profit_factor", 0):.2f}</td></tr>
<tr><td>Median Hold</td><td>{tp.get("median_holding_periods", 0):.0f} periods</td></tr>
</table>\n"""
            if "trade_profile" in plot_paths:
                html += '<img src="plots/trade_profile.png" alt="Trade Profile">\n'

        if "edge_attribution" in inv and inv["edge_attribution"].get("status") == "ok":
            ea = inv["edge_attribution"]
            html += f"""<h3>Edge Attribution</h3>
<table><tr><th>Variant</th><th>Sharpe</th><th>Edge</th></tr>
<tr><td>Full Strategy</td><td class="metric">{ea.get("full_sharpe", 0):.3f}</td><td>—</td></tr>
<tr><td>Random Entry</td><td>{ea.get("random_entry_sharpe", 0):.3f}</td><td>Signal: {ea.get("signal_edge", 0):+.3f}</td></tr>
<tr><td>Flat Sizing</td><td>{ea.get("flat_sizing_sharpe", 0):.3f}</td><td>Sizing: {ea.get("sizing_edge", 0):+.3f}</td></tr>
</table>\n"""

        if "regime_decomposition" in inv and inv["regime_decomposition"].get("status") == "ok":
            rd = inv["regime_decomposition"]
            html += "<h3>Regime Analysis</h3>\n"
            if "regime_decomposition" in plot_paths:
                html += '<img src="plots/regime_decomposition.png" alt="Regime Performance">\n'
            if "monthly_heatmap" in plot_paths:
                html += '<img src="plots/monthly_heatmap.png" alt="Monthly Heatmap">\n'
            if rd.get("crisis_metrics"):
                html += "<h4>Crisis Events</h4><table><tr><th>Event</th><th>Return</th></tr>\n"
                for event, m in rd["crisis_metrics"].items():
                    html += f"<tr><td>{event}</td><td>{m['total_return']:.1%}</td></tr>\n"
                html += "</table>\n"
        html += "</div>\n"

    # Validation section
    if validation_results and validation_results.get("tests"):
        html += '<div class="section"><h2>Validation Battery</h2>\n'
        html += "<table><tr><th>Test</th><th>Verdict</th><th>Key Metric</th></tr>\n"
        for name, result in validation_results["tests"].items():
            verdict = result.get("verdict", "error")
            badge = _VERDICT_BADGE.get(verdict, verdict)
            key_metric = _extract_key_metric(name, result)
            html += f"<tr><td>{name}</td><td>{badge}</td><td>{key_metric}</td></tr>\n"
        html += "</table>\n"

        for name in validation_results["tests"]:
            if name in plot_paths:
                html += f'<h3>{name}</h3><img src="plots/{name}.png" alt="{name}">\n'
        html += "</div>\n"

    html += "</body></html>"

    report_path = output_dir / "report.html"
    report_path.write_text(html)

    # Save raw results as JSON
    raw = {"investigation": investigation_results, "validation": validation_results}
    (output_dir / "results.json").write_text(json.dumps(raw, indent=2, default=str))

    return str(report_path)


def _extract_key_metric(test_name, result):
    """Extract the most important metric from a test result for the summary table."""
    if result.get("status") in ("error", "skipped"):
        return result.get("error", "skipped")
    extractors = {
        "stress_test": lambda r: f"Breakeven: {r.get('breakeven_bps', '?')} bps",
        "bootstrap_sharpe": lambda r: (
            f"5th pct: {r.get('percentiles', {}).get(5, '?'):.2f}"
            if isinstance(r.get("percentiles", {}).get(5), (int, float))
            else "N/A"
        ),
        "walk_forward_multi": lambda r: "; ".join(
            f"{k}d: {v.get('frac_positive', 0):.0%}" for k, v in r.get("results", {}).items()
        ),
        "subsample_stability": lambda r: (
            f"50% drop: Sharpe {r.get('results', {}).get(0.5, {}).get('mean', '?'):.2f}"
            if isinstance(r.get("results", {}).get(0.5, {}).get("mean"), (int, float))
            else "N/A"
        ),
        "cpcv_validate": lambda r: f"PBO: {r.get('pbo', '?'):.2f}",
        "multi_seed_test": lambda r: f"std: {r.get('std', 0):.4f}",
        "tail_risk_analysis": lambda r: (
            f"CVaR(5%): {r.get('cvar', {}).get(0.05, '?'):.4f}"
            if isinstance(r.get("cvar", {}).get(0.05), (int, float))
            else "N/A"
        ),
        "drawdown_analysis": lambda r: f"MaxDD: {r.get('max_drawdown', 0):.1%}",
        "rolling_stability": lambda r: f"{len(r.get('flagged_periods', []))} flagged periods",
        "sub_period_analysis": lambda r: _sub_period_summary(r),
        "correlation_stability": lambda r: f"Mean: {r.get('mean_corr', 0):.2f}",
        "target_vol_frontier": lambda r: "informational",
    }
    fn = extractors.get(test_name)
    if fn:
        try:
            return fn(result)
        except Exception:
            return "—"
    return "—"


def _sub_period_summary(r):
    results = r.get("results", {})
    parts = []
    for label, m in results.items():
        if label == "full":
            continue
        s = m.get("sharpe")
        if s is not None:
            parts.append(f"{label}: {s:.2f}")
    return "; ".join(parts[:4]) if parts else "N/A"


def generate_project_summary(project_id, all_candidate_results, output_dir):
    """Generate project-level HTML summary with leaderboard."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{project_id} — Project Summary</title>{_CSS}</head>
<body>
<h1>Project: {project_id}</h1>
<h2>Candidate Leaderboard</h2>
<table>
<tr><th>Candidate</th><th>Sharpe@30</th><th>Sharpe@50</th><th>DSR</th>
<th>PBO</th><th>Bootstrap 5th</th><th>MaxDD</th><th>Verdict</th></tr>
"""

    rows = []
    for cid, data in all_candidate_results.items():
        val = data.get("validation", {})
        inv = data.get("investigation", {})
        tests = val.get("tests", {})

        sharpe_30 = _get_nested(tests, "stress_test", "results", 30, "sharpe")
        sharpe_50 = _get_nested(tests, "stress_test", "results", 50, "sharpe")
        dsr = _get_nested(data, "dsr")
        pbo = _get_nested(tests, "cpcv_validate", "pbo")
        boot5 = _get_nested(tests, "bootstrap_sharpe", "percentiles", 5)
        maxdd = _get_nested(tests, "drawdown_analysis", "max_drawdown")
        verdict = val.get("overall_verdict", "N/A")

        rows.append((cid, sharpe_30, sharpe_50, dsr, pbo, boot5, maxdd, verdict))

    # Sort by Sharpe@30 descending
    rows.sort(key=lambda r: r[1] if isinstance(r[1], (int, float)) else -999, reverse=True)

    for cid, s30, s50, dsr, pbo, b5, mdd, verdict in rows:
        badge_class = {"PASS": "pass", "FAIL": "fail", "CONDITIONAL": "conditional"}.get(verdict, "error")
        html += f"""<tr>
<td><a href="candidates/{cid}/report.html">{cid}</a></td>
<td>{_fmt(s30)}</td><td>{_fmt(s50)}</td><td>{_fmt(dsr)}</td>
<td>{_fmt(pbo)}</td><td>{_fmt(b5)}</td><td>{_fmt(mdd, pct=True)}</td>
<td><span class="badge badge-{badge_class}">{verdict}</span></td>
</tr>\n"""

    html += """</table>
</div></body></html>"""

    summary_path = output_dir / "index.html"
    summary_path.write_text(html)
    return str(summary_path)


def _get_nested(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
        if d is None:
            return None
    return d


def _fmt(v, pct=False):
    if v is None:
        return "—"
    if pct:
        return f"{v:.1%}"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)
