#!/usr/bin/env python3
"""Audit Contract 004 wandb runs with Deflated Sharpe Ratio analysis.

Pulls all contract_004 tagged runs from wandb, groups by step (sweep, regime,
ensemble, novel), computes DSR for each group, and generates an audit report.

Usage:
    python scripts/audit_contract_004.py
"""

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sparky.tracking.experiment import ExperimentTracker
from sparky.tracking.metrics import analytical_dsr, expected_max_sharpe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

STEPS = ["sweep", "regime", "ensemble", "novel"]
OUTPUT_DIR = Path("results")


def fetch_contract_004_runs():
    """Fetch all contract_004 runs from wandb, grouped by step tag."""
    tracker = ExperimentTracker(experiment_name="contract_004")

    # Fetch all runs with contract_004 tag
    runs = tracker._fetch_runs(filters={"tags": {"$in": ["contract_004"]}})

    grouped = defaultdict(list)
    for run in runs:
        tags = run.tags if hasattr(run, "tags") else []
        summary = run.summary or {}
        for step in STEPS:
            if step in tags:
                grouped[step].append({
                    "run_id": run.id,
                    "name": run.name,
                    "sharpe": summary.get("sharpe"),
                    "dsr": summary.get("dsr"),
                    "max_drawdown": summary.get("max_drawdown"),
                    "n_observations": summary.get("n_observations"),
                    "skewness": summary.get("skewness"),
                    "kurtosis": summary.get("kurtosis"),
                    "config": dict(run.config) if hasattr(run, "config") else {},
                    "tags": tags,
                })
                break
        else:
            grouped["untagged"].append({
                "run_id": run.id,
                "name": run.name,
                "sharpe": summary.get("sharpe"),
            })

    return grouped


def analyze_group(group_name, runs):
    """Analyze a group of runs: compute expected max Sharpe and DSR.

    Uses n_observations from run summaries for T when available (median of
    available values), falling back to 8760*5 (5 years of hourly data).
    """
    sharpes = [r["sharpe"] for r in runs if r["sharpe"] is not None]
    if not sharpes:
        return None

    n_trials = len(sharpes)
    best_sharpe = max(sharpes)
    best_run = max([r for r in runs if r["sharpe"] is not None], key=lambda r: r["sharpe"])

    # Dynamic T: use median n_observations from runs, fallback to 8760*5
    n_obs_values = [r.get("n_observations") for r in runs if r.get("n_observations") is not None]
    if n_obs_values:
        T = int(np.median(n_obs_values))
        T_source = f"median of {len(n_obs_values)} runs"
    else:
        T = 8760 * 5  # ~5 years of hourly data
        T_source = "default (8760*5)"

    exp_max_sr = expected_max_sharpe(n_trials, T)

    # Recompute DSR for best run using analytical_dsr (more reliable than stored value)
    best_skew = best_run.get("skewness")  # None if not logged
    best_kurt = best_run.get("kurtosis")  # None if not logged (raw, normal=3)
    best_T = best_run.get("n_observations") or T
    recomputed_dsr = analytical_dsr(
        sr=best_sharpe,
        skewness=best_skew,
        kurtosis=best_kurt,
        T=best_T,
        n_trials=n_trials,
    )
    dsr_source = "recomputed"
    if best_skew is None or best_kurt is None:
        dsr_source = "recomputed (Gaussian approx)"

    return {
        "group": group_name,
        "n_runs": len(runs),
        "n_with_sharpe": n_trials,
        "best_sharpe": best_sharpe,
        "best_run_name": best_run["name"],
        "best_run_id": best_run["run_id"],
        "best_dsr": recomputed_dsr,
        "best_dsr_source": dsr_source,
        "stored_dsr": best_run.get("dsr"),
        "best_max_dd": best_run.get("max_drawdown"),
        "expected_max_sharpe": exp_max_sr,
        "sharpe_vs_expected": best_sharpe - exp_max_sr,
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
        "all_sharpes": sharpes,
        "T": T,
        "T_source": T_source,
    }


def generate_report(analyses, total_runs):
    """Generate the audit report markdown."""
    lines = [
        "# Contract 004 Audit Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Total runs analyzed:** {total_runs}",
        "",
        "## Summary",
        "",
        "The Deflated Sharpe Ratio (DSR) corrects for multiple testing. A DSR > 0.95 means",
        "<5% probability the result is a statistical fluke. Expected Max Sharpe shows what",
        "the best Sharpe from pure noise would be given the number of trials.",
        "",
        "## Per-Step Analysis",
        "",
        "| Step | Runs | Best Sharpe | Expected Max (noise) | Best vs Expected | Best DSR | Best Run |",
        "|------|------|-------------|---------------------|-----------------|----------|----------|",
    ]

    for a in analyses:
        if a is None:
            continue
        dsr_str = f"{a['best_dsr']:.3f}" if a["best_dsr"] is not None else "N/A"
        lines.append(
            f"| {a['group']} | {a['n_with_sharpe']} | {a['best_sharpe']:.3f} | "
            f"{a['expected_max_sharpe']:.3f} | {a['sharpe_vs_expected']:+.3f} | "
            f"{dsr_str} | {a['best_run_name']} |"
        )

    lines.extend(["", "## Detailed Per-Step Breakdown", ""])

    for a in analyses:
        if a is None:
            continue
        lines.extend([
            f"### {a['group'].title()}",
            "",
            f"- **Runs with Sharpe data:** {a['n_with_sharpe']}",
            f"- **Best Sharpe:** {a['best_sharpe']:.3f} ({a['best_run_name']})",
            f"- **Expected Max Sharpe (from noise alone with {a['n_with_sharpe']} trials, T={a.get('T', 'N/A')} [{a.get('T_source', 'N/A')}]):** {a['expected_max_sharpe']:.3f}",
            f"- **Best DSR:** {a['best_dsr']:.3f}" if a.get("best_dsr") is not None else "- **Best DSR:** N/A",
            f"- **Best Max Drawdown:** {a['best_max_dd']:.1%}" if a.get("best_max_dd") is not None else "- **Best Max Drawdown:** N/A",
            f"- **Mean Sharpe:** {a['mean_sharpe']:.3f}",
            f"- **Median Sharpe:** {a['median_sharpe']:.3f}",
            f"- **Std Sharpe:** {a['std_sharpe']:.3f}",
            "",
        ])

    # Overall verdict
    all_sharpes = []
    for a in analyses:
        if a and a["all_sharpes"]:
            all_sharpes.extend(a["all_sharpes"])

    total_trials = len(all_sharpes)
    overall_exp_max = expected_max_sharpe(max(total_trials, 2), 8760 * 5) if total_trials > 0 else 0
    overall_best = max(all_sharpes) if all_sharpes else 0

    lines.extend([
        "## Overall Verdict",
        "",
        f"- **Total configs tested across all steps:** {total_trials}",
        f"- **Overall expected max Sharpe (noise with {total_trials} trials):** {overall_exp_max:.3f}",
        f"- **Actual best Sharpe:** {overall_best:.3f}",
        f"- **Best vs Expected:** {overall_best - overall_exp_max:+.3f}",
        "",
    ])

    if overall_best > overall_exp_max:
        lines.append("**Conclusion:** Best result exceeds noise threshold. Further DSR validation recommended.")
    else:
        lines.append("**Conclusion:** Best result does NOT convincingly exceed noise. Consider more diverse strategies.")

    lines.extend(["", "---", "", "*Generated by `scripts/audit_contract_004.py`*", ""])

    return "\n".join(lines)


def generate_histogram(analyses):
    """Generate Sharpe distribution histogram with dynamic grid sizing."""
    n_steps = len(STEPS)
    n_cols = 2
    n_rows = math.ceil(n_steps / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    fig.suptitle("Contract 004 — Sharpe Ratio Distributions by Step", fontsize=14)

    # Flatten axes for uniform iteration
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for idx, ax in enumerate(axes_flat):
        if idx >= n_steps:
            ax.set_visible(False)
            continue
        step = STEPS[idx]
        analysis = next((a for a in analyses if a and a["group"] == step), None)
        if analysis and analysis["all_sharpes"]:
            sharpes = analysis["all_sharpes"]
            ax.hist(sharpes, bins=min(20, max(5, len(sharpes) // 2)), edgecolor="black", alpha=0.7)
            ax.axvline(
                analysis["expected_max_sharpe"],
                color="red",
                linestyle="--",
                label=f"Expected max (noise): {analysis['expected_max_sharpe']:.2f}",
            )
            ax.axvline(1.062, color="green", linestyle="--", label="Donchian baseline: 1.062")
            ax.set_title(f"{step.title()} (n={len(sharpes)})")
            ax.set_xlabel("Sharpe Ratio")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
        else:
            ax.set_title(f"{step.title()} (no data)")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "sharpe_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Histogram saved to {output_path}")
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Contract 004 runs from wandb...")
    grouped = fetch_contract_004_runs()

    total_runs = sum(len(v) for v in grouped.values())
    logger.info(f"Found {total_runs} total runs")
    for step, runs in grouped.items():
        logger.info(f"  {step}: {len(runs)} runs")

    # Analyze each step
    analyses = []
    for step in STEPS:
        runs = grouped.get(step, [])
        if runs:
            analysis = analyze_group(step, runs)
            analyses.append(analysis)
            if analysis:
                logger.info(
                    f"  {step}: best={analysis['best_sharpe']:.3f}, "
                    f"expected_max={analysis['expected_max_sharpe']:.3f}"
                )
        else:
            logger.info(f"  {step}: no runs found")

    # Generate report
    report = generate_report(analyses, total_runs)
    # Intentionally named contract_005_audit.md — this is the Contract 005 audit
    # OF Contract 004 results. The workflow's done_when checks for this filename.
    report_path = OUTPUT_DIR / "contract_005_audit.md"
    report_path.write_text(report)
    logger.info(f"Report written to {report_path}")

    # Generate histogram
    generate_histogram(analyses)

    # Also save raw analysis as JSON for programmatic access
    json_data = {
        a["group"]: {k: v for k, v in a.items() if k != "all_sharpes"}
        for a in analyses
        if a
    }
    json_path = OUTPUT_DIR / "contract_004_dsr_analysis.json"
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    logger.info(f"JSON analysis saved to {json_path}")


if __name__ == "__main__":
    main()
