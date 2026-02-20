#!/usr/bin/env python3
"""Single-shot OOS evaluation of a champion portfolio.

Usage:
    SPARKY_OOS_ENABLED=1 .venv/bin/python scripts/oos_evaluate.py configs/oos/champion_btc82_eth83.yaml

Exit codes: 0=PASS, 1=FAIL, 2=ERROR
Output: markdown report to /tmp/sparky_oos_reports/ (or --output-dir)
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from sparky.data.loader import load
from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.tracking.metrics import compute_all_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_leg_returns(leg: dict, cost_bps: int) -> pd.Series:
    """Build net returns for one portfolio leg over the full IS+OOS range."""
    # Load IS data (from data/) and OOS data (from data/holdout/), merge
    is_df = load(leg["dataset"], purpose="analysis")
    oos_df = load(leg["dataset"], purpose="evaluation")
    df = pd.concat([is_df, oos_df])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    prices = df["close"]
    signals = donchian_channel_strategy(prices, leg["entry_period"], leg["exit_period"])

    # Shift combined signal+scale to match research code: position[T] = signal[T-1] * scale[T-1]
    if leg.get("sizing") == "inverse_vol":
        sp = leg["sizing_params"]
        rvol = np.log(prices / prices.shift(1)).rolling(sp["vol_window"]).std() * np.sqrt(1095)
        scale = (sp["target_vol"] / rvol).clip(0.1, 1.5).fillna(0.5)
        positions = (signals * scale).shift(1).fillna(0)
    else:
        positions = signals.shift(1).fillna(0)

    price_returns = prices.pct_change().fillna(0)

    # Gross strategy returns
    gross = positions * price_returns

    # Transaction costs on position changes
    cost_frac = cost_bps / 10_000
    trades = positions.diff().abs().fillna(0)
    costs = trades * cost_frac
    net = gross - costs
    return net


def build_portfolio_returns(config: dict) -> pd.Series:
    """Build weighted portfolio returns from all legs."""
    legs = config["portfolio"]["legs"]
    cost_bps = config["portfolio"]["cost_bps"]

    leg_returns = {}
    for leg in legs:
        key = f"{leg['asset']}_{leg['entry_period']}_{leg['exit_period']}"
        leg_returns[key] = (build_leg_returns(leg, cost_bps), leg["weight"])

    # Align on common index
    all_series = [s for s, _ in leg_returns.values()]
    common_idx = all_series[0].index
    for s in all_series[1:]:
        common_idx = common_idx.intersection(s.index)

    portfolio = pd.Series(0.0, index=common_idx)
    for s, w in leg_returns.values():
        portfolio += s.reindex(common_idx).fillna(0) * w

    return portfolio


def extract_oos_returns(returns: pd.Series, oos_start: str) -> pd.Series:
    oos_ts = pd.Timestamp(oos_start, tz="UTC")
    return returns[returns.index >= oos_ts]


def extract_is_returns(returns: pd.Series, oos_start: str) -> pd.Series:
    oos_ts = pd.Timestamp(oos_start, tz="UTC")
    return returns[returns.index < oos_ts]


def evaluate(oos_returns: pd.Series, config: dict) -> dict:
    ppy = config["portfolio"]["periods_per_year"]
    metrics = compute_all_metrics(oos_returns, n_trials=1, periods_per_year=ppy)

    thresholds = config["thresholds"]
    passed = True
    verdicts = []

    sharpe = metrics["sharpe"]
    max_dd = metrics["max_drawdown"]

    if sharpe < thresholds["sharpe_min"]:
        passed = False
        verdicts.append(f"FAIL: Sharpe {sharpe:.3f} < {thresholds['sharpe_min']}")
    else:
        verdicts.append(f"PASS: Sharpe {sharpe:.3f} >= {thresholds['sharpe_min']}")

    if max_dd < thresholds["max_drawdown_max"]:
        passed = False
        verdicts.append(f"FAIL: MaxDD {max_dd:.3f} < {thresholds['max_drawdown_max']}")
    else:
        verdicts.append(f"PASS: MaxDD {max_dd:.3f} >= {thresholds['max_drawdown_max']}")

    return {"passed": passed, "verdicts": verdicts, "metrics": metrics}


def write_report(
    config: dict,
    full_metrics: dict,
    is_metrics: dict,
    oos_metrics: dict,
    result: dict,
    report_dir: Path,
):
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"oos_evaluation_{ts}.md"

    legs_desc = []
    for leg in config["portfolio"]["legs"]:
        legs_desc.append(
            f"  - {leg['asset'].upper()} Don8h({leg['entry_period']},{leg['exit_period']}) weight={leg['weight']}"
        )

    om = oos_metrics
    fm = full_metrics
    im = is_metrics
    verdict = "PASS" if result["passed"] else "FAIL"

    oos_start = config["oos"]["start"]
    lines = [
        f"# OOS Evaluation Report — {verdict}",
        f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Portfolio",
        f"- Cost: {config['portfolio']['cost_bps']} bps/side",
        f"- PPY: {config['portfolio']['periods_per_year']}",
        *legs_desc,
        "",
        f"## OOS Results ({oos_start} to present)",
        f"- **Sharpe**: {om['sharpe']:.3f}",
        f"- **Sortino**: {om['sortino']:.3f}",
        f"- **MaxDD**: {om['max_drawdown']:.3f}",
        f"- **Calmar**: {om['calmar']:.3f}",
        f"- **Total Return**: {om['total_return']:.1%}",
        f"- **Win Rate**: {om['win_rate']:.1%}",
        f"- **Profit Factor**: {om['profit_factor']:.2f}",
        f"- **N Observations**: {om['n_observations']}",
        f"- **DSR (n_trials=1)**: {om['dsr']:.3f}",
        "",
        f"## In-Sample Results (to {oos_start})",
        f"- **Sharpe**: {im['sharpe']:.3f}",
        f"- **Sortino**: {im['sortino']:.3f}",
        f"- **MaxDD**: {im['max_drawdown']:.3f}",
        f"- **Total Return**: {im['total_return']:.1%}",
        f"- **N Observations**: {im['n_observations']}",
        "",
        "## Full-Period Results (IS + OOS)",
        f"- **Sharpe**: {fm['sharpe']:.3f}",
        f"- **Sortino**: {fm['sortino']:.3f}",
        f"- **MaxDD**: {fm['max_drawdown']:.3f}",
        f"- **Total Return**: {fm['total_return']:.1%}",
        f"- **N Observations**: {fm['n_observations']}",
        "",
        "## Verdict",
    ]
    for v in result["verdicts"]:
        lines.append(f"- {v}")
    lines.append("")
    lines.append(f"**Overall: {verdict}**")

    report_path.write_text("\n".join(lines) + "\n")
    print(f"Report written to {report_path}")
    return report_path


def main():
    if len(sys.argv) < 2:
        print("Usage: SPARKY_OOS_ENABLED=1 .venv/bin/python scripts/oos_evaluate.py <config.yaml>")
        sys.exit(2)

    # Ensure CWD is project root so loader's relative DATA_DIRS resolve correctly
    os.chdir(ROOT)

    config_path = sys.argv[1]
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(2)

    try:
        portfolio_returns = build_portfolio_returns(config)
    except PermissionError as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    oos_start = config["oos"]["start"]
    oos_returns = extract_oos_returns(portfolio_returns, oos_start)
    is_returns = extract_is_returns(portfolio_returns, oos_start)

    if len(oos_returns) == 0:
        print(f"ERROR: No OOS data found after {oos_start}")
        sys.exit(2)

    ppy = config["portfolio"]["periods_per_year"]
    oos_metrics = compute_all_metrics(oos_returns, n_trials=1, periods_per_year=ppy)
    is_metrics = compute_all_metrics(is_returns, n_trials=1, periods_per_year=ppy)
    full_metrics = compute_all_metrics(portfolio_returns, n_trials=1, periods_per_year=ppy)
    result = evaluate(oos_returns, config)

    # Write report to --output-dir if given, else /tmp (sparky-oos can't write to project)
    report_dir = Path("/tmp/sparky_oos_reports")  # noqa: S108
    for i, arg in enumerate(sys.argv):
        if arg == "--output-dir" and i + 1 < len(sys.argv):
            report_dir = Path(sys.argv[i + 1])
    write_report(config, full_metrics, is_metrics, oos_metrics, result, report_dir)

    for v in result["verdicts"]:
        print(v)

    if result["passed"]:
        print("\nOOS EVALUATION: PASS")
        sys.exit(0)
    else:
        print("\nOOS EVALUATION: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
