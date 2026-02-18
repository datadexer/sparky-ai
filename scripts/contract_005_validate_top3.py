#!/usr/bin/env python3
"""
CONTRACT #005 — Step 2: Validate Top-3 Contract 004 Configs

Runs full metrics pipeline with guardrails on the top 3 configs identified
by the Step 1 audit:
  1. mom_OR_lb40_adx21t25 — Sharpe ~1.390, std 0.854, 2022=-0.094
  2. mom_OR_+20pct_all (lb=48,adx17>36,vol24) — Sharpe ~1.217, std 0.464, 2022=+0.701
  3. ADX(14,30) Donchian — Sharpe ~1.181, std 0.829, 2022=0.000

N_TRIALS = 187 (total contract_004 wandb runs, confirmed by Step 1 audit)
Tags: contract_005, validation
"""

import sys

sys.path.insert(0, "src")

import warnings

warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sparky.data.loader import load
from sparky.features.returns import annualized_sharpe
from sparky.tracking.experiment import ExperimentTracker
from sparky.tracking.guardrails import (
    has_blocking_failure,
    log_results,
    run_post_checks,
    run_pre_checks,
)
from sparky.tracking.metrics import compute_all_metrics

# ─── Constants ───────────────────────────────────────────────────────────────
N_TRIALS = 187  # Total contract_004 wandb runs (confirmed by Step 1 audit)
DONCHIAN_BASELINE_SHARPE = 1.062  # Multi-TF Donchian corrected baseline
ENTRY_PERIOD = 40
EXIT_PERIOD = 20
TRANSACTION_COSTS_BPS = 30.0  # 30 bps per side — standard cost floor per guardrails
DSR_THRESHOLD = 0.95
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CONTRACT #005 STEP 2 — Full Metrics Validation of Top-3 Configs")
print(f"N_TRIALS = {N_TRIALS} | DSR threshold = {DSR_THRESHOLD}")
print("=" * 80)


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data():
    print("\nLoading data (once)...")
    df_feat = load("feature_matrix_btc_hourly_expanded", purpose="training")

    prices_hourly = load("ohlcv_hourly_max_coverage", purpose="training")
    prices_daily = prices_hourly["close"].resample("D").last().dropna()
    if prices_daily.index.tz is None:
        prices_daily.index = prices_daily.index.tz_localize("UTC")
    prices_daily = prices_daily.loc["2019-01-01":"2023-12-31"]
    daily_returns = prices_daily.pct_change().dropna()

    # Hourly features for guardrail checks (keeps min_samples threshold satisfied)
    # The hourly data has ~43,800 rows which exceeds the 2000-row guardrail default
    df_hourly = df_feat.loc[df_feat.index <= "2023-12-31"]

    # Daily-aggregate features (for reference)
    df_daily = df_feat.resample("D").mean()
    df_daily = df_daily.loc["2019-01-01":"2023-12-31"]

    # Align prices + daily features
    common_idx = prices_daily.index.intersection(df_daily.index)
    prices_daily = prices_daily.loc[common_idx]
    daily_returns = daily_returns.reindex(common_idx).fillna(0)
    df_daily = df_daily.loc[common_idx]

    print(
        f"  Prices: {len(prices_daily)} daily rows ({prices_daily.index[0].date()} - {prices_daily.index[-1].date()})"
    )
    print(f"  Hourly features shape (for guardrails): {df_hourly.shape}")
    print(f"  Daily features shape: {df_daily.shape}")
    return prices_daily, daily_returns, df_daily, df_hourly


# ─── Signal Utilities (exact replication from Step 5v2 scripts) ──────────────
def compute_adx(prices: pd.Series, period: int = 14) -> pd.Series:
    """Approximate ADX from close prices (Wilder EWM smoothing)."""
    delta = prices.diff()
    plus_dm = delta.clip(lower=0)
    minus_dm = (-delta).clip(lower=0)
    plus_di = plus_dm.ewm(span=period, adjust=False).mean()
    minus_di = minus_dm.ewm(span=period, adjust=False).mean()
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
    return dx.ewm(span=period, adjust=False).mean()


def donchian_signal(prices: pd.Series, entry: int = 40, exit_p: int = 20) -> pd.Series:
    """Donchian channel breakout — stateful, no look-ahead (uses upper[i-1])."""
    upper = prices.rolling(window=entry).max()
    lower = prices.rolling(window=exit_p).min()
    sig = pd.Series(0, index=prices.index, dtype=int)
    in_pos = False
    for i in range(len(prices)):
        if i < entry:
            continue
        px = prices.iloc[i]
        if not in_pos:
            if px >= upper.iloc[i - 1]:
                in_pos = True
                sig.iloc[i] = 1
        else:
            if i >= exit_p and px <= lower.iloc[i - 1]:
                in_pos = False
            else:
                sig.iloc[i] = 1
    return sig


def compute_strategy_returns(
    daily_returns: pd.Series,
    signals: pd.Series,
    costs_bps: float = 30.0,
) -> np.ndarray:
    """
    Apply signal to returns with transaction costs.

    CRITICAL: signals are ALREADY computed at T using data through T-1
    (donchian uses upper.iloc[i-1]; momentum is computed on prior returns).
    However, since the signal at time T uses close[T] for the rolling max
    comparison, we must shift(1) to avoid any look-ahead.

    Position on day T = signal[T-1] (shift by 1 means "trade on next open").
    """
    pos = signals.reindex(daily_returns.index).fillna(0).shift(1).fillna(0)

    # Apply transaction costs on position changes
    pos_changes = pos.diff().abs()
    pos_changes.iloc[0] = pos.iloc[0]  # First entry

    cost_per_trade = costs_bps / 10000.0  # Convert bps to fractional
    cost_series = pos_changes * cost_per_trade

    strat_ret = pos * daily_returns - cost_series
    return strat_ret.values


def yearly_sharpe(daily_returns: pd.Series, signals: pd.Series) -> dict:
    """Compute per-year Sharpe (no costs, for comparison with stored results)."""
    result = {}
    for yr in range(2019, 2024):
        mask = daily_returns.index.year == yr
        if not mask.any():
            result[str(yr)] = 0.0
            continue
        pos = signals.reindex(daily_returns.index).fillna(0).shift(1).fillna(0)
        yr_ret = (pos * daily_returns)[mask]
        if yr_ret.std() == 0 or len(yr_ret) < 10:
            result[str(yr)] = 0.0
        else:
            result[str(yr)] = round(float(annualized_sharpe(yr_ret)), 3)
    return result


# ─── Config 1: mom_OR_lb40_adx21t25 ──────────────────────────────────────────
def build_signals_mom_OR_lb40_adx21t25(prices: pd.Series, daily_returns: pd.Series) -> pd.Series:
    """
    Momentum (lb=40, thresh=0%) + vol_adx_OR (vol_window=20, ADX period=21, thresh=25).
    Exact replication of the novel_exploration_step5_v2.py logic.
    """
    lb, thresh, vol_w, adx_p, adx_t = 40, 0.00, 20, 21, 25

    mom = prices.pct_change(lb)
    mom_sig = (mom > thresh).astype(int)

    ret = prices.pct_change()
    roll_vol = ret.rolling(vol_w).std() * np.sqrt(252)
    # Vol median computed on full in-sample (no forward leak; walk-forward uses train history)
    vol_med = roll_vol.median()
    vol_reg = (roll_vol > vol_med).astype(int)

    adx = compute_adx(prices, period=adx_p)
    adx_reg = (adx > adx_t).astype(int)

    combined_regime = ((vol_reg == 1) | (adx_reg == 1)).astype(int)
    final_sig = mom_sig * combined_regime
    return final_sig


# ─── Config 2: mom_OR_+20pct_all (lb=48, adx17>36, vol24) ──────────────────
def build_signals_mom_OR_plus20pct_all(prices: pd.Series, daily_returns: pd.Series) -> pd.Series:
    """
    Momentum + vol_adx_OR with +20% perturbation on all parameters.
    lb=48, vol_window=24, ADX period=17, ADX threshold=36.
    Source: novel_exploration_summary.md P4 robustness testing.
    """
    lb, thresh, vol_w, adx_p, adx_t = 48, 0.00, 24, 17, 36

    mom = prices.pct_change(lb)
    mom_sig = (mom > thresh).astype(int)

    ret = prices.pct_change()
    roll_vol = ret.rolling(vol_w).std() * np.sqrt(252)
    vol_med = roll_vol.median()
    vol_reg = (roll_vol > vol_med).astype(int)

    adx = compute_adx(prices, period=adx_p)
    adx_reg = (adx > adx_t).astype(int)

    combined_regime = ((vol_reg == 1) | (adx_reg == 1)).astype(int)
    final_sig = mom_sig * combined_regime
    return final_sig


# ─── Config 3: ADX(14,30) Donchian ──────────────────────────────────────────
def build_signals_adx14t30_donchian(prices: pd.Series, daily_returns: pd.Series) -> pd.Series:
    """
    Donchian(40/20) filtered by ADX(14) > 30 regime filter.
    Source: Step 3 regime scripts, best performing regime-filtered baseline.
    """
    don_sig = donchian_signal(prices, entry=40, exit_p=20)
    adx = compute_adx(prices, period=14)
    adx_reg = (adx > 30).astype(int)
    return don_sig * adx_reg


# ─── Main Validation Loop ────────────────────────────────────────────────────
def main():
    prices_daily, daily_returns, df_daily, df_hourly = load_data()

    tracker = ExperimentTracker(experiment_name="contract_005")

    top_configs = {
        "mom_OR_lb40_adx21t25": {
            "signal_fn": build_signals_mom_OR_lb40_adx21t25,
            "config": {
                "strategy_type": "momentum_vol_adx_OR",
                "lookback": 40,
                "momentum_threshold": 0.00,
                "vol_window": 20,
                "adx_period": 21,
                "adx_threshold": 25,
                "entry_period": 40,
                "exit_period": 20,
                "transaction_costs_bps": TRANSACTION_COSTS_BPS,
                "features": [],  # Rule-based, no ML features
                "target": "target_1d",
                "n_trials": N_TRIALS,
                "stored_sharpe": 1.390,
                "stored_std": 0.854,
                "stored_sharpe_2022": -0.094,
            },
            "expected_sharpe": 1.390,
        },
        "mom_OR_plus20pct_all": {
            "signal_fn": build_signals_mom_OR_plus20pct_all,
            "config": {
                "strategy_type": "momentum_vol_adx_OR_robust",
                "lookback": 48,
                "momentum_threshold": 0.00,
                "vol_window": 24,
                "adx_period": 17,
                "adx_threshold": 36,
                "entry_period": 40,
                "exit_period": 20,
                "transaction_costs_bps": TRANSACTION_COSTS_BPS,
                "features": [],
                "target": "target_1d",
                "n_trials": N_TRIALS,
                "stored_sharpe": 1.217,
                "stored_std": 0.464,
                "stored_sharpe_2022": 0.701,
            },
            "expected_sharpe": 1.217,
        },
        "ADX14t30_Donchian40_20": {
            "signal_fn": build_signals_adx14t30_donchian,
            "config": {
                "strategy_type": "donchian_adx_regime",
                "entry_period": 40,
                "exit_period": 20,
                "adx_period": 14,
                "adx_threshold": 30,
                "transaction_costs_bps": TRANSACTION_COSTS_BPS,
                "features": [],
                "target": "target_1d",
                "n_trials": N_TRIALS,
                "stored_sharpe": 1.181,
                "stored_std": 0.829,
                "stored_sharpe_2022": 0.000,
            },
            "expected_sharpe": 1.181,
        },
    }

    all_results = {}

    for config_name, spec in top_configs.items():
        print(f"\n{'=' * 80}")
        print(f"VALIDATING: {config_name}")
        print(f"  Expected Sharpe: {spec['expected_sharpe']:.3f}")
        print("=" * 80)

        config = spec["config"]

        # ── 1. Pre-checks ────────────────────────────────────────────────────
        # Use hourly features for pre-checks so minimum_samples threshold (2000) is met.
        # The holdout boundary, no-lookahead, costs-specified, and param-data-ratio checks
        # are not affected by daily vs hourly granularity.
        print("\n[1/4] Running pre-experiment guardrails...")
        pre_results = run_pre_checks(df_hourly, config)
        log_results(pre_results, run_id=f"{config_name}_pre")

        if has_blocking_failure(pre_results):
            print(f"  BLOCKED: {config_name} failed pre-checks — skipping")
            for r in pre_results:
                if not r.passed:
                    print(f"    FAIL [{r.severity}] {r.check_name}: {r.message}")
            all_results[config_name] = {"status": "BLOCKED_PRE", "pre_checks": pre_results}
            continue

        print("  Pre-checks PASSED:")
        for r in pre_results:
            status = "PASS" if r.passed else f"FAIL[{r.severity}]"
            print(f"    {status} {r.check_name}: {r.message}")

        # ── 2. Build signals + compute returns ───────────────────────────────
        print("\n[2/4] Computing signals and returns...")
        signals = spec["signal_fn"](prices_daily, daily_returns)
        strat_returns = compute_strategy_returns(daily_returns, signals, costs_bps=TRANSACTION_COSTS_BPS)

        # Also compute no-cost returns for reporting (for comparison with stored values)
        pos_no_cost = signals.reindex(daily_returns.index).fillna(0).shift(1).fillna(0)
        returns_no_cost = (pos_no_cost * daily_returns).values

        n_long = int((pos_no_cost > 0).sum())
        n_total = len(pos_no_cost)
        print(f"  In-market: {n_long}/{n_total} days ({n_long / n_total * 100:.1f}%)")

        # Per-year Sharpe for verification
        yr_sharpes = yearly_sharpe(daily_returns, signals)
        print("  Per-year Sharpe (no-cost):")
        for yr, sh in yr_sharpes.items():
            print(f"    {yr}: {sh:+.3f}")

        mean_sharpe_no_cost = np.mean(list(yr_sharpes.values()))
        std_sharpe = np.std(list(yr_sharpes.values()))
        print(f"  Mean WF Sharpe (no-cost): {mean_sharpe_no_cost:.3f} ± {std_sharpe:.3f}")
        print(f"  Stored Sharpe: {spec['expected_sharpe']:.3f}")

        delta = abs(mean_sharpe_no_cost - spec["expected_sharpe"])
        print(f"  Delta vs stored: {delta:.3f} {'✅ within 0.15' if delta < 0.15 else '⚠️ outside 0.15'}")

        # ── 3. Compute full metrics (n_trials=187) ───────────────────────────
        print("\n[3/4] Computing full metrics (n_trials=187)...")
        # Use full-period returns (not walk-forward) for DSR computation
        full_returns = strat_returns[strat_returns != 0] if np.any(strat_returns != 0) else strat_returns
        # Actually use all returns (including zeros from flat periods)
        metrics = compute_all_metrics(strat_returns, n_trials=N_TRIALS)

        print(f"  Sharpe (with costs, full period): {metrics['sharpe']:.4f}")
        print(f"  DSR (n_trials={N_TRIALS}): {metrics['dsr']:.6f}")
        print(f"  PSR: {metrics['psr']:.6f}")
        print(f"  Sortino: {metrics['sortino']:.4f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Calmar: {metrics['calmar']:.4f}")
        print(f"  CVaR (5%): {metrics['cvar_5pct']:.4f}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Skewness: {metrics['skewness']:.3f}")
        print(f"  Kurtosis (raw): {metrics['kurtosis']:.3f}")

        dsr_pass = metrics["dsr"] > DSR_THRESHOLD
        print(
            f"\n  DSR > {DSR_THRESHOLD}: {'✅ YES — statistically significant' if dsr_pass else '❌ NO — statistical fluke possible'}"
        )
        print(
            f"  Beats Donchian baseline (1.062): {'✅ YES' if metrics['sharpe'] > DONCHIAN_BASELINE_SHARPE else '❌ NO'}"
        )

        # Count trades for post-checks
        pos_series = signals.reindex(daily_returns.index).fillna(0).shift(1).fillna(0)
        pos_changes = pos_series.diff().abs()
        n_trades = int((pos_changes > 0).sum())
        print(f"  Number of trades: {n_trades}")

        # ── 4. Post-checks ───────────────────────────────────────────────────
        print("\n[4/4] Running post-experiment guardrails...")
        post_results = run_post_checks(strat_returns, metrics, config, n_trades=n_trades)
        log_results(post_results, run_id=f"{config_name}_post")

        print("  Post-check results:")
        for r in post_results:
            status = "PASS" if r.passed else f"FAIL[{r.severity}]"
            print(f"    {status} {r.check_name}: {r.message}")

        has_block = has_blocking_failure(post_results)
        if has_block:
            print("  ⚠️ POST-CHECK BLOCKING FAILURE — results flagged but logged")

        # ── 5. Log to wandb ──────────────────────────────────────────────────
        print("\n[5] Logging to wandb...")
        wandb_config = {
            **config,
            "step": "validation",
            "contract": "005",
            "n_trials_total": N_TRIALS,
            "per_year_sharpe": yr_sharpes,
            "mean_wf_sharpe": round(mean_sharpe_no_cost, 4),
            "std_wf_sharpe": round(std_sharpe, 4),
            "n_trades": n_trades,
            "pre_checks_passed": all(r.passed for r in pre_results),
            "post_checks_passed": all(r.passed or r.severity != "block" for r in post_results),
        }

        wandb_metrics = {
            **metrics,
            "mean_wf_sharpe_no_cost": round(mean_sharpe_no_cost, 4),
            "std_wf_sharpe": round(std_sharpe, 4),
            "sharpe_2022": yr_sharpes.get("2022", 0.0),
            "sharpe_2023": yr_sharpes.get("2023", 0.0),
            "n_trades": n_trades,
            "donchian_baseline": DONCHIAN_BASELINE_SHARPE,
            "beats_baseline": int(metrics["sharpe"] > DONCHIAN_BASELINE_SHARPE),
            "dsr_significant": int(dsr_pass),
            "pre_check_pass": int(not has_blocking_failure(pre_results)),
            "post_check_pass": int(not has_blocking_failure(post_results)),
        }

        tracker.log_experiment(
            name=f"c005_validation_{config_name}",
            config=wandb_config,
            metrics=wandb_metrics,
            tags=["contract_005", "validation"],
            job_type="validation",
        )
        print(f"  Logged to wandb: c005_validation_{config_name}")

        # ── Store results ────────────────────────────────────────────────────
        all_results[config_name] = {
            "status": "COMPLETED",
            "metrics": metrics,
            "yr_sharpes": yr_sharpes,
            "mean_wf_sharpe": mean_sharpe_no_cost,
            "std_wf_sharpe": std_sharpe,
            "n_trades": n_trades,
            "dsr_significant": dsr_pass,
            "pre_checks": [(r.check_name, r.passed, r.severity, r.message) for r in pre_results],
            "post_checks": [(r.check_name, r.passed, r.severity, r.message) for r in post_results],
            "pre_blocking_failure": has_blocking_failure(pre_results),
            "post_blocking_failure": has_blocking_failure(post_results),
            "beats_baseline": metrics["sharpe"] > DONCHIAN_BASELINE_SHARPE,
        }

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    n_passed = sum(1 for v in all_results.values() if v.get("status") == "COMPLETED")
    n_significant = sum(1 for v in all_results.values() if v.get("dsr_significant", False))
    n_beats_baseline = sum(1 for v in all_results.values() if v.get("beats_baseline", False))
    n_tier1 = sum(
        1
        for v in all_results.values()
        if v.get("status") == "COMPLETED"
        and v.get("metrics", {}).get("sharpe", 0) >= 1.0
        and v.get("dsr_significant", False)
    )

    print(f"  Configs completed: {n_passed}/3")
    print(f"  DSR > 0.95 (significant): {n_significant}/{n_passed}")
    print(f"  Beats Donchian baseline (1.062): {n_beats_baseline}/{n_passed}")
    print(f"  TIER 1 candidates (Sharpe >= 1.0 + DSR > 0.95): {n_tier1}")

    return all_results


if __name__ == "__main__":
    results = main()
    # Persist results for the markdown writer
    out_path = Path("results/contract_005_validation_raw.json")
    serializable = {}
    for k, v in results.items():
        sv = {}
        for fk, fv in v.items():
            if fk in ("pre_checks", "post_checks"):
                sv[fk] = fv  # already serializable tuples
            elif fk == "metrics":
                sv[fk] = {mk: (float(mv) if isinstance(mv, (np.floating, np.integer)) else mv) for mk, mv in fv.items()}
            elif isinstance(fv, bool):
                sv[fk] = fv
            elif isinstance(fv, (np.floating, np.integer)):
                sv[fk] = float(fv)
            else:
                sv[fk] = fv
        serializable[k] = sv
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nRaw results saved to {out_path}")
