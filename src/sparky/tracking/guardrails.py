"""Pre/post experiment guardrails for Sparky AI.

Provides automated checks that run before and after experiments to
ensure data integrity, prevent common mistakes, and flag suspicious
results. Each check returns a GuardrailResult with pass/fail status
and severity level.

Usage:
    from sparky.tracking.guardrails import run_pre_checks, run_post_checks, has_blocking_failure

    # Before training
    pre_results = run_pre_checks(data, config)
    if has_blocking_failure(pre_results):
        raise RuntimeError("Pre-experiment checks failed")

    # After backtest
    post_results = run_post_checks(returns, metrics, config)
    log_results(pre_results + post_results, run_id="my_run")
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

logger = logging.getLogger(__name__)

__all__ = [
    "GuardrailResult",
    "run_pre_checks",
    "run_post_checks",
    "has_blocking_failure",
    "log_results",
]


_VALID_SEVERITIES = {"block", "warn", "info"}


@dataclass
class GuardrailResult:
    """Result of a single guardrail check."""

    passed: bool
    check_name: str
    message: str
    severity: str  # "block" | "warn" | "info"

    def __post_init__(self):
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(f"Invalid severity {self.severity!r}, must be one of {_VALID_SEVERITIES}")


# === PRE-EXPERIMENT CHECKS ===


def check_holdout_boundary(data: pd.DataFrame, asset: str = "btc") -> GuardrailResult:
    """BLOCK: Verify data does not extend past holdout boundary.

    Uses the same HoldoutGuard as the data loader to check that the data
    doesn't include any holdout period data. Fails closed (block) if the
    index is not a DatetimeIndex — cannot verify holdout safety.
    """
    from sparky.oversight.holdout_guard import HoldoutGuard

    guard = HoldoutGuard()
    max_date = guard.get_max_training_date(asset)

    if not isinstance(data.index, pd.DatetimeIndex):
        return GuardrailResult(
            passed=False,
            check_name="holdout_boundary",
            message="Data index is not DatetimeIndex — cannot verify holdout boundary",
            severity="block",
        )

    data_max = data.index.max()

    # Normalize both timestamps to UTC for safe comparison
    if data_max.tzinfo is None:
        data_max = data_max.tz_localize("UTC")
    else:
        data_max = data_max.tz_convert("UTC")

    if hasattr(max_date, "tzinfo") and max_date.tzinfo is None:
        max_date = max_date.tz_localize("UTC")
    elif hasattr(max_date, "tz_convert"):
        max_date = max_date.tz_convert("UTC")

    if data_max > max_date:
        return GuardrailResult(
            passed=False,
            check_name="holdout_boundary",
            message=f"Data extends to {data_max}, past holdout boundary {max_date}",
            severity="block",
        )
    return GuardrailResult(
        passed=True,
        check_name="holdout_boundary",
        message=f"Data within boundary (max={data_max}, limit={max_date})",
        severity="block",
    )


def check_minimum_samples(data: pd.DataFrame, min_samples: int = 2000) -> GuardrailResult:
    """BLOCK: Ensure enough data for meaningful statistical analysis."""
    n = len(data)
    if n < min_samples:
        return GuardrailResult(
            passed=False,
            check_name="minimum_samples",
            message=f"Only {n} samples, need at least {min_samples}",
            severity="block",
        )
    return GuardrailResult(
        passed=True,
        check_name="minimum_samples",
        message=f"{n} samples (min={min_samples})",
        severity="block",
    )


def check_no_lookahead(data: pd.DataFrame, config: dict) -> GuardrailResult:
    """BLOCK: Verify target column is not in feature list."""
    features = config.get("features", [])
    target = config.get("target", "target_1h")

    if target in features:
        return GuardrailResult(
            passed=False,
            check_name="no_lookahead",
            message=f"Target '{target}' found in feature list — look-ahead bias!",
            severity="block",
        )
    return GuardrailResult(
        passed=True,
        check_name="no_lookahead",
        message=f"Target '{target}' not in {len(features)} features",
        severity="block",
    )


def check_costs_specified(config: dict, min_costs_bps: float = 30.0) -> GuardrailResult:
    """BLOCK: Ensure transaction costs are specified at >= 30 bps per side.

    Standard: 30 bps (Coinbase limit / DEX L2). Stress test: 50 bps (market orders).
    """
    costs = config.get("transaction_costs_bps", None)
    if costs is None:
        return GuardrailResult(
            passed=False,
            check_name="costs_specified",
            message="No transaction_costs_bps in config — must specify costs (standard: 30 bps, stress: 50 bps)",
            severity="block",
        )
    if costs < min_costs_bps:
        return GuardrailResult(
            passed=False,
            check_name="costs_specified",
            message=f"Transaction costs {costs} bps below minimum {min_costs_bps} bps. Standard is 30 bps per side.",
            severity="block",
        )
    return GuardrailResult(
        passed=True,
        check_name="costs_specified",
        message=f"Transaction costs: {costs} bps",
        severity="block",
    )


def check_param_data_ratio(config: dict, data: pd.DataFrame, max_ratio: float = 0.1) -> GuardrailResult:
    """WARN: Check that parameter count doesn't exceed data ratio threshold.

    A high parameter-to-data ratio suggests overfitting risk.
    """
    n_params = len(config.get("features", [])) + len(
        [
            k
            for k in config
            if k not in ("features", "target", "transaction_costs_bps") and isinstance(config[k], (int, float))
        ]
    )
    n_samples = len(data)
    ratio = n_params / n_samples if n_samples > 0 else float("inf")

    if ratio > max_ratio:
        return GuardrailResult(
            passed=False,
            check_name="param_data_ratio",
            message=f"Param/data ratio {ratio:.4f} exceeds {max_ratio} ({n_params} params, {n_samples} samples)",
            severity="warn",
        )
    return GuardrailResult(
        passed=True,
        check_name="param_data_ratio",
        message=f"Param/data ratio {ratio:.4f} (limit={max_ratio})",
        severity="warn",
    )


# === POST-EXPERIMENT CHECKS ===


def check_sharpe_sanity(metrics: dict, max_sharpe: float = 4.0) -> GuardrailResult:
    """BLOCK: Flag suspiciously high Sharpe ratios."""
    sharpe = metrics.get("sharpe", 0)
    if abs(sharpe) > max_sharpe:
        return GuardrailResult(
            passed=False,
            check_name="sharpe_sanity",
            message=f"Sharpe {sharpe:.3f} exceeds sanity limit {max_sharpe} — likely a bug",
            severity="block",
        )
    return GuardrailResult(
        passed=True,
        check_name="sharpe_sanity",
        message=f"Sharpe {sharpe:.3f} within sanity bounds",
        severity="block",
    )


def check_minimum_trades(
    returns: np.ndarray,
    config: dict,
    min_trades: int = 30,
    n_trades: Optional[int] = None,
) -> GuardrailResult:
    """WARN: Ensure enough trades for statistical significance.

    Trade count priority: (1) explicit n_trades arg, (2) config["n_trades"],
    (3) sign-change heuristic fallback from returns array.
    """
    returns = np.asarray(returns)

    # Priority: (1) explicit arg, (2) config, (3) heuristic
    if n_trades is not None:
        trade_count = n_trades
        source = "explicit"
    elif "n_trades" in config:
        trade_count = config["n_trades"]
        source = "config"
    else:
        # Estimate trades from sign changes
        signs = np.sign(returns[returns != 0]) if np.any(returns != 0) else np.array([])
        if len(signs) > 1:
            trade_count = int(np.sum(np.diff(signs) != 0)) + 1
        else:
            trade_count = len(signs)
        source = "heuristic"

    if trade_count < min_trades:
        return GuardrailResult(
            passed=False,
            check_name="minimum_trades",
            message=f"Only ~{trade_count} trades detected ({source}), need at least {min_trades}",
            severity="warn",
        )
    return GuardrailResult(
        passed=True,
        check_name="minimum_trades",
        message=f"~{trade_count} trades detected ({source}, min={min_trades})",
        severity="warn",
    )


def check_dsr_threshold(metrics: dict, threshold: float = 0.80) -> GuardrailResult:
    """INFO: Report DSR status relative to threshold."""
    dsr = metrics.get("dsr", None)
    if dsr is None:
        return GuardrailResult(
            passed=False,
            check_name="dsr_threshold",
            message="DSR not computed — add n_trials to compute_all_metrics()",
            severity="info",
        )
    if dsr < threshold:
        return GuardrailResult(
            passed=False,
            check_name="dsr_threshold",
            message=f"DSR {dsr:.3f} below {threshold} — may be statistical fluke",
            severity="info",
        )
    return GuardrailResult(
        passed=True,
        check_name="dsr_threshold",
        message=f"DSR {dsr:.3f} above {threshold}",
        severity="info",
    )


def check_max_drawdown(metrics: dict, max_dd: float = -0.40) -> GuardrailResult:
    """WARN: Flag excessive drawdowns."""
    dd = metrics.get("max_drawdown", 0)
    if dd < max_dd:
        return GuardrailResult(
            passed=False,
            check_name="max_drawdown",
            message=f"Max drawdown {dd:.1%} exceeds limit {max_dd:.1%}",
            severity="warn",
        )
    return GuardrailResult(
        passed=True,
        check_name="max_drawdown",
        message=f"Max drawdown {dd:.1%} within limit",
        severity="warn",
    )


def check_returns_distribution(returns: np.ndarray) -> GuardrailResult:
    """WARN: Flag extreme kurtosis suggesting data quality issues."""
    returns = np.asarray(returns)
    if len(returns) < 10:
        return GuardrailResult(
            passed=True,
            check_name="returns_distribution",
            message="Too few returns to check distribution",
            severity="warn",
        )
    # Use excess kurtosis (normal = 0)
    kurt = float(scipy_kurtosis(returns, fisher=True))
    if kurt > 20:
        return GuardrailResult(
            passed=False,
            check_name="returns_distribution",
            message=f"Excess kurtosis {kurt:.1f} > 20 — fat tails suggest data issues",
            severity="warn",
        )
    return GuardrailResult(
        passed=True,
        check_name="returns_distribution",
        message=f"Excess kurtosis {kurt:.1f}",
        severity="warn",
    )


def check_consistency(metrics: dict, max_rolling_std: float = 1.5) -> GuardrailResult:
    """WARN: Flag inconsistent performance across time."""
    rolling_std = metrics.get("rolling_sharpe_std", None)
    if rolling_std is None or np.isnan(rolling_std):
        return GuardrailResult(
            passed=True,
            check_name="consistency",
            message="Rolling Sharpe std not available",
            severity="warn",
        )
    if rolling_std > max_rolling_std:
        return GuardrailResult(
            passed=False,
            check_name="consistency",
            message=f"Rolling Sharpe std {rolling_std:.3f} > {max_rolling_std} — inconsistent performance",
            severity="warn",
        )
    return GuardrailResult(
        passed=True,
        check_name="consistency",
        message=f"Rolling Sharpe std {rolling_std:.3f}",
        severity="warn",
    )


# === ORCHESTRATORS ===


def run_pre_checks(
    data: pd.DataFrame, config: dict, asset: str = "btc", min_samples: int = 2000
) -> list[GuardrailResult]:
    """Run all pre-experiment checks.

    Args:
        data: Training data DataFrame with DatetimeIndex.
        config: Experiment config dict. Expected keys:
            - features: list of feature column names
            - target: target column name (default: "target_1h")
            - transaction_costs_bps: costs in basis points
        asset: Asset for holdout boundary check.
        min_samples: Minimum sample count. Default 2000 (hourly).
            Use 500 for daily data.

    Returns:
        List of GuardrailResult objects.
    """
    return [
        check_holdout_boundary(data, asset),
        check_minimum_samples(data, min_samples=min_samples),
        check_no_lookahead(data, config),
        check_costs_specified(config),
        check_param_data_ratio(config, data),
    ]


def run_post_checks(
    returns: np.ndarray,
    metrics: dict,
    config: dict,
    n_trades: Optional[int] = None,
) -> list[GuardrailResult]:
    """Run all post-experiment checks.

    Args:
        returns: Array of strategy returns.
        metrics: Dict from compute_all_metrics(). DSR should already be computed.
        config: Experiment config dict.
        n_trades: Optional explicit trade count (overrides heuristic detection).

    Returns:
        List of GuardrailResult objects.
    """
    return [
        check_sharpe_sanity(metrics),
        check_minimum_trades(returns, config, n_trades=n_trades),
        check_dsr_threshold(metrics),
        check_max_drawdown(metrics),
        check_returns_distribution(returns),
        check_consistency(metrics),
    ]


def has_blocking_failure(results: list[GuardrailResult]) -> bool:
    """Check if any result has severity='block' and passed=False."""
    return any(not r.passed and r.severity == "block" for r in results)


def log_results(
    results: list[GuardrailResult],
    run_id: str,
    logfile: str = "results/guardrail_log.jsonl",
) -> None:
    """Append guardrail results to JSONL log file.

    Args:
        results: List of GuardrailResult objects.
        run_id: Identifier for this experiment run.
        logfile: Path to JSONL log file.
    """
    logpath = Path(logfile)
    logpath.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "checks": [asdict(r) for r in results],
        "has_blocking_failure": has_blocking_failure(results),
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "blocked": sum(1 for r in results if not r.passed and r.severity == "block"),
        },
    }

    with open(logpath, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    logger.info(
        f"[GUARDRAILS] {entry['summary']['passed']}/{entry['summary']['total']} checks passed "
        f"(run_id={run_id}, blocked={entry['summary']['blocked']})"
    )
