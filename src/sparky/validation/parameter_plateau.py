from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class PlateauResult:
    best_sharpe: float
    plateau_lower: float  # best * (1 - threshold_frac)
    plateau_upper: float  # best
    n_in_plateau: int
    n_total: int
    coverage: float  # n_in_plateau / n_total
    passed: bool  # coverage >= min_coverage


def parameter_plateau_test(
    sharpe_by_config: pd.DataFrame,  # columns: param cols + 'sharpe'
    threshold_frac: float = 0.30,
    min_coverage: float = 0.50,
) -> PlateauResult:
    """Test whether Sharpe values form a plateau (robust) or a spike (fragile)."""
    sharpes = sharpe_by_config["sharpe"].values
    if len(sharpes) == 0:
        raise ValueError("sharpe_by_config is empty — no configurations to evaluate")
    best = float(np.max(sharpes))
    lower = best - abs(best) * threshold_frac
    n_in = int(np.sum(sharpes >= lower))
    n_total = len(sharpes)
    coverage = n_in / n_total if n_total > 0 else 0.0
    return PlateauResult(
        best_sharpe=best,
        plateau_lower=lower,
        plateau_upper=best,
        n_in_plateau=n_in,
        n_total=n_total,
        coverage=coverage,
        passed=coverage >= min_coverage,
    )


def parameter_sensitivity_1d(
    sharpe_by_config: pd.DataFrame,
    param_name: str,
) -> dict:
    """Analyze sensitivity of Sharpe to a single parameter."""
    grouped = sharpe_by_config.groupby(param_name)["sharpe"].mean().sort_index()
    values = grouped.index.tolist()
    mean_sharpes = grouped.values.tolist()

    if len(mean_sharpes) < 2:
        return {
            "param_values": values,
            "mean_sharpes": mean_sharpes,
            "is_monotonic": True,
            "max_gradient": 0.0,
        }

    diffs = np.diff(mean_sharpes)
    param_steps = np.diff(np.array(values, dtype=float))
    is_mono = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    nonzero_steps = np.where(param_steps != 0, param_steps, 1.0)
    max_grad = float(np.max(np.abs(diffs / nonzero_steps)))

    return {
        "param_values": values,
        "mean_sharpes": mean_sharpes,
        "is_monotonic": is_mono,
        "max_gradient": max_grad,
    }
