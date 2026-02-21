import numpy as np
import pandas as pd


def regime_attribution(
    returns: pd.Series,
    positions: pd.Series,
    regime_labels: pd.Series,
    periods_per_year: int = 365,
) -> pd.DataFrame:
    """Break down strategy performance by market regime."""
    idx = returns.index.intersection(positions.index).intersection(regime_labels.index)
    r, p, reg = returns.loc[idx], positions.loc[idx], regime_labels.loc[idx]
    strat_ret = r * p.shift(1).fillna(0)
    total_pnl = strat_ret.sum()

    rows = []
    for regime in sorted(reg.unique()):
        mask = reg == regime
        sr = strat_ret[mask]
        n = int(mask.sum())
        mean_r = sr.mean()
        std_r = sr.std(ddof=1)
        sharpe = float(mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 0 else 0.0
        rows.append(
            {
                "regime": regime,
                "sharpe": sharpe,
                "total_return": float(sr.sum()),
                "frac_time": n / len(idx),
                "frac_pnl": float(sr.sum() / total_pnl) if total_pnl != 0 else 0.0,
                "n_obs": n,
            }
        )
    return pd.DataFrame(rows)


def signal_contribution(
    returns: pd.Series,
    signal_dict: dict[str, pd.Series],
    positions: pd.Series,
) -> pd.DataFrame:
    """Measure each signal's correlation with returns and position."""
    idx = returns.index.intersection(positions.index)
    for s in signal_dict.values():
        idx = idx.intersection(s.index)

    r = returns.loc[idx]
    p = positions.loc[idx]
    strat_ret = r * p.shift(1).fillna(0)
    forward_r = r.shift(-1)

    rows = []
    for name, sig in signal_dict.items():
        s = sig.loc[idx]
        corr_ret = float(np.corrcoef(s.values, strat_ret.values)[0, 1]) if s.std() > 0 else 0.0
        corr_pos = float(np.corrcoef(s.values, p.values)[0, 1]) if s.std() > 0 else 0.0
        valid = s.index[forward_r.notna()]
        marginal_ic = float(s.loc[valid].rank().corr(forward_r.loc[valid].rank()))
        rows.append(
            {
                "signal": name,
                "corr_with_returns": corr_ret,
                "corr_with_position": corr_pos,
                "marginal_ic": marginal_ic,
            }
        )
    return pd.DataFrame(rows)


def temporal_stability(
    returns: pd.Series,
    positions: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """Compute rolling performance metrics for temporal stability analysis."""
    idx = returns.index.intersection(positions.index)
    r, p = returns.loc[idx], positions.loc[idx]
    strat_ret = r * p.shift(1).fillna(0)

    rolling_ret = strat_ret.rolling(window).mean() * window
    rolling_vol = strat_ret.rolling(window).std(ddof=1) * np.sqrt(window)
    rolling_sharpe = rolling_ret / rolling_vol

    df = pd.DataFrame(
        {
            "date": idx,
            "rolling_sharpe": rolling_sharpe.values,
            "rolling_return": rolling_ret.values,
            "rolling_vol": rolling_vol.values,
        }
    ).dropna()
    return df
