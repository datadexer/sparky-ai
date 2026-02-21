"""GARCH volatility modeling and forecasting."""

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "fit_garch",
    "rolling_garch_forecast",
    "garch_parameter_stability",
    "ewma_volatility",
]


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1):
    """Fit GARCH(p,q) model to returns. Returns arch.ARCHModelResult."""
    from arch import arch_model

    scaled = returns * 100
    model = arch_model(scaled, vol="Garch", p=p, q=q, dist="Normal", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp="off")
    return result


def rolling_garch_forecast(
    returns: pd.Series,
    window: int = 252,
    p: int = 1,
    q: int = 1,
    refit_every: int = 21,
) -> pd.Series:
    """Rolling one-step-ahead GARCH volatility forecast.

    Refits every `refit_every` days. Falls back to EWMA on convergence failure.
    Output is annualized volatility.
    """
    from arch import arch_model

    n = len(returns)
    forecasts = pd.Series(np.nan, index=returns.index)

    last_result = None
    periods_since_refit = refit_every  # force initial fit

    for i in range(window, n):
        if periods_since_refit >= refit_every:
            train = returns.iloc[i - window : i] * 100
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = arch_model(train, vol="Garch", p=p, q=q, dist="Normal", rescale=False)
                    last_result = model.fit(disp="off")
                periods_since_refit = 0
            except Exception:
                logger.warning("[GARCH] Convergence failure at index %d, using EWMA fallback", i)
                last_result = None

        if last_result is not None:
            try:
                fcast = last_result.forecast(horizon=1, reindex=False)
                daily_vol = np.sqrt(fcast.variance.values[-1, 0]) / 100
                forecasts.iloc[i] = daily_vol * np.sqrt(365)
            except Exception:
                forecasts.iloc[i] = returns.iloc[i - window : i].ewm(span=30).std().iloc[-1] * np.sqrt(365)
        else:
            forecasts.iloc[i] = returns.iloc[i - window : i].ewm(span=30).std().iloc[-1] * np.sqrt(365)

        periods_since_refit += 1

    return forecasts


def garch_parameter_stability(
    returns: pd.Series,
    window: int = 252,
    step: int = 63,
) -> pd.DataFrame:
    """Sliding window GARCH(1,1) parameter estimates.

    Returns DataFrame with columns: omega, alpha, beta, persistence.
    """
    from arch import arch_model

    records = []
    for i in range(window, len(returns), step):
        train = returns.iloc[i - window : i] * 100
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(train, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
                result = model.fit(disp="off")
                params = result.params
                records.append(
                    {
                        "date": returns.index[i],
                        "omega": params.get("omega", np.nan),
                        "alpha": params.get("alpha[1]", np.nan),
                        "beta": params.get("beta[1]", np.nan),
                        "persistence": params.get("alpha[1]", 0) + params.get("beta[1]", 0),
                    }
                )
        except Exception:
            records.append(
                {
                    "date": returns.index[i],
                    "omega": np.nan,
                    "alpha": np.nan,
                    "beta": np.nan,
                    "persistence": np.nan,
                }
            )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("date")


def ewma_volatility(returns: pd.Series, span: int = 30) -> pd.Series:
    """Exponentially weighted moving average volatility."""
    return returns.ewm(span=span).std()
