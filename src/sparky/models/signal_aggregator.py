"""Signal aggregation: convert hourly model predictions to daily trading signals.

The 1h XGBoost model predicts P(up) for each hour. This module aggregates
24 hourly predictions into a daily confidence score for trading decisions.

Aggregation methods:
1. Mean: daily_signal = mean(P(up) for 24 hours) > threshold
2. Weighted: exponentially weight recent hours more
3. Regime-aware: separate thresholds for high/low volatility periods

Usage:
    aggregator = HourlyToDailyAggregator(method="mean", threshold=0.5)
    daily_signals = aggregator.aggregate(hourly_probas, hourly_features)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HourlyToDailyAggregator:
    """Aggregate hourly P(up) predictions into daily trading signals.

    Args:
        method: Aggregation method ("mean", "weighted", or "regime").
        threshold: Probability threshold for LONG signal (default 0.5).
        ema_halflife: Half-life in hours for weighted method (default 6).
    """

    def __init__(
        self,
        method: str = "mean",
        threshold: float = 0.5,
        ema_halflife: int = 6,
    ):
        if method not in ("mean", "weighted", "regime"):
            raise ValueError(f"Unknown method: {method}. Use 'mean', 'weighted', or 'regime'.")
        self.method = method
        self.threshold = threshold
        self.ema_halflife = ema_halflife

    def aggregate(
        self,
        hourly_probas: pd.Series,
        hourly_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Aggregate hourly probabilities to daily signals.

        Args:
            hourly_probas: Series of P(up) indexed by hourly timestamps.
            hourly_features: Optional hourly features (needed for 'regime' method).

        Returns:
            DataFrame with columns: ['daily_proba', 'signal', 'n_hours', 'std']
            indexed by date.
        """
        if self.method == "mean":
            return self._aggregate_mean(hourly_probas)
        elif self.method == "weighted":
            return self._aggregate_weighted(hourly_probas)
        elif self.method == "regime":
            if hourly_features is None:
                raise ValueError("hourly_features required for 'regime' method")
            return self._aggregate_regime(hourly_probas, hourly_features)

    def _aggregate_mean(self, hourly_probas: pd.Series) -> pd.DataFrame:
        """Simple mean of 24 hourly P(up) predictions.

        daily_signal = mean(P(up) for all hours in day) > threshold → LONG
        """
        daily = hourly_probas.resample("D").agg(["mean", "std", "count"])
        daily.columns = ["daily_proba", "std", "n_hours"]

        # Only generate signals for days with >= 20 hours of data
        daily["signal"] = 0
        valid = daily["n_hours"] >= 20
        daily.loc[valid, "signal"] = (daily.loc[valid, "daily_proba"] > self.threshold).astype(int)

        # Ensure timezone-aware (UTC)
        if daily.index.tz is None:
            daily.index = daily.index.tz_localize("UTC")

        logger.info(
            f"Aggregated {len(hourly_probas):,} hourly predictions to {len(daily)} daily signals "
            f"(method=mean, threshold={self.threshold})"
        )
        return daily

    def _aggregate_weighted(self, hourly_probas: pd.Series) -> pd.DataFrame:
        """Exponentially weighted mean — recent hours count more.

        Uses EMA with configurable half-life to weight recent predictions higher.
        """
        # Group by day, apply exponential weights within each day
        results = []
        for date, group in hourly_probas.groupby(hourly_probas.index.date):
            if len(group) < 20:
                continue
            # Exponential weights: most recent hour gets highest weight
            n = len(group)
            decay = np.log(2) / self.ema_halflife
            weights = np.exp(decay * np.arange(n))
            weights /= weights.sum()

            weighted_proba = np.sum(weights * group.values)
            results.append(
                {
                    "date": pd.Timestamp(date),
                    "daily_proba": weighted_proba,
                    "std": group.std(),
                    "n_hours": n,
                    "signal": int(weighted_proba > self.threshold),
                }
            )

        daily = pd.DataFrame(results).set_index("date")
        logger.info(
            f"Aggregated {len(hourly_probas):,} hourly predictions to {len(daily)} daily signals "
            f"(method=weighted, halflife={self.ema_halflife}h)"
        )
        return daily

    def _aggregate_regime(
        self,
        hourly_probas: pd.Series,
        hourly_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Regime-aware aggregation — adjust threshold by volatility.

        In high-volatility regimes, require stronger conviction (higher threshold).
        In low-volatility regimes, use standard threshold.
        """
        # Get daily volatility from hourly features
        if "realized_vol_24h" in hourly_features.columns:
            daily_vol = hourly_features["realized_vol_24h"].resample("D").last()
        else:
            daily_vol = hourly_features.iloc[:, 0].resample("D").std()

        # Median volatility as regime boundary
        vol_median = daily_vol.median()

        # Base aggregation
        daily = self._aggregate_mean(hourly_probas)

        # Adjust threshold: high vol → threshold + 0.02, low vol → threshold - 0.01
        if daily_vol.index.tz is None:
            daily_vol.index = daily_vol.index.tz_localize("UTC")

        daily_vol_aligned = daily_vol.reindex(daily.index)
        high_vol = daily_vol_aligned > vol_median

        # Re-compute signals with regime-adjusted thresholds
        daily["signal"] = 0
        valid = daily["n_hours"] >= 20

        high_vol_mask = valid & high_vol.fillna(False)
        low_vol_mask = valid & ~high_vol.fillna(False)

        daily.loc[high_vol_mask, "signal"] = (daily.loc[high_vol_mask, "daily_proba"] > self.threshold + 0.02).astype(
            int
        )
        daily.loc[low_vol_mask, "signal"] = (daily.loc[low_vol_mask, "daily_proba"] > self.threshold - 0.01).astype(int)

        logger.info(
            f"Aggregated with regime adjustment: "
            f"high_vol_threshold={self.threshold + 0.02}, "
            f"low_vol_threshold={self.threshold - 0.01}"
        )
        return daily


class RegimeAwareAggregator:
    """Research-validated regime-aware signal aggregator.

    Implements regime-aware position sizing and dynamic thresholds based on
    volatility regimes (LOW/MEDIUM/HIGH). Research shows this approach achieves
    Sharpe 0.829 (IMCA 2025) vs static models.

    Regime rules:
    - HIGH (>60% vol): 50% position, threshold 0.55
    - MEDIUM (30-60%): 75% position, threshold 0.52
    - LOW (<30%): 100% position, threshold 0.50

    Args:
        regime_window: Window for volatility regime computation (default 30 days * 24 hours).
        frequency: Data frequency ("1h" or "1d") for annualization.
    """

    REGIME_RULES = {
        "high": {"position_size": 0.50, "threshold": 0.55},
        "medium": {"position_size": 0.75, "threshold": 0.52},
        "low": {"position_size": 1.00, "threshold": 0.50},
    }

    def __init__(
        self,
        regime_window: int = 30 * 24,
        frequency: str = "1h",
    ):
        self.regime_window = regime_window
        self.frequency = frequency

    def aggregate_to_daily(
        self,
        hourly_probas: pd.Series,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """Aggregate hourly predictions to daily signals with regime awareness.

        Args:
            hourly_probas: Series of P(up) indexed by hourly timestamps.
            prices: Hourly close prices for regime computation.

        Returns:
            DataFrame with columns:
            - date: trading date
            - daily_proba: mean probability for the day
            - regime: volatility regime (low/medium/high)
            - threshold: regime-specific probability threshold
            - signal: 1 (LONG) or 0 (FLAT)
            - position_size: regime-specific position size (0.5/0.75/1.0)
            - n_hours: number of hourly predictions that day
        """
        from sparky.features.regime_indicators import compute_volatility_regime

        # Compute volatility regime
        regimes = compute_volatility_regime(prices, window=self.regime_window, frequency=self.frequency)

        # Aggregate hourly probabilities to daily
        daily_proba = hourly_probas.resample("D").mean()
        daily_std = hourly_probas.resample("D").std()
        daily_n_hours = hourly_probas.resample("D").count()

        # Get daily regime (most common regime that day)
        daily_regime = regimes.resample("D").agg(lambda x: x.mode()[0] if len(x) > 0 else "medium")

        # Ensure timezone-aware (UTC)
        if daily_proba.index.tz is None:
            daily_proba.index = daily_proba.index.tz_localize("UTC")
            daily_std.index = daily_std.index.tz_localize("UTC")
            daily_n_hours.index = daily_n_hours.index.tz_localize("UTC")
        if daily_regime.index.tz is None:
            daily_regime.index = daily_regime.index.tz_localize("UTC")

        # Generate regime-aware signals
        results = []
        for date in daily_proba.index:
            if daily_n_hours[date] < 20:
                # Skip days with insufficient data
                continue

            prob = daily_proba[date]
            regime = daily_regime[date]
            rules = self.REGIME_RULES[regime]

            threshold = rules["threshold"]
            signal = 1 if prob > threshold else 0
            position_size = rules["position_size"] if signal == 1 else 0.0

            results.append(
                {
                    "date": date,
                    "daily_proba": prob,
                    "std": daily_std[date],
                    "regime": regime,
                    "threshold": threshold,
                    "signal": signal,
                    "position_size": position_size,
                    "n_hours": daily_n_hours[date],
                }
            )

        df = pd.DataFrame(results).set_index("date")

        n_long = (df["signal"] == 1).sum()
        regime_counts = df["regime"].value_counts()

        logger.info(f"Aggregated {len(hourly_probas):,} hourly predictions to {len(df)} daily signals (regime-aware)")
        logger.info(
            f"  Signals: {n_long} LONG ({n_long / len(df) * 100:.1f}%), "
            f"{len(df) - n_long} FLAT ({(len(df) - n_long) / len(df) * 100:.1f}%)"
        )
        logger.info(
            f"  Regimes: low={regime_counts.get('low', 0)}, "
            f"medium={regime_counts.get('medium', 0)}, "
            f"high={regime_counts.get('high', 0)}"
        )

        return df
