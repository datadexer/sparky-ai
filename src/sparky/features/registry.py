"""Feature registry and matrix builder.

Central catalog of all features with metadata. Handles temporal alignment,
valid_from dates (critical for ETH features), and feature matrix construction.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """Metadata for a registered feature."""

    name: str
    category: str  # "technical", "onchain_btc", "onchain_eth", "market_context"
    compute_fn: Callable  # Function that computes the feature
    input_columns: list[str]  # Column names needed from the data
    lookback: int = 0  # Lookback window in days
    data_source: str = ""  # e.g., "binance", "bgeometrics", "coinmetrics"
    expected_range: Optional[tuple[float, float]] = None
    valid_from: Optional[str] = None  # "YYYY-MM-DD" — NaN before this date
    description: str = ""
    asset: str = "all"  # "btc", "eth", or "all"


class FeatureRegistry:
    """Central feature catalog with metadata and matrix builder.

    Usage:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(
            name="rsi_14",
            category="technical",
            compute_fn=lambda df: rsi(df["close"], 14),
            input_columns=["close"],
            lookback=14,
        ))
        matrix = registry.build_feature_matrix(
            asset="btc", feature_names=["rsi_14"], data={"price": price_df}
        )
    """

    def __init__(self):
        self._features: dict[str, FeatureDefinition] = {}

    def register(self, feature_def: FeatureDefinition) -> None:
        """Register a feature definition."""
        if feature_def.name in self._features:
            logger.warning(f"Overwriting existing feature: {feature_def.name}")
        self._features[feature_def.name] = feature_def

    def get(self, name: str) -> FeatureDefinition:
        """Get a feature definition by name."""
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not registered. Available: {list(self._features.keys())}")
        return self._features[name]

    def list_features(
        self,
        category: Optional[str] = None,
        asset: Optional[str] = None,
    ) -> list[str]:
        """List registered feature names, optionally filtered."""
        features = self._features.values()
        if category:
            features = [f for f in features if f.category == category]
        if asset:
            features = [f for f in features if f.asset in (asset, "all")]
        return [f.name for f in features]

    def build_feature_matrix(
        self,
        asset: str,
        feature_names: list[str],
        data: dict[str, pd.DataFrame],
        drop_na_rows: bool = True,
    ) -> pd.DataFrame:
        """Build a feature matrix from registered features.

        Args:
            asset: "btc" or "eth".
            feature_names: List of feature names to compute.
            data: Dict of DataFrames keyed by source name (e.g., "price", "onchain").
                  Each should have a DatetimeIndex.
            drop_na_rows: If True, drop rows where ALL features are NaN
                          (from lookback periods). Rows with some NaN are kept
                          (features with different valid_from dates).

        Returns:
            DataFrame with DatetimeIndex and one column per feature.
        """
        result = {}
        active_features = []
        skipped_features = []

        for name in feature_names:
            feat_def = self.get(name)

            # Check asset compatibility
            if feat_def.asset not in (asset, "all"):
                skipped_features.append((name, f"not applicable for {asset}"))
                continue

            try:
                # Compute feature
                series = feat_def.compute_fn(data)

                if not isinstance(series, pd.Series):
                    skipped_features.append((name, "compute_fn did not return a Series"))
                    continue

                # Apply valid_from mask
                if feat_def.valid_from:
                    valid_from_ts = pd.Timestamp(feat_def.valid_from, tz="UTC")
                    # Ensure series index is tz-aware for comparison
                    if series.index.tz is None:
                        series.index = series.index.tz_localize("UTC")
                    series = series.copy()
                    series[series.index < valid_from_ts] = np.nan

                result[name] = series
                active_features.append(name)

            except Exception as e:
                skipped_features.append((name, str(e)))
                logger.warning(f"Failed to compute feature '{name}': {e}")

        if not result:
            logger.warning("No features computed successfully")
            return pd.DataFrame()

        matrix = pd.DataFrame(result)
        matrix = matrix.sort_index()

        # Drop rows where ALL features are NaN (pure lookback/pre-valid rows)
        if drop_na_rows:
            before = len(matrix)
            matrix = matrix.dropna(how="all")
            if len(matrix) < before:
                logger.info(
                    f"Dropped {before - len(matrix)} all-NaN rows (lookback periods)"
                )

        if skipped_features:
            for name, reason in skipped_features:
                logger.info(f"Skipped feature '{name}': {reason}")

        logger.info(
            f"Feature matrix for {asset}: {len(matrix)} rows, "
            f"{len(active_features)} features active, "
            f"{len(skipped_features)} skipped"
        )
        return matrix

    @property
    def feature_count(self) -> int:
        """Number of registered features."""
        return len(self._features)
