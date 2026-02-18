"""Feature pipeline protocol for feature engineering."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FeaturePipelineProtocol(Protocol):
    """Protocol for feature engineering pipelines.

    Defines the interface for transforming raw data into feature
    matrices suitable for model training.
    """

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into features.

        Args:
            data: Raw OHLCV or enriched DataFrame.

        Returns:
            DataFrame with computed features.
        """
        ...

    def get_feature_names(self) -> list[str]:
        """Return the list of feature names produced by this pipeline.

        Returns:
            List of feature column names.
        """
        ...
