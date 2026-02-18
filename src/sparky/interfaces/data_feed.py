"""Data feed protocol for dataset loading."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataFeedProtocol(Protocol):
    """Protocol for data loading with holdout enforcement.

    Any callable or class that provides load() and list_datasets()
    satisfies this protocol.
    """

    def load(self, dataset: str, purpose: str = "training") -> pd.DataFrame:
        """Load a dataset with holdout enforcement.

        Args:
            dataset: Dataset name or path.
            purpose: "training", "validation", or "analysis".

        Returns:
            DataFrame with DatetimeIndex.
        """
        ...

    def list_datasets(self) -> list[str]:
        """List available dataset names.

        Returns:
            List of dataset name strings.
        """
        ...
